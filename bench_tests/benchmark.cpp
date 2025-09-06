#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include "dequant_kernel.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

void generate_random_data(std::vector<uint8_t>& quantized_data, 
                         std::vector<__half>& grid_data,
                         int out_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> quant_dist(0, 255);
    std::uniform_real_distribution<float> grid_dist(-1.0f, 1.0f);
    
    // Generate random quantized data
    quantized_data.resize(out_dim);
    for (int i = 0; i < out_dim; ++i) {
        quantized_data[i] = quant_dist(gen);
    }
    
    // Generate random float16 grid (256, 2)
    grid_data.resize(256 * 2);
    for (int i = 0; i < 256 * 2; ++i) {
        grid_data[i] = __float2half(grid_dist(gen));
    }
}

double benchmark_kernel(const uint8_t* d_x, 
                       const uint32_t* d_grid,
                       uint32_t* d_out,
                       int64_t out_dim,
                       cudaStream_t stream,
                       void (*kernel_func)(const uint8_t*, const uint32_t*, uint32_t*, int64_t, cudaStream_t),
                       int warmup_runs = 10,
                       int benchmark_runs = 500) {
    
    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        kernel_func(d_x, d_grid, d_out, out_dim, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    // Benchmark runs
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        kernel_func(d_x, d_grid, d_out, out_dim, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return duration.count() / 1000.0 / benchmark_runs; // Return average time in milliseconds
}

int main() {
    std::cout << "Higgs Dequantization Kernel Benchmark" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Test different output dimensions (up to 1024 * 1024 * 64 = 67,108,864)
    // 2x more points with same range
    std::vector<int> test_sizes = {
        1024,           // 1K
        2048,           // 2K
        4096,           // 4K
        8192,           // 8K
        16384,          // 16K
        32768,          // 32K
        65536,          // 64K
        131072,         // 128K
        262144,         // 256K
        524288,         // 512K
        1048576,        // 1M
        2097152,        // 2M
        4194304,        // 4M
        8388608,        // 8M
        16777216,       // 16M
        33554432,       // 32M
        67108864        // 64M (1024 * 1024 * 64)
    };
    
    for (int out_dim : test_sizes) {
        std::cout << "\nTesting with output dimension: " << out_dim << std::endl;
        
        // Generate random data
        std::vector<uint8_t> h_x;
        std::vector<__half> h_grid;
        generate_random_data(h_x, h_grid, out_dim);
        
        // Allocate GPU memory
        uint8_t* d_x;
        uint32_t* d_grid;
        uint32_t* d_out;
        
        CUDA_CHECK(cudaMalloc(&d_x, out_dim * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_grid, 256 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_out, out_dim * sizeof(uint32_t)));
        
        // Copy data to GPU (reinterpret h_grid as uint32_t*)
        CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), out_dim * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid, reinterpret_cast<const uint32_t*>(h_grid.data()), 256 * sizeof(uint32_t), cudaMemcpyHostToDevice));
        
        // Create stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        
        // Run benchmark for original kernel
        double avg_time_ms_orig = benchmark_kernel(d_x, d_grid, d_out, out_dim, stream, launch_higgs_dequantize_2_256_kernel);
        
        // Run benchmark for new kernel
        double avg_time_ms_new = benchmark_kernel(d_x, d_grid, d_out, out_dim, stream, new_launch_higgs_dequantize_2_256_kernel);
        
        // Calculate throughput for original kernel
        double elements_per_second_orig = (out_dim / avg_time_ms_orig) * 1000.0;
        double gb_per_second_orig = (elements_per_second_orig * sizeof(uint8_t)) / (1024.0 * 1024.0 * 1024.0);
        
        // Calculate throughput for new kernel
        double elements_per_second_new = (out_dim / avg_time_ms_new) * 1000.0;
        double gb_per_second_new = (elements_per_second_new * sizeof(uint8_t)) / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << "  Original kernel:" << std::endl;
        std::cout << "    Average time: " << avg_time_ms_orig << " ms" << std::endl;
        std::cout << "    Throughput: " << elements_per_second_orig / 1e6 << " M elements/sec" << std::endl;
        std::cout << "    Bandwidth: " << gb_per_second_orig << " GB/s" << std::endl;
        
        std::cout << "  New kernel:" << std::endl;
        std::cout << "    Average time: " << avg_time_ms_new << " ms" << std::endl;
        std::cout << "    Throughput: " << elements_per_second_new / 1e6 << " M elements/sec" << std::endl;
        std::cout << "    Bandwidth: " << gb_per_second_new << " GB/s" << std::endl;
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_grid));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
    std::cout << "\nBenchmark completed!" << std::endl;
    return 0;
}