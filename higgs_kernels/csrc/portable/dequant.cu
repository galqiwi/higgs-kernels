#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// Each thread processes 4 indices: loads 4 bytes (uint32), stores 16 bytes (uint4)
static constexpr int ELEMENTS_PER_THREAD = 4;

static __global__ void higgs_dequantize_2_256_ptr_cuda_portable_kernel(
	const uint8_t* __restrict__ x,
	const uint32_t* __restrict__ grid_packed,
	uint32_t* __restrict__ out_packed,
	long long out_dim) {
	__shared__ uint32_t s_grid[256];

	// Load codebook to shared memory
	for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
		s_grid[idx] = grid_packed[idx];
	}
	__syncthreads();

	// Each thread processes 4 indices
	long long base_idx = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

	if (base_idx >= out_dim) return;

	// Check if we have a full 4 elements to process
	if (base_idx + ELEMENTS_PER_THREAD <= out_dim) {
		// Vectorized load: load 4 uint8 indices as uint32 (4 bytes)
		uint32_t indices_packed = *reinterpret_cast<const uint32_t*>(&x[base_idx]);

		// Extract individual bytes
		uint8_t idx0 = indices_packed & 0xFF;
		uint8_t idx1 = (indices_packed >> 8) & 0xFF;
		uint8_t idx2 = (indices_packed >> 16) & 0xFF;
		uint8_t idx3 = (indices_packed >> 24) & 0xFF;

		// Lookup all 4 values
		uint32_t val0 = s_grid[idx0];
		uint32_t val1 = s_grid[idx1];
		uint32_t val2 = s_grid[idx2];
		uint32_t val3 = s_grid[idx3];

		// Vectorized store: write 4 uint32 values as uint4 (16 bytes)
		uint4 result = make_uint4(val0, val1, val2, val3);
		*reinterpret_cast<uint4*>(&out_packed[base_idx]) = result;
	} else {
		// Handle remainder (less than 4 elements at the end)
		for (long long i = base_idx; i < out_dim; i++) {
			out_packed[i] = s_grid[x[i]];
		}
	}
}

extern "C" void higgs_dequantize_2_256_ptr_cuda_portable(
	uint64_t x_ptr,
	uint64_t grid_ptr,
	uint64_t out_ptr,
	int64_t out_dim) {
	const uint8_t* x = reinterpret_cast<const uint8_t*>(x_ptr);
	const uint32_t* grid_packed = reinterpret_cast<const uint32_t*>(grid_ptr);
	uint32_t* out_packed = reinterpret_cast<uint32_t*>(out_ptr);

	constexpr int threads_per_block = 256;
	constexpr int elements_per_block = threads_per_block * ELEMENTS_PER_THREAD;
	int blocks = static_cast<int>((out_dim + elements_per_block - 1) / elements_per_block);

	auto stream = at::cuda::getCurrentCUDAStream();
	higgs_dequantize_2_256_ptr_cuda_portable_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
		x, grid_packed, out_packed, static_cast<long long>(out_dim));

	C10_CUDA_KERNEL_LAUNCH_CHECK();
}
