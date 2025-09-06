#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static __global__ void higgs_dequantize_2_256_kernel(
	const uint8_t* __restrict__ x,
	const uint32_t* __restrict__ grid,
	uint32_t* __restrict__ out,
	long long out_dim) {
	__shared__ uint32_t s_grid[256];

	for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
		s_grid[idx] = grid[idx];
	}
	__syncthreads();

	long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= out_dim) return;

	uint8_t code = x[i];
	out[i] = s_grid[code];
}

extern "C" void launch_higgs_dequantize_2_256_kernel(
	const uint8_t* x,
	const uint32_t* grid,
	uint32_t* out,
	int64_t out_dim,
	cudaStream_t stream) {
	
	constexpr int threads_per_block = 256;
	int blocks = static_cast<int>((out_dim + threads_per_block - 1) / threads_per_block);

	higgs_dequantize_2_256_kernel<<<blocks, threads_per_block, 0, stream>>>(
		x, grid, out, static_cast<long long>(out_dim));
}

extern "C" void new_launch_higgs_dequantize_2_256_kernel(
	const uint8_t* x,
	const uint32_t* grid,
	uint32_t* out,
	int64_t out_dim,
	cudaStream_t stream) {
	
	constexpr int threads_per_block = 256;
	int blocks = static_cast<int>((out_dim + threads_per_block - 1) / threads_per_block);

	higgs_dequantize_2_256_kernel<<<blocks, threads_per_block, 0, stream>>>(
		x, grid, out, static_cast<long long>(out_dim));
}