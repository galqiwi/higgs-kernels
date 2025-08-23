#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

static __global__ void higgs_dequantize_2_256_ptr_cuda_portable_kernel(
	const uint8_t* __restrict__ x,
	const uint32_t* __restrict__ grid_packed,
	uint32_t* __restrict__ out_packed,
	long long out_dim) {
	__shared__ uint32_t s_grid[256];

	for (int idx = threadIdx.x; idx < 256; idx += blockDim.x) {
		s_grid[idx] = grid_packed[idx];
	}
	__syncthreads();

	long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i >= out_dim) return;

	uint8_t code = x[i];
	out_packed[i] = s_grid[code];
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
	int blocks = static_cast<int>((out_dim + threads_per_block - 1) / threads_per_block);

	auto stream = at::cuda::getCurrentCUDAStream();
	higgs_dequantize_2_256_ptr_cuda_portable_kernel<<<blocks, threads_per_block, 0, stream.stream()>>>(
		x, grid_packed, out_packed, static_cast<long long>(out_dim));

	C10_CUDA_KERNEL_LAUNCH_CHECK();
}
