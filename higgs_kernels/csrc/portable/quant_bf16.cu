#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

static constexpr int kCodebookSize = 256;
static constexpr int kInDim = 2;

__device__ __forceinline__ float bf16_to_fp32(uint16_t v) {
	uint32_t u = static_cast<uint32_t>(v) << 16;
	return __uint_as_float(u);
}
__device__ __forceinline__ uint16_t fp32_to_bf16_rne(float f) {
	uint32_t x = __float_as_uint(f);
	uint32_t lsb = (x >> 16) & 1U;
	uint32_t rounding_bias = 0x00007FFFU + lsb;
	x += rounding_bias;
	return static_cast<uint16_t>(x >> 16);
}

__device__ __forceinline__ uint16_t bf16_mul(uint16_t a_bf16, uint16_t b_bf16) {
	float a = bf16_to_fp32(a_bf16);
	float b = bf16_to_fp32(b_bf16);
	return fp32_to_bf16_rne(a * b);
}

__global__ void higgs_quantize_2_256_ptr_bf16_cuda_portable_kernel(
	const uint16_t* __restrict__ x,
	const uint16_t* __restrict__ grid,
	const uint16_t* __restrict__ grid_norms,
	unsigned char* __restrict__ out,
	int64_t out_dim)
{
	__shared__ float s_grid[kCodebookSize][kInDim];
	__shared__ uint16_t s_norms_bf16[kCodebookSize];

	for (int idx = threadIdx.x; idx < kCodebookSize * kInDim; idx += blockDim.x) {
		int r = idx / kInDim;
		int c = idx % kInDim;
		s_grid[r][c] = bf16_to_fp32(grid[r * kInDim + c]);
	}
	for (int idx = threadIdx.x; idx < kCodebookSize; idx += blockDim.x) {
		s_norms_bf16[idx] = grid_norms[idx];
	}
	__syncthreads();

	int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= out_dim) return;

	const uint16_t two_bf16 = fp32_to_bf16_rne(2.0f);

	uint16_t x0_b = x[row * kInDim + 0];
	uint16_t x1_b = x[row * kInDim + 1];
	float x0 = bf16_to_fp32(x0_b);
	float x1 = bf16_to_fp32(x1_b);

	float best_score = -CUDART_INF_F;
	unsigned int best_index = 0u;

	#pragma unroll 8
	for (int c = 0; c < kCodebookSize; ++c) {
		float g0 = s_grid[c][0];
		float g1 = s_grid[c][1];
		float dot_fp32 = x0 * g0 + x1 * g1;
		uint16_t dot_bf16 = fp32_to_bf16_rne(dot_fp32);

		uint16_t twice_dot_bf16 = bf16_mul(dot_bf16, two_bf16);
		float twice_dot = bf16_to_fp32(twice_dot_bf16);

		uint16_t grid_norm_bf16 = s_norms_bf16[c];
		float score = bf16_to_fp32(fp32_to_bf16_rne(twice_dot - bf16_to_fp32(grid_norm_bf16)));

		if (score > best_score) {
			best_score = score;
			best_index = static_cast<unsigned int>(c);
		}
	}

	out[row] = static_cast<unsigned char>(best_index);
}

extern "C" void higgs_quantize_2_256_ptr_bf16_cuda_portable(
	uint64_t x_ptr,
	uint64_t grid_ptr,
	uint64_t grid_norms_ptr,
	uint64_t out_ptr,
	int64_t out_dim)
{
	const uint16_t* x = reinterpret_cast<const uint16_t*>(x_ptr);
	const uint16_t* grid = reinterpret_cast<const uint16_t*>(grid_ptr);
	const uint16_t* grid_norms = reinterpret_cast<const uint16_t*>(grid_norms_ptr);
	unsigned char* out = reinterpret_cast<unsigned char*>(out_ptr);

	int threads = 256;
	int blocks = static_cast<int>((out_dim + threads - 1) / threads);

	auto stream = at::cuda::getCurrentCUDAStream();
	higgs_quantize_2_256_ptr_bf16_cuda_portable_kernel<<<blocks, threads, 0, stream>>>(x, grid, grid_norms, out, out_dim);

	C10_CUDA_KERNEL_LAUNCH_CHECK();
}