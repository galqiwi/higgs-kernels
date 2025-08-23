#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

static constexpr int codebookSize = 256;
static constexpr int codebookDim = 2;

static __global__ void higgs_quantize_2_256_ptr_f16_cuda_portable_kernel(
	const __half* __restrict__ x,
	const __half* __restrict__ grid,
	const __half* __restrict__ grid_norms,
	uint8_t* __restrict__ out,
	int64_t out_dim)
{
	__shared__ float s_grid[codebookSize][codebookDim];
	__shared__ __half s_norms[codebookSize];

	for (int idx = threadIdx.x; idx < codebookSize * codebookDim; idx += blockDim.x) {
		int r = idx / codebookDim;
		int c = idx % codebookDim;
		s_grid[r][c] = __half2float(grid[r * codebookDim + c]);
	}
	for (int idx = threadIdx.x; idx < codebookSize; idx += blockDim.x) {
		s_norms[idx] = grid_norms[idx];
	}
	__syncthreads();

	int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (row >= out_dim) return;

	const __half two_h = __float2half(2.0f);

	const __half x0_h = x[row * codebookDim + 0];
	const __half x1_h = x[row * codebookDim + 1];

	float best_score = -FLT_MAX;
	uint8_t best_index = 0;

	#pragma unroll 8
	for (int c = 0; c < codebookSize; ++c) {
		const float g0 = s_grid[c][0];
		const float g1 = s_grid[c][1];
		const float dot_f = __half2float(x0_h) * g0 + __half2float(x1_h) * g1;
		const __half dot_h = __float2half_rn(dot_f);
		const __half twice_dot_h = __hmul(dot_h, two_h);
		const __half score_h = __hsub(twice_dot_h, s_norms[c]);
		const float score_f = __half2float(score_h);
		if (score_f > best_score) {
			best_score = score_f;
			best_index = static_cast<uint8_t>(c);
		}
	}

	out[row] = best_index;
}

extern "C" void higgs_quantize_2_256_ptr_f16_cuda_portable(
	uint64_t x_ptr,
	uint64_t grid_ptr,
	uint64_t grid_norms_ptr,
	uint64_t out_ptr,
	int64_t out_dim)
{
	const __half* x = reinterpret_cast<const __half*>(x_ptr);
	const __half* grid = reinterpret_cast<const __half*>(grid_ptr);
	const __half* grid_norms = reinterpret_cast<const __half*>(grid_norms_ptr);
	uint8_t* out = reinterpret_cast<uint8_t*>(out_ptr);

	const int threads = 256;
	const int blocks = static_cast<int>((out_dim + threads - 1) / threads);

	auto stream = at::cuda::getCurrentCUDAStream();
	higgs_quantize_2_256_ptr_f16_cuda_portable_kernel<<<blocks, threads, 0, stream>>>(x, grid, grid_norms, out, out_dim);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}