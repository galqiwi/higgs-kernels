#pragma once

#include <cstdint>
#include <cuda_runtime.h>

extern "C" void launch_higgs_dequantize_2_256_kernel(
	const uint8_t* x,
	const uint32_t* grid,
	uint32_t* out,
	int64_t out_dim,
	cudaStream_t stream);

extern "C" void new_launch_higgs_dequantize_2_256_kernel(
	const uint8_t* x,
	const uint32_t* grid,
	uint32_t* out,
	int64_t out_dim,
	cudaStream_t stream);