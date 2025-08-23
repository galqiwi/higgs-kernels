#include <torch/extension.h>
#include <cstdint>

extern "C" void higgs_quantize_2_256_ptr_f16_cuda_portable(
	uint64_t x_ptr,
	uint64_t grid_ptr,
	uint64_t grid_norms_ptr,
	uint64_t out_ptr,
	int64_t out_dim);

extern "C" void higgs_quantize_2_256_ptr_bf16_cuda_portable(
	uint64_t x_ptr,
	uint64_t grid_ptr,
	uint64_t grid_norms_ptr,
	uint64_t out_ptr,
	int64_t out_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"higgs_quantize_2_256_ptr_bf16_cuda_portable",
		&higgs_quantize_2_256_ptr_bf16_cuda_portable,
		""
	);
	m.def(
		"higgs_quantize_2_256_ptr_f16_cuda_portable",
		&higgs_quantize_2_256_ptr_f16_cuda_portable,
		""
	);
}