#include <torch/extension.h>
#include <cstdint>

extern "C" void higgs_dequantize_2_256_ptr_cuda_portable(
	uint64_t x_ptr,
	uint64_t grid_ptr,
	uint64_t out_ptr,
	int64_t out_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def(
		"higgs_dequantize_2_256_ptr_cuda_portable",
		&higgs_dequantize_2_256_ptr_cuda_portable,
		""
	);
}


