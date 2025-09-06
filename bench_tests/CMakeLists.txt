cmake_minimum_required(VERSION 3.18)
project(higgs_dequant_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architectures (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 2070 SUPER

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Add executable
add_executable(bench_dequant
    benchmark.cpp
    dequant_kernel.cu
)

# Set properties for CUDA compilation
set_property(TARGET bench_dequant PROPERTY CUDA_STANDARD 17)
set_property(TARGET bench_dequant PROPERTY CUDA_STANDARD_REQUIRED ON)

# Link CUDA libraries using modern CMake approach
target_link_libraries(bench_dequant 
    CUDA::cudart 
    CUDA::curand
)

# Compiler-specific options
target_compile_options(bench_dequant PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
    >
)