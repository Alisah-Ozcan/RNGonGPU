# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/benchmark)

include(FetchContent)
FetchContent_Declare(
    nvbench
    GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
    GIT_TAG main
)
FetchContent_MakeAvailable(nvbench)

add_executable(aes_benchmark benchmark_aes.cu)
target_link_libraries(aes_benchmark PRIVATE rngongpu CUDA::cudart nvbench::main)
set_target_properties(aes_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

add_executable(cuda_benchmark benchmark_cuda.cu)
target_link_libraries(cuda_benchmark PRIVATE rngongpu CUDA::cudart nvbench::main)
set_target_properties(cuda_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)