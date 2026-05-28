# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/benchmark)

option(RNGonGPU_BENCH_BUILD_AES "Build AES CTR-DRBG benchmarks" ON)
option(RNGonGPU_BENCH_BUILD_CUDA "Build cuRAND-backed benchmarks" ON)

function(add_rngongpu_benchmark target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${target} PRIVATE rngongpu CUDA::cudart)
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
endfunction()

if(RNGonGPU_BENCH_BUILD_AES)
    add_rngongpu_benchmark(aes_benchmark benchmark_aes.cu)
endif()

if(RNGonGPU_BENCH_BUILD_CUDA)
    add_rngongpu_benchmark(cuda_benchmark benchmark_cuda.cu)
endif()
