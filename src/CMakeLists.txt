# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(LIB_SOURCE_FILES
    lib/common/aes.cu
    lib/common/base_rng.cu
    lib/rand_aes/aes_rng.cu
    lib/rand_cuda/cuda_rng.cu
    lib/rand_cuda/cuda_rng_kernels.cu
)

add_library(rngongpu STATIC ${LIB_SOURCE_FILES})
add_library(RNGonGPU::rngongpu ALIAS rngongpu)
if(CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(rngongpu PUBLIC cxx_std_17)
else()
    message(WARNING
        "CMake does not know C++ compile features for ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION}; falling back to -std=c++17"
    )
    target_compile_options(rngongpu PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
endif()

if(CMAKE_CUDA_COMPILE_FEATURES)
    target_compile_features(rngongpu PRIVATE cuda_std_17)
else()
    target_compile_options(rngongpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17>)
endif()
target_compile_options(rngongpu PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -g -DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    $<$<CONFIG:Debug>:-g>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
    $<$<AND:$<CONFIG:RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:--generate-line-info>
)
option(RNGonGPU_ENABLE_WARNINGS "Enable extra warnings for RNGonGPU" OFF)
if(RNGonGPU_ENABLE_WARNINGS)
    target_compile_options(rngongpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )
endif()

set_target_properties(rngongpu PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  OUTPUT_NAME "rngongpu-${PROJECT_VERSION}"
)

target_link_libraries(rngongpu PUBLIC 
    GPUNTT::ntt
    CUDA::curand
    OpenSSL::SSL
    OpenSSL::Crypto
)

set_target_properties(rngongpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

target_include_directories(
    rngongpu
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/RNGonGPU-${PROJECT_VERSION}>
)

install(TARGETS rngongpu
  EXPORT ${RNGonGPU_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${INCLUDES_INSTALL_DIR}
)
