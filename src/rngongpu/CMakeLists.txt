# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

file(GLOB_RECURSE LIB_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    lib/common/*.cu
    lib/rand_aes/*.cu
    lib/rand_aes_cpu/*.cu
    lib/rand_cuda/*.cu
    lib/common/*.cpp
    lib/rand_aes/*.cpp
    lib/rand_aes_cpu/*.cpp
    lib/rand_cuda/*.cpp
)

file(GLOB_RECURSE LIB_HEADER_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    include/common/*.cuh
    include/rand_aes/*.cuh
    include/rand_aes_cpu/*.cuh
    include/rand_cuda/*.cuh
    include/common/*.h
    include/rand_aes/*.h
    include/rand_aes_cpu/*.h
    include/rand_cuda/*.h
)

add_library(rngongpu STATIC ${LIB_SOURCE_FILES})

set_target_properties(rngongpu PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  OUTPUT_NAME "rngongpu-${PROJECT_VERSION}"
)

target_link_libraries(rngongpu PUBLIC 
    ${GPUNTT_TARGET}
    CUDA::curand
    OpenSSL::SSL
    OpenSSL::Crypto
)

set_target_properties(rngongpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

target_include_directories(
    rngongpu
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rand_aes>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rand_aes_cpu>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/rand_cuda>
    $<INSTALL_INTERFACE:include/RNGonGPU-${PROJECT_VERSION}>
    PRIVATE ${OPENSSL_INCLUDE_DIR}
)

install(TARGETS rngongpu
  EXPORT ${RNGonGPU_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(FILES ${LIB_HEADER_FILES} DESTINATION ${INCLUDES_INSTALL_DIR})
