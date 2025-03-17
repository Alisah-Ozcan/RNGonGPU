# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

cmake_minimum_required(VERSION 3.26)
project(RNGonGPU VERSION 1.0 LANGUAGES C CXX CUDA ASM)

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenSSL)
if(NOT OPENSSL_FOUND)
    message(FATAL_ERROR "OpenSSL not found on the system.
Please install OpenSSL.
For example, on Ubuntu run:
    sudo apt-get install libssl-dev
or refer to https://www.openssl.org for more instructions.")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} for RNGonGPU")

##################
# Export Library #
##################

set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/RNGonGPU-${PROJECT_VERSION})

set(RNGonGPU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(RNGonGPU_VERSION_CONFIG "${RNGonGPU_GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(RNGonGPU_PROJECT_CONFIG "${RNGonGPU_GENERATED_DIR}/${PROJECT_NAME}Config.cmake")
set(RNGonGPU_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(RNGonGPU_CONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}-${PROJECT_VERSION}")
set(RNGonGPU_NAMESPACE "${PROJECT_NAME}::")
set(RNGonGPU_VERSION 1.0)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${RNGonGPU_VERSION_CONFIG}" VERSION ${RNGonGPU_VERSION} COMPATIBILITY SameMajorVersion
)
configure_file("${RNGonGPU_SOURCE_DIR}/cmake/Config.cmake.in" "${RNGonGPU_PROJECT_CONFIG}" @ONLY)

install(
    FILES "${RNGonGPU_PROJECT_CONFIG}" "${RNGonGPU_VERSION_CONFIG}"
    DESTINATION "${RNGonGPU_CONFIG_INSTALL_DIR}")

install(
    EXPORT "${RNGonGPU_TARGETS_EXPORT_NAME}"
    NAMESPACE "${RNGonGPU_NAMESPACE}"
    DESTINATION "${RNGonGPU_CONFIG_INSTALL_DIR}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

option(RNGonGPU_USE_INTERNAL_GPUNTT "Use the internal GPU-NTT provided by RNGonGPU" ON)

if(RNGonGPU_USE_INTERNAL_GPUNTT)
    add_subdirectory(thirdparty)
    set(GPUNTT_TARGET ntt)
    message(STATUS "Internal GPU-NTT used")
else()
    set(GPUNTT_TARGET ntt)
    message(STATUS "External GPU-NTT used")
endif()

add_subdirectory(src)

############
# Examples #
############

option(RNGonGPU_BUILD_EXAMPLES "Build RNGonGPU Examples" OFF)
message(STATUS "RNGonGPU_BUILD_EXAMPLES: ${RNGonGPU_BUILD_EXAMPLES}")
if(RNGonGPU_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

#########
# Tests #
#########

option(RNGonGPU_BUILD_TESTS "Build RNGonGPU Tests" OFF)
message(STATUS "RNGonGPU_BUILD_TESTS: ${RNGonGPU_BUILD_TESTS}")
if(RNGonGPU_BUILD_TESTS)
    add_subdirectory(test)
endif()

##############
# Benchmarks #
##############

option(RNGonGPU_BUILD_BENCHMARKS "Build RNGonGPU Benchmarks" OFF)
message(STATUS "RNGonGPU_BUILD_BENCHMARKS: ${RNGonGPU_BUILD_BENCHMARKS}")
if(RNGonGPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()