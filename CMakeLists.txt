# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

cmake_minimum_required(VERSION 3.2)
project(RNGonGPU VERSION 1.0 LANGUAGES C CXX CUDA ASM)

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} for RNGonGPU")

add_subdirectory(thirdparty)
add_subdirectory(src)

############
# Examples #
############

option(RNGonGPU_BUILD_EXAMPLES "Build RNGonGPU Examples" OFF)
message(STATUS "RNGonGPU_BUILD_EXAMPLES: ${RNGonGPU_BUILD_EXAMPLES}")
if(RNGonGPU_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()
