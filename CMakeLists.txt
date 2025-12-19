# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

cmake_minimum_required(VERSION 3.26)
project(RNGonGPU VERSION 1.0 LANGUAGES C CXX CUDA ASM)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" )
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug)
message(STATUS "Build RNGonGPU version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

find_package(OpenSSL REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "CUDA GPU architectures" FORCE)
endif()
message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} for RNGonGPU")

##################
# Export Library #
##################

set(RUNTIME_DESTINATION ${CMAKE_INSTALL_BINDIR})
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_LIBDIR})
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_LIBDIR})
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/RNGonGPU-${PROJECT_VERSION})

set(RNGonGPU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(RNGonGPU_VERSION_CONFIG "${RNGonGPU_GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(RNGonGPU_PROJECT_CONFIG "${RNGonGPU_GENERATED_DIR}/${PROJECT_NAME}Config.cmake")
set(RNGonGPU_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(RNGonGPU_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}-${PROJECT_VERSION}")
set(RNGonGPU_NAMESPACE "${PROJECT_NAME}::")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${RNGonGPU_VERSION_CONFIG}" VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
configure_file("${RNGonGPU_SOURCE_DIR}/cmake/Config.cmake.in" "${RNGonGPU_PROJECT_CONFIG}" @ONLY)

install(
    FILES "${RNGonGPU_PROJECT_CONFIG}" "${RNGonGPU_VERSION_CONFIG}"
    DESTINATION "${RNGonGPU_CONFIG_INSTALL_DIR}")

install(
    EXPORT "${RNGonGPU_TARGETS_EXPORT_NAME}"
    NAMESPACE "${RNGonGPU_NAMESPACE}"
    DESTINATION "${RNGonGPU_CONFIG_INSTALL_DIR}")

option(RNGonGPU_USE_INTERNAL_GPUNTT "Use the internal GPU-NTT provided by RNGonGPU" ON)

if(RNGonGPU_USE_INTERNAL_GPUNTT)
    add_subdirectory(thirdparty)
    if(TARGET ntt AND NOT TARGET GPUNTT::ntt)
        add_library(GPUNTT::ntt ALIAS ntt)
    endif()
    message(STATUS "Internal GPU-NTT used")
else()
    # Prefer an in-tree GPUNTT target when RNGonGPU is built as a subproject
    # (e.g., a superbuild that already adds GPU-NTT via add_subdirectory).
    if(TARGET GPUNTT::ntt)
        message(STATUS "Using in-tree GPUNTT target (GPUNTT::ntt)")
    elseif(TARGET ntt)
        add_library(GPUNTT::ntt ALIAS ntt)
        message(STATUS "Using in-tree GPUNTT target (ntt -> GPUNTT::ntt)")
    else()
        find_package(GPUNTT REQUIRED)
        message(STATUS "External GPU-NTT used (find_package(GPUNTT))")
    endif()
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
