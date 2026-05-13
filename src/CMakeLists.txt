# Copyright 2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(LIB_SOURCE_FILES
    lib/common/aes.hip
    lib/common/base_rng.hip
    lib/rand_aes/aes_rng.hip
    lib/rand_hip/hip_rng.hip
    lib/rand_hip/hip_rng_kernels.hip
)

add_library(rngonAMDGPU STATIC ${LIB_SOURCE_FILES})
add_library(RNGonAMDGPU::rngonAMDGPU ALIAS rngonAMDGPU)

# ─────────────────────────────────────────────
# C++ standard
# ─────────────────────────────────────────────
target_compile_features(rngonAMDGPU PUBLIC cxx_std_17)

# ─────────────────────────────────────────────
# Build flags
# ─────────────────────────────────────────────
target_compile_options(rngonAMDGPU PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O3 -g -DNDEBUG>
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>
    $<$<CONFIG:Debug>:-g>
)

# ─────────────────────────────────────────────
# Optional warnings
# ─────────────────────────────────────────────
option(RNGonGPU_ENABLE_WARNINGS "Enable extra warnings" OFF)

if(RNGonGPU_ENABLE_WARNINGS)
    target_compile_options(rngonAMDGPU PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

# ─────────────────────────────────────────────
# Properties
# ─────────────────────────────────────────────
set_target_properties(rngonAMDGPU PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    OUTPUT_NAME "rngonAMDGPU-${PROJECT_VERSION}"
    POSITION_INDEPENDENT_CODE ON
)

# ─────────────────────────────────────────────
# Dependencies (HIP ONLY)
# ─────────────────────────────────────────────
find_package(OpenSSL REQUIRED)

target_link_libraries(rngonAMDGPU PUBLIC
    GPUNTT::ntt
    roc::rocrand
    OpenSSL::SSL
    OpenSSL::Crypto
)

# ─────────────────────────────────────────────
# Include directories
# ─────────────────────────────────────────────
target_include_directories(rngonAMDGPU
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ─────────────────────────────────────────────
# Install rules (safe + standard)
# ─────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS rngonAMDGPU
    EXPORT ${RNGONAMDGPU_TARGETS_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
