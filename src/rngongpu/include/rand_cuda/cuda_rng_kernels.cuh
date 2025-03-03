// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CudarandRNG_kernels_H
#define CudarandRNG_kernels_H

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include "aes.cuh"   // Assumes definitions for Data32, Data64, f32, and f64

namespace rngongpu {

//======================================================================
// Configuration Constants
//======================================================================
extern const int  BLOCKS;            // defined in the CU file
extern const int  THREADS_PER_BLOCK; // defined in the CU file
extern const int  TOTAL_THREADS;     // defined in the CU file

//======================================================================
// Templated Kernel Declarations
//======================================================================

// Initialization kernel
template <typename RNGState>
__global__ void init_states(RNGState *states, Data64 baseSeed);

// 32-bit Uniform Random Number Generation Kernel
template <typename RNGState>
__global__ void generate_uniform_32(RNGState *states, Data32 *random_numbers, int n);

// 64-bit Uniform Random Number Generation Kernel
template <typename RNGState>
__global__ void generate_uniform_64(RNGState *states, Data64 *random_numbers, int n);

// 32-bit Normal (Gaussian) Random Number Generation Kernel
template <typename RNGState>
__global__ void generate_normal_32(RNGState *states, f32 *random_numbers, int n);

// 64-bit Normal (Gaussian) Random Number Generation Kernel
template <typename RNGState>
__global__ void generate_normal_64(RNGState *states, f64 *random_numbers, int n);

} // end namespace rngongpu

#endif // CudarandRNG_kernels_H
