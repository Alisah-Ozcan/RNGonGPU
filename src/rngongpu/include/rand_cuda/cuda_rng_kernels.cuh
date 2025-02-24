// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_RNG_KERNELS_H
#define Cuda_RNG_KERNELS_H

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include "aes.cuh"





namespace rngongpu {


// Configuration Constants

static const int  BLOCKS                = 24;
static const int  THREADS_PER_BLOCK     = 256;
static const int  TOTAL_THREADS         = BLOCKS * THREADS_PER_BLOCK;



// Global  State Pointers

extern curandStateXORWOW_t        *d_xorwowStates;
extern curandStateMRG32k3a_t      *d_mrg32k3aStates;
extern curandStatePhilox4_32_10_t *d_philoxStates;
extern curandStateMtgp32_t        *d_mtgpStates64;

// Kernel  for Initialization

__global__ void init_xorwow_states(curandStateXORWOW_t *states, unsigned long long baseSeed);
__global__ void init_mrg32k3a_states(curandStateMRG32k3a_t *states, unsigned long long baseSeed);
__global__ void init_philox_states(curandStatePhilox4_32_10_t *states, unsigned long long baseSeed);


// Kernel Prototypes for 32-bit Random Number Generation (Uniform)

__global__ void generate_random_xorwow(curandStateXORWOW_t *states,
                                       Data32 *random_numbers, int n);
__global__ void generate_random_mrg32k3a(curandStateMRG32k3a_t *states,
                                         Data32 *random_numbers, int n);
__global__ void generate_random_philox(curandStatePhilox4_32_10_t *states,
                                         Data32  *random_numbers, int n);


// Kernel  for 64-bit Random Number Generation (Uniform)

__global__ void generate_random_xorwow_64(curandStateXORWOW_t *states,
                                          Data64 *random_numbers, int n);
__global__ void generate_random_mrg32k3a_64(curandStateMRG32k3a_t *states,
                                            Data64 *random_numbers, int n);
__global__ void generate_random_philox_64(curandStatePhilox4_32_10_t *states,
                                          Data64 *random_numbers, int n);





// Kernel  for Normal Random Number Generation (32-bit)

__global__ void generate_random_xorwow_normal(curandStateXORWOW_t *states,
                                              f32 *random_numbers, int n);
__global__ void generate_random_mrg32k3a_normal(curandStateMRG32k3a_t *states,
                                                f32 *random_numbers, int n);
__global__ void generate_random_philox_normal(curandStatePhilox4_32_10_t *states,
                                               f32 *random_numbers, int n);


// Kernel  for Normal Random Number Generation (64-bit)

__global__ void generate_random_xorwow_normal_64(curandStateXORWOW_t *states,
                                                 f64 *random_numbers, int n);
__global__ void generate_random_mrg32k3a_normal_64(curandStateMRG32k3a_t *states,
                                                   f64 *random_numbers, int n);
__global__ void generate_random_philox_normal_64(curandStatePhilox4_32_10_t *states,
                                                 f64 *random_numbers, int n);



} // end namespace rngongpu

#endif // CUDA_RNG_KERNELS_H
