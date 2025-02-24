// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng_kernels.cuh"
#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>







namespace rngongpu {


// Initialization Kernels


__global__ void init_xorwow_states(curandStateXORWOW_t *states, unsigned long long baseSeed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TOTAL_THREADS) {
        curand_init(baseSeed + tid, 0, 0, &states[tid]);
    }
}

__global__ void init_mrg32k3a_states(curandStateMRG32k3a_t *states, unsigned long long baseSeed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TOTAL_THREADS) {
        curand_init(baseSeed + tid, 0, 0, &states[tid]);
    }
}

__global__ void init_philox_states(curandStatePhilox4_32_10_t *states, unsigned long long baseSeed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TOTAL_THREADS) {
        curand_init(baseSeed + tid, 0, 0, &states[tid]);
    }
}


// Uniform Generation Kernels (32-bit)


__global__ void generate_random_xorwow(curandStateXORWOW_t *states,
                                       Data32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateXORWOW_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand(&localState);
    }
    states[tid] = localState;
}

__global__ void generate_random_mrg32k3a(curandStateMRG32k3a_t *states,
                                         Data32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateMRG32k3a_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand(&localState);
    }
    states[tid] = localState;
}

__global__ void generate_random_philox(curandStatePhilox4_32_10_t *states,
                                         Data32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStatePhilox4_32_10_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand(&localState);
    }
    states[tid] = localState;
}


// Uniform Generation Kernels (64-bit)


__global__ void generate_random_xorwow_64(curandStateXORWOW_t *states,
                                          Data64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateXORWOW_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        unsigned int r1 = curand(&localState);
        unsigned int r2 = curand(&localState);
        unsigned long long merged =
            (static_cast<unsigned long long>(r1) << 32) | (static_cast<unsigned long long>(r2) & 0xffffffffULL);
        random_numbers[idx] = merged;
    }
    states[tid] = localState;
}

__global__ void generate_random_mrg32k3a_64(curandStateMRG32k3a_t *states,
                                            Data64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateMRG32k3a_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        unsigned int r1 = curand(&localState);
        unsigned int r2 = curand(&localState);
        unsigned long long merged =
            (static_cast<unsigned long long>(r1) << 32) | (static_cast<unsigned long long>(r2) & 0xffffffffULL);
        random_numbers[idx] = merged;
    }
    states[tid] = localState;
}

__global__ void generate_random_philox_64(curandStatePhilox4_32_10_t *states,
                                          Data64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStatePhilox4_32_10_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        unsigned int r1 = curand(&localState);
        unsigned int r2 = curand(&localState);
        unsigned long long merged =
            (static_cast<unsigned long long>(r1) << 32) | (static_cast<unsigned long long>(r2) & 0xffffffffULL);
        random_numbers[idx] = merged;
    }
    states[tid] = localState;
}

// Normal Generation Kernels (32-bit) using curand_normal


__global__ void generate_random_xorwow_normal(curandStateXORWOW_t *states,
                                              f32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateXORWOW_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal(&localState);
    }
    states[tid] = localState;
}

__global__ void generate_random_mrg32k3a_normal(curandStateMRG32k3a_t *states,
                                                f32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateMRG32k3a_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal(&localState);
    }
    states[tid] = localState;
}

__global__ void generate_random_philox_normal(curandStatePhilox4_32_10_t *states,
                                               f32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStatePhilox4_32_10_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal(&localState);
    }
    states[tid] = localState;
}


// Normal Generation Kernels (64-bit) using curand_normal_double


__global__ void generate_random_xorwow_normal_64(curandStateXORWOW_t *states,
                                                 f64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateXORWOW_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal_double(&localState);
    }
    states[tid] = localState;
}

__global__ void generate_random_mrg32k3a_normal_64(curandStateMRG32k3a_t *states,
                                                   f64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStateMRG32k3a_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal_double(&localState);
    }
    states[tid] = localState;
}

__global__ void generate_random_philox_normal_64(curandStatePhilox4_32_10_t *states,
                                                 f64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    curandStatePhilox4_32_10_t localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal_double(&localState);
    }
    states[tid] = localState;
}





} // end namespace rngongpu



