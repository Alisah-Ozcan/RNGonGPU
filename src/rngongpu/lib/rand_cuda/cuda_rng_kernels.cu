// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng_kernels.cuh"
#include <curand_kernel.h>

namespace rngongpu {

//======================================================================
// Configuration Constants Definitions
//======================================================================
const int BLOCKS            = 24;
const int THREADS_PER_BLOCK = 256;
const int TOTAL_THREADS     = BLOCKS * THREADS_PER_BLOCK;





template <typename RNGState>
__global__ void init_states(RNGState *states, Data64 baseSeed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TOTAL_THREADS) {
        curand_init(baseSeed + tid, 0, 0, &states[tid]);
    }
}


template <typename RNGState>
__global__ void generate_uniform_32(RNGState *states, Data32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    RNGState localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand(&localState);
    }
    states[tid] = localState;
}


template <typename RNGState>
__global__ void generate_uniform_64(RNGState *states, Data64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    RNGState localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        unsigned int r1 = curand(&localState);
        unsigned int r2 = curand(&localState);
        random_numbers[idx] =
            (static_cast<Data64>(r1) << 32) |
            (static_cast<Data64>(r2) & 0xffffffffULL);
    }
    states[tid] = localState;
}

template <typename RNGState>
__global__ void generate_normal_32(RNGState *states, f32 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    RNGState localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal(&localState);
    }
    states[tid] = localState;
}


template <typename RNGState>
__global__ void generate_normal_64(RNGState *states, f64 *random_numbers, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_THREADS) return;
    RNGState localState = states[tid];
    for (int idx = tid; idx < n; idx += TOTAL_THREADS) {
        random_numbers[idx] = curand_normal_double(&localState);
    }
    states[tid] = localState;
}


template __global__ void init_states<curandStateXORWOW>(curandStateXORWOW*, Data64);
template __global__ void init_states<curandStateMRG32k3a>(curandStateMRG32k3a*, Data64);
template __global__ void init_states<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, Data64);

template __global__ void generate_uniform_32<curandStateXORWOW>(curandStateXORWOW*, Data32*, int);
template __global__ void generate_uniform_32<curandStateMRG32k3a>(curandStateMRG32k3a*, Data32*, int);
template __global__ void generate_uniform_32<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, Data32*, int);

template __global__ void generate_uniform_64<curandStateXORWOW>(curandStateXORWOW*, Data64*, int);
template __global__ void generate_uniform_64<curandStateMRG32k3a>(curandStateMRG32k3a*, Data64*, int);
template __global__ void generate_uniform_64<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, Data64*, int);

template __global__ void generate_normal_32<curandStateXORWOW>(curandStateXORWOW*, f32*, int);
template __global__ void generate_normal_32<curandStateMRG32k3a>(curandStateMRG32k3a*, f32*, int);
template __global__ void generate_normal_32<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, f32*, int);

template __global__ void generate_normal_64<curandStateXORWOW>(curandStateXORWOW*, f64*, int);
template __global__ void generate_normal_64<curandStateMRG32k3a>(curandStateMRG32k3a*, f64*, int);
template __global__ void generate_normal_64<curandStatePhilox4_32_10>(curandStatePhilox4_32_10*, f64*, int);
} // end namespace rngongpu
