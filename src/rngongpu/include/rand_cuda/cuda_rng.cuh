// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_RNG_H
#define CUDA_RNG_H

#include "cuda_rng.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <iostream>
#include "aes.cuh"
#include "cuda_rng_kernels.cuh"
#include "base_rng.cuh"

namespace rngongpu {

// CudarandRNG is a templated class for GPU-based random number generation.
// It initializes a set of RNG states on the device and provides methods
// to generate uniform and normal random numbers as well as modulo-reduced variants.
template <typename RNGState>
class CudarandRNG {
private:
    // Initializes the RNG states on the GPU using the init_states kernel.
    void initState();

    // Base seed used to initialize all RNG states.
    Data64 baseSeed;

    
    void* d_states;

   
    int numStates;

    // (Optional) Additional state count (e.g., for MTGP32 variants).
    int mtgp32_numStates;

public:
    // Constructs the RNG with a given seed.
    explicit CudarandRNG(Data64 seed);

    // Destructor: frees allocated device memory.
    ~CudarandRNG();

    // Generates N random 32-bit unsigned integers.
    void gen_random_u32(int N, Data32* res);

    // Generates N random 64-bit unsigned integers.
    void gen_random_u64(int N, Data64* res);

    // Generates N random 32-bit floating-point (normal distribution) numbers.
    void gen_random_f32(int N, f32* res);

    // Generates N random 64-bit floating-point (normal distribution) numbers.
    void gen_random_f64(int N, f64* res);

    // Generates N random 32-bit unsigned integers modulo a given modulus.
    void gen_random_u32_mod_p(int N, Modulus32* p, Data32* res);

    // Generates N random 32-bit unsigned integers modulo a given modulus,
    // using an extra parameter (p_num).
    void gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res);

    // Generates N random 64-bit unsigned integers modulo a given modulus.
    void gen_random_u64_mod_p(int N, Modulus64* p, Data64* res);

    // Generates N random 64-bit unsigned integers modulo a given modulus,
    // using an extra parameter (p_num).
    void gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res);
};

} // namespace rngongpu



#endif // CUDA_RNG_H
