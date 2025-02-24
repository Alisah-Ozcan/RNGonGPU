// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_RNG_H
#define CUDA_RNG_H

#include <string>
#include "aes.cuh"
#include "cuda_rng_kernels.cuh"  // Contains kernel declarations.
#include <curand_kernel.h>

namespace rngongpu {

//---------------------------------------------------------------------
// Generator type structs for different RNG generators
//---------------------------------------------------------------------


struct XORWOW_generator {
    using StateType = curandStateXORWOW_t;
    
    static void initStates(StateType* states, unsigned long long seed, int numStates, int threadsPerBlock) {
        int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
        init_xorwow_states<<<blocks, threadsPerBlock>>>(states, seed);
    }
    
    static void generate_u32(StateType* states, Data32* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_xorwow<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_u64(StateType* states, Data64* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_xorwow_64<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_f32(StateType* states, f32* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_xorwow_normal<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_f64(StateType* states, f64* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_xorwow_normal_64<<<blocks, threadsPerBlock>>>(states, res, N);
    }
};


struct MRG32k3a_generator {
    using StateType = curandStateMRG32k3a_t;
    
    static void initStates(StateType* states, unsigned long long seed, int numStates, int threadsPerBlock) {
        int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
        init_mrg32k3a_states<<<blocks, threadsPerBlock>>>(states, seed);
    }
    
    static void generate_u32(StateType* states, Data32* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_mrg32k3a<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_u64(StateType* states, Data64* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_mrg32k3a_64<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_f32(StateType* states, f32* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_mrg32k3a_normal<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_f64(StateType* states, f64* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_mrg32k3a_normal_64<<<blocks, threadsPerBlock>>>(states, res, N);
    }
};


struct Philox_generator {
    using StateType = curandStatePhilox4_32_10_t;
    
    static void initStates(StateType* states, unsigned long long seed, int numStates, int threadsPerBlock) {
        int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
        init_philox_states<<<blocks, threadsPerBlock>>>(states, seed);
    }
    
    static void generate_u32(StateType* states, Data32* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_philox<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_u64(StateType* states, Data64* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_philox_64<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_f32(StateType* states, f32* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_philox_normal<<<blocks, threadsPerBlock>>>(states, res, N);
    }
    
    static void generate_f64(StateType* states, f64* res, int N, int threadsPerBlock) {
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        generate_random_philox_normal_64<<<blocks, threadsPerBlock>>>(states, res, N);
    }
};

//---------------------------------------------------------------------
// Templated RNG class that uses a generator struct to define generator-specific behavior.
//---------------------------------------------------------------------

template <typename generator>
class CudarandRNG {
private:
    using StateType = typename generator::StateType;

    // Initializes the states using the generator-specific kernel.
    void initState();

    // The base seed used to initialize all the states.
    unsigned long long baseSeed;

    // Device pointer to the generator states.
    void* d_states;

    // Number of states to allocate.
    int numStates;


public:
    // Constructs the RNG with a given seed.
    explicit CudarandRNG(unsigned long long seed);
    ~CudarandRNG();

    // Generates N random 32-bit unsigned integers.
    void gen_random_u32(int N, Data32* res);

    // Generates N random 64-bit unsigned integers.
    void gen_random_u64(int N, Data64* res);

    // Generates N random 32-bit floating-point numbers (f32).
    void gen_random_f32(int N, f32* res);

    // Generates N random 64-bit floating-point numbers (f64).
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

// Include the template implementation if separated (alternatively, you could define these inline)


#endif // CUDA_RNG_H
