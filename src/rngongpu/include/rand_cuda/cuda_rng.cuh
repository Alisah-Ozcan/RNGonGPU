// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_RNG_H
#define CUDA_RNG_H

#include <string>
#include "aes.cuh"  

namespace rngongpu {

class CudarandRNG {
private:
    
    void initState();

   
    std::string generator_type;
    
    unsigned long long baseSeed;


    void* d_states;


    int numStates;         
    int mtgp32_numStates; 

public:
   
    CudarandRNG(unsigned long long seed, const std::string& generatorName);
    ~CudarandRNG();

    // Generates N random 32-bit unsigned integers.
    void gen_random_u32(int N, Data32* res);

    // Generates N random 64-bit unsigned integers.
    void gen_random_u64(int N, Data64* res);

    // Generates N random 32-bit floating-point numbers using f32.
    void gen_random_f32(int N, f32* res);

    // Generates N random 64-bit floating-point numbers using f64.
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
