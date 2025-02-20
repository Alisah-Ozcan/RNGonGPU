// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef UNIFORM_RNG_CUH
#define UNIFORM_RNG_CUH

#include "aes_rng.cuh"
#include "aes.cuh"
#include "modular_arith.cuh"

#define THREADS 256
#define BLOCKS 4

namespace rngongpu {
    class UniformRNG : public BaseRNG_AES {
        private:
            void initState() override;
        public:
            UniformRNG();
            void gen_random_u64(int N, Data64* res) override;
    
            void gen_random_u64_mod_p(int N, Modulus64* p, Data64* res) override;
    
            void gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) override;
    
            void gen_random_u32(int N, Data32* res) override;
    
            void gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) override;
    
            void gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) override;
    
            void gen_random_f32(int N, f32* res) override;

            void gen_random_f64(int N, f64* res) override;
    };
} // namespace rngongpu

#endif