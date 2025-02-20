// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef NORMAL_RNG_CUH
#define NORMAL_RNG_CUH

#include "aes.cuh"
#include "aes_rng.cuh"

namespace rngongpu {
    class NormalRNG : public BaseRNG {
        private:
            void initState() override;
        public:
            NormalRNG();
            void gen_random_u32(int N, Data32* res);
            void gen_random_u32_mod_p(int N, Modulus32* p, Data32* res);
            void gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res);
            void gen_random_u64(int N, Data64* res);
            void gen_random_u64_mod_p(int N, Modulus64* p, Data64* res);
            void gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res);
            void gen_random_f32(int N, f32* res);
            void gen_random_f64(int N, f64* res);
    };
} // namespace rngongpu

#endif