// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "normal_rng.cuh"
#include "aes.cuh"

#define THREADS 256
#define BLOCKS 4

namespace rngongpu {
        void NormalRNG::initState() {}
        NormalRNG::NormalRNG() {}
        void NormalRNG::gen_random_u32(int N, Data32* res) {}
        void NormalRNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) {}
        void NormalRNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) {}
        void NormalRNG::gen_random_u64(int N, Data64* res) {}
        void NormalRNG::gen_random_u64_mod_p(int N, Modulus64* p, Data64* res) {}
        void NormalRNG::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) {}
        void NormalRNG::gen_random_f32(int N, f32* res) {
            Data64* res_u64;
            int num_u32 = (N + 1) / 2;
            cudaMalloc(&res_u64, num_u32 * sizeof(Data32));
            this -> gen_random_bytes(num_u32 * sizeof(Data32), BLOCKS, THREADS, res_u64);

            const int CTA_size = 256;
            const int grid_size = (N + CTA_size - 1) / (CTA_size * 2);
            
            Data32* d_res_as_u32 = reinterpret_cast<Data32*>(res_u64);
            box_muller_u32<<<grid_size, CTA_size>>>(d_res_as_u32, res, N);
            cudaDeviceSynchronize();
        }
        void NormalRNG::gen_random_f64(int N, f64* res) {
            Data64* res_u64;
            int num_u64 = (N + 1) / 2;
            cudaMalloc(&res_u64, num_u64 * sizeof(Data64));
            this -> gen_random_bytes(num_u64 * sizeof(Data64), BLOCKS, THREADS, res_u64);

            const int CTA_size = 256;
            const int grid_size = (N + CTA_size - 1) / (CTA_size * 2);
            
            box_muller_u64<<<grid_size, CTA_size>>>(res_u64, res, N);
            cudaDeviceSynchronize();
        }

}