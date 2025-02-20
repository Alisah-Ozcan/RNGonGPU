// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "uniform_rng.cuh"
#include "modular_arith.cuh"
#include "aes.cuh"

namespace rngongpu {
    void UniformRNG::initState() {}
    UniformRNG::UniformRNG() {}
    void UniformRNG::gen_random_u64(int N, Data64* res) {
        this -> gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res);
    }

    void UniformRNG::gen_random_u64_mod_p(int N,  Modulus64* p, Data64* res) {
        this -> gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res);
        
        Modulus64* d_p;
        cudaMalloc(&d_p, sizeof(Modulus64));
        cudaMemcpy(d_p, p, sizeof(Modulus64), cudaMemcpyHostToDevice);

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / CTA_size;

        mod_reduce_u64<<<grid_size, CTA_size>>>(res, d_p, N);
        cudaDeviceSynchronize();
    }

    void UniformRNG::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) {
        this -> gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res);

        Modulus64* d_p;
        cudaMalloc(&d_p, p_num * sizeof(Modulus64));
        cudaMemcpy(d_p, p, p_num * sizeof(Modulus64), cudaMemcpyHostToDevice);
        mod_reduce_u64<<<dim3(BLOCKS, p_num, 1), THREADS>>>(res, d_p, p_num, N);
        cudaDeviceSynchronize();
    }

    void UniformRNG::gen_random_u32(int N, Data32* res) {
        Data64* res_u64 = (Data64*) res;
        this -> gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64);
        cudaDeviceSynchronize();
        res = (Data32*) res_u64;
    }

    void UniformRNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) {
        Data64* res_u64 = (Data64*) res;
        this -> gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64);
        cudaDeviceSynchronize();
        res = (Data32*) res_u64;

        Modulus32* d_p;
        cudaMalloc(&d_p, sizeof(Modulus32));
        cudaMemcpy(d_p, p, sizeof(Modulus32), cudaMemcpyHostToDevice);

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / CTA_size;

        mod_reduce_u32<<<grid_size, CTA_size>>>(res, d_p, N);
        cudaDeviceSynchronize();
    }

    void UniformRNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) {
        Data64* res_u64 = (Data64*) res;
        this -> gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64);
        res = (Data32*) res_u64;

        Modulus32* d_p;
        cudaMalloc(&d_p, p_num * sizeof(Modulus32));
        cudaMemcpy(d_p, p, p_num * sizeof(Modulus32), cudaMemcpyHostToDevice);

        mod_reduce_u32<<<dim3(BLOCKS, p_num, 1), THREADS>>>(res, d_p, p_num, N);
        cudaDeviceSynchronize();
    }

    void UniformRNG::gen_random_f32(int N, f32* res) {}

    void UniformRNG::gen_random_f64(int N, f64* res) {}
} //namespace rngongpu