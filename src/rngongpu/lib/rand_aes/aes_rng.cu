// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "base_rng.cuh"
#include <random>

namespace rngongpu
{
    void test_aes() {
        std::cout << "TEST SUCCESS\n";
    }
    void BaseRNG_AES::init() {
        this -> seed = new Data32[4];
        this -> nonce = new Data32[4];
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<Data32> dist(0, std::numeric_limits<Data32>::max());
        
        this -> seed[0] = dist(gen);
        this -> seed[1] = dist(gen);
        this -> seed[2] = dist(gen);
        this -> seed[3] = dist(gen); 

        this -> nonce[0] = dist(gen);
        this -> nonce[1] = dist(gen);
        this -> nonce[2] = dist(gen);
        this -> nonce[3] = dist(gen);

        // this -> nonce[0] = 0x3243F6A8U;
        // this -> nonce[1] = 0x885A308DU;
        // this -> nonce[2] = 0x313198A2U;
        // this -> nonce[3] = 0x00000000U;

        // this -> seed[0] = 0x2B7E1516U;
        // this -> seed[1] = 0x28AED2A6U;
        // this -> seed[2] = 0xABF71588U;
        // this -> seed[3] = 0x09CF4F3CU;

        // Allocate RCON values
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> rcon), RCON_SIZE * sizeof(Data32)));
        for (int i = 0; i < RCON_SIZE; i++) {
            this -> rcon[i] = RCON32[i];
        }

        // Allocate Tables
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t0), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t1), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t2), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t3), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t4), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t4_0), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t4_1), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t4_2), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> t4_3), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> SAES_d), 256 * sizeof(Data8))); // Cihangir
        for (int i = 0; i < TABLE_SIZE; i++) {
            this -> t0[i] = T0[i];
            this -> t1[i] = T1[i];
            this -> t2[i] = T2[i];
            this -> t3[i] = T3[i];
            this -> t4[i] = T4[i];
            this -> t4_0[i] = T4_0[i];
            this -> t4_1[i] = T4_1[i];
            this -> t4_2[i] = T4_2[i];
            this -> t4_3[i] = T4_3[i];
        }
        for (int i = 0; i < 256; i++) this -> SAES_d[i] = SAES[i]; // Cihangir

        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this -> roundKeys), AES_128_KEY_SIZE_INT * sizeof(Data32)));
        
        cudaMalloc(&(this -> d_nonce), 4 * sizeof(Data32));
        cudaMemcpy(this -> d_nonce, this -> nonce, 4 * sizeof(Data32), cudaMemcpyHostToDevice);

        // Key expansion
        keyExpansion(this -> seed, this -> roundKeys);
        initState();
    }
    void BaseRNG_AES::initState() {}
    void BaseRNG_AES::increment_nonce(Data32 N) {
        if (this -> nonce[3] + N < this -> nonce[3]) {
            this -> nonce[2] += 1;
        }
        this -> nonce[3] += N;
        
        cudaMemcpy(this -> d_nonce, this -> nonce, 4 * sizeof(Data32), cudaMemcpyHostToDevice);
    }

    // generate random bits on the device. Write N bytes to res 
    // using BLOCKS blocks with THREADS threads each.
    void BaseRNG_AES::gen_random_bytes(int N, int BLOCKS, int THREADS, Data64* res) {
        int num_u64 = (N + 7) / 8;
        // Calculate the range for each thread
        Data64* range;
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&range, sizeof(Data64)));
        int threadCount = BLOCKS * THREADS;
        double threadCount_d = (double) num_u64;
        double threadRange = threadCount_d / (threadCount * 2);
        *range = ceil(threadRange);

        printf("N: %u, range: %llu, BLOCKS: %u, THREADS: %u\n", num_u64, *range, BLOCKS, THREADS);
        printf("Calling kernel to generate %u numbers, range: %llu\n", num_u64, *range);
        counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir<<<BLOCKS, THREADS>>>(this -> d_nonce, this -> roundKeys, this -> t0, this -> t4, range, this -> SAES_d, res, num_u64);
        cudaDeviceSynchronize();
        //printLastCUDAError();

        // Free alocated arrays
        cudaFree(range);

        this -> increment_nonce(num_u64 + 1 / 2);
    }
    BaseRNG_AES::BaseRNG_AES() : seed(nullptr), nonce(nullptr) {this -> init();}

    // tune the object for desired output in the next function call 
    // ex: set the stddev and mean for Normal distribution objects
    // virtual void set_state() = 0;

    // virtual void gen_random_u32(int N, u32* res) = 0;
    // virtual void gen_random_u32_mod_p(int N, u32 p, u32* res) = 0;
    // virtual void gen_random_u32_mod_p(int N, u32* p, u32 p_num, u32* res) = 0;
    // virtual void gen_random_u64(int N, u64* res) = 0;
    // virtual void gen_random_u64_mod_p(int N, u64 p, u64* res) = 0;
    // virtual void gen_random_u64_mod_p(int N, u64* p, u32 p_num, u64* res) = 0;
    // virtual void gen_random_f32(int N, f32* res) = 0;
    // virtual void gen_random_f64(int N, f64* res) = 0;

    BaseRNG_AES::~BaseRNG_AES() {
        cudaFree(this -> t0);
        cudaFree(this -> t1);
        cudaFree(this -> t2);
        cudaFree(this -> t3);
        cudaFree(this -> t4);
        cudaFree(this -> t4_0);
        cudaFree(this -> t4_1);
        cudaFree(this -> t4_2);
        cudaFree(this -> t4_3);
        cudaFree(this -> rcon);
        cudaFree(this -> SAES_d);
        cudaFree(this -> d_nonce);
        cudaFree(this -> roundKeys);
    }
} // namespace rngongpu