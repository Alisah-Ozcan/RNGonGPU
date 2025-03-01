// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "base_rng.cuh"
#include <random>

#define BLOCKS 4
#define THREADS 256

namespace rngongpu
{
    void AES_RNG::init() {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<Data32> dist(0, std::numeric_limits<Data8>::max());
        
        for (int i = 0; i < 32; i++) this -> seed.push_back(dist(gen));

        // Results of Block_Encrypt(0, 1) and Block_Encrypt(0, 2)
        std::vector<unsigned char> temp = {0x58, 0xE2, 0xFC, 0xCE,
                                        0xFA, 0x7E, 0x30, 0x61,
                                        0x36, 0x7F, 0x1D, 0x57,
                                        0xA4, 0xE7, 0x45, 0x5A,
                                        0x03, 0x88, 0xDA, 0xCE,
                                        0x60, 0xB6, 0xA3, 0x92,
                                        0xF3, 0x28, 0xC2, 0xB9,
                                        0x71, 0xB2, 0xFE, 0x78};

        for (int i = 0; i < 16; i++) {
            this -> key.push_back(temp[i] ^ seed[i]);
            this -> nonce.push_back(temp[i+16] ^ seed[i+16]);
        }

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
        cudaMemcpy(this -> d_nonce, (this -> nonce).data(), 4 * sizeof(Data32), cudaMemcpyHostToDevice);

        // Key expansion
        keyExpansion(this -> key, this -> roundKeys);
    }
    void AES_RNG::increment_nonce(Data32 N) {
        for (int i = nonce.size() - 1; i >= 0; i--) { 
            if (nonce[i] < 255) {
                nonce[i]++;
                break;
            }
            nonce[i] = 0;
        }
        
        cudaMemcpy(this -> d_nonce, (this -> nonce).data(), 4 * sizeof(Data32), cudaMemcpyHostToDevice);
    }

    // generate random bits on the device. Write N bytes to res 
    // using BLOCKS blocks with THREADS threads each.
    void AES_RNG::gen_random_bytes(int N, int nBLOCKS, int nTHREADS, Data64* res) {
        if (this -> isPredictionResistanceEnabled || this -> reseedCounter >= RESEED_INTERVAL) {
            reseed(std::vector<unsigned char>());
        }

        int num_u64 = (N + 7) / 8;
        // Calculate the range for each thread
        Data64* range;
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&range, sizeof(Data64)));
        int threadCount = nBLOCKS * nTHREADS;
        double threadCount_d = (double) num_u64;
        double threadRange = threadCount_d / (threadCount * 2);
        *range = ceil(threadRange);

        printf("N: %u, range: %llu, BLOCKS: %u, THREADS: %u\n", num_u64, *range, nBLOCKS, nTHREADS);
        printf("Calling kernel to generate %u numbers, range: %llu\n", num_u64, *range);
        counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir<<<nBLOCKS, nTHREADS>>>(this -> d_nonce, this -> roundKeys, this -> t0, this -> t4, range, this -> SAES_d, res, num_u64);
        Data64* h_res_u64 = new Data64[num_u64];
        cudaMemcpy(h_res_u64, res, num_u64 * sizeof(Data64), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        //printLastCUDAError();

        // Free alocated arrays
        cudaFree(range);

        this -> increment_nonce(num_u64 + 1 / 2);
        this -> update(std::vector<unsigned char>());
        this -> reseedCounter += (N / MAX_BYTES_PER_REQUEST + 1);
    }
    AES_RNG::AES_RNG(bool _isPredictionResistanceEnabled) : reseedCounter(1UL), isPredictionResistanceEnabled(_isPredictionResistanceEnabled) {this -> init();}

    AES_RNG::~AES_RNG() {
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
    void AES_RNG::gen_random_f32(int N, f32* res) {
        Data64* res_u64;
        int num_u32 = N;
        cudaMalloc(&res_u64, num_u32 * sizeof(Data32));
        this -> gen_random_bytes(num_u32 * sizeof(Data32), BLOCKS, THREADS, res_u64);

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / (CTA_size * 2);
        
        Data32* d_res_as_u32 = reinterpret_cast<Data32*>(res_u64);
        box_muller_u32<<<grid_size, CTA_size>>>(d_res_as_u32, res, N);
        cudaDeviceSynchronize();
    }
    void AES_RNG::gen_random_f64(int N, f64* res) {
        Data64* res_u64;
        int num_u64 = N;
        cudaMalloc(&res_u64, num_u64 * sizeof(Data64));
        this -> gen_random_bytes(num_u64 * sizeof(Data64), BLOCKS, THREADS, res_u64);

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / (CTA_size * 2);
        
        box_muller_u64<<<grid_size, CTA_size>>>(res_u64, res, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::gen_random_u64(int N, Data64* res) {
        this -> gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res);
    }

    void AES_RNG::gen_random_u64_mod_p(int N,  Modulus64* p, Data64* res) {
        this -> gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res);
        
        Modulus64* d_p;
        cudaMalloc(&d_p, sizeof(Modulus64));
        cudaMemcpy(d_p, p, sizeof(Modulus64), cudaMemcpyHostToDevice);

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / CTA_size;

        mod_reduce_u64<<<grid_size, CTA_size>>>(res, d_p, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) {
        this -> gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res);

        Modulus64* d_p;
        cudaMalloc(&d_p, p_num * sizeof(Modulus64));
        cudaMemcpy(d_p, p, p_num * sizeof(Modulus64), cudaMemcpyHostToDevice);
        mod_reduce_u64<<<dim3(BLOCKS, p_num, 1), THREADS>>>(res, d_p, p_num, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::gen_random_u32(int N, Data32* res) {
        Data64* res_u64 = (Data64*) res;
        this -> gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64);
        cudaDeviceSynchronize();
        res = (Data32*) res_u64;
    }

    void AES_RNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) {
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

    void AES_RNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) {
        Data64* res_u64 = (Data64*) res;
        this -> gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64);
        res = (Data32*) res_u64;

        Modulus32* d_p;
        cudaMalloc(&d_p, p_num * sizeof(Modulus32));
        cudaMemcpy(d_p, p, p_num * sizeof(Modulus32), cudaMemcpyHostToDevice);

        mod_reduce_u32<<<dim3(BLOCKS, p_num, 1), THREADS>>>(res, d_p, p_num, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::update(std::vector<unsigned char> additionalInput) {
        // Do 2 encryptions on V and V + 1.
        // Set V to V + 2.
        // XOR the blocks with additionalInput
        // Set Key to first, V to second block

        if (additionalInput.size() < 32) {
            for (int i = 1; i <= 32 - additionalInput.size(); i++) additionalInput.push_back(0);
        }

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error("CTR_DRBG_Update: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), nullptr, (this -> key).data(), nullptr))
            throw std::runtime_error("CTR_DRBG_Update: EVP_EncryptInit_ex failed");

        EVP_CIPHER_CTX_set_padding(ctx, 0);
        std::vector<unsigned char> temp;
        temp.reserve(16);
        const std::size_t blockSize = 16;
        std::vector<unsigned char> outputBlock(blockSize);
        std::vector<unsigned char> Vtemp(this->nonce);

        for (std::size_t i = 0; i < 32 / blockSize; i++) { 
            // Increment Vtemp in big-endian order.
            for (int j = blockSize - 1; j >= 0; j--) {
                if (++Vtemp[j] != 0)
                    break;
            }           
            int outlen = 0;
            if (1 != EVP_EncryptUpdate(ctx, outputBlock.data(), &outlen, Vtemp.data(), blockSize))
                throw std::runtime_error("CTR_DRBG_Update: EVP_EncryptUpdate failed");
            if (outlen != static_cast<int>(blockSize))
                throw std::runtime_error("CTR_DRBG_Update: Unexpected block size");
            temp.insert(temp.end(), outputBlock.begin(), outputBlock.end());
        }
        EVP_CIPHER_CTX_free(ctx);

        for (int i = 0; i < blockSize; i++) {
            this -> key[i] = temp[i] ^ additionalInput[i];
            this -> nonce[i] = temp[i+16] ^ additionalInput[i+16];
        }
        keyExpansion(key, roundKeys);
        cudaMemcpy(this -> d_nonce, (this -> nonce).data(), 4 * sizeof(Data32), cudaMemcpyHostToDevice);
    }   

    void AES_RNG::resetReseedCounter() {
        this -> reseedCounter = 1;
    }

    void AES_RNG::reseed(std::vector<unsigned char> additionalInput) {
        if (additionalInput.size() < 32) {
            for (int i = 0; i < 32 - additionalInput.size(); i++) additionalInput.push_back(0);
        }
        std::vector<unsigned char> entropyInput(32, 0);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<Data32> dist(0, std::numeric_limits<Data8>::max());
        
        for (int i = 0; i < 32; i++) entropyInput.push_back(dist(gen) ^ additionalInput[i]);
        this -> update(entropyInput);
        this -> resetReseedCounter();
    }

    void AES_RNG::printWorkingState() {
        std::cout << "------DRBG State------\n";
        std::cout << "Key: " << std::hex << std::uppercase;
        for (int i = 0; i < 13; i+=4) std::cout << (int) this->key[i] << (int)  this->key[i+1] << (int)  this->key[i+2] << (int)  this->key[i+3] << " "; 
        std:: cout << std::endl << "V: ";
        for (int i = 0; i < 13; i+=4) std::cout << (int) this->nonce[i] << (int) this->nonce[i+1] << (int) this->nonce[i+2] << (int) this->nonce[i+3] << " "; 
        std::cout << std::endl;
        std::cout << std::dec << "Reseed Counter: " << reseedCounter << std::endl;
    }
} // namespace rngongpu