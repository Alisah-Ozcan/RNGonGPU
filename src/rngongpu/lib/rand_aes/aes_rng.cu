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
    const EVP_CIPHER* AES_RNG::getEVPCipherECB() const
    {
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                return EVP_aes_128_ecb();
            case SecurityLevel::AES192:
                return EVP_aes_192_ecb();
            case SecurityLevel::AES256:
                return EVP_aes_256_ecb();
            default:
                throw std::runtime_error("Unsupported security level in ECB");
        }
    }
    void AES_RNG::init()
    {
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                keyLen = 16;
                break;
            case SecurityLevel::AES192:
                keyLen = 24;
                break;
            case SecurityLevel::AES256:
                keyLen = 32;
                break;
            default:
                throw std::runtime_error("Unsupported security level");
        }
        seedLen = keyLen + 16; // 16 bytes for the block size (V)

        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<Data32> dist(
            0, std::numeric_limits<Data8>::max());

        for (int i = 0; i < seedLen; i++)
            this->seed.push_back(dist(gen));

        std::vector<unsigned char> seedMaterial = DF(seed, seedLen);

        this -> key = std::vector<unsigned char>(keyLen, 0);
        this -> nonce = std::vector<unsigned char>(16, 0);

        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                RNGONGPU_CUDA_CHECK(cudaMallocManaged(
                    &(this->roundKeys), AES_128_KEY_SIZE_INT * sizeof(Data32)));
                break;
            case SecurityLevel::AES192:
                RNGONGPU_CUDA_CHECK(cudaMallocManaged(
                    &(this->roundKeys), AES_192_KEY_SIZE_INT * sizeof(Data32)));
                break;
            case SecurityLevel::AES256:
                RNGONGPU_CUDA_CHECK(cudaMallocManaged(
                    &(this->roundKeys), AES_256_KEY_SIZE_INT * sizeof(Data32)));
                break;
            default:
                throw std::runtime_error("Unsupported security level");
        }

        this -> update(seedMaterial);

        // Allocate RCON values
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->rcon), RCON_SIZE * sizeof(Data32)));
        for (int i = 0; i < RCON_SIZE; i++)
        {
            this->rcon[i] = RCON32[i];
        }

        // Allocate Tables
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t0), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t1), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t2), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t3), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t4), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t4_0), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t4_1), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t4_2), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(
            cudaMallocManaged(&(this->t4_3), TABLE_SIZE * sizeof(Data32)));
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&(this->SAES_d),
                                              256 * sizeof(Data8))); // Cihangir
        for (int i = 0; i < TABLE_SIZE; i++)
        {
            this->t0[i] = T0[i];
            this->t1[i] = T1[i];
            this->t2[i] = T2[i];
            this->t3[i] = T3[i];
            this->t4[i] = T4[i];
            this->t4_0[i] = T4_0[i];
            this->t4_1[i] = T4_1[i];
            this->t4_2[i] = T4_2[i];
            this->t4_3[i] = T4_3[i];
        }
        for (int i = 0; i < 256; i++)
            this->SAES_d[i] = SAES[i]; // Cihangir

        cudaMalloc(&(this->d_nonce), 4 * sizeof(Data32));
        cudaMemcpy(this->d_nonce, (this->nonce).data(), 4 * sizeof(Data32),
                   cudaMemcpyHostToDevice);
    }
    void AES_RNG::increment_nonce(Data32 N)
    {
        for (int i = nonce.size() - 1; i >= 0; i--)
        {
            if (nonce[i] < 255)
            {
                nonce[i]++;
                break;
            }
            nonce[i] = 0;
        }

        cudaMemcpy(this->d_nonce, (this->nonce).data(), 4 * sizeof(Data32),
                   cudaMemcpyHostToDevice);
    }

    // generate random bits on the device. Write N bytes to res
    // using BLOCKS blocks with THREADS threads each.
    void AES_RNG::gen_random_bytes(int N, int nBLOCKS, int nTHREADS,
                                   Data64* res,
                                   std::vector<unsigned char> additionalInput)
    {
        if (this->isPredictionResistanceEnabled ||
            this->reseedCounter >= RESEED_INTERVAL)
        {
            reseed(std::vector<unsigned char>());
        }

        if (additionalInput.size() != 0)
        {
            additionalInput = DF(additionalInput, seedLen);
            update(additionalInput);
        } else 
        {
            additionalInput = std::vector<unsigned char>(seedLen, 0);
        }

        int num_u64 = (N + 7) / 8;
        // Calculate the range for each thread
        Data64* range;
        RNGONGPU_CUDA_CHECK(cudaMallocManaged(&range, sizeof(Data64)));
        int threadCount = nBLOCKS * nTHREADS;
        double threadCount_d = (double) num_u64;
        double threadRange = threadCount_d / (threadCount * 2);
        *range = ceil(threadRange);

        printf("N: %u, range: %llu, BLOCKS: %u, THREADS: %u\n", num_u64, *range,
               nBLOCKS, nTHREADS);
        printf("Calling kernel to generate %u numbers, range: %llu\n", num_u64,
               *range);
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir<<<
                    nBLOCKS, nTHREADS>>>(this->d_nonce, this->roundKeys, this->t0,
                                        this->t4, range, this->SAES_d, res, num_u64);
                break;
            case SecurityLevel::AES192:
                counter192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<
                    nBLOCKS, nTHREADS>>>(this->d_nonce, this->roundKeys, this->t0,
                                        this->t4, range, res, num_u64);
                break;
            case SecurityLevel::AES256:
                counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<
                    nBLOCKS, nTHREADS>>>(this->d_nonce, this->roundKeys, this->t0,
                                        this->t4, range, res, num_u64);
                break;
            default:
                throw std::runtime_error("Unsupported security level");
        }
        Data64* h_res_u64 = new Data64[num_u64];
        cudaMemcpy(h_res_u64, res, num_u64 * sizeof(Data64),
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        // printLastCUDAError();

        // Free alocated arrays
        cudaFree(range);

        this->increment_nonce(num_u64 + 1 / 2);
        this->update(additionalInput);
        this->reseedCounter += (N / MAX_BYTES_PER_REQUEST + 1);
    }
    AES_RNG::AES_RNG(bool _isPredictionResistanceEnabled, SecurityLevel _securityLevel)
        : reseedCounter(1UL),
          isPredictionResistanceEnabled(_isPredictionResistanceEnabled),
          securityLevel(_securityLevel)
    {
        this->init();
    }

    AES_RNG::~AES_RNG()
    {
        cudaFree(this->t0);
        cudaFree(this->t1);
        cudaFree(this->t2);
        cudaFree(this->t3);
        cudaFree(this->t4);
        cudaFree(this->t4_0);
        cudaFree(this->t4_1);
        cudaFree(this->t4_2);
        cudaFree(this->t4_3);
        cudaFree(this->rcon);
        cudaFree(this->SAES_d);
        cudaFree(this->d_nonce);
        cudaFree(this->roundKeys);
    }
    void AES_RNG::gen_random_f32(int N, f32* res)
    {
        Data64* res_u64;
        int num_u32 = N;
        cudaMalloc(&res_u64, num_u32 * sizeof(Data32));
        this->gen_random_bytes(num_u32 * sizeof(Data32), BLOCKS, THREADS,
                               res_u64, std::vector<unsigned char>());

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / (CTA_size * 2);

        Data32* d_res_as_u32 = reinterpret_cast<Data32*>(res_u64);
        box_muller_u32<<<grid_size, CTA_size>>>(d_res_as_u32, res, N);
        cudaDeviceSynchronize();
    }
    void AES_RNG::gen_random_f64(int N, f64* res)
    {
        Data64* res_u64;
        int num_u64 = N;
        cudaMalloc(&res_u64, num_u64 * sizeof(Data64));
        this->gen_random_bytes(num_u64 * sizeof(Data64), BLOCKS, THREADS,
                               res_u64, std::vector<unsigned char>());

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / (CTA_size * 2);

        box_muller_u64<<<grid_size, CTA_size>>>(res_u64, res, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::gen_random_u64(int N, Data64* res)
    {
        this->gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res,
                               std::vector<unsigned char>());
    }

    void AES_RNG::gen_random_u64_mod_p(int N, Modulus64* p, Data64* res)
    {
        this->gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res,
                               std::vector<unsigned char>());

        Modulus64* d_p;
        cudaMalloc(&d_p, sizeof(Modulus64));
        cudaMemcpy(d_p, p, sizeof(Modulus64), cudaMemcpyHostToDevice);

        const int CTA_size = 256;
        const int grid_size = (N + CTA_size - 1) / CTA_size;

        mod_reduce_u64<<<grid_size, CTA_size>>>(res, d_p, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num,
                                       Data64* res)
    {
        this->gen_random_bytes(N * sizeof(Data64), BLOCKS, THREADS, res,
                               std::vector<unsigned char>());

        Modulus64* d_p;
        cudaMalloc(&d_p, p_num * sizeof(Modulus64));
        cudaMemcpy(d_p, p, p_num * sizeof(Modulus64), cudaMemcpyHostToDevice);
        mod_reduce_u64<<<dim3(BLOCKS, p_num, 1), THREADS>>>(res, d_p, p_num, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::gen_random_u32(int N, Data32* res)
    {
        Data64* res_u64 = (Data64*) res;
        this->gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64,
                               std::vector<unsigned char>());
        cudaDeviceSynchronize();
        res = (Data32*) res_u64;
    }

    void AES_RNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res)
    {
        Data64* res_u64 = (Data64*) res;
        this->gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64,
                               std::vector<unsigned char>());
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

    void AES_RNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num,
                                       Data32* res)
    {
        Data64* res_u64 = (Data64*) res;
        this->gen_random_bytes(N * sizeof(Data32), BLOCKS, THREADS, res_u64,
                               std::vector<unsigned char>());
        res = (Data32*) res_u64;

        Modulus32* d_p;
        cudaMalloc(&d_p, p_num * sizeof(Modulus32));
        cudaMemcpy(d_p, p, p_num * sizeof(Modulus32), cudaMemcpyHostToDevice);

        mod_reduce_u32<<<dim3(BLOCKS, p_num, 1), THREADS>>>(res, d_p, p_num, N);
        cudaDeviceSynchronize();
    }

    void AES_RNG::update(std::vector<unsigned char> additionalInput)
    {

        if (additionalInput.size() < seedLen)
        {
            for (int i = 1; i <= seedLen - additionalInput.size(); i++)
                additionalInput.push_back(0);
        }

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error(
                "CTR_DRBG_Update: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, this->getEVPCipherECB(), nullptr,
                                    (this->key).data(), nullptr))
            throw std::runtime_error(
                "CTR_DRBG_Update: EVP_EncryptInit_ex failed");

        EVP_CIPHER_CTX_set_padding(ctx, 0);
        std::vector<unsigned char> temp;
        temp.reserve(seedLen);
        const std::size_t blockSize = 16;
        std::vector<unsigned char> outputBlock(blockSize);
        std::vector<unsigned char> Vtemp(this->nonce);
            
        for (std::size_t i = 0; i < seedLen / blockSize; i++)
        {
            // Increment Vtemp in big-endian order.
            for (int j = blockSize - 1; j >= 0; j--)
            {
                if (++Vtemp[j] != 0)
                    break;
            }
            int outlen = 0;
            if (1 != EVP_EncryptUpdate(ctx, outputBlock.data(), &outlen,
                                       Vtemp.data(), blockSize))
                throw std::runtime_error(
                    "CTR_DRBG_Update: EVP_EncryptUpdate failed");
            if (outlen != static_cast<int>(blockSize))
                throw std::runtime_error(
                    "CTR_DRBG_Update: Unexpected block size");
            temp.insert(temp.end(), outputBlock.begin(), outputBlock.end());
        }
        EVP_CIPHER_CTX_free(ctx);

        if (!additionalInput.empty())
        {
            if (additionalInput.size() != seedLen)
                throw std::runtime_error("CTR_DRBG_Update: additional input "
                                         "must be of length seedLen");
            for (std::size_t i = 0; i < seedLen; i++)
            {
                temp[i] ^= additionalInput[i];
            }
        }
        // Update internal state: new key is first keyLen bytes; new V is the
        // remaining 16 bytes.
        key.assign(temp.begin(), temp.begin() + keyLen);
        nonce.assign(temp.begin() + keyLen, temp.end());
        
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                keyExpansion(this->key, this->roundKeys);
                break;
            case SecurityLevel::AES192:
                keyExpansion192(this->key, this->roundKeys);
                break;
            case SecurityLevel::AES256:
                keyExpansion256(this->key, this->roundKeys);
                break;
            default:
                throw std::runtime_error("Unsupported security level");
        }
        cudaMemcpy(this->d_nonce, (this->nonce).data(), 4 * sizeof(Data32),
                   cudaMemcpyHostToDevice);
    }

    void AES_RNG::resetReseedCounter()
    {
        this->reseedCounter = 1;
    }

    void AES_RNG::reseed(std::vector<unsigned char> additionalInput)
    {
        if (additionalInput.size() < seedLen - keyLen)
        {
            for (int i = 0; i < seedLen - keyLen - additionalInput.size(); i++)
                additionalInput.push_back(0);
        }
        std::vector<unsigned char> entropyInput(keyLen, 0);
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<Data32> dist(
            0, std::numeric_limits<Data8>::max());

        for (int i = 0; i < keyLen; i++)
            entropyInput[i] = dist(gen);

        for (int i = 0; i < seedLen - keyLen; i++)
            entropyInput.push_back(additionalInput[i]);

        std::vector<unsigned char> seedMaterial = DF(entropyInput, seedLen);
        this->update(seedMaterial);
        this->resetReseedCounter();
    }

    void AES_RNG::printWorkingState()
    {
        std::cout << "------DRBG State------\n";
        std::cout << "Key: " << std::hex << std::uppercase;
        for (int i = 0; i < keyLen - 3; i += 4)
            std::cout << (int) this->key[i] << (int) this->key[i + 1]
                      << (int) this->key[i + 2] << (int) this->key[i + 3]
                      << " ";
        std::cout << std::endl << "V: ";
        for (int i = 0; i < 13; i += 4)
            std::cout << (int) this->nonce[i] << (int) this->nonce[i + 1]
                      << (int) this->nonce[i + 2] << (int) this->nonce[i + 3]
                      << " ";
        std::cout << std::endl;
        std::cout << std::dec << "Reseed Counter: " << reseedCounter
                  << std::endl;
    }

    // DF (Derivation Function) per NIST SP 800‑90A.
    // According to NIST, the DF input should be constructed as follows:
    // [requestedOutputBits (4 bytes) || inputLengthBits (4 bytes) || input ||
    // 0x80 || padding] Then, encrypt S using AES-CBC with a zero key and zero
    // IV to produce the seed.
    std::vector<unsigned char>
    AES_RNG::DF(const std::vector<unsigned char>& input, std::size_t outputLen)
    {
        unsigned int requestedBits = static_cast<unsigned int>(outputLen * 8);
        std::vector<unsigned char> S;
        S.push_back((requestedBits >> 24) & 0xFF);
        S.push_back((requestedBits >> 16) & 0xFF);
        S.push_back((requestedBits >> 8) & 0xFF);
        S.push_back(requestedBits & 0xFF);

        unsigned int inputBits = static_cast<unsigned int>(input.size() * 8);

        S.push_back((inputBits >> 24) & 0xFF);
        S.push_back((inputBits >> 16) & 0xFF);
        S.push_back((inputBits >> 8) & 0xFF);
        S.push_back(inputBits & 0xFF);

        S.insert(S.end(), input.begin(), input.end());
        S.push_back(0x80);
        while (S.size() % 16 != 0)
            S.push_back(0x00);

        // Use a zero key whose length is equal to keyLen.
        std::vector<unsigned char> zeroKey(16, 0x00);
        unsigned char zeroIV[16] = {0};

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error("DF: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, this->getEVPCipherECB(), nullptr,
                                    zeroKey.data(), zeroIV))
            throw std::runtime_error("DF: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> cipher(S.size() + 16);
        int outlen1 = 0, outlen2 = 0;
        if (1 !=
            EVP_EncryptUpdate(ctx, cipher.data(), &outlen1, S.data(), S.size()))
            throw std::runtime_error("DF: EVP_EncryptUpdate failed");
        if (1 != EVP_EncryptFinal_ex(ctx, cipher.data() + outlen1, &outlen2))
            throw std::runtime_error("DF: EVP_EncryptFinal_ex failed");
        EVP_CIPHER_CTX_free(ctx);

        cipher.resize(outlen1 + outlen2);
        if (cipher.size() > outputLen)
            cipher.resize(outputLen);
        return cipher;
    }
} // namespace rngongpu