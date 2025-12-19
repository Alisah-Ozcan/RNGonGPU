// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "rngongpu/rand_aes/aes_rng.cuh"
#include "rngongpu/common/base_rng.cuh"
#include <random>

namespace rngongpu
{
    __host__ void RNGTraits<Mode::AES>::initialize(
        ModeFeature<Mode::AES>& features,
        const std::vector<unsigned char>& entropy_input,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalization_string,
        SecurityLevel security_level, bool prediction_resistance_enabled)
    {
        if (entropy_input.size() < 16)
        {
            throw std::runtime_error("Error: Invalid key size!");
        }
        int device;
        cudaGetDevice(&device);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaDeviceGetAttribute(&features.num_blocks_,
                               cudaDevAttrMultiProcessorCount, device);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        features.reseed_counter_ = 1ULL;
        features.is_prediction_resistance_enabled_ =
            prediction_resistance_enabled;
        features.security_level_ = security_level;

        switch (features.security_level_)
        {
            case SecurityLevel::AES128:
                features.key_len_ = 16;
                break;
            case SecurityLevel::AES192:
                features.key_len_ = 24;
                break;
            case SecurityLevel::AES256:
                features.key_len_ = 32;
                break;
            default:
                throw std::runtime_error("Error: Unsupported security level!");
        }
        features.seed_len_ = features.key_len_ + features.nonce_len_;
        features.seed_ = entropy_input;
        features.seed_.insert(features.seed_.end(), nonce.begin(), nonce.end());
        features.seed_.insert(features.seed_.end(),
                              personalization_string.begin(),
                              personalization_string.end());
        std::vector<unsigned char> seed_material =
            derivation_function(features, features.seed_, features.seed_len_);
        features.key_ = std::vector<unsigned char>(features.key_len_, 0);
        features.nonce_ = std::vector<unsigned char>(features.nonce_len_, 0);

        switch (features.security_level_)
        {
            case SecurityLevel::AES128:
                cudaMalloc(&(features.round_keys_),
                                  AES_128_KEY_SIZE_INT * sizeof(Data32));
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
                break;
            case SecurityLevel::AES192:
                cudaMalloc(&(features.round_keys_),
                                  AES_192_KEY_SIZE_INT * sizeof(Data32));
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
                break;
            case SecurityLevel::AES256:
                cudaMalloc(&(features.round_keys_),
                                  AES_256_KEY_SIZE_INT * sizeof(Data32));
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
                break;
            default:
                throw std::runtime_error("Error: Unsupported security level!");
        }

        cudaMalloc(&features.d_nonce_, 4 * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        update(features, seed_material);
        
        cudaMalloc(&features.rcon_, RCON_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.rcon_, RCON32, RCON_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t0_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t0_, T0, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t1_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t1_, T1, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t2_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t2_, T2, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t3_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t3_, T3, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t4_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t4_, T4, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t4_0_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t4_0_, T4_0, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t4_1_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t4_1_, T4_1, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t4_2_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t4_2_, T4_2, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMalloc(&features.t4_3_, TABLE_SIZE * sizeof(Data32));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.t4_3_, T4_3, TABLE_SIZE * sizeof(Data32), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
                
        cudaMalloc(&features.SAES_d_, 256 * sizeof(Data8));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaMemcpy(features.SAES_d_, SAES, 256 * sizeof(Data8), cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
 
        std::vector<unsigned char> nonce_rev = features.nonce_;
        std::reverse(nonce_rev.begin(), nonce_rev.end());
        cudaMemcpy(features.d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                   cudaMemcpyHostToDevice);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void RNGTraits<Mode::AES>::clear(ModeFeature<Mode::AES>& features)
    {
        RNGONGPU_CUDA_CHECK(cudaFree(features.t0_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t1_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t2_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t3_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t4_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t4_0_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t4_1_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t4_2_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.t4_3_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.rcon_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.SAES_d_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.d_nonce_));
        RNGONGPU_CUDA_CHECK(cudaFree(features.round_keys_));
    }

    __host__ const EVP_CIPHER*
    RNGTraits<Mode::AES>::get_EVP_cipher_ECB(ModeFeature<Mode::AES>& features)
    {
        switch (features.security_level_)
        {
            case SecurityLevel::AES128:
                return EVP_aes_128_ecb();
            case SecurityLevel::AES192:
                return EVP_aes_192_ecb();
            case SecurityLevel::AES256:
                return EVP_aes_256_ecb();
            default:
                throw std::runtime_error(
                    "Error: Unsupported security level in ECB!");
        }
    }

    __host__ std::vector<unsigned char>
    RNGTraits<Mode::AES>::uint32_to_bytes(unsigned int x)
    {
        std::vector<unsigned char> bytes(4);
        bytes[0] = (x >> 24) & 0xFF;
        bytes[1] = (x >> 16) & 0xFF;
        bytes[2] = (x >> 8) & 0xFF;
        bytes[3] = x & 0xFF;
        return bytes;
    }

    __host__ std::vector<unsigned char>
    RNGTraits<Mode::AES>::derivation_function(
        ModeFeature<Mode::AES>& features,
        const std::vector<unsigned char>& input_string,
        std::size_t no_of_bits_to_return)
    {
        unsigned int input_bits =
            static_cast<unsigned int>(input_string.size());
        std::vector<unsigned char> S = uint32_to_bytes(input_bits);

        unsigned int requested_bit =
            static_cast<unsigned int>(no_of_bits_to_return);
        std::vector<unsigned char> len_bytes = uint32_to_bytes(requested_bit);

        S.reserve(S.size() + len_bytes.size() + input_string.size());
        S.insert(S.end(), len_bytes.begin(), len_bytes.end());
        if (!input_string.empty())
        {
            S.insert(S.end(), input_string.begin(), input_string.end());
        }

        S.push_back(0x80);
        while (S.size() % features.out_len_ != 0)
            S.push_back(0x00);

        std::vector<unsigned char> temp;

        uint32_t i = 0;

        std::vector<unsigned char> K;
        for (uint32_t j = 0; j < features.key_len_; j++)
        {
            K.push_back(static_cast<unsigned char>(j));
        }

        while (temp.size() < (features.key_len_ + features.out_len_))
        {
            std::vector<unsigned char> IV = uint32_to_bytes(i);
            while (IV.size() < features.out_len_)
            {
                IV.push_back(0x00);
            }

            std::vector<unsigned char> dataForBCC;
            dataForBCC.insert(dataForBCC.end(), IV.begin(), IV.end());
            dataForBCC.insert(dataForBCC.end(), S.begin(), S.end());

            std::vector<unsigned char> bccResult = BCC(features, K, dataForBCC);
            temp.insert(temp.end(), bccResult.begin(), bccResult.end());
            i++;
        }

        std::vector<unsigned char> newK(temp.begin(),
                                        temp.begin() + features.key_len_);
        K = newK;

        std::vector<unsigned char> X(temp.begin() + features.key_len_,
                                     temp.begin() + features.key_len_ +
                                         features.out_len_);

        temp.clear();

        while (temp.size() < no_of_bits_to_return)
        {
            X = block_encrypt(features, K, X);
            temp.insert(temp.end(), X.begin(), X.end());
        }

        std::vector<unsigned char> requested_bits(
            temp.begin(), temp.begin() + no_of_bits_to_return);

        return requested_bits;
    }

    __host__ void
    RNGTraits<Mode::AES>::update(ModeFeature<Mode::AES>& features,
                                 std::vector<unsigned char> additional_input,
                                 cudaStream_t stream)
    {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error("Error: Failed to create "
                                     "EVP_CIPHER_CTX in CTR_DRBG_Update!");

        if (1 != EVP_EncryptInit_ex(ctx, get_EVP_cipher_ECB(features), nullptr,
                                    (features.key_).data(), nullptr))
            throw std::runtime_error(
                "Error: EVP_EncryptInit_ex failed in CTR_DRBG_Update!");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> temp;
        temp.reserve(features.seed_len_);
        std::vector<unsigned char> output_block(features.block_len_);
        std::vector<unsigned char> Vtemp(features.nonce_);

        while (temp.size() < features.seed_len_)
        {
            for (int j = features.block_len_ - 1; j >= 0; j--)
            {
                if (++Vtemp[j] != 0)
                    break;
            }
            int outlen = 0;
            if (1 != EVP_EncryptUpdate(ctx, output_block.data(), &outlen,
                                       Vtemp.data(), features.block_len_))
                throw std::runtime_error("update: EVP_EncryptUpdate failed");
            if (outlen != static_cast<int>(features.block_len_))
                throw std::runtime_error("update: Unexpected block size");
            temp.insert(temp.end(), output_block.begin(), output_block.end());
        }
        EVP_CIPHER_CTX_free(ctx);

        if (!additional_input.empty())
        {
            if (additional_input.size() != features.seed_len_)
                throw std::runtime_error(
                    "Error: additional input "
                    "must be of length seedLen in CTR_DRBG_Update!");
            for (std::size_t i = 0; i < features.seed_len_; i++)
            {
                temp[i] ^= additional_input[i];
            }
        }

        features.key_.assign(temp.begin(), temp.begin() + features.key_len_);
        features.nonce_.assign(temp.begin() + features.key_len_,
                               temp.begin() + features.seed_len_);

        switch (features.security_level_)
        {
            case SecurityLevel::AES128:
                keyExpansion(features.key_, features.round_keys_);
                break;
            case SecurityLevel::AES192:
                keyExpansion192(features.key_, features.round_keys_);
                break;
            case SecurityLevel::AES256:
                keyExpansion256(features.key_, features.round_keys_);
                break;
            default:
                throw std::runtime_error("Error: Unsupported security "
                                         "level in CTR_DRBG_Update!");
        }

        std::vector<unsigned char> nonce_rev = features.nonce_;
        std::reverse(nonce_rev.begin(), nonce_rev.end());
        cudaMemcpyAsync(features.d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                        cudaMemcpyHostToDevice, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    RNGTraits<Mode::AES>::increment_nonce(ModeFeature<Mode::AES>& features,
                                          Data32 size, cudaStream_t stream)
    {
        Data32 carry = size;
        for (int i = features.nonce_.size() - 1; i >= 0 && carry > 0; i--)
        {
            Data32 sum = features.nonce_[i] + carry;
            features.nonce_[i] = sum & 0xFF; // 0-255
            carry = sum >> 8; // remainder after div 256.
        }

        std::vector<unsigned char> nonce_rev = features.nonce_;
        std::reverse(nonce_rev.begin(), nonce_rev.end());
        cudaMemcpyAsync(features.d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                        cudaMemcpyHostToDevice, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void RNGTraits<Mode::AES>::gen_random_bytes(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Data64 requested_number_of_bytes,
        const std::vector<unsigned char>& entropy_input,
        const std::vector<unsigned char>& additional_input, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input_in;
        if (features.is_prediction_resistance_enabled_ ||
            features.reseed_counter_ >= features.reseed_interval_)
        {
            reseed(features, entropy_input, additional_input, stream);
            additional_input_in =
                std::vector<unsigned char>(features.seed_len_, 0);
        }
        else
        {
            if (additional_input.size() != 0)
            {
                additional_input_in = derivation_function(
                    features, additional_input, features.seed_len_);
                update(features, additional_input_in, stream);
            }
            else
            {
                additional_input_in =
                    std::vector<unsigned char>(features.seed_len_, 0);
            }
        }

        Data32 num_u64 =
            static_cast<Data32>((requested_number_of_bytes + 7) / 8);

        Data32 threadCount = features.num_blocks_ * features.thread_per_block_;
        double threadCount_d = static_cast<double>(num_u64);
        double threadRange = threadCount_d / (threadCount * 2);
        Data64 range = ceil(threadRange);

        switch (features.security_level_)
        {
            case SecurityLevel::AES128:
                counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir<<<
                    features.num_blocks_, features.thread_per_block_, 0,
                    stream>>>(features.d_nonce_, features.round_keys_,
                              features.t0_, features.t4_, range,
                              features.SAES_d_, threadCount, pointer, num_u64);
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
                break;
            case SecurityLevel::AES192:
                counter192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<
                    features.num_blocks_, features.thread_per_block_, 0,
                    stream>>>(features.d_nonce_, features.round_keys_,
                              features.t0_, features.t4_, range, threadCount,
                              pointer, num_u64);
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
                break;
            case SecurityLevel::AES256:
                counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<
                    features.num_blocks_, features.thread_per_block_, 0,
                    stream>>>(features.d_nonce_, features.round_keys_,
                              features.t0_, features.t4_, range, threadCount,
                              pointer, num_u64);
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
                break;
            default:
                throw std::runtime_error("Error: Unsupported security "
                                         "level in gen_random_bytes!");
        }

        increment_nonce(features, (num_u64 + 1) / 2, stream);
        update(features, additional_input_in, stream);
        features.reseed_counter_ +=
            (requested_number_of_bytes / features.max_bytes_per_request_ + 1);
    }

    __host__ void RNGTraits<Mode::AES>::reseed(
        ModeFeature<Mode::AES>& features,
        const std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::vector<unsigned char> additional_input_in = additional_input;
        std::vector<unsigned char> seed_material = entropy_input;
        seed_material.insert(seed_material.end(), additional_input_in.begin(),
                             additional_input_in.end());
        seed_material =
            derivation_function(features, seed_material, features.seed_len_);

        update(features, seed_material, stream);
        features.reseed_counter_ = 1;
    }

    __host__ std::vector<unsigned char> RNGTraits<Mode::AES>::block_encrypt(
        ModeFeature<Mode::AES>& features, const std::vector<unsigned char>& key,
        const std::vector<unsigned char>& plaintext)
    {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
        {
            throw std::runtime_error("EVP_CIPHER_CTX_new failed");
        }

        std::vector<unsigned char> ciphertext(plaintext.size() +
                                              EVP_MAX_BLOCK_LENGTH);
        int len = 0, ciphertext_len = 0;
        if (1 != EVP_EncryptInit_ex(ctx, get_EVP_cipher_ECB(features), nullptr,
                                    key.data(), nullptr))
        {
            throw std::runtime_error("EVP_EncryptInit_ex failed");
        }

        EVP_CIPHER_CTX_set_padding(ctx, 0);

        if (1 != EVP_EncryptUpdate(ctx, ciphertext.data(), &len,
                                   plaintext.data(), plaintext.size()))
        {
            throw std::runtime_error("EVP_EncryptUpdate failed");
        }

        ciphertext_len = len;

        if (1 != EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len))
        {
            throw std::runtime_error("EVP_EncryptFinal_ex failed");
        }

        ciphertext_len += len;
        EVP_CIPHER_CTX_free(ctx);

        ciphertext.resize(ciphertext_len);
        return ciphertext;
    }

    __host__ std::vector<unsigned char>
    RNGTraits<Mode::AES>::BCC(ModeFeature<Mode::AES>& features,
                              const std::vector<unsigned char>& key,
                              const std::vector<unsigned char>& data)
    {
        if (data.size() % features.out_len_ != 0)
        {
            throw std::runtime_error(
                "BCC input data length is not a multiple of block size");
        }
        std::vector<unsigned char> X(features.out_len_, 0x00);
        size_t num_blocks = data.size() / features.out_len_;
        for (size_t i = 0; i < num_blocks; i++)
        {
            std::vector<unsigned char> block(
                data.begin() + i * features.out_len_,
                data.begin() + (i + 1) * features.out_len_);
            for (size_t j = 0; j < features.out_len_; j++)
            {
                X[j] ^= block[j];
            }
            X = block_encrypt(features, key, X);
        }
        return X;
    }

    // --

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_uniform_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Modulus<T> modulus,
        Data32 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        mod_reduce_kernel<<<features.num_blocks_, features.thread_per_block_, 0,
                            stream>>>(pointer, modulus, size, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
        Data32 log_size, int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 size = 1ULL << log_size;
        Data64 total_byte_count =
            static_cast<Data64>(size * repeat_count * mod_count) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        mod_reduce_kernel<<<features.num_blocks_, features.thread_per_block_, 0,
                            stream>>>(pointer, modulus, log_size, mod_count,
                                      repeat_count, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
        Data32 log_size, int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 size = 1ULL << log_size;
        Data64 total_byte_count =
            static_cast<Data64>(size * repeat_count * mod_count) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        mod_reduce_kernel<<<features.num_blocks_, features.thread_per_block_, 0,
                            stream>>>(pointer, modulus, log_size, mod_count,
                                      mod_index, repeat_count, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    // --

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_normal_random_number(
        ModeFeature<Mode::AES>& features, T std_dev, T* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64;
        Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
        cudaMallocAsync(&pointer64, total_byte_count, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        if constexpr (std::is_same_v<T, f32>)
        {
            Data32* pointer32 = reinterpret_cast<Data32*>(pointer64);
            box_muller_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                std_dev, pointer32, pointer, size, total_thread);
        }
        else if constexpr (std::is_same_v<T, f64>)
        {
            box_muller_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                std_dev, pointer64, pointer, size, total_thread);
        }

        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
    }

    template <typename T, typename U>
    __host__ void RNGTraits<Mode::AES>::generate_modular_normal_random_number(
        ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
        Modulus<T> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        box_muller_kernel<<<features.num_blocks_, features.thread_per_block_, 0,
                            stream>>>(std_dev, pointer, modulus, size,
                                      total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T, typename U>
    __host__ void RNGTraits<Mode::AES>::generate_modular_normal_random_number(
        ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
        Modulus<T>* modulus, Data32 log_size, int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64;
        Data64 size = 1ULL << log_size;
        Data64 total_byte_count =
            static_cast<Data64>(size * repeat_count) * sizeof(T);
        cudaMallocAsync(&pointer64, total_byte_count, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        T* pointer_T = reinterpret_cast<T*>(pointer64);
        int total_thread = features.num_blocks_ * features.thread_per_block_;

        box_muller_kernel<<<features.num_blocks_, features.thread_per_block_, 0,
                            stream>>>(std_dev, pointer_T, pointer, modulus,
                                      log_size, mod_count, repeat_count,
                                      total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
    }

    template <typename T, typename U>
    __host__ void RNGTraits<Mode::AES>::generate_modular_normal_random_number(
        ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
        Modulus<T>* modulus, Data32 log_size, int mod_count, int* mod_index,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64;
        Data64 size = 1ULL << log_size;
        Data64 total_byte_count =
            static_cast<Data64>(size * repeat_count) * sizeof(T);
        cudaMallocAsync(&pointer64, total_byte_count, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        T* pointer_T = reinterpret_cast<T*>(pointer64);
        int total_thread = features.num_blocks_ * features.thread_per_block_;
        box_muller_kernel<<<features.num_blocks_, features.thread_per_block_, 0,
                            stream>>>(std_dev, pointer_T, pointer, modulus,
                                      log_size, mod_count, mod_index,
                                      repeat_count, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
    }

    // --

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_ternary_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        ternary_number_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
            pointer, size, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Modulus<T> modulus,
        Data32 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
        Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        int total_thread = features.num_blocks_ * features.thread_per_block_;
        ternary_number_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
            pointer, modulus, size, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
    }

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
        Data32 log_size, int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64;
        Data64 size = 1ULL << log_size;
        Data64 total_byte_count =
            static_cast<Data64>(size * repeat_count) * sizeof(T);
        cudaMallocAsync(&pointer64, total_byte_count, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        T* pointer_T = reinterpret_cast<T*>(pointer64);
        int total_thread = features.num_blocks_ * features.thread_per_block_;
        ternary_number_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
            pointer_T, pointer, modulus, log_size, mod_count, repeat_count,
            total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
    }

    template <typename T>
    __host__ void RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
        ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
        Data32 log_size, int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        Data64* pointer64;
        Data64 size = 1ULL << log_size;
        Data64 total_byte_count =
            static_cast<Data64>(size * repeat_count) * sizeof(T);
        cudaMallocAsync(&pointer64, total_byte_count, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        gen_random_bytes(features, pointer64, total_byte_count, entropy_input,
                         additional_input, stream);

        T* pointer_T = reinterpret_cast<T*>(pointer64);
        int total_thread = features.num_blocks_ * features.thread_per_block_;
        ternary_number_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
            pointer_T, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, total_thread);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
    }

    ///////////////////////////////////////////////////////////////////////

    RNG<Mode::AES>::RNG(
        const std::vector<unsigned char>& entropyInput,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalization_string,
        SecurityLevel security_level, bool prediction_resistance_enabled)
    {
        RNGTraits<Mode::AES>::initialize(*this, entropyInput, nonce,
                                         personalization_string, security_level,
                                         prediction_resistance_enabled);
    }

    RNG<Mode::AES>::~RNG()
    {
        RNGTraits<Mode::AES>::clear(*this);
    }

    void RNG<Mode::AES>::print_params(std::ostream& out)
    {
        out << "\tKey\t= ";
        for (unsigned char byte : this->key_)
        {
            out << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(byte);
        }
        out << std::endl;

        out << "\tV\t= ";
        for (unsigned char byte : this->nonce_)
        {
            out << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(byte);
        }
        out << std::dec << std::endl << std::endl;
    }

    void RNG<Mode::AES>::set(
        const std::vector<unsigned char>& entropy_input,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalization_string,
        cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock((*this).mutex_);

        if (entropy_input.size() < 16)
        {
            throw std::runtime_error("Error: Invalid key size!");
        }

        (*this).reseed_counter_ = 1ULL;
        switch ((*this).security_level_)
        {
            case SecurityLevel::AES128:
                (*this).key_len_ = 16;
                break;
            case SecurityLevel::AES192:
                (*this).key_len_ = 24;
                break;
            case SecurityLevel::AES256:
                (*this).key_len_ = 32;
                break;
            default:
                throw std::runtime_error("Error: Unsupported security level!");
        }

        (*this).seed_len_ = (*this).key_len_ + (*this).nonce_len_;
        (*this).seed_ = entropy_input;
        (*this).seed_.insert((*this).seed_.end(), nonce.begin(), nonce.end());
        (*this).seed_.insert((*this).seed_.end(),
                             personalization_string.begin(),
                             personalization_string.end());
        std::vector<unsigned char> seed_material =
            RNGTraits<Mode::AES>::derivation_function(*this, (*this).seed_,
                                                      (*this).seed_len_);
        (*this).key_ = std::vector<unsigned char>((*this).key_len_, 0);
        (*this).nonce_ = std::vector<unsigned char>((*this).nonce_len_, 0);

        RNGTraits<Mode::AES>::update(*this, seed_material);

        std::vector<unsigned char> nonce_rev = (*this).nonce_;
        std::reverse(nonce_rev.begin(), nonce_rev.end());
        cudaMemcpyAsync((*this).d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                        cudaMemcpyHostToDevice, stream);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void
    RNG<Mode::AES>::reseed(const std::vector<unsigned char>& entropy_input,
                           const std::vector<unsigned char>& additional_input,
                           cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock((*this).mutex_);
        RNGTraits<Mode::AES>::reseed(*this, entropy_input, additional_input,
                                     stream);
        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::uniform_random_number(
        T* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_uniform_random_number(
            *this, pointer, size, generated_entropy, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::uniform_random_number(
        T* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::AES>::generate_uniform_random_number(
            *this, pointer, size, entropy_input, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_uniform_random_number(
        T* pointer, Modulus<T> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
            *this, pointer, modulus, size, generated_entropy, additional_input,
            stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_uniform_random_number(
        T* pointer, Modulus<T> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
            *this, pointer, modulus, size, entropy_input, additional_input,
            stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_uniform_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char> additional_input,
        cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
            *this, pointer, modulus, log_size, mod_count, repeat_count,
            generated_entropy, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_uniform_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
            *this, pointer, modulus, log_size, mod_count, repeat_count,
            entropy_input, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_uniform_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
            *this, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, generated_entropy, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_uniform_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        RNGTraits<Mode::AES>::generate_modular_uniform_random_number(
            *this, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, entropy_input, additional_input, stream);
    }

    // --

    template <typename T>
    __host__ void RNG<Mode::AES>::normal_random_number(
        T std_dev, T* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_normal_random_number(
            *this, std_dev, pointer, size, generated_entropy, additional_input,
            stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::normal_random_number(
        T std_dev, T* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::AES>::generate_normal_random_number(
            *this, std_dev, pointer, size, entropy_input, additional_input,
            stream);
    }

    template <typename T, typename U>
    __host__ void RNG<Mode::AES>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, size, generated_entropy,
            additional_input, stream);
    }

    template <typename T, typename U>
    __host__ void RNG<Mode::AES>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::AES>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, size, entropy_input,
            additional_input, stream);
    }

    template <typename T, typename U>
    __host__ void RNG<Mode::AES>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, log_size, mod_count, repeat_count,
            generated_entropy, additional_input, stream);
    }

    template <typename T, typename U>
    __host__ void RNG<Mode::AES>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        RNGTraits<Mode::AES>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, log_size, mod_count, repeat_count,
            entropy_input, additional_input, stream);
    }

    template <typename T, typename U>
    __host__ void RNG<Mode::AES>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, generated_entropy, additional_input, stream);
    }

    template <typename T, typename U>
    __host__ void RNG<Mode::AES>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        RNGTraits<Mode::AES>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, entropy_input, additional_input, stream);
    }

    // --

    template <typename T>
    __host__ void RNG<Mode::AES>::ternary_random_number(
        T* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_ternary_random_number(
            *this, pointer, size, generated_entropy, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::ternary_random_number(
        T* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::AES>::generate_ternary_random_number(
            *this, pointer, size, entropy_input, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_ternary_random_number(
        T* pointer, Modulus<T> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
            *this, pointer, modulus, size, generated_entropy, additional_input,
            stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_ternary_random_number(
        T* pointer, Modulus<T> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
            *this, pointer, modulus, size, entropy_input, additional_input,
            stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_ternary_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char> additional_input,
        cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
            *this, pointer, modulus, log_size, mod_count, repeat_count,
            generated_entropy, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_ternary_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
            *this, pointer, modulus, log_size, mod_count, repeat_count,
            entropy_input, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_ternary_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        std::vector<unsigned char> generated_entropy(this->key_len_);
        if (1 != RAND_bytes(generated_entropy.data(), generated_entropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
            *this, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, generated_entropy, additional_input, stream);
    }

    template <typename T>
    __host__ void RNG<Mode::AES>::modular_ternary_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        RNGTraits<Mode::AES>::generate_modular_ternary_random_number(
            *this, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, entropy_input, additional_input, stream);
    }

    template __host__ void
    RNGTraits<Mode::AES>::generate_uniform_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_uniform_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_uniform_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer,
        Modulus<Data32> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_uniform_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Modulus<Data64> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_uniform_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_uniform_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_uniform_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_uniform_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    // --

    template __host__ void
    RNGTraits<Mode::AES>::generate_normal_random_number<f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, f32* pointer,
        Data32 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_normal_random_number<f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, f64* pointer,
        Data32 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data32, f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data32, f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data64, f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data64, f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data32, f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data32, f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data64, f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data64, f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data32, f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data32, f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data64, f32>(
        ModeFeature<Mode::AES>& features, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_normal_random_number<Data64, f64>(
        ModeFeature<Mode::AES>& features, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    // --

    template __host__ void
    RNGTraits<Mode::AES>::generate_ternary_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_ternary_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_ternary_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer,
        Modulus<Data32> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_ternary_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Modulus<Data64> modulus, Data32 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_ternary_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_ternary_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int repeat_count, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_ternary_random_number<Data32>(
        ModeFeature<Mode::AES>& features, Data32* pointer,
        Modulus<Data32>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::AES>::generate_modular_ternary_random_number<Data64>(
        ModeFeature<Mode::AES>& features, Data64* pointer,
        Modulus<Data64>* modulus, Data32 log_size, int mod_count,
        int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    // --

    template __host__ void RNG<Mode::AES>::uniform_random_number<Data32>(
        Data32* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::uniform_random_number<Data64>(
        Data64* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::uniform_random_number<Data32>(
        Data32* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::uniform_random_number<Data64>(
        Data64* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    // --

    template __host__ void RNG<Mode::AES>::normal_random_number<f32>(
        f32 std_dev, f32* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::normal_random_number<f64>(
        f64 std_dev, f64* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::normal_random_number<f32>(
        f32 std_dev, f32* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::normal_random_number<f64>(
        f64 std_dev, f64* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f32>(
        f32 std_dev, Data32* pointer, Modulus<Data32> modulus,
        const Data64 size, std::vector<unsigned char> additional_input,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f64>(
        f64 std_dev, Data32* pointer, Modulus<Data32> modulus,
        const Data64 size, std::vector<unsigned char> additional_input,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f32>(
        f32 std_dev, Data64* pointer, Modulus<Data64> modulus,
        const Data64 size, std::vector<unsigned char> additional_input,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f64>(
        f64 std_dev, Data64* pointer, Modulus<Data64> modulus,
        const Data64 size, std::vector<unsigned char> additional_input,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f32>(
        f32 std_dev, Data32* pointer, Modulus<Data32> modulus,
        const Data64 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f64>(
        f64 std_dev, Data32* pointer, Modulus<Data32> modulus,
        const Data64 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f32>(
        f32 std_dev, Data64* pointer, Modulus<Data64> modulus,
        const Data64 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f64>(
        f64 std_dev, Data64* pointer, Modulus<Data64> modulus,
        const Data64 size, std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f32>(
        f32 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f64>(
        f64 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f32>(
        f32 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f64>(
        f64 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f32>(
        f32 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f64>(
        f64 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f32>(
        f32 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f64>(
        f64 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f32>(
        f32 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f64>(
        f64 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f32>(
        f32 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f64>(
        f64 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f32>(
        f32 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data32, f64>(
        f64 std_dev, Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f32>(
        f32 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_normal_random_number<Data64, f64>(
        f64 std_dev, Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    // --

    template __host__ void RNG<Mode::AES>::ternary_random_number<Data32>(
        Data32* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::ternary_random_number<Data64>(
        Data64* pointer, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::ternary_random_number<Data32>(
        Data32* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void RNG<Mode::AES>::ternary_random_number<Data64>(
        Data64* pointer, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);
    template __host__ void
    RNG<Mode::AES>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count,
        std::vector<unsigned char>& entropy_input,
        std::vector<unsigned char> additional_input, cudaStream_t stream);

} // namespace rngongpu
