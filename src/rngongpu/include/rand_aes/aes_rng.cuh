// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AES_RNG_H
#define AES_RNG_H

#include "aes.cuh"
#include "base_rng.cuh"
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>

namespace rngongpu
{
    enum class SecurityLevel
    {
        AES128,
        AES192,
        AES256
    };

    template <> struct ModeFeature<Mode::AES>
    {
      protected:
        std::vector<unsigned char> seed_;
        std::vector<unsigned char> key_;
        std::vector<unsigned char> nonce_;

        bool is_prediction_resistance_enabled_;

        // NIST SP 800‑90A recommends that the number of blocks generated before
        // a reseed be limited.
        const Data64 reseed_interval_ = (1ULL << 48);
        const Data32 max_bytes_per_request_ = 1 << 19;
        SecurityLevel security_level_;
        Data32 key_len_; // Key length in bytes (16 for AES-128, 24 for AES-192,
                         // 32 for AES-256)
        const Data32 nonce_len_ =
            16; // Nonce length in bytes (16 for all AES-128, AES-192, AES-256)
        Data32 seed_len_; // seedLen = keyLen + nonce_len_
        const Data32 out_len_ = 16; // for AES
        const Data32 block_len_ = 16; // for AES
        Data64 reseed_counter_;

        // AES-128 relevant fields
        Data32* t0_;
        Data32* t1_;
        Data32* t2_;
        Data32* t3_;
        Data32* t4_;
        Data32* t4_0_;
        Data32* t4_1_;
        Data32* t4_2_;
        Data32* t4_3_;

        Data8* SAES_d_;
        Data32* rcon_;
        Data32* round_keys_;
        Data32* d_nonce_;

        const int thread_per_block_ = 1024;
        int num_blocks_;

        friend struct RNGTraits<Mode::AES>;
    };

    template <> struct RNGTraits<Mode::AES>
    {
        /**
         * @brief Instantiates the DRBG using a derivation function.
         *
         * According to NIST Special Publication 800-90A, when instantiation is
         * performed using this method, the entropy input may not have full
         * entropy; therefore, a nonce is required. Let @c df be the derivation
         * function specified in Section 10.3.2. Unlike the method in
         * Section 10.2.1.3.1, which does not require a nonce due to the full
         * entropy provided, this instantiation method mandates a nonce.
         *
         * @param entropy_input The bit string obtained from the randomness
         * source.
         * @param nonce A bit string as specified in Section 8.6.7.
         * @param personalization_string The personalization string provided by
         * the consuming application. Note that this string may be empty.
         * @param security_level The security strength for the instantiation.
         * This parameter is optional for CTR_DRBG, as it is not used.
         * @param prediction_resistance_enabled If set to true, a reseed is
         * performed on each generate_bytes() call.
         */
        static __host__ void
        initialize(ModeFeature<Mode::AES>& features,
                   const std::vector<unsigned char>& entropy_input,
                   const std::vector<unsigned char>& nonce,
                   const std::vector<unsigned char>& personalization_string,
                   SecurityLevel security_level,
                   bool prediction_resistance_enabled)
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
                    throw std::runtime_error(
                        "Error: Unsupported security level!");
            }
            features.seed_len_ = features.key_len_ + features.nonce_len_;
            features.seed_ = entropy_input;
            features.seed_.insert(features.seed_.end(), nonce.begin(),
                                  nonce.end());
            features.seed_.insert(features.seed_.end(),
                                  personalization_string.begin(),
                                  personalization_string.end());
            std::vector<unsigned char> seed_material = derivation_function(
                features, features.seed_, features.seed_len_);
            features.key_ = std::vector<unsigned char>(features.key_len_, 0);
            features.nonce_ =
                std::vector<unsigned char>(features.nonce_len_, 0);

            switch (features.security_level_)
            {
                case SecurityLevel::AES128:
                    cudaMallocManaged(&(features.round_keys_),
                                      AES_128_KEY_SIZE_INT * sizeof(Data32));
                    RNGONGPU_CUDA_CHECK(cudaGetLastError());
                    break;
                case SecurityLevel::AES192:
                    cudaMallocManaged(&(features.round_keys_),
                                      AES_192_KEY_SIZE_INT * sizeof(Data32));
                    RNGONGPU_CUDA_CHECK(cudaGetLastError());
                    break;
                case SecurityLevel::AES256:
                    cudaMallocManaged(&(features.round_keys_),
                                      AES_256_KEY_SIZE_INT * sizeof(Data32));
                    RNGONGPU_CUDA_CHECK(cudaGetLastError());
                    break;
                default:
                    throw std::runtime_error(
                        "Error: Unsupported security level!");
            }

            cudaMalloc(&features.d_nonce_, 4 * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            update(features, seed_material);

            cudaMallocManaged(&features.rcon_, RCON_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            for (int i = 0; i < RCON_SIZE; i++)
            {
                features.rcon_[i] = RCON32[i];
            }

            cudaMallocManaged(&features.t0_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t1_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t2_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t3_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t4_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t4_0_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t4_1_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t4_2_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.t4_3_, TABLE_SIZE * sizeof(Data32));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaMallocManaged(&features.SAES_d_, 256 * sizeof(Data8));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            for (int i = 0; i < TABLE_SIZE; i++)
            {
                features.t0_[i] = T0[i];
                features.t1_[i] = T1[i];
                features.t2_[i] = T2[i];
                features.t3_[i] = T3[i];
                features.t4_[i] = T4[i];
                features.t4_0_[i] = T4_0[i];
                features.t4_1_[i] = T4_1[i];
                features.t4_2_[i] = T4_2[i];
                features.t4_3_[i] = T4_3[i];
            }
            for (int i = 0; i < 256; i++)
                features.SAES_d_[i] = SAES[i];
            std::vector<unsigned char> nonce_rev = features.nonce_;
            std::reverse(nonce_rev.begin(), nonce_rev.end());
            cudaMemcpy(features.d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                       cudaMemcpyHostToDevice);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        static __host__ void clear(ModeFeature<Mode::AES>& features)
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

        static __host__ const EVP_CIPHER*
        get_EVP_cipher_ECB(ModeFeature<Mode::AES>& features)
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

        static __host__ std::vector<unsigned char>
        uint32_to_bytes(unsigned int x)
        {
            std::vector<unsigned char> bytes(4);
            bytes[0] = (x >> 24) & 0xFF;
            bytes[1] = (x >> 16) & 0xFF;
            bytes[2] = (x >> 8) & 0xFF;
            bytes[3] = x & 0xFF;
            return bytes;
        }

        /**
         * @brief Derivation function as specified in NIST Special Publication
         * 800-90A.
         *
         * This derivation function is used by the CTR_DRBG described in
         * Section 10.2. BCC and Block_Encrypt are discussed in Section 10.3.3.
         * Let @c out_len_ denote the output block length, which is a multiple
         * of eight bits for the approved block cipher algorithms, and let @c
         * key_len_ denote the key length.
         *
         * @param input_string The string to be processed. It must have a length
         * that is a multiple of eight bits.
         * @param no_of_bits_to_return The number of bits to be returned by
         * Block_Cipher_df. The maximum allowable value is 512 bits for the
         * currently approved block cipher algorithms.
         */
        static __host__ std::vector<unsigned char>
        derivation_function(ModeFeature<Mode::AES>& features,
                            const std::vector<unsigned char>& input_string,
                            std::size_t no_of_bits_to_return)
        {
            unsigned int input_bits =
                static_cast<unsigned int>(input_string.size());
            std::vector<unsigned char> S = uint32_to_bytes(input_bits);

            unsigned int requested_bit =
                static_cast<unsigned int>(no_of_bits_to_return);
            std::vector<unsigned char> len_bytes =
                uint32_to_bytes(requested_bit);

            S.reserve(S.size() + len_bytes.size() + input_string.size());
            S.insert(S.end(), len_bytes.begin(), len_bytes.end());
            if (!input_string.empty()) {
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

                std::vector<unsigned char> bccResult =
                    BCC(features, K, dataForBCC);
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

        /**
         * @brief Updates the internal state of the CTR_DRBG.
         *
         * This function updates the internal state of the CTR_DRBG using the
         * provided data. The values for @c block_len_, @c key_len_, and @c
         * seed_len_ are specified in Table 3 of Section 10.2.1. The value of @c
         * ctr_len_ is determined by the implementation. In step 2.2 of the
         * CTR_DRBG_UPDATE process, the block cipher operation employs the
         * selected block cipher algorithm, as discussed in Section 10.3.3.
         *
         * @param provided_data The data to be used for the update. It must be
         * exactly @c seed_len_ bits in length, a condition ensured by the
         * construction of the provided data in the instantiate, reseed, and
         * generate functions.
         */
        static __host__ void update(ModeFeature<Mode::AES>& features,
                                    std::vector<unsigned char> additional_input)
        {
            EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
            if (!ctx)
                throw std::runtime_error("Error: Failed to create "
                                         "EVP_CIPHER_CTX in CTR_DRBG_Update!");

            if (1 != EVP_EncryptInit_ex(ctx, get_EVP_cipher_ECB(features),
                                        nullptr, (features.key_).data(),
                                        nullptr))
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
                    throw std::runtime_error(
                        "update: EVP_EncryptUpdate failed");
                if (outlen != static_cast<int>(features.block_len_))
                    throw std::runtime_error("update: Unexpected block size");
                temp.insert(temp.end(), output_block.begin(),
                            output_block.end());
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

            features.key_.assign(temp.begin(),
                                 temp.begin() + features.key_len_);
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
            cudaMemcpy(features.d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                       cudaMemcpyHostToDevice);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        static __host__ void increment_nonce(ModeFeature<Mode::AES>& features,
                                             Data32 size)
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
            cudaMemcpy(features.d_nonce_, nonce_rev.data(), 4 * sizeof(Data32),
                       cudaMemcpyHostToDevice);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        static __host__ void
        gen_random_bytes(ModeFeature<Mode::AES>& features, Data64* pointer,
                         Data64 requested_number_of_bytes,
                         const std::vector<unsigned char>& entropy_input,
                         const std::vector<unsigned char>& additional_input,
                         cudaStream_t stream)
        {
            std::vector<unsigned char> additional_input_in;
            if (features.is_prediction_resistance_enabled_ ||
                features.reseed_counter_ >= features.reseed_interval_)
            {
                reseed(features, entropy_input, additional_input);
                additional_input_in =
                    std::vector<unsigned char>(features.seed_len_, 0);
            }
            else
            {
                if (additional_input.size() != 0)
                {
                    additional_input_in = derivation_function(
                        features, additional_input, features.seed_len_);
                    update(features, additional_input_in);
                }
                else
                {
                    additional_input_in =
                        std::vector<unsigned char>(features.seed_len_, 0);
                }
            }

            Data32 num_u64 =
                static_cast<Data32>((requested_number_of_bytes + 7) / 8);

            Data64* range;
            cudaMallocManaged(&range, sizeof(Data64));
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            Data32 threadCount =
                features.num_blocks_ * features.thread_per_block_;
            double threadCount_d = static_cast<double>(num_u64);
            double threadRange = threadCount_d / (threadCount * 2);
            *range = ceil(threadRange);

            switch (features.security_level_)
            {
                case SecurityLevel::AES128:
                    counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir<<<
                        features.num_blocks_, features.thread_per_block_, 0,
                        stream>>>(features.d_nonce_, features.round_keys_,
                                  features.t0_, features.t4_, range,
                                  features.SAES_d_, threadCount, pointer,
                                  num_u64);
                    RNGONGPU_CUDA_CHECK(cudaGetLastError());
                    break;
                case SecurityLevel::AES192:
                    counter192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<
                        features.num_blocks_, features.thread_per_block_, 0,
                        stream>>>(features.d_nonce_, features.round_keys_,
                                  features.t0_, features.t4_, range,
                                  threadCount, pointer, num_u64);
                    RNGONGPU_CUDA_CHECK(cudaGetLastError());
                    break;
                case SecurityLevel::AES256:
                    counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<
                        features.num_blocks_, features.thread_per_block_, 0,
                        stream>>>(features.d_nonce_, features.round_keys_,
                                  features.t0_, features.t4_, range,
                                  threadCount, pointer, num_u64);
                    RNGONGPU_CUDA_CHECK(cudaGetLastError());
                    break;
                default:
                    throw std::runtime_error("Error: Unsupported security "
                                             "level in gen_random_bytes!");
            }

            cudaFree(range); // Remove it!
            increment_nonce(features, (num_u64 + 1) / 2);
            update(features, additional_input_in);
            features.reseed_counter_ +=
                (requested_number_of_bytes / features.max_bytes_per_request_ +
                 1);
        }

        /**
         * @brief Reseeds the DRBG when a derivation function is used.
         *
         * According to NIST Special Publication 800-90A, let @c df be the
         * derivation function specified in Section 10.3.2. The following
         * process, or its equivalent, is used as the reseed algorithm for this
         * DRBG mechanism (see step 6 of the reseed process in Section 9.2):
         *
         * @param entropy_input The bit string obtained from the randomness
         * source.
         * @param additional_input The additional input string provided by the
         * consuming application. Note that the additional input string may be
         * empty.
         */
        static __host__ void
        reseed(ModeFeature<Mode::AES>& features,
               const std::vector<unsigned char>& entropy_input,
               std::vector<unsigned char> additional_input)
        {
            std::vector<unsigned char> additional_input_in = additional_input;
            std::vector<unsigned char> seed_material = entropy_input;
            seed_material.insert(seed_material.end(),
                                 additional_input_in.begin(),
                                 additional_input_in.end());
            seed_material = derivation_function(features, seed_material,
                                                features.seed_len_);

            update(features, seed_material);
            features.reseed_counter_ = 1;
        }

        static __host__ std::vector<unsigned char>
        block_encrypt(ModeFeature<Mode::AES>& features,
                      const std::vector<unsigned char>& key,
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
            if (1 != EVP_EncryptInit_ex(ctx, get_EVP_cipher_ECB(features),
                                        nullptr, key.data(), nullptr))
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

        static __host__ std::vector<unsigned char>
        BCC(ModeFeature<Mode::AES>& features,
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
        static __host__ void generate_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Data32 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T> modulus,
            Data32 size, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            mod_reduce_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                pointer, modulus, size, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            int size = 1 << log_size;
            Data64 total_byte_count =
                static_cast<Data64>(size * repeat_count) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            mod_reduce_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                pointer, modulus, log_size, mod_count, repeat_count,
                total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            int size = 1 << log_size;
            Data64 total_byte_count =
                static_cast<Data64>(size * repeat_count) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            mod_reduce_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                pointer, modulus, log_size, mod_count, mod_index, repeat_count,
                total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // --

        template <typename T>
        static __host__ void generate_normal_random_number(
            ModeFeature<Mode::AES>& features, T std_dev, T* pointer,
            Data32 size, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64;
            Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
            cudaMallocAsync(&pointer64, total_byte_count, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            box_muller_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                std_dev, pointer64, pointer, size, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
            Modulus<T> modulus, Data32 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            box_muller_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                std_dev, pointer, modulus, size, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data32 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64;
            int size = 1 << log_size;
            Data64 total_byte_count =
                static_cast<Data64>(size * repeat_count) * sizeof(T);
            cudaMallocAsync(&pointer64, total_byte_count, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            T* pointer_T = reinterpret_cast<T*>(pointer64);
            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            box_muller_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                std_dev, pointer_T, pointer, modulus, log_size, mod_count,
                repeat_count, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data32 log_size, int mod_count, int* mod_index,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64;
            int size = 1 << log_size;
            Data64 total_byte_count =
                static_cast<Data64>(size * repeat_count) * sizeof(T);
            cudaMallocAsync(&pointer64, total_byte_count, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            T* pointer_T = reinterpret_cast<T*>(pointer64);
            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            box_muller_kernel<<<features.num_blocks_,
                                features.thread_per_block_, 0, stream>>>(
                std_dev, pointer_T, pointer, modulus, log_size, mod_count,
                mod_index, repeat_count, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
        }

        // --

        template <typename T>
        static __host__ void generate_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Data32 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            ternary_number_kernel<<<features.num_blocks_,
                                    features.thread_per_block_, 0, stream>>>(
                pointer, size, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T> modulus,
            Data32 size, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64 = reinterpret_cast<Data64*>(pointer);
            Data64 total_byte_count = static_cast<Data64>(size) * sizeof(T);
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            ternary_number_kernel<<<features.num_blocks_,
                                    features.thread_per_block_, 0, stream>>>(
                pointer, modulus, size, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64;
            int size = 1 << log_size;
            Data64 total_byte_count =
                static_cast<Data64>(size * repeat_count) * sizeof(T);
            cudaMallocAsync(&pointer64, total_byte_count, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            T* pointer_T = reinterpret_cast<T*>(pointer64);
            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            ternary_number_kernel<<<features.num_blocks_,
                                    features.thread_per_block_, 0, stream>>>(
                pointer_T, pointer, modulus, log_size, mod_count, repeat_count,
                total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream)
        {
            Data64* pointer64;
            int size = 1 << log_size;
            Data64 total_byte_count =
                static_cast<Data64>(size * repeat_count) * sizeof(T);
            cudaMallocAsync(&pointer64, total_byte_count, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            gen_random_bytes(features, pointer64, total_byte_count,
                             entropy_input, additional_input, stream);

            T* pointer_T = reinterpret_cast<T*>(pointer64);
            int total_thread =
                features.num_blocks_ * features.thread_per_block_;
            ternary_number_kernel<<<features.num_blocks_,
                                    features.thread_per_block_, 0, stream>>>(
                pointer_T, pointer, modulus, log_size, mod_count, mod_index,
                repeat_count, total_thread);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            RNGONGPU_CUDA_CHECK(cudaFree(pointer64));
        }
    };

    template <> class RNG<Mode::AES> : public ModeFeature<Mode::AES>
    {
      public:
        __host__ explicit RNG(
            const std::vector<unsigned char>& key,
            const std::vector<unsigned char>& nonce,
            const std::vector<unsigned char>& personalization_string,
            SecurityLevel security_level,
            bool prediction_resistance_enabled = false);

        ~RNG();

        void print_params(std::ostream& out = std::cout);

        const std::vector<unsigned char>& get_key() const { return this->key_; }

        const std::vector<unsigned char>& get_nonce() const
        {
            return this->nonce_;
        }

        void reseed(const std::vector<unsigned char>& entropy_input,
                    const std::vector<unsigned char>& additional_input);

        /**
         * @brief Generates uniform random numbers.
         *
         * This function generates uniformly distributed random numbers.
         * The numbers are written to the memory pointed to by @p pointer, which
         * must reside on the device or in unified memory. If the pointer does
         * not reference device or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param size The number of random numbers to generate.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         */
        template <typename T>
        __host__ void
        uniform_random_number(T* pointer, const Data64 size,
                              std::vector<unsigned char> additional_input,
                              cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates uniform random numbers.
         *
         * This function generates uniformly distributed random numbers.
         * The numbers are written to the memory pointed to by @p pointer, which
         * must reside on the device or in unified memory. If the pointer does
         * not reference device or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param size The number of random numbers to generate.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         */
        template <typename T>
        __host__ void
        uniform_random_number(T* pointer, const Data64 size,
                              std::vector<unsigned char>& entropy_input,
                              std::vector<unsigned char> additional_input,
                              cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo.
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulus. The numbers are written to the memory pointed to by
         * @p pointer, which must reside on the GPU or in unified memory. If the
         * pointer does not reference GPU or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers.
         * @param size The number of random numbers to generate.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         */
        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T> modulus, const Data64 size,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo.
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulus. The numbers are written to the memory pointed to by
         * @p pointer, which must reside on the GPU or in unified memory. If the
         * pointer does not reference GPU or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers.
         * @param size The number of random numbers to generate.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         */
        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T> modulus, const Data64 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order.
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q0, array3 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order.
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q1, array2 % q0, array3 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order.
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0, array1]
         *
         *   - output array : [array0 % q0, array1 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q3, array2 % q0, array3 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular uniform random numbers according to given
         * modulo order.
         *
         * This function produces uniform random numbers that are reduced modulo
         * a given modulo order. The numbers are written to the memory pointed
         * to by @p pointer, which must reside on the GPU or in unified memory.
         * If the pointer does not reference GPU or unified memory, an error is
         * thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0, array1]
         *
         *   - output array : [array0 % q0, array1 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1, array2, array3]
         *
         *   - output array : [array0 % q0, array1 % q3, array2 % q0, array3 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        // --

        /**
         * @brief Generates Gaussian-distributed random numbers.
         *
         * This function generates Gaussian-distributed random numbers.
         * The numbers are written to the memory pointed to by @p pointer, which
         * must reside on the device or in unified memory. If the pointer does
         * not reference device or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param size The number of random numbers to generate.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         */
        template <typename T>
        __host__ void
        normal_random_number(T std_dev, T* pointer, const Data64 size,
                             std::vector<unsigned char> additional_input,
                             cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers.
         *
         * This function generates Gaussian-distributed random numbers.
         * The numbers are written to the memory pointed to by @p pointer, which
         * must reside on the device or in unified memory. If the pointer does
         * not reference device or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param size The number of random numbers to generate.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         */
        template <typename T>
        __host__ void
        normal_random_number(T std_dev, T* pointer, const Data64 size,
                             std::vector<unsigned char>& entropy_input,
                             std::vector<unsigned char> additional_input,
                             cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * domain
         *
         * This function generates Gaussian-distributed random numbers in given
         * modulo domain. The numbers are written to the memory pointed to by @p
         * pointer, which must reside on the GPU or in unified memory. If the
         * pointer does not reference GPU or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @tparam U The data type of the standart deviation. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers.
         * @param size The number of random numbers to generate.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         */
        template <typename T, typename U>
        __host__ void modular_normal_random_number(
            U std_dev, T* pointer, Modulus<T> modulus, const Data64 size,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * domain
         *
         * This function generates Gaussian-distributed random numbers in given
         * modulo domain. The numbers are written to the memory pointed to by @p
         * pointer, which must reside on the GPU or in unified memory. If the
         * pointer does not reference GPU or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @tparam U The data type of the standart deviation. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers.
         * @param size The number of random numbers to generate.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         */
        template <typename T, typename U>
        __host__ void modular_normal_random_number(
            U std_dev, T* pointer, Modulus<T> modulus, const Data64 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order.
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @tparam U The data type of the standart deviation. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T, typename U>
        __host__ void modular_normal_random_number(
            U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
            int mod_count, int repeat_count,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order.
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @tparam U The data type of the standart deviation. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - array order  : [array0, array1] since repeat_count = 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T, typename U>
        __host__ void modular_normal_random_number(
            U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
            int mod_count, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order.
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @tparam U The data type of the standart deviation. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T, typename U>
        __host__ void modular_normal_random_number(
            U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Gaussian-distributed random numbers in given modulo
         * order.
         *
         * This function produces Gaussian-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @tparam U The data type of the standart deviation. (f32 or f64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T, typename U>
        __host__ void modular_normal_random_number(
            U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        // --

        /**
         * @brief Generates Ternary-distributed random numbers. (-1,0,1)
         *
         * This function generates Ternary-distributed random numbers.
         * The numbers are written to the memory pointed to by @p pointer, which
         * must reside on the device or in unified memory. If the pointer does
         * not reference device or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param size The number of random numbers to generate.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         */
        template <typename T>
        __host__ void
        ternary_random_number(T* pointer, const Data64 size,
                              std::vector<unsigned char> additional_input,
                              cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers. (-1,0,1)
         *
         * This function generates Ternary-distributed random numbers.
         * The numbers are written to the memory pointed to by @p pointer, which
         * must reside on the device or in unified memory. If the pointer does
         * not reference device or unified memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param size The number of random numbers to generate.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         */
        template <typename T>
        __host__ void
        ternary_random_number(T* pointer, const Data64 size,
                              std::vector<unsigned char>& entropy_input,
                              std::vector<unsigned char> additional_input,
                              cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular Ternary-distributed random numbers according
         * to given modulo. (-1,0,1)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulus. The numbers are written to the memory
         * pointed to by @p pointer, which must reside on the GPU or in unified
         * memory. If the pointer does not reference GPU or unified memory, an
         * error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers.
         * @param size The number of random numbers to generate.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         */
        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T> modulus, const Data64 size,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates modular Ternary-distributed random numbers according
         * to given modulo. (-1,0,1)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulus. The numbers are written to the memory
         * pointed to by @p pointer, which must reside on the GPU or in unified
         * memory. If the pointer does not reference GPU or unified memory, an
         * error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers.
         * @param size The number of random numbers to generate.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         */
        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T> modulus, const Data64 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order.(-1,0,1)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order.(-1,0,1)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 3, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q1, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q1, array1 % q0, array1 %
         * q1]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order. (-1,0,1)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @note  If needed, an entropy input is generated randomly within the
         * function.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates Ternary-distributed random numbers in given modulo
         * order. (-1,0,1)
         *
         * This function produces Ternary-distributed random numbers that are
         * reduced modulo a given modulo order. The numbers are written to the
         * memory pointed to by @p pointer, which must reside on the GPU or in
         * unified memory. If the pointer does not reference GPU or unified
         * memory, an error is thrown.
         *
         * @tparam T The data type of the random numbers. (Data32 or Data64)
         * @param std_dev Standart deviation.
         * @param pointer Pointer to the memory for storing random numbers
         * (device or unified memory required).
         * @param modulus The modulus used to reduce the generated random
         * numbers (device or unified memory required).
         * @param log_size The log domain number of random numbers to generate.
         * (log_size should be power of 2)
         * @param mod_count The mod count indicates how many different moduli
         * are used.
         * @param mod_index The mod index indicates which modules are used
         * (device or unified memory required).
         * @param repeat_count The repeat count indicates how many times the
         * modulus order is repeated.
         * @param entropy_input The entropy input string of bits obtained from
         * the randomness source.
         * @param additional_input The additional input string received from the
         * consuming application. Note that the length of the additional_input
         * string may be zero.
         *
         * @example
         * example1: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 1
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    2  ]
         *
         *   - array order  : [array0] since repeat_count = 1
         *
         *   - output array : [array0 % q0, array0 % q2]
         *
         * example2: each array size = 2^log_size, mod_count = 2, repeat_count =
         * 2
         *   - modulus order: [  q0 ,   q1 ,   q2 ,   q3 ]
         *   - mod_index    : [  0   ,    3  ]
         *
         *   - array order  : [array0, array1] since repeat_count = 2
         *
         *   - output array : [array0 % q0, array0 % q3, array1 % q0, array1 %
         * q3]
         *
         * @note Total generated random number count = (2^log_size) x mod_count
         * x repeat_count
         */
        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input,
            cudaStream_t stream = cudaStreamDefault);
    };

} // namespace rngongpu

#endif // AES_RNG_H