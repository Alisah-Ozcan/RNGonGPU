// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AES_RNG_H
#define AES_RNG_H

#include "rngongpu/common/base_rng.cuh"
#include "rngongpu/common/aes.cuh"
//#include "rngongpu/common/base_rng.cuh"
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <mutex>

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

        std::mutex mutex_;

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
                   bool prediction_resistance_enabled);

        static __host__ void clear(ModeFeature<Mode::AES>& features);

        static __host__ const EVP_CIPHER*
        get_EVP_cipher_ECB(ModeFeature<Mode::AES>& features);

        static __host__ std::vector<unsigned char>
        uint32_to_bytes(unsigned int x);

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
                            std::size_t no_of_bits_to_return);

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
                                    std::vector<unsigned char> additional_input,
                                    cudaStream_t stream = cudaStreamDefault);

        static __host__ void
        increment_nonce(ModeFeature<Mode::AES>& features, Data32 size,
                        cudaStream_t stream = cudaStreamDefault);

        static __host__ void
        gen_random_bytes(ModeFeature<Mode::AES>& features, Data64* pointer,
                         Data64 requested_number_of_bytes,
                         const std::vector<unsigned char>& entropy_input,
                         const std::vector<unsigned char>& additional_input,
                         cudaStream_t stream);

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
               std::vector<unsigned char> additional_input,
               cudaStream_t stream);

        static __host__ std::vector<unsigned char>
        block_encrypt(ModeFeature<Mode::AES>& features,
                      const std::vector<unsigned char>& key,
                      const std::vector<unsigned char>& plaintext);

        static __host__ std::vector<unsigned char>
        BCC(ModeFeature<Mode::AES>& features,
            const std::vector<unsigned char>& key,
            const std::vector<unsigned char>& data);

        // --

        template <typename T>
        static __host__ void generate_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Data32 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T> modulus,
            Data32 size, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        // --

        template <typename T>
        static __host__ void generate_normal_random_number(
            ModeFeature<Mode::AES>& features, T std_dev, T* pointer,
            Data32 size, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
            Modulus<T> modulus, Data32 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data32 log_size, int mod_count,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::AES>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data32 log_size, int mod_count, int* mod_index,
            int repeat_count, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        // --

        template <typename T>
        static __host__ void generate_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Data32 size,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T> modulus,
            Data32 size, std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::AES>& features, T* pointer, Modulus<T>* modulus,
            Data32 log_size, int mod_count, int* mod_index, int repeat_count,
            std::vector<unsigned char>& entropy_input,
            std::vector<unsigned char> additional_input, cudaStream_t stream);
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

        void set(const std::vector<unsigned char>& entropy_input,
                 const std::vector<unsigned char>& nonce,
                 const std::vector<unsigned char>& personalization_string,
                 cudaStream_t stream = cudaStreamDefault);

        void reseed(const std::vector<unsigned char>& entropy_input,
                    const std::vector<unsigned char>& additional_input,
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
