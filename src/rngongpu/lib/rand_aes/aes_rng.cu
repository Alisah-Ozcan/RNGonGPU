// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "base_rng.cuh"
#include <random>

namespace rngongpu
{
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