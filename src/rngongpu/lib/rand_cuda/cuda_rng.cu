
// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng.cuh"

namespace rngongpu
{
    template <typename State> RNG<Mode::CUDA, State>::RNG(Data64 seed)
    {
        RNGTraits<Mode::CUDA, State>::initialize(*this, seed);
    }

    template <typename State> RNG<Mode::CUDA, State>::~RNG()
    {
        RNGTraits<Mode::CUDA, State>::clear(*this);
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNG<Mode::CUDA, State>::uniform_random_number(T* pointer, const Data64 size,
                                                  cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::CUDA, State>::generate_uniform_random_number(
            *this, pointer, size, stream);
    }

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::modular_uniform_random_number(
        T* pointer, Modulus<T> modulus, const Data64 size, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::CUDA, State>::generate_modular_uniform_random_number(
            *this, pointer, modulus, size, stream);
    }

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::modular_uniform_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int repeat_count, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        RNGTraits<Mode::CUDA, State>::generate_modular_uniform_random_number(
            *this, pointer, modulus, log_size, mod_count, repeat_count, stream);
    }

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::modular_uniform_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        RNGTraits<Mode::CUDA, State>::generate_modular_uniform_random_number(
            *this, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, stream);
    }

    // --

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::normal_random_number(
        T std_dev, T* pointer, const Data64 size, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::CUDA, State>::generate_normal_random_number(
            *this, std_dev, pointer, size, stream);
    }

    template <typename State>
    template <typename T, typename U>
    __host__ void RNG<Mode::CUDA, State>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T> modulus, const Data64 size,
        cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::CUDA, State>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, size, stream);
    }

    template <typename State>
    template <typename T, typename U>
    __host__ void RNG<Mode::CUDA, State>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        RNGTraits<Mode::CUDA, State>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, log_size, mod_count, repeat_count,
            stream);
    }

    template <typename State>
    template <typename T, typename U>
    __host__ void RNG<Mode::CUDA, State>::modular_normal_random_number(
        U std_dev, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        RNGTraits<Mode::CUDA, State>::generate_modular_normal_random_number(
            *this, std_dev, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, stream);
    }

    // --

    template <typename State>
    template <typename T>
    __host__ void
    RNG<Mode::CUDA, State>::ternary_random_number(T* pointer, const Data64 size,
                                                  cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::CUDA, State>::generate_ternary_random_number(
            *this, pointer, size, stream);
    }

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::modular_ternary_random_number(
        T* pointer, Modulus<T> modulus, const Data64 size, cudaStream_t stream)
    {
        if (size == 0)
            return;

        CheckCudaPointer(pointer);

        RNGTraits<Mode::CUDA, State>::generate_modular_ternary_random_number(
            *this, pointer, modulus, size, stream);
    }

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::modular_ternary_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int repeat_count, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);

        RNGTraits<Mode::CUDA, State>::generate_modular_ternary_random_number(
            *this, pointer, modulus, log_size, mod_count, repeat_count, stream);
    }

    template <typename State>
    template <typename T>
    __host__ void RNG<Mode::CUDA, State>::modular_ternary_random_number(
        T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, cudaStream_t stream)
    {
        if ((log_size == 0) || (repeat_count == 0))
            return;

        CheckCudaPointer(pointer);
        CheckCudaPointer(modulus);
        CheckCudaPointer(mod_index);

        RNGTraits<Mode::CUDA, State>::generate_modular_ternary_random_number(
            *this, pointer, modulus, log_size, mod_count, mod_index,
            repeat_count, stream);
    }

    template class RNG<Mode::CUDA, curandStateXORWOW>;
    template class RNG<Mode::CUDA, curandStateMRG32k3a>;
    template class RNG<Mode::CUDA, curandStatePhilox4_32_10>;

    // --

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::uniform_random_number<Data32>(
        Data32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::uniform_random_number<Data64>(
        Data64* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::uniform_random_number<Data32>(
        Data32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::uniform_random_number<Data64>(
        Data64* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::uniform_random_number<Data32>(
        Data32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::uniform_random_number<Data64>(
        Data64* pointer, const Data64 size, cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_uniform_random_number<
        Data32>(Data32* pointer, Modulus<Data32> modulus, const Data64 size,
                cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_uniform_random_number<
        Data64>(Data64* pointer, Modulus<Data64> modulus, const Data64 size,
                cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_uniform_random_number<
        Data32>(Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
                int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_uniform_random_number<
        Data64>(Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
                int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_uniform_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_uniform_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_uniform_random_number<
        Data32>(Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
                int mod_count, int* mod_index, int repeat_count,
                cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_uniform_random_number<
        Data64>(Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
                int mod_count, int* mod_index, int repeat_count,
                cudaStream_t stream);

    // --

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::normal_random_number<f32>(
        f32 std_dev, f32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::normal_random_number<f64>(
        f64 std_dev, f64* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::normal_random_number<f32>(
        f32 std_dev, f32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::normal_random_number<f64>(
        f64 std_dev, f64* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::normal_random_number<f32>(
        f32 std_dev, f32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::normal_random_number<f64>(
        f64 std_dev, f64* pointer, const Data64 size, cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64> modulus,
                     const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64> modulus,
                     const Data64 size, cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int repeat_count,
                     cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data32, f32>(f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data32, f64>(f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data64, f32>(f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_normal_random_number<
        Data64, f64>(f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
                     Data64 log_size, int mod_count, int* mod_index,
                     int repeat_count, cudaStream_t stream);

    // --

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::ternary_random_number<Data32>(
        Data32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::ternary_random_number<Data64>(
        Data64* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::ternary_random_number<Data32>(
        Data32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::ternary_random_number<Data64>(
        Data64* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::ternary_random_number<Data32>(
        Data32* pointer, const Data64 size, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::ternary_random_number<Data64>(
        Data64* pointer, const Data64 size, cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64> modulus, const Data64 size,
        cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_ternary_random_number<
        Data32>(Data32* pointer, Modulus<Data32> modulus, const Data64 size,
                cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_ternary_random_number<
        Data64>(Data64* pointer, Modulus<Data64> modulus, const Data64 size,
                cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_ternary_random_number<
        Data32>(Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
                int mod_count, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_ternary_random_number<
        Data64>(Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
                int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateXORWOW>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_ternary_random_number<Data32>(
        Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStateMRG32k3a>::modular_ternary_random_number<Data64>(
        Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_ternary_random_number<
        Data32>(Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
                int mod_count, int* mod_index, int repeat_count,
                cudaStream_t stream);
    template __host__ void
    RNG<Mode::CUDA, curandStatePhilox4_32_10>::modular_ternary_random_number<
        Data64>(Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
                int mod_count, int* mod_index, int repeat_count,
                cudaStream_t stream);

} // end namespace rngongpu
