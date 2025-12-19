
// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "rngongpu/rand_cuda/cuda_rng.cuh"

namespace rngongpu
{
    template <typename State>
    __host__ void RNGTraits<Mode::CUDA, State>::initialize(
        ModeFeature<Mode::CUDA, State>& features, Data64 seed)
    {
        int device;
        cudaGetDevice(&device);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        cudaDeviceGetAttribute(&features.num_blocks_,
                               cudaDevAttrMultiProcessorCount, device);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        features.num_states_ =
            features.thread_per_block_ * features.num_blocks_;
        features.seed_ = seed;

        cudaMalloc(&features.device_states_,
                   features.num_states_ * sizeof(State));
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        init_state_kernel<<<features.num_blocks_, features.thread_per_block_>>>(
            features.device_states_, features.seed_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    }

    template <typename State>
    __host__ void RNGTraits<Mode::CUDA, State>::clear(
        ModeFeature<Mode::CUDA, State>& features)
    {
        if (features.device_states_)
        {
            cudaFree(features.device_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    template <typename State>
    template <typename T>
    __host__ void RNGTraits<Mode::CUDA, State>::generate_uniform_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer, Data64 size,
        cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        uniform_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, size, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_uniform_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer,
        Modulus<T> modulus, Data64 size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        uniform_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, modulus, size,
            features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_uniform_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer,
        Modulus<T>* modulus, Data64 log_size, int mod_count, int repeat_count,
        cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        uniform_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, modulus, log_size, mod_count,
            repeat_count, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_uniform_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer,
        Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
        int repeat_count, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        uniform_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, modulus, log_size, mod_count,
            mod_index, repeat_count, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // --

    template <typename State>
    template <typename T>
    __host__ void RNGTraits<Mode::CUDA, State>::generate_normal_random_number(
        ModeFeature<Mode::CUDA, State>& features, T std_dev, T* pointer,
        Data64 size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        normal_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, std_dev, pointer, size,
            features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T, typename U>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_normal_random_number(
        ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
        Modulus<T> modulus, Data64 size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        normal_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, std_dev, pointer, modulus, size,
            features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T, typename U>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_normal_random_number(
        ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
        Modulus<T>* modulus, Data64 log_size, int mod_count, int repeat_count,
        cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        normal_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, std_dev, pointer, modulus, log_size,
            mod_count, repeat_count, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T, typename U>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_normal_random_number(
        ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
        Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
        int repeat_count, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        normal_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, std_dev, pointer, modulus, log_size,
            mod_count, mod_index, repeat_count, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // --

    template <typename State>
    template <typename T>
    __host__ void RNGTraits<Mode::CUDA, State>::generate_ternary_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer, Data64 size,
        cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        ternary_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, size, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_ternary_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer,
        Modulus<T> modulus, Data64 size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        ternary_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, modulus, size,
            features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_ternary_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer,
        Modulus<T>* modulus, Data64 log_size, int mod_count, int repeat_count,
        cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        ternary_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, modulus, log_size, mod_count,
            repeat_count, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    template <typename State>
    template <typename T>
    __host__ void
    RNGTraits<Mode::CUDA, State>::generate_modular_ternary_random_number(
        ModeFeature<Mode::CUDA, State>& features, T* pointer,
        Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
        int repeat_count, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(features.mutex_);

        ternary_random_number_generation_kernel<<<
            features.num_blocks_, features.thread_per_block_, 0, stream>>>(
            features.device_states_, pointer, modulus, log_size, mod_count,
            mod_index, repeat_count, features.num_states_);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

        RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    ///////////////////////////////////////////////////////////////////////

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

    template class RNGTraits<Mode::CUDA, curandStateXORWOW>;
    template class RNGTraits<Mode::CUDA, curandStateMRG32k3a>;
    template class RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>;

    // ----

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateXORWOW>::generate_uniform_random_number<
        Data32>(ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
                Data32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateXORWOW>::generate_uniform_random_number<
        Data64>(ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
                Data64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateMRG32k3a>::generate_uniform_random_number<
        Data32>(ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
                Data32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateMRG32k3a>::generate_uniform_random_number<
        Data64>(ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
                Data64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_uniform_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_uniform_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    // --

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateXORWOW>::generate_normal_random_number<
        f32>(ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
             f32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateXORWOW>::generate_normal_random_number<
        f64>(ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
             f64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateMRG32k3a>::generate_normal_random_number<
        f32>(ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
             f32 std_dev, f32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateMRG32k3a>::generate_normal_random_number<
        f64>(ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
             f64 std_dev, f64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_normal_random_number<f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, f32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_normal_random_number<f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, f64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f32 std_dev,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f64 std_dev,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f32 std_dev,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f64 std_dev,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f32 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f64 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f32 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f64 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
            Data64 log_size, int mod_count, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
            Data64 log_size, int mod_count, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
            Data64 log_size, int mod_count, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
            Data64 log_size, int mod_count, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f32 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features, f64 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f32 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f64 std_dev,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f32 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features, f64 std_dev,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data32, f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, Data32* pointer, Modulus<Data32>* modulus,
            Data64 log_size, int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data32, f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, Data32* pointer, Modulus<Data32>* modulus,
            Data64 log_size, int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data64, f32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f32 std_dev, Data64* pointer, Modulus<Data64>* modulus,
            Data64 log_size, int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_normal_random_number<Data64, f64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            f64 std_dev, Data64* pointer, Modulus<Data64>* modulus,
            Data64 log_size, int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    // --

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateXORWOW>::generate_ternary_random_number<
        Data32>(ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
                Data32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateXORWOW>::generate_ternary_random_number<
        Data64>(ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
                Data64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateMRG32k3a>::generate_ternary_random_number<
        Data32>(ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
                Data32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void
    RNGTraits<Mode::CUDA, curandStateMRG32k3a>::generate_ternary_random_number<
        Data64>(ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
                Data64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Data64 size, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Modulus<Data32> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Modulus<Data64> modulus, Data64 size,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int repeat_count, cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateXORWOW>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateXORWOW>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStateMRG32k3a>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStateMRG32k3a>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_ternary_random_number<Data32>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data32* pointer, Modulus<Data32>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    template __host__ void RNGTraits<Mode::CUDA, curandStatePhilox4_32_10>::
        generate_modular_ternary_random_number<Data64>(
            ModeFeature<Mode::CUDA, curandStatePhilox4_32_10>& features,
            Data64* pointer, Modulus<Data64>* modulus, Data64 log_size,
            int mod_count, int* mod_index, int repeat_count,
            cudaStream_t stream);

    // --

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
