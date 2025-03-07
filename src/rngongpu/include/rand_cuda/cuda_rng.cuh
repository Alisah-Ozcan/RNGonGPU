// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_RNG_H
#define CUDA_RNG_H

#include "cuda_rng.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <iostream>
#include "aes.cuh"
#include "cuda_rng_kernels.cuh"
#include "base_rng.cuh"

namespace rngongpu
{
    template <typename State> struct ModeFeature<Mode::CUDA, State>
    {
      protected:
        const int thread_per_block_ = 512;
        int num_blocks_;
        int num_states_;
        State* device_states_;
        Data64 seed_;
        friend struct RNGTraits<Mode::CUDA, State>;
    };

    template <typename State> struct RNGTraits<Mode::CUDA, State>
    {
        static __host__ void
        initialize(ModeFeature<Mode::CUDA, State>& features, Data64 seed)
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

            init_state_kernel<<<features.num_blocks_,
                                features.thread_per_block_>>>(
                features.device_states_, features.seed_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }

        static __host__ void clear(ModeFeature<Mode::CUDA, State>& features)
        {
            if (features.device_states_)
            {
                cudaFree(features.device_states_);
                RNGONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }

        template <typename T>
        static __host__ void
        generate_uniform_random_number(ModeFeature<Mode::CUDA, State>& features,
                                       T* pointer, Data64 size)
        {
            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, size, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T> modulus, Data64 size)
        {
            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, modulus, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count)
        {
            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count)
        {
            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                mod_index, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // --

        template <typename T>
        static __host__ void
        generate_normal_random_number(ModeFeature<Mode::CUDA, State>& features,
                                      T std_dev, T* pointer, Data64 size)
        {
            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, std_dev, pointer, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T> modulus, Data64 size)
        {
            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, std_dev, pointer, modulus, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count)
        {
            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, std_dev, pointer, modulus, log_size,
                mod_count, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count)
        {
            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, std_dev, pointer, modulus, log_size,
                mod_count, mod_index, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        // --

        template <typename T>
        static __host__ void
        generate_ternary_random_number(ModeFeature<Mode::CUDA, State>& features,
                                       T* pointer, Data64 size)
        {
            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, size, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T> modulus, Data64 size)
        {
            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, modulus, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count)
        {
            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count)
        {
            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                mod_index, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());
        }
    };

    template <typename State>
    class RNG<Mode::CUDA, State> : public ModeFeature<Mode::CUDA, State>
    {
      public:
        __host__ explicit RNG(Data64 seed);

        ~RNG();

        template <typename T>
        __host__ void uniform_random_number(T* pointer, const Data64 size);

        template <typename T>
        __host__ void modular_uniform_random_number(T* pointer,
                                                    Modulus<T> modulus,
                                                    const Data64 size);

        template <typename T>
        __host__ void
        modular_uniform_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int repeat_count = 1);

        template <typename T>
        __host__ void
        modular_uniform_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int* mod_index, int repeat_count = 1);

        // --

        template <typename T>
        __host__ void normal_random_number(T std_dev, T* pointer,
                                           const Data64 size);

        template <typename T, typename U>
        __host__ void modular_normal_random_number(U std_dev, T* pointer,
                                                   Modulus<T> modulus,
                                                   const Data64 size);

        template <typename T, typename U>
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T>* modulus,
                                     Data64 log_size, int mod_count,
                                     int repeat_count = 1);

        template <typename T, typename U>
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T>* modulus,
                                     Data64 log_size, int mod_count,
                                     int* mod_index, int repeat_count = 1);

        // --

        template <typename T>
        __host__ void ternary_random_number(T* pointer, const Data64 size);

        template <typename T>
        __host__ void modular_ternary_random_number(T* pointer,
                                                    Modulus<T> modulus,
                                                    const Data64 size);

        template <typename T>
        __host__ void
        modular_ternary_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int repeat_count = 1);

        template <typename T>
        __host__ void
        modular_ternary_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int* mod_index, int repeat_count = 1);
    };

} // namespace rngongpu

#endif // CUDA_RNG_H
