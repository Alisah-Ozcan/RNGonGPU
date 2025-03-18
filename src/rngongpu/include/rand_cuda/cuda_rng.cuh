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
#include <mutex>

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
        std::mutex mutex_;
        cudaEvent_t event_control_;
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
                                       T* pointer, Data64 size,
                                       cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, size, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T> modulus, Data64 size, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, modulus, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            uniform_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                mod_index, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // --

        template <typename T>
        static __host__ void
        generate_normal_random_number(ModeFeature<Mode::CUDA, State>& features,
                                      T std_dev, T* pointer, Data64 size,
                                      cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, std_dev, pointer, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T> modulus, Data64 size, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, std_dev, pointer, modulus, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, std_dev, pointer, modulus, log_size,
                mod_count, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            normal_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, std_dev, pointer, modulus, log_size,
                mod_count, mod_index, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // --

        template <typename T>
        static __host__ void
        generate_ternary_random_number(ModeFeature<Mode::CUDA, State>& features,
                                       T* pointer, Data64 size,
                                       cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, size, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T> modulus, Data64 size, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, modulus, size,
                features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count, cudaStream_t stream)
        {
            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                if (features.event_control_)
                {
                    cudaStreamWaitEvent(stream, features.event_control_, 0);
                    cudaEventDestroy(features.event_control_);
                    features.event_control_ = nullptr;
                }
            }

            ternary_random_number_generation_kernel<<<
                features.num_blocks_, features.thread_per_block_, 0, stream>>>(
                features.device_states_, pointer, modulus, log_size, mod_count,
                mod_index, repeat_count, features.num_states_);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            cudaEvent_t event_in;
            cudaEventCreate(&event_in);
            cudaEventRecord(event_in, stream);
            RNGONGPU_CUDA_CHECK(cudaGetLastError());

            {
                std::lock_guard<std::mutex> lock(features.mutex_);
                features.event_control_ = event_in;
            }

            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    template <typename State>
    class RNG<Mode::CUDA, State> : public ModeFeature<Mode::CUDA, State>
    {
      public:
        __host__ explicit RNG(Data64 seed);

        ~RNG();

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
         */
        template <typename T>
        __host__ void
        uniform_random_number(T* pointer, const Data64 size,
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
         */
        template <typename T>
        __host__ void
        modular_uniform_random_number(T* pointer, Modulus<T> modulus,
                                      const Data64 size,
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
            int repeat_count, cudaStream_t stream = cudaStreamDefault);

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
        __host__ void
        modular_uniform_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int* mod_index, int repeat_count,
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
         */
        template <typename T>
        __host__ void
        normal_random_number(T std_dev, T* pointer, const Data64 size,
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
         */
        template <typename T, typename U>
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T> modulus,
                                     const Data64 size,
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
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T>* modulus,
                                     Data64 log_size, int mod_count,
                                     int repeat_count,
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
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T>* modulus,
                                     Data64 log_size, int mod_count,
                                     int* mod_index, int repeat_count,
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
         */
        template <typename T>
        __host__ void
        ternary_random_number(T* pointer, const Data64 size,
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
         */
        template <typename T>
        __host__ void
        modular_ternary_random_number(T* pointer, Modulus<T> modulus,
                                      const Data64 size,
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
            int repeat_count, cudaStream_t stream = cudaStreamDefault);
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
        __host__ void
        modular_ternary_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int* mod_index, int repeat_count,
                                      cudaStream_t stream = cudaStreamDefault);
    };

} // namespace rngongpu

#endif // CUDA_RNG_H
