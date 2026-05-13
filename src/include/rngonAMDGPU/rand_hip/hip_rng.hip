// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HIP_RNG_H
#define HIP_RNG_H

#include "rngonAMDGPU/rand_hip/hip_rng.hip"
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <hiprand/hiprand_mtgp32_host.h>
// #include <hiprand/hiprand_mtgp32dc_p_11213.h>   // doesn't exist in HIP; constants are in the above header
#include <iostream>
#include "rngonAMDGPU/common/aes.h"
#include "rngonAMDGPU/rand_hip/hip_rng_kernels.hip"
#include "rngonAMDGPU/common/base_rng.h"
#include <mutex>

namespace rngonAMDGPU
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
        friend struct RNGTraits<Mode::CUDA, State>;
    };

    template <typename State> struct RNGTraits<Mode::CUDA, State>
    {
        static __host__ void
        initialize(ModeFeature<Mode::CUDA, State>& features, Data64 seed);

        static __host__ void clear(ModeFeature<Mode::CUDA, State>& features);

        template <typename T>
        static __host__ void
        generate_uniform_random_number(ModeFeature<Mode::CUDA, State>& features,
                                       T* pointer, Data64 size,
                                       hipStream_t stream);

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T> modulus, Data64 size, hipStream_t stream);

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, hipStream_t stream);

        template <typename T>
        static __host__ void generate_modular_uniform_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count, hipStream_t stream);

        // --

        template <typename T>
        static __host__ void
        generate_normal_random_number(ModeFeature<Mode::CUDA, State>& features,
                                      T std_dev, T* pointer, Data64 size,
                                      hipStream_t stream);

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T> modulus, Data64 size, hipStream_t stream);

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, hipStream_t stream);

        template <typename T, typename U>
        static __host__ void generate_modular_normal_random_number(
            ModeFeature<Mode::CUDA, State>& features, U std_dev, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count, hipStream_t stream);

        // --

        template <typename T>
        static __host__ void
        generate_ternary_random_number(ModeFeature<Mode::CUDA, State>& features,
                                       T* pointer, Data64 size,
                                       hipStream_t stream);

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T> modulus, Data64 size, hipStream_t stream);

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, hipStream_t stream);

        template <typename T>
        static __host__ void generate_modular_ternary_random_number(
            ModeFeature<Mode::CUDA, State>& features, T* pointer,
            Modulus<T>* modulus, Data64 log_size, int mod_count, int* mod_index,
            int repeat_count, hipStream_t stream);
    };

    template <typename State>
    class RNG<Mode::CUDA, State> : public ModeFeature<Mode::CUDA, State>
    {
      public:
        __host__ explicit RNG(Data64 seed);

        ~RNG();

        template <typename T>
        __host__ void
        uniform_random_number(T* pointer, const Data64 size,
                              hipStream_t stream = hipStreamDefault);

        template <typename T>
        __host__ void
        modular_uniform_random_number(T* pointer, Modulus<T> modulus,
                                      const Data64 size,
                                      hipStream_t stream = hipStreamDefault);

        template <typename T>
        __host__ void modular_uniform_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, hipStream_t stream = hipStreamDefault);

        template <typename T>
        __host__ void
        modular_uniform_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int* mod_index, int repeat_count,
                                      hipStream_t stream = hipStreamDefault);

        // --

        template <typename T>
        __host__ void
        normal_random_number(T std_dev, T* pointer, const Data64 size,
                             hipStream_t stream = hipStreamDefault);

        template <typename T, typename U>
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T> modulus,
                                     const Data64 size,
                                     hipStream_t stream = hipStreamDefault);

        template <typename T, typename U>
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T>* modulus,
                                     Data64 log_size, int mod_count,
                                     int repeat_count,
                                     hipStream_t stream = hipStreamDefault);

        template <typename T, typename U>
        __host__ void
        modular_normal_random_number(U std_dev, T* pointer, Modulus<T>* modulus,
                                     Data64 log_size, int mod_count,
                                     int* mod_index, int repeat_count,
                                     hipStream_t stream = hipStreamDefault);

        // --

        template <typename T>
        __host__ void
        ternary_random_number(T* pointer, const Data64 size,
                              hipStream_t stream = hipStreamDefault);

        template <typename T>
        __host__ void
        modular_ternary_random_number(T* pointer, Modulus<T> modulus,
                                      const Data64 size,
                                      hipStream_t stream = hipStreamDefault);

        template <typename T>
        __host__ void modular_ternary_random_number(
            T* pointer, Modulus<T>* modulus, Data64 log_size, int mod_count,
            int repeat_count, hipStream_t stream = hipStreamDefault);

        template <typename T>
        __host__ void
        modular_ternary_random_number(T* pointer, Modulus<T>* modulus,
                                      Data64 log_size, int mod_count,
                                      int* mod_index, int repeat_count,
                                      hipStream_t stream = hipStreamDefault);
    };

} // namespace rngonAMDGPU

#endif // HIP_RNG_H
