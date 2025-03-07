// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng_kernels.cuh"
#include <curand_kernel.h>

namespace rngongpu
{
    template <typename State>
    __global__ void init_state_kernel(State* state, Data64 seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        curand_init(seed, idx, 0, &state[idx]);
    }

    template <typename State, typename T>
    __global__ void
    uniform_random_number_generation_kernel(State* state, T* pointer,
                                            Data64 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                pointer[i] = curand(&thread_state);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                uint32_t num_lo = curand(&thread_state);
                uint32_t num_hi = curand(&thread_state);

                uint64_t combined = (static_cast<uint64_t>(num_hi) << 32) |
                                    static_cast<uint64_t>(num_lo);

                pointer[i] = static_cast<Data64>(combined);
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T>
    __global__ void
    uniform_random_number_generation_kernel(State* state, T* pointer,
                                            Modulus<T> modulus, Data64 size,
                                            int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                T num = curand(&thread_state);
                pointer[i] = OPERATOR_GPU<T>::reduce_forced(num, modulus);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                uint32_t num_lo = curand(&thread_state);
                uint32_t num_hi = curand(&thread_state);

                uint64_t combined = (static_cast<uint64_t>(num_hi) << 32) |
                                    static_cast<uint64_t>(num_lo);

                T num = static_cast<Data64>(combined);
                pointer[i] = OPERATOR_GPU<T>::reduce_forced(num, modulus);
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T>
    __global__ void uniform_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];
        int size_wo_repeat = mod_count << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;
                int index = i >> log_size;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    T num = curand(&thread_state);
                    pointer[global_index] =
                        OPERATOR_GPU<T>::reduce_forced(num, modulus[index]);
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    uint32_t num_lo = curand(&thread_state);
                    uint32_t num_hi = curand(&thread_state);

                    uint64_t combined = (static_cast<uint64_t>(num_hi) << 32) |
                                        static_cast<uint64_t>(num_lo);

                    T num = static_cast<Data64>(combined);
                    pointer[global_index] =
                        OPERATOR_GPU<T>::reduce_forced(num, modulus[index]);
                }
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T>
    __global__ void uniform_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];
        int size_wo_repeat = mod_count << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;
                int index = i >> log_size;
                int new_index = mod_index[index];

                if constexpr (std::is_same_v<T, Data32>)
                {
                    T num = curand(&thread_state);
                    pointer[global_index] =
                        OPERATOR_GPU<T>::reduce_forced(num, modulus[new_index]);
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    uint32_t num_lo = curand(&thread_state);
                    uint32_t num_hi = curand(&thread_state);

                    uint64_t combined = (static_cast<uint64_t>(num_hi) << 32) |
                                        static_cast<uint64_t>(num_lo);

                    T num = static_cast<Data64>(combined);
                    pointer[global_index] =
                        OPERATOR_GPU<T>::reduce_forced(num, modulus[new_index]);
                }
            }
        }

        state[idx] = thread_state;
    }

    // -

    template <typename State, typename T>
    __global__ void
    normal_random_number_generation_kernel(State* state, T std_dev, T* pointer,
                                           Data64 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, f32>)
            {
                float num_f = curand_normal(&thread_state);
                num_f = num_f * std_dev;
                pointer[i] = num_f;
            }
            else if constexpr (std::is_same_v<T, f64>)
            {
                double num_f = curand_normal_double(&thread_state);
                num_f = num_f * std_dev;
                pointer[i] = num_f;
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T, typename U>
    __global__ void
    normal_random_number_generation_kernel(State* state, U std_dev, T* pointer,
                                           Modulus<T> modulus, Data64 size,
                                           int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                float num_f = curand_normal(&thread_state);
                num_f = num_f * std_dev;

                uint32_t flag =
                    static_cast<uint32_t>(-static_cast<int32_t>(num_f < 0));

                pointer[i] = static_cast<T>(num_f) + (flag & modulus.value);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                double num_f = curand_normal_double(&thread_state);
                num_f = num_f * std_dev;

                uint64_t flag =
                    static_cast<uint64_t>(-static_cast<int64_t>(num_f < 0));

                pointer[i] = static_cast<T>(num_f) + (flag & modulus.value);
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T, typename U>
    __global__ void normal_random_number_generation_kernel(
        State* state, U std_dev, T* pointer, Modulus<T>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];
        int size_wo_repeat = 1 << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    float num_f = curand_normal(&thread_state);
                    num_f = num_f * std_dev;

                    uint32_t flag =
                        static_cast<uint32_t>(-static_cast<int32_t>(num_f < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            static_cast<T>(num_f) + (flag & modulus[k].value);
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    double num_f = curand_normal_double(&thread_state);
                    num_f = num_f * std_dev;

                    uint64_t flag =
                        static_cast<uint64_t>(-static_cast<int64_t>(num_f < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            static_cast<T>(num_f) + (flag & modulus[k].value);
                    }
                }
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T, typename U>
    __global__ void
    normal_random_number_generation_kernel(State* state, U std_dev, T* pointer,
                                           Modulus<T>* modulus, Data64 log_size,
                                           int mod_count, int* mod_index,
                                           int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];
        int size_wo_repeat = 1 << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    float num_f = curand_normal(&thread_state);
                    num_f = num_f * std_dev;

                    uint32_t flag =
                        static_cast<uint32_t>(-static_cast<int32_t>(num_f < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            static_cast<T>(num_f) +
                            (flag & modulus[new_index].value);
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    double num_f = curand_normal_double(&thread_state);
                    num_f = num_f * std_dev;

                    uint64_t flag =
                        static_cast<uint64_t>(-static_cast<int64_t>(num_f < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            static_cast<T>(num_f) +
                            (flag & modulus[new_index].value);
                    }
                }
            }
        }

        state[idx] = thread_state;
    }

    // -

    template <typename State, typename T>
    __global__ void
    ternary_random_number_generation_kernel(State* state, T* pointer,
                                            Data64 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                T num = curand(&thread_state);
                pointer[i] = num & 1;
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                T num = static_cast<Data64>(curand(&thread_state));
                pointer[i] = num & 1ULL;
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T>
    __global__ void
    ternary_random_number_generation_kernel(State* state, T* pointer,
                                            Modulus<T> modulus, Data64 size,
                                            int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                T num = curand(&thread_state);
                num = num & 3;
                if (num == 3)
                {
                    num -= 3;
                }

                uint32_t flag =
                    static_cast<uint32_t>(-static_cast<int32_t>(num == 0));

                pointer[i] = num + (flag & modulus.value) - 1;
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                T num = curand(&thread_state);
                num = num & 3ULL;
                if (num == 3ULL)
                {
                    num -= 3ULL;
                }

                uint64_t flag =
                    static_cast<uint64_t>(-static_cast<int64_t>(num == 0));

                pointer[i] = num + (flag & modulus.value) - 1;
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T>
    __global__ void ternary_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];
        int size_wo_repeat = 1 << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    T num = curand(&thread_state);
                    num = num & 3;
                    if (num == 3)
                    {
                        num -= 3;
                    }

                    uint32_t flag =
                        static_cast<uint32_t>(-static_cast<int32_t>(num == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            num + (flag & modulus[k].value) - 1;
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    T num = curand(&thread_state);
                    num = num & 3ULL;
                    if (num == 3ULL)
                    {
                        num -= 3ULL;
                    }

                    uint64_t flag =
                        static_cast<uint64_t>(-static_cast<int64_t>(num == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            num + (flag & modulus[k].value) - 1;
                    }
                }
            }
        }

        state[idx] = thread_state;
    }

    template <typename State, typename T>
    __global__ void ternary_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        State thread_state = state[idx];
        int size_wo_repeat = 1 << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    T num = curand(&thread_state);
                    num = num & 3;
                    if (num == 3)
                    {
                        num -= 3;
                    }

                    uint32_t flag =
                        static_cast<uint32_t>(-static_cast<int32_t>(num == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            num + (flag & modulus[new_index].value) - 1;
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    T num = curand(&thread_state);
                    num = num & 3ULL;
                    if (num == 3ULL)
                    {
                        num -= 3ULL;
                    }

                    uint64_t flag =
                        static_cast<uint64_t>(-static_cast<int64_t>(num == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        pointer[global_index + in_offset] =
                            num + (flag & modulus[new_index].value) - 1;
                    }
                }
            }
        }

        state[idx] = thread_state;
    }

    template __global__ void
    init_state_kernel<curandStateXORWOW>(curandStateXORWOW* state, Data64 seed);
    template __global__ void
    init_state_kernel<curandStateMRG32k3a>(curandStateMRG32k3a* state,
                                           Data64 seed);
    template __global__ void
    init_state_kernel<curandStatePhilox4_32_10>(curandStatePhilox4_32_10* state,
                                                Data64 seed);

    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer, Data64 size,
        int max_state_num);

    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);

    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);

    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);

    // -

    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, f32>(
        curandStateXORWOW* state, f32 std_dev, f32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, f64>(
        curandStateXORWOW* state, f64 std_dev, f64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, f32>(
        curandStateMRG32k3a* state, f32 std_dev, f32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, f64>(
        curandStateMRG32k3a* state, f64 std_dev, f64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, f32>(
        curandStatePhilox4_32_10* state, f32 std_dev, f32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, f64>(
        curandStatePhilox4_32_10* state, f64 std_dev, f64* pointer, Data64 size,
        int max_state_num);

    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data32, f32>(
        curandStateXORWOW* state, f32 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data32, f64>(
        curandStateXORWOW* state, f64 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data64, f32>(
        curandStateXORWOW* state, f32 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data64, f64>(
        curandStateXORWOW* state, f64 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data32, f32>(
        curandStateMRG32k3a* state, f32 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data32, f64>(
        curandStateMRG32k3a* state, f64 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data64, f32>(
        curandStateMRG32k3a* state, f32 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data64, f64>(
        curandStateMRG32k3a* state, f64 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data32,
                                           f32>(curandStatePhilox4_32_10* state,
                                                f32 std_dev, Data32* pointer,
                                                Modulus<Data32> modulus,
                                                Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data32,
                                           f64>(curandStatePhilox4_32_10* state,
                                                f64 std_dev, Data32* pointer,
                                                Modulus<Data32> modulus,
                                                Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data64,
                                           f32>(curandStatePhilox4_32_10* state,
                                                f32 std_dev, Data64* pointer,
                                                Modulus<Data64> modulus,
                                                Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data64,
                                           f64>(curandStatePhilox4_32_10* state,
                                                f64 std_dev, Data64* pointer,
                                                Modulus<Data64> modulus,
                                                Data64 size, int max_state_num);

    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data32, f32>(
        curandStateXORWOW* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data32, f64>(
        curandStateXORWOW* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data64, f32>(
        curandStateXORWOW* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data64, f64>(
        curandStateXORWOW* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data32, f32>(
        curandStateMRG32k3a* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data32, f64>(
        curandStateMRG32k3a* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data64, f32>(
        curandStateMRG32k3a* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data64, f64>(
        curandStateMRG32k3a* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data32,
                                           f32>(curandStatePhilox4_32_10* state,
                                                f32 std_dev, Data32* pointer,
                                                Modulus<Data32>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data32,
                                           f64>(curandStatePhilox4_32_10* state,
                                                f64 std_dev, Data32* pointer,
                                                Modulus<Data32>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data64,
                                           f32>(curandStatePhilox4_32_10* state,
                                                f32 std_dev, Data64* pointer,
                                                Modulus<Data64>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStatePhilox4_32_10, Data64,
                                           f64>(curandStatePhilox4_32_10* state,
                                                f64 std_dev, Data64* pointer,
                                                Modulus<Data64>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);

    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data32, f32>(
        curandStateXORWOW* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data32, f64>(
        curandStateXORWOW* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data64, f32>(
        curandStateXORWOW* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateXORWOW, Data64, f64>(
        curandStateXORWOW* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data32, f32>(
        curandStateMRG32k3a* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data32, f64>(
        curandStateMRG32k3a* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data64, f32>(
        curandStateMRG32k3a* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<curandStateMRG32k3a, Data64, f64>(
        curandStateMRG32k3a* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        curandStatePhilox4_32_10, Data32, f32>(curandStatePhilox4_32_10* state,
                                               f32 std_dev, Data32* pointer,
                                               Modulus<Data32>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        curandStatePhilox4_32_10, Data32, f64>(curandStatePhilox4_32_10* state,
                                               f64 std_dev, Data32* pointer,
                                               Modulus<Data32>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        curandStatePhilox4_32_10, Data64, f32>(curandStatePhilox4_32_10* state,
                                               f32 std_dev, Data64* pointer,
                                               Modulus<Data64>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        curandStatePhilox4_32_10, Data64, f64>(curandStatePhilox4_32_10* state,
                                               f64 std_dev, Data64* pointer,
                                               Modulus<Data64>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);

    // -

    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer, Data64 size,
        int max_state_num);

    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);

    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);

    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data32>(
        curandStateXORWOW* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateXORWOW, Data64>(
        curandStateXORWOW* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data32>(
        curandStateMRG32k3a* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStateMRG32k3a, Data64>(
        curandStateMRG32k3a* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data32>(
        curandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<curandStatePhilox4_32_10, Data64>(
        curandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);

} // end namespace rngongpu
