// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "rngonAMDGPU/rand_hip/hip_rng_kernels.hip"
#include <hiprand/hiprand_kernel.h>

namespace rngonAMDGPU
{
    template <typename State>
    __global__ void init_state_kernel(State* state, Data64 seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        hiprand_init(seed, idx, 0, &state[idx]);
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
                pointer[i] = hiprand(&thread_state);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                uint32_t num_lo = hiprand(&thread_state);
                uint32_t num_hi = hiprand(&thread_state);

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
                T num = hiprand(&thread_state);
                pointer[i] = OPERATOR_GPU<T>::reduce_forced(num, modulus);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                uint32_t num_lo = hiprand(&thread_state);
                uint32_t num_hi = hiprand(&thread_state);

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
                    T num = hiprand(&thread_state);
                    pointer[global_index] =
                        OPERATOR_GPU<T>::reduce_forced(num, modulus[index]);
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    uint32_t num_lo = hiprand(&thread_state);
                    uint32_t num_hi = hiprand(&thread_state);

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
                    T num = hiprand(&thread_state);
                    pointer[global_index] =
                        OPERATOR_GPU<T>::reduce_forced(num, modulus[new_index]);
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    uint32_t num_lo = hiprand(&thread_state);
                    uint32_t num_hi = hiprand(&thread_state);

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
                float num_f = hiprand_normal(&thread_state);
                num_f = num_f * std_dev;
                pointer[i] = num_f;
            }
            else if constexpr (std::is_same_v<T, f64>)
            {
                double num_f = hiprand_normal_double(&thread_state);
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
                float num_f = hiprand_normal(&thread_state);
                num_f = num_f * std_dev;

                uint32_t flag =
                    static_cast<uint32_t>(-static_cast<int32_t>(num_f < 0));

                pointer[i] = static_cast<T>(num_f) + (flag & modulus.value);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                double num_f = hiprand_normal_double(&thread_state);
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
                    float num_f = hiprand_normal(&thread_state);
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
                    double num_f = hiprand_normal_double(&thread_state);
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
                    float num_f = hiprand_normal(&thread_state);
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
                    double num_f = hiprand_normal_double(&thread_state);
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
                T num = hiprand(&thread_state);
                pointer[i] = num & 1;
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                T num = static_cast<Data64>(hiprand(&thread_state));
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
                T num = hiprand(&thread_state);
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
                T num = hiprand(&thread_state);
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
                    T num = hiprand(&thread_state);
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
                    T num = hiprand(&thread_state);
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
                    T num = hiprand(&thread_state);
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
                    T num = hiprand(&thread_state);
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

    // --- explicit template instantiations (using HIP types) ---

    template __global__ void
    init_state_kernel<hiprandStateXORWOW_t>(hiprandStateXORWOW_t* state, Data64 seed);
    template __global__ void
    init_state_kernel<hiprandStateMRG32k3a_t>(hiprandStateMRG32k3a_t* state,
                                           Data64 seed);
    template __global__ void
    init_state_kernel<hiprandStatePhilox4_32_10>(hiprandStatePhilox4_32_10* state,
                                                Data64 seed);

    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer, Data64 size,
        int max_state_num);

    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);

    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);

    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    uniform_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);

    // -

    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, f32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, f64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, f32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, f64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, f32>(
        hiprandStatePhilox4_32_10* state, f32 std_dev, f32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, f64>(
        hiprandStatePhilox4_32_10* state, f64 std_dev, f64* pointer, Data64 size,
        int max_state_num);

    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data32, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data32, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data64, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data64, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32,
                                           f32>(hiprandStatePhilox4_32_10* state,
                                                f32 std_dev, Data32* pointer,
                                                Modulus<Data32> modulus,
                                                Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32,
                                           f64>(hiprandStatePhilox4_32_10* state,
                                                f64 std_dev, Data32* pointer,
                                                Modulus<Data32> modulus,
                                                Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64,
                                           f32>(hiprandStatePhilox4_32_10* state,
                                                f32 std_dev, Data64* pointer,
                                                Modulus<Data64> modulus,
                                                Data64 size, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64,
                                           f64>(hiprandStatePhilox4_32_10* state,
                                                f64 std_dev, Data64* pointer,
                                                Modulus<Data64> modulus,
                                                Data64 size, int max_state_num);

    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data32, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data32, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data64, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data64, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32,
                                           f32>(hiprandStatePhilox4_32_10* state,
                                                f32 std_dev, Data32* pointer,
                                                Modulus<Data32>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32,
                                           f64>(hiprandStatePhilox4_32_10* state,
                                                f64 std_dev, Data32* pointer,
                                                Modulus<Data32>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64,
                                           f32>(hiprandStatePhilox4_32_10* state,
                                                f32 std_dev, Data64* pointer,
                                                Modulus<Data64>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64,
                                           f64>(hiprandStatePhilox4_32_10* state,
                                                f64 std_dev, Data64* pointer,
                                                Modulus<Data64>* modulus,
                                                Data64 log_size, int mod_count,
                                                int repeat_count,
                                                int max_state_num);

    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data32, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data32, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data64, f32>(
        hiprandStateXORWOW_t* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateXORWOW_t, Data64, f64>(
        hiprandStateXORWOW_t* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64, f32>(
        hiprandStateMRG32k3a_t* state, f32 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    normal_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64, f64>(
        hiprandStateMRG32k3a_t* state, f64 std_dev, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        hiprandStatePhilox4_32_10, Data32, f32>(hiprandStatePhilox4_32_10* state,
                                               f32 std_dev, Data32* pointer,
                                               Modulus<Data32>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        hiprandStatePhilox4_32_10, Data32, f64>(hiprandStatePhilox4_32_10* state,
                                               f64 std_dev, Data32* pointer,
                                               Modulus<Data32>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        hiprandStatePhilox4_32_10, Data64, f32>(hiprandStatePhilox4_32_10* state,
                                               f32 std_dev, Data64* pointer,
                                               Modulus<Data64>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);
    template __global__ void normal_random_number_generation_kernel<
        hiprandStatePhilox4_32_10, Data64, f64>(hiprandStatePhilox4_32_10* state,
                                               f64 std_dev, Data64* pointer,
                                               Modulus<Data64>* modulus,
                                               Data64 log_size, int mod_count,
                                               int* mod_index, int repeat_count,
                                               int max_state_num);

    // -

    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer, Data64 size,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer, Data64 size,
        int max_state_num);

    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Modulus<Data32> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Modulus<Data64> modulus,
        Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32> modulus, Data64 size, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64> modulus, Data64 size, int max_state_num);

    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int repeat_count, int max_state_num);

    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data32>(
        hiprandStateXORWOW_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateXORWOW_t, Data64>(
        hiprandStateXORWOW_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data32>(
        hiprandStateMRG32k3a_t* state, Data32* pointer, Modulus<Data32>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStateMRG32k3a_t, Data64>(
        hiprandStateMRG32k3a_t* state, Data64* pointer, Modulus<Data64>* modulus,
        Data64 log_size, int mod_count, int* mod_index, int repeat_count,
        int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data32>(
        hiprandStatePhilox4_32_10* state, Data32* pointer,
        Modulus<Data32>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);
    template __global__ void
    ternary_random_number_generation_kernel<hiprandStatePhilox4_32_10, Data64>(
        hiprandStatePhilox4_32_10* state, Data64* pointer,
        Modulus<Data64>* modulus, Data64 log_size, int mod_count,
        int* mod_index, int repeat_count, int max_state_num);

} // end namespace rngonAMDGPU
