// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "base_rng.cuh"

namespace rngongpu
{
    void CheckCudaPointer(const void* ptr)
    {
        // Check for a null pointer first.
        if (ptr == nullptr)
        {
            throw std::runtime_error("Error: Provided pointer is null!");
        }

        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
        RNGONGPU_CUDA_CHECK(cudaGetLastError());

#if CUDART_VERSION >= 10000
        // For CUDA 10.0 and later, use attr.type.
        switch (attr.type)
        {
            case cudaMemoryTypeDevice:
                // The pointer is in device memory. All is well.
                break;
            case cudaMemoryTypeHost:
            {
                std::ostringstream oss;
                oss << "Error: Pointer " << ptr << " is in CPU host memory!";
                throw std::runtime_error(oss.str());
            }
            case cudaMemoryTypeManaged:
            {
                // The pointer is in Unified (Managed) memory. All is well.
                break;
            }
            default:
            {
                std::ostringstream oss;
                oss << "Error: Pointer " << ptr
                    << " has an unknown memory type: " << attr.type;
                throw std::runtime_error(oss.str());
            }
        }
#else
        // For older CUDA versions, use attr.memoryType.
        switch (attr.memoryType)
        {
            case cudaMemoryTypeDevice:
                break;
            case cudaMemoryTypeHost:
            {
                std::ostringstream oss;
                oss << "Error: Pointer " << ptr << " is in CPU host memory!";
                throw std::runtime_error(oss.str());
            }
            default:
            {
                std::ostringstream oss;
                oss << "Error: Pointer " << ptr
                    << " has an unknown memory type: " << attr.memoryType;
                throw std::runtime_error(oss.str());
            }
        }
#endif
    }

    // --

    template <typename T>
    __global__ void mod_reduce_kernel(T* pointer, Modulus<T> modulus,
                                      Data32 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = idx; i < size; i += max_state_num)
        {
            T number = pointer[i];
            number = OPERATOR_GPU<T>::reduce_forced(number, modulus);
            pointer[i] = number;
        }
    }

    template <typename T>
    __global__ void mod_reduce_kernel(T* pointer, Modulus<T>* modulus,
                                      Data32 log_size, int mod_count,
                                      int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int size_wo_repeat = mod_count << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;
                int index = i >> log_size;

                T number = pointer[global_index];
                number = OPERATOR_GPU<T>::reduce_forced(number, modulus[index]);
                pointer[global_index] = number;
            }
        }
    }

    template <typename T>
    __global__ void mod_reduce_kernel(T* pointer, Modulus<T>* modulus,
                                      Data32 log_size, int mod_count,
                                      int* mod_index, int repeat_count,
                                      int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int size_wo_repeat = mod_count << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;
                int index = i >> log_size;
                int new_index = mod_index[index];

                T number = pointer[global_index];
                number =
                    OPERATOR_GPU<T>::reduce_forced(number, modulus[new_index]);
                pointer[global_index] = number;
            }
        }
    }

    // --

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* input, U* output,
                                      Data32 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        T max_value;
        if constexpr (std::is_same_v<T, Data32>)
        {
            max_value = MAX_U32;
        }
        else if constexpr (std::is_same_v<T, Data64>)
        {
            max_value = MAX_U64;
        }

        for (int i = idx; 2 * i + 1 < size; i += max_state_num)
        {
            U u1 = static_cast<U>(input[2 * i]) / max_value;
            U u2 = static_cast<U>(input[2 * i + 1]) / max_value;

            U radius = sqrt(-static_cast<U>(2.0) * log(u1));
            U theta = static_cast<U>(2.0) * static_cast<U>(M_PI) * u2;

            U z0 = radius * cos(theta);
            U z1 = radius * sin(theta);

            output[2 * i] = z0 * std_dev;
            output[2 * i + 1] = z1 * std_dev;
        }
    }

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* pointer, Modulus<T> modulus,
                                      Data32 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        T max_value;
        if constexpr (std::is_same_v<T, Data32>)
        {
            max_value = MAX_U32;
        }
        else if constexpr (std::is_same_v<T, Data64>)
        {
            max_value = MAX_U64;
        }

        for (int i = idx; 2 * i + 1 < size; i += max_state_num)
        {
            U u1 = static_cast<U>(pointer[2 * i]) / max_value;
            U u2 = static_cast<U>(pointer[2 * i + 1]) / max_value;

            U radius = sqrt(-static_cast<U>(2.0) * log(u1));
            U theta = static_cast<U>(2.0) * static_cast<U>(M_PI) * u2;

            U z0 = radius * cos(theta);
            U z1 = radius * sin(theta);

            z0 = z0 * std_dev;
            z1 = z1 * std_dev;

            if constexpr (std::is_same_v<T, Data32>)
            {
                uint32_t flag0 =
                    static_cast<uint32_t>(-static_cast<int32_t>(z0 < 0));
                uint32_t flag1 =
                    static_cast<uint32_t>(-static_cast<int32_t>(z1 < 0));

                pointer[2 * i] = static_cast<T>(z0) + (flag0 & modulus.value);
                pointer[2 * i + 1] =
                    static_cast<T>(z1) + (flag1 & modulus.value);
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                uint64_t flag0 =
                    static_cast<uint64_t>(-static_cast<int64_t>(z0 < 0));
                uint64_t flag1 =
                    static_cast<uint64_t>(-static_cast<int64_t>(z1 < 0));

                pointer[2 * i] = static_cast<T>(z0) + (flag0 & modulus.value);
                pointer[2 * i + 1] =
                    static_cast<T>(z1) + (flag1 & modulus.value);
            }
        }
    }

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* input, T* output,
                                      Modulus<T>* modulus, Data32 log_size,
                                      int mod_count, int repeat_count,
                                      int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int size_wo_repeat = 1 << log_size;

        T max_value;
        if constexpr (std::is_same_v<T, Data32>)
        {
            max_value = MAX_U32;
        }
        else if constexpr (std::is_same_v<T, Data64>)
        {
            max_value = MAX_U64;
        }

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; 2 * i + 1 < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + (2 * i);

                U u1 = static_cast<U>(input[2 * i]) / max_value;
                U u2 = static_cast<U>(input[2 * i + 1]) / max_value;

                U radius = sqrt(-static_cast<U>(2.0) * log(u1));
                U theta = static_cast<U>(2.0) * static_cast<U>(M_PI) * u2;

                U z0 = radius * cos(theta);
                U z1 = radius * sin(theta);

                z0 = z0 * std_dev;
                z1 = z1 * std_dev;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    uint32_t flag0 =
                        static_cast<uint32_t>(-static_cast<int32_t>(z0 < 0));
                    uint32_t flag1 =
                        static_cast<uint32_t>(-static_cast<int32_t>(z1 < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            static_cast<T>(z0) + (flag0 & modulus[k].value);
                        output[global_index + 1 + in_offset] =
                            static_cast<T>(z1) + (flag1 & modulus[k].value);
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    uint64_t flag0 =
                        static_cast<uint64_t>(-static_cast<int64_t>(z0 < 0));
                    uint64_t flag1 =
                        static_cast<uint64_t>(-static_cast<int64_t>(z1 < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            static_cast<T>(z0) + (flag0 & modulus[k].value);
                        output[global_index + 1 + in_offset] =
                            static_cast<T>(z1) + (flag1 & modulus[k].value);
                    }
                }
            }
        }
    }

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* input, T* output,
                                      Modulus<T>* modulus, Data32 log_size,
                                      int mod_count, int* mod_index,
                                      int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int size_wo_repeat = 1 << log_size;

        T max_value;
        if constexpr (std::is_same_v<T, Data32>)
        {
            max_value = MAX_U32;
        }
        else if constexpr (std::is_same_v<T, Data64>)
        {
            max_value = MAX_U64;
        }

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; 2 * i + 1 < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + (2 * i);

                U u1 = static_cast<U>(input[2 * global_index]) / max_value;
                U u2 = static_cast<U>(input[2 * global_index + 1]) / max_value;

                U radius = sqrt(-static_cast<U>(2.0) * log(u1));
                U theta = static_cast<U>(2.0) * static_cast<U>(M_PI) * u2;

                U z0 = radius * cos(theta);
                U z1 = radius * sin(theta);

                if constexpr (std::is_same_v<T, Data32>)
                {
                    uint32_t flag0 =
                        static_cast<uint32_t>(-static_cast<int32_t>(z0 < 0));
                    uint32_t flag1 =
                        static_cast<uint32_t>(-static_cast<int32_t>(z1 < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            static_cast<T>(z0) +
                            (flag0 & modulus[new_index].value);
                        output[global_index + 1 + in_offset] =
                            static_cast<T>(z1) +
                            (flag1 & modulus[new_index].value);
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    uint64_t flag0 =
                        static_cast<uint64_t>(-static_cast<int64_t>(z0 < 0));
                    uint64_t flag1 =
                        static_cast<uint64_t>(-static_cast<int64_t>(z1 < 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            static_cast<T>(z0) +
                            (flag0 & modulus[new_index].value);
                        output[global_index + 1 + in_offset] =
                            static_cast<T>(z1) +
                            (flag1 & modulus[new_index].value);
                    }
                }
            }
        }
    }

    // --

    template <typename T>
    __global__ void ternary_number_kernel(T* pointer, Data32 size,
                                          int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                T number = pointer[i];
                pointer[i] = number & 1;
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                T number = pointer[i];
                pointer[i] = number & 1ULL;
            }
        }
    }

    template <typename T>
    __global__ void ternary_number_kernel(T* pointer, Modulus<T> modulus,
                                          Data32 size, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        for (int i = idx; i < size; i += max_state_num)
        {
            if constexpr (std::is_same_v<T, Data32>)
            {
                T number = pointer[i];
                number = number & 3;
                if (number == 3)
                {
                    number -= 3;
                }

                uint32_t flag =
                    static_cast<uint32_t>(-static_cast<int32_t>(number == 0));

                pointer[i] = number + (flag & modulus.value) - 1;
            }
            else if constexpr (std::is_same_v<T, Data64>)
            {
                T number = pointer[i];
                number = number & 3ULL;
                if (number == 3ULL)
                {
                    number -= 3ULL;
                }

                uint64_t flag =
                    static_cast<uint64_t>(-static_cast<int64_t>(number == 0));

                pointer[i] = number + (flag & modulus.value) - 1;
            }
        }
    }

    template <typename T>
    __global__ void ternary_number_kernel(T* input, T* output,
                                          Modulus<T>* modulus, Data32 log_size,
                                          int mod_count, int repeat_count,
                                          int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int size_wo_repeat = 1 << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    T number = input[i];
                    number = number & 3;
                    if (number == 3)
                    {
                        number -= 3;
                    }

                    uint32_t flag = static_cast<uint32_t>(
                        -static_cast<int32_t>(number == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            number + (flag & modulus[k].value) - 1;
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    T number = input[i];
                    number = number & 3ULL;
                    if (number == 3ULL)
                    {
                        number -= 3ULL;
                    }

                    uint64_t flag = static_cast<uint64_t>(
                        -static_cast<int64_t>(number == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            number + (flag & modulus[k].value) - 1;
                    }
                }
            }
        }
    }

    template <typename T>
    __global__ void ternary_number_kernel(T* input, T* output,
                                          Modulus<T>* modulus, Data32 log_size,
                                          int mod_count, int* mod_index,
                                          int repeat_count, int max_state_num)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int size_wo_repeat = 1 << log_size;

        for (int j = 0; j < repeat_count; j++)
        {
            int offset = j * size_wo_repeat * mod_count;
            for (int i = idx; i < size_wo_repeat; i += max_state_num)
            {
                int global_index = offset + i;

                if constexpr (std::is_same_v<T, Data32>)
                {
                    T number = input[i];
                    number = number & 3;
                    if (number == 3)
                    {
                        number -= 3;
                    }

                    uint32_t flag = static_cast<uint32_t>(
                        -static_cast<int32_t>(number == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            number + (flag & modulus[new_index].value) - 1;
                    }
                }
                else if constexpr (std::is_same_v<T, Data64>)
                {
                    T number = input[i];
                    number = number & 3ULL;
                    if (number == 3ULL)
                    {
                        number -= 3ULL;
                    }

                    uint64_t flag = static_cast<uint64_t>(
                        -static_cast<int64_t>(number == 0));

#pragma unroll
                    for (int k = 0; k < mod_count; k++)
                    {
                        int new_index = mod_index[k];
                        int in_offset = k << log_size;
                        output[global_index + in_offset] =
                            number + (flag & modulus[new_index].value) - 1;
                    }
                }
            }
        }
    }

    template __global__ void mod_reduce_kernel<Data32>(Data32* pointer,
                                                       Modulus<Data32> modulus,
                                                       Data32 size,
                                                       int max_state_num);
    template __global__ void mod_reduce_kernel<Data64>(Data64* pointer,
                                                       Modulus<Data64> modulus,
                                                       Data32 size,
                                                       int max_state_num);

    template __global__ void
    mod_reduce_kernel<Data32>(Data32* pointer, Modulus<Data32>* modulus,
                              Data32 log_size, int mod_count, int repeat_count,
                              int max_state_num);
    template __global__ void
    mod_reduce_kernel<Data64>(Data64* pointer, Modulus<Data64>* modulus,
                              Data32 log_size, int mod_count, int repeat_count,
                              int max_state_num);

    template __global__ void
    mod_reduce_kernel<Data32>(Data32* pointer, Modulus<Data32>* modulus,
                              Data32 log_size, int mod_count, int* mod_index,
                              int repeat_count, int max_state_num);
    template __global__ void
    mod_reduce_kernel<Data64>(Data64* pointer, Modulus<Data64>* modulus,
                              Data32 log_size, int mod_count, int* mod_index,
                              int repeat_count, int max_state_num);

    template __global__ void
    box_muller_kernel<Data32, f32>(f32 std_dev, Data32* input, f32* output,
                                   Data32 size, int max_state_num);
    template __global__ void
    box_muller_kernel<Data32, f64>(f64 std_dev, Data32* input, f64* output,
                                   Data32 size, int max_state_num);
    template __global__ void
    box_muller_kernel<Data64, f32>(f32 std_dev, Data64* input, f32* output,
                                   Data32 size, int max_state_num);
    template __global__ void
    box_muller_kernel<Data64, f64>(f64 std_dev, Data64* input, f64* output,
                                   Data32 size, int max_state_num);

    template __global__ void
    box_muller_kernel<Data32, f32>(f32 std_dev, Data32* pointer,
                                   Modulus<Data32> modulus, Data32 size,
                                   int max_state_num);
    template __global__ void
    box_muller_kernel<Data32, f64>(f64 std_dev, Data32* pointer,
                                   Modulus<Data32> modulus, Data32 size,
                                   int max_state_num);
    template __global__ void
    box_muller_kernel<Data64, f32>(f32 std_dev, Data64* pointer,
                                   Modulus<Data64> modulus, Data32 size,
                                   int max_state_num);
    template __global__ void
    box_muller_kernel<Data64, f64>(f64 std_dev, Data64* pointer,
                                   Modulus<Data64> modulus, Data32 size,
                                   int max_state_num);

    template __global__ void box_muller_kernel<Data32, f32>(
        f32 std_dev, Data32* input, Data32* output, Modulus<Data32>* modulus,
        Data32 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void box_muller_kernel<Data32, f64>(
        f64 std_dev, Data32* input, Data32* output, Modulus<Data32>* modulus,
        Data32 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void box_muller_kernel<Data64, f32>(
        f32 std_dev, Data64* input, Data64* output, Modulus<Data64>* modulus,
        Data32 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void box_muller_kernel<Data64, f64>(
        f64 std_dev, Data64* input, Data64* output, Modulus<Data64>* modulus,
        Data32 log_size, int mod_count, int repeat_count, int max_state_num);

    template __global__ void
    box_muller_kernel<Data32, f32>(f32 std_dev, Data32* input, Data32* output,
                                   Modulus<Data32>* modulus, Data32 log_size,
                                   int mod_count, int* mod_index,
                                   int repeat_count, int max_state_num);
    template __global__ void
    box_muller_kernel<Data32, f64>(f64 std_dev, Data32* input, Data32* output,
                                   Modulus<Data32>* modulus, Data32 log_size,
                                   int mod_count, int* mod_index,
                                   int repeat_count, int max_state_num);
    template __global__ void
    box_muller_kernel<Data64, f32>(f32 std_dev, Data64* input, Data64* output,
                                   Modulus<Data64>* modulus, Data32 log_size,
                                   int mod_count, int* mod_index,
                                   int repeat_count, int max_state_num);
    template __global__ void
    box_muller_kernel<Data64, f64>(f64 std_dev, Data64* input, Data64* output,
                                   Modulus<Data64>* modulus, Data32 log_size,
                                   int mod_count, int* mod_index,
                                   int repeat_count, int max_state_num);

    template __global__ void ternary_number_kernel<Data32>(Data32* pointer,
                                                           Data32 size,
                                                           int max_state_num);
    template __global__ void ternary_number_kernel<Data64>(Data64* pointer,
                                                           Data32 size,
                                                           int max_state_num);

    template __global__ void
    ternary_number_kernel<Data32>(Data32* pointer, Modulus<Data32> modulus,
                                  Data32 size, int max_state_num);
    template __global__ void
    ternary_number_kernel<Data64>(Data64* pointer, Modulus<Data64> modulus,
                                  Data32 size, int max_state_num);

    template __global__ void ternary_number_kernel<Data32>(
        Data32* input, Data32* output, Modulus<Data32>* modulus,
        Data32 log_size, int mod_count, int repeat_count, int max_state_num);
    template __global__ void ternary_number_kernel<Data64>(
        Data64* input, Data64* output, Modulus<Data64>* modulus,
        Data32 log_size, int mod_count, int repeat_count, int max_state_num);

    template __global__ void
    ternary_number_kernel<Data32>(Data32* input, Data32* output,
                                  Modulus<Data32>* modulus, Data32 log_size,
                                  int mod_count, int* mod_index,
                                  int repeat_count, int max_state_num);
    template __global__ void
    ternary_number_kernel<Data64>(Data64* input, Data64* output,
                                  Modulus<Data64>* modulus, Data32 log_size,
                                  int mod_count, int* mod_index,
                                  int repeat_count, int max_state_num);

} // namespace rngongpu