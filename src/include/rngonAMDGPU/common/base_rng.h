// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef BASE_RNG_H
#define BASE_RNG_H

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <hip/hip_runtime.h>
#include "gpuntt/common/modular_arith.hip"
#include "rngonAMDGPU/common/aes.h"

namespace rngonAMDGPU
{
    enum class Mode
    {
        CUDA,
        AES
    };

    template <Mode mode, typename State = void> struct ModeFeature;

    template <Mode mode, typename State = void> struct RNGTraits;

    template <Mode mode, typename State = void> class RNG;

    void CheckCudaPointer(const void* ptr);

    // --

    template <typename T>
    __global__ void mod_reduce_kernel(T* pointer, Modulus<T> modulus,
                                      Data32 size, int max_state_num);

    template <typename T>
    __global__ void mod_reduce_kernel(T* pointer, Modulus<T>* modulus,
                                      Data32 log_size, int mod_count,
                                      int repeat_count, int max_state_num);

    template <typename T>
    __global__ void mod_reduce_kernel(T* pointer, Modulus<T>* modulus,
                                      Data32 log_size, int mod_count,
                                      int* mod_index, int repeat_count,
                                      int max_state_num);

    // --

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* input, U* output,
                                      Data32 size, int max_state_num);

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* pointer, Modulus<T> modulus,
                                      Data32 size, int max_state_num);

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* input, T* output,
                                      Modulus<T>* modulus, Data32 log_size,
                                      int mod_count, int repeat_count,
                                      int max_state_num);

    template <typename T, typename U>
    __global__ void box_muller_kernel(U std_dev, T* input, T* output,
                                      Modulus<T>* modulus, Data32 log_size,
                                      int mod_count, int* mod_index,
                                      int repeat_count, int max_state_num);

    // --

    template <typename T>
    __global__ void ternary_number_kernel(T* pointer, Data32 size,
                                          int max_state_num);

    template <typename T>
    __global__ void ternary_number_kernel(T* pointer, Modulus<T> modulus,
                                          Data32 size, int max_state_num);

    template <typename T>
    __global__ void ternary_number_kernel(T* input, T* output,
                                          Modulus<T>* modulus, Data32 log_size,
                                          int mod_count, int repeat_count,
                                          int max_state_num);

    template <typename T>
    __global__ void ternary_number_kernel(T* input, T* output,
                                          Modulus<T>* modulus, Data32 log_size,
                                          int mod_count, int* mod_index,
                                          int repeat_count, int max_state_num);

    class CudaException : public std::exception
    {
      public:
        CudaException(const std::string& file, int line, hipError_t error)
            : file_(file), line_(line), error_(error)
        {
        }

        const char* what() const noexcept override
        {
            return m_error_string.c_str();
        }

      private:
        std::string file_;
        int line_;
        hipError_t error_;
        std::string m_error_string = "CUDA Error in " + file_ + " at line " +
                                     std::to_string(line_) + ": " +
                                     hipGetErrorString(error_);
    };

    __global__ void box_muller_u32(Data32* nums, f32* res, Data32 N);

    __global__ void box_muller_u64(Data64* nums, f64* res, Data32 N);

    __global__ void mod_reduce_u64(Data64* nums, Modulus64* p, Data32 N);

    __global__ void mod_reduce_u64(Data64* nums, Modulus64* p, Data32 p_N,
                                   Data32 N);

    __global__ void mod_reduce_u32(Data32* nums, Modulus32* p, Data32 p_N,
                                   Data32 N);

    __global__ void mod_reduce_u32(Data32* nums, Modulus32* p, Data32 N);

} // namespace rngonAMDGPU

#define RNGONGPU_CUDA_CHECK(err)                                               \
    do                                                                         \
    {                                                                          \
        hipError_t error = err;                                               \
        if (error != hipSuccess)                                              \
        {                                                                      \
            throw CudaException(__FILE__, __LINE__, error);                    \
        }                                                                      \
    } while (0)

#endif // BASE_RNG_H
