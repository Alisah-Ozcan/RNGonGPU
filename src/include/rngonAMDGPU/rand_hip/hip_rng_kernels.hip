// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HIP_RNG_KERNEL_H
#define HIP_RNG_KERNEL_H

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <hiprand/hiprand_mtgp32_host.h>          // contains mtgp32dc_params_fast_11213
#include "rngonAMDGPU/common/aes.h"

namespace rngonAMDGPU
{
    template <typename State>
    __global__ void init_state_kernel(State* state, Data64 seed);

    // -

    template <typename State, typename T>
    __global__ void
    uniform_random_number_generation_kernel(State* state, T* pointer,
                                            Data64 size, int max_state_num);

    template <typename State, typename T>
    __global__ void
    uniform_random_number_generation_kernel(State* state, T* pointer,
                                            Modulus<T> modulus, Data64 size,
                                            int max_state_num);

    template <typename State, typename T>
    __global__ void uniform_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count, int max_state_num);

    template <typename State, typename T>
    __global__ void uniform_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, int max_state_num);

    // -

    template <typename State, typename T>
    __global__ void
    normal_random_number_generation_kernel(State* state, T std_dev, T* pointer,
                                           Data64 size, int max_state_num);

    template <typename State, typename T, typename U>
    __global__ void
    normal_random_number_generation_kernel(State* state, U std_dev, T* pointer,
                                           Modulus<T> modulus, Data64 size,
                                           int max_state_num);

    template <typename State, typename T, typename U>
    __global__ void normal_random_number_generation_kernel(
        State* state, U std_dev, T* pointer, Modulus<T>* modulus,
        Data64 log_size, int mod_count, int repeat_count, int max_state_num);

    template <typename State, typename T, typename U>
    __global__ void
    normal_random_number_generation_kernel(State* state, U std_dev, T* pointer,
                                           Modulus<T>* modulus, Data64 log_size,
                                           int mod_count, int* mod_index,
                                           int repeat_count, int max_state_num);

    // -

    template <typename State, typename T>
    __global__ void
    ternary_random_number_generation_kernel(State* state, T* pointer,
                                            Data64 size, int max_state_num);

    template <typename State, typename T>
    __global__ void
    ternary_random_number_generation_kernel(State* state, T* pointer,
                                            Modulus<T> modulus, Data64 size,
                                            int max_state_num);

    template <typename State, typename T>
    __global__ void ternary_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int repeat_count, int max_state_num);

    template <typename State, typename T>
    __global__ void ternary_random_number_generation_kernel(
        State* state, T* pointer, Modulus<T>* modulus, Data64 log_size,
        int mod_count, int* mod_index, int repeat_count, int max_state_num);

} // end namespace rngonAMDGPU

#endif // HIP_RNG_KERNEL_H
