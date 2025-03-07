// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDA_RNG_KERNEL_H
#define CUDA_RNG_KERNEL_H

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include "aes.cuh"

namespace rngongpu
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

} // end namespace rngongpu

#endif // CUDA_RNG_KERNEL_H
