
// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng.cuh"

namespace rngongpu {

template <typename RNGState>
CudarandRNG<RNGState>::CudarandRNG(Data64 seed)
    : baseSeed(seed),
      d_states(nullptr),
      numStates(rngongpu::TOTAL_THREADS),
      mtgp32_numStates(0)
{
    // Allocate memory for RNG states on the device.
    cudaError_t err = cudaMalloc(&d_states, numStates * sizeof(RNGState));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for states: " 
                  << cudaGetErrorString(err) << std::endl;
        d_states = nullptr;
    }
    initState();
}

template <typename RNGState>
CudarandRNG<RNGState>::~CudarandRNG() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}

template <typename RNGState>
void CudarandRNG<RNGState>::initState() {
    if (!d_states) return;
    // Launch the initialization kernel using the defined grid and block sizes.
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    init_states<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), baseSeed);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error during state initialization synchronization: " 
                  << cudaGetErrorString(err) << std::endl;
    }
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_u32(int N, Data32* res) {
    if (!d_states) return;
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_uniform_32<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_u32 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_u64(int N, Data64* res) {
    if (!d_states) return;
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_uniform_64<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_u64 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_f32(int N, f32* res) {
    if (!d_states) return;
    // Launch the kernel generating normal 32-bit floats.
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_normal_32<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_f32 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_f64(int N, f64* res) {
    if (!d_states) return;
    // Launch the kernel generating normal 64-bit doubles.
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_normal_64<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_f64 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_u64_mod_p(int N, Modulus64* p, Data64* res) {
    if (!d_states) return;
    // Generate 64-bit uniform numbers.
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_uniform_64<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u64_mod_p generation kernel: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    // Copy modulus value to device.
    Modulus64* d_p;
    cudaMalloc(&d_p, sizeof(Modulus64));
    cudaMemcpy(d_p, p, sizeof(Modulus64), cudaMemcpyHostToDevice);
    const int CTA_size = 256;
    int grid_size = (N + CTA_size - 1) / CTA_size;
    mod_reduce_u64<<<grid_size, CTA_size>>>(res, d_p, N);
    cudaDeviceSynchronize();
    cudaFree(d_p);
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) {
    if (!d_states) return;
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_uniform_64<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u64_mod_p generation kernel: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    Modulus64* d_p;
    cudaMalloc(&d_p, p_num * sizeof(Modulus64));
    cudaMemcpy(d_p, p, p_num * sizeof(Modulus64), cudaMemcpyHostToDevice);
    mod_reduce_u64<<<dim3(BLOCKS, p_num, 1), THREADS_PER_BLOCK>>>(res, d_p, p_num, N);
    cudaDeviceSynchronize();
    cudaFree(d_p);
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) {
    if (!d_states) return;
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_uniform_32<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u32_mod_p generation kernel: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    Modulus32* d_p;
    cudaMalloc(&d_p, sizeof(Modulus32));
    cudaMemcpy(d_p, p, sizeof(Modulus32), cudaMemcpyHostToDevice);
    const int CTA_size = 256;
    int grid_size = (N + CTA_size - 1) / CTA_size;
    mod_reduce_u32<<<grid_size, CTA_size>>>(res, d_p, N);
    cudaDeviceSynchronize();
    cudaFree(d_p);
}

template <typename RNGState>
void CudarandRNG<RNGState>::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) {
    if (!d_states) return;
    dim3 grid(BLOCKS);
    dim3 block(THREADS_PER_BLOCK);
    generate_uniform_32<<<grid, block>>>(reinterpret_cast<RNGState*>(d_states), res, N);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u32_mod_p generation kernel: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    Modulus32* d_p;
    cudaMalloc(&d_p, p_num * sizeof(Modulus32));
    cudaMemcpy(d_p, p, p_num * sizeof(Modulus32), cudaMemcpyHostToDevice);
    mod_reduce_u32<<<dim3(BLOCKS, p_num, 1), THREADS_PER_BLOCK>>>(res, d_p, p_num, N);
    cudaDeviceSynchronize();
    cudaFree(d_p);
}


template class CudarandRNG<curandStateXORWOW>;
template class CudarandRNG<curandStateMRG32k3a>;
template class CudarandRNG<curandStatePhilox4_32_10>;

} // end namespace rngongpu
