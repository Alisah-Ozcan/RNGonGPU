// CudarandRNG_template_impl.cuh
// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>
#include <iostream>
#include "aes.cuh"
#include "cuda_rng_kernels.cuh"

namespace rngongpu {

template <typename generator>
CudarandRNG<generator>::CudarandRNG(unsigned long long seed)
    : baseSeed(seed),
      d_states(nullptr),
      numStates(rngongpu::TOTAL_THREADS),
      mtgp32_numStates(0)
{
    cudaError_t err = cudaMalloc(&d_states, numStates * sizeof(typename generator::StateType));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory for states: " 
                  << cudaGetErrorString(err) << std::endl;
        d_states = nullptr;
    }
    initState();
}

template <typename generator>
CudarandRNG<generator>::~CudarandRNG() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}

template <typename generator>
void CudarandRNG<generator>::initState() {
    if (!d_states) return;
    int threadsPerBlock = 256;
    // Removed unused variable "blocks"
    generator::initStates(reinterpret_cast<typename generator::StateType*>(d_states),
                            baseSeed, numStates, threadsPerBlock);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error during state initialization synchronization: " 
                  << cudaGetErrorString(err) << std::endl;
    }
}

template <typename generator>
void CudarandRNG<generator>::gen_random_u32(int N, Data32* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_u32(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_u32 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename generator>
void CudarandRNG<generator>::gen_random_u64(int N, Data64* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_u64(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_u64 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename generator>
void CudarandRNG<generator>::gen_random_f32(int N, f32* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_f32(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_f32 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

template <typename generator>
void CudarandRNG<generator>::gen_random_f64(int N, f64* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_f64(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_f64 kernel: " 
                  << cudaGetErrorString(err) << std::endl;
}

//---------------------------------------------------------------------
// Modulo generation variants
//---------------------------------------------------------------------

template <typename generator>
void CudarandRNG<generator>::gen_random_u64_mod_p(int N, Modulus64* p, Data64* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_u64(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u64_mod_p generation kernel: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }

    Modulus64* d_p;
    cudaMalloc(&d_p, sizeof(Modulus64));
    cudaMemcpy(d_p, p, sizeof(Modulus64), cudaMemcpyHostToDevice);

    const int CTA_size = 256;
    int grid_size = (N + CTA_size - 1) / CTA_size;
    mod_reduce_u64<<<grid_size, CTA_size>>>(res, d_p, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_p);
}

template <typename generator>
void CudarandRNG<generator>::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_u64(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
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

template <typename generator>
void CudarandRNG<generator>::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_u32(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
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

template <typename generator>
void CudarandRNG<generator>::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) {
    if (!d_states) return;
    int threadsPerBlock = 256;
    generator::generate_u32(reinterpret_cast<typename generator::StateType*>(d_states),
                            res, N, threadsPerBlock);
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

} // end namespace rngongpu

// Explicit instantiations for all generator types
namespace rngongpu {
    template class CudarandRNG<XORWOW_generator>;
    template class CudarandRNG<MRG32k3a_generator>;
    template class CudarandRNG<Philox_generator>;
}
