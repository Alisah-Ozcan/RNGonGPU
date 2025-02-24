// CudarandRNG.cpp
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


CudarandRNG::CudarandRNG(unsigned long long seed, const std::string& generatorName)
    : generator_type(generatorName), baseSeed(seed), d_states(nullptr)
{
    
    if (generator_type == "XORWOW" ||
        generator_type == "MRG32k3a" ||
        generator_type == "Philox")
    {
        numStates = rngongpu::TOTAL_THREADS; 
       
        cudaError_t err = cudaMalloc(&d_states, numStates * sizeof(curandStateXORWOW_t));
        if (err != cudaSuccess) {
            std::cerr << "Error allocating device memory for states: "
                      << cudaGetErrorString(err) << std::endl;
            d_states = nullptr;
        }
    }
    else {
        std::cerr << "Unknown generator type: " << generator_type << std::endl;
        d_states = nullptr;
    }

    // Initialize the states.
    initState();
}

CudarandRNG::~CudarandRNG() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
}


void CudarandRNG::initState() {
    if (!d_states) return;

    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::init_xorwow_states<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states),
            baseSeed);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::init_mrg32k3a_states<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states),
            baseSeed);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::init_philox_states<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states),
            baseSeed);
    }
    else {
        std::cerr << "Unknown generator type in initState: " << generator_type << std::endl;
        return;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error during state initialization synchronization: "
                  << cudaGetErrorString(err) << std::endl;
    }
}

//(Non-modulo generation variants)

void CudarandRNG::gen_random_u32(int N, unsigned int* res) {
    if (!d_states) return;
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states),
            res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states),
            res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states),
            res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_u32: " << generator_type << std::endl;
        return;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_u32 kernel: " << cudaGetErrorString(err) << std::endl;
}

void CudarandRNG::gen_random_u64(int N, unsigned long long* res) {
    if (!d_states) return;
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states),
            res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states),
            res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states),
            res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_u64: " << generator_type << std::endl;
        return;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_u64 kernel: " << cudaGetErrorString(err) << std::endl;
}

void CudarandRNG::gen_random_f32(int N, f32* res) {
    if (!d_states) return;
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow_normal<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states),
            res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a_normal<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states),
            res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox_normal<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states),
            res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_f32: " << generator_type << std::endl;
        return;
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_f32 kernel: " << cudaGetErrorString(err) << std::endl;
}

void CudarandRNG::gen_random_f64(int N, f64* res) {
    if (!d_states) return;
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow_normal_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states),
            res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a_normal_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states),
            res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox_normal_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states),
            res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_f64: " << generator_type << std::endl;
        return;
    }
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cerr << "Error in gen_random_f64 kernel: " << cudaGetErrorString(err) << std::endl;
}

//---------------------------------------------------------------------
// (Modulo generation variants)
//---------------------------------------------------------------------

// 64-bit single-modulus version
void CudarandRNG::gen_random_u64_mod_p(int N, Modulus64* p, Data64* res) {
    if (!d_states) return;
    
    // Generate N random 64-bit numbers
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states), res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states), res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states), res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_u64_mod_p: " << generator_type << std::endl;
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u64_mod_p generation kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    Modulus64* d_p;
    cudaMalloc(&d_p, sizeof(Modulus64));
    cudaMemcpy(d_p, p, sizeof(Modulus64), cudaMemcpyHostToDevice);


    const int CTA_size = 256;
    const int grid_size = (N + CTA_size - 1) / CTA_size;
    mod_reduce_u64<<<grid_size, CTA_size>>>(res, d_p, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_p);
}


void CudarandRNG::gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) {
    if (!d_states) return;
    
    // Generate N random 64-bit numbers
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states), res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states), res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox_64<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states), res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_u64_mod_p: " << generator_type << std::endl;
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u64_mod_p generation kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    Modulus64* d_p;
    cudaMalloc(&d_p, p_num * sizeof(Modulus64));
    cudaMemcpy(d_p, p, p_num * sizeof(Modulus64), cudaMemcpyHostToDevice);
    mod_reduce_u64<<<dim3(BLOCKS, p_num, 1), THREADS_PER_BLOCK>>>(res, d_p, p_num, N);
    cudaDeviceSynchronize();

    cudaFree(d_p);
}


void CudarandRNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) {
    if (!d_states) return;
    

    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states), res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states), res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states), res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_u32_mod_p: " << generator_type << std::endl;
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u32_mod_p generation kernel: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    

    Modulus32* d_p;
    cudaMalloc(&d_p, sizeof(Modulus32));
    cudaMemcpy(d_p, p, sizeof(Modulus32), cudaMemcpyHostToDevice);

    const int CTA_size = 256;
    const int grid_size = (N + CTA_size - 1) / CTA_size;
    mod_reduce_u32<<<grid_size, CTA_size>>>(res, d_p, N);
    cudaDeviceSynchronize();
    
    cudaFree(d_p);
}

// 32-bit array-of-moduli version
void CudarandRNG::gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) {
    if (!d_states) return;
    
    // Generate N random 32-bit numbers
    if (generator_type == "XORWOW") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_xorwow<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateXORWOW_t*>(d_states), res, N);
    }
    else if (generator_type == "MRG32k3a") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_mrg32k3a<<<blocks, threadsPerBlock>>>(
            static_cast<curandStateMRG32k3a_t*>(d_states), res, N);
    }
    else if (generator_type == "Philox") {
        int threadsPerBlock = 256;
        int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        rngongpu::generate_random_philox<<<blocks, threadsPerBlock>>>(
            static_cast<curandStatePhilox4_32_10_t*>(d_states), res, N);
    }
    else {
        std::cerr << "Unknown generator type in gen_random_u32_mod_p: " << generator_type << std::endl;
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in gen_random_u32_mod_p generation kernel: " << cudaGetErrorString(err) << std::endl;
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
