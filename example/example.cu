// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "cuda_rng.cuh"
#include "aes.cuh"
#include "base_rng.cuh"
#include <iostream>

using namespace std;

#define N (1 << 20)

int main(int argc, char* argv[])
{   
    // Instantiate the DRBG object with prediction resistance.
    rngongpu::AES_RNG* rng = new rngongpu::AES_RNG(false);
    rng -> printWorkingState();

    // Reseed with no additional input
    rng -> reseed(std::vector<unsigned char>());
    rng -> printWorkingState();

    // Generate 4096 doubles
    cout << "**********NORMAL DISTRIBUTION F64**********\n";
    f64* d_res, *h_res;
    cudaMalloc(&d_res, N * sizeof(f64));
    rng -> gen_random_f64(N, d_res);
    h_res = new f64[N];
    cudaMemcpy(h_res, d_res, N * sizeof(f64), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i <= N-4; i+=4) printf("%.6f %.6f %.6f %.6f\n", h_res[i], h_res[i+1], h_res[i+2], h_res[i+3]);

    rng -> printWorkingState();

    // Generate 4096 u64 mod p
    cout << "**********UNIFORM DISTRIBUTION U64 MOD P***********\n";
    Data64* d_res_u64, *h_res_u64;
    cudaMalloc(&d_res_u64, N * sizeof(Data64));
    Modulus64* p = new Modulus64(1265904160645881121UL);

    rng -> gen_random_u64_mod_p(N, p, d_res_u64);

    h_res_u64 = new Data64[N];
    cudaMemcpy(h_res_u64, d_res_u64, N * sizeof(Data64), cudaMemcpyDeviceToHost);

    for (int i = 0 ; i<= N-4; i+=4) printf("%llu %llu %llu %llu\n", h_res_u64[i],  h_res_u64[i+1], h_res_u64[i+2], h_res_u64[i+3]);

    rng -> printWorkingState();

    delete rng;

    return EXIT_SUCCESS;
}