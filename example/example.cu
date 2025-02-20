// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "cuda_rng.cuh"
#include "normal_rng.cuh"
#include "uniform_rng.cuh"
#include "aes.cuh"
#include "base_rng.cuh"
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
    // rngongpu::NormalRNG* rng = new rngongpu::NormalRNG();

    // f64* d_res, *h_res;
    // cudaMalloc(&d_res, 4096 * sizeof(f64));
    // rng -> gen_random_f64(4096, d_res);

    // h_res = new f64[4096];
    // cudaMemcpy(h_res, d_res, 4096 * sizeof(f64), cudaMemcpyDeviceToHost);
    // cout << "**********NORMAL DISTRIBUTION F64**********\n";
    // for (int i = 0; i <= 4092; i+=4) printf("%.6f %.6f %.6f %.6f\n", h_res[i], h_res[i+1], h_res[i+2], h_res[i+3]);

    cout << "**********UNIFORM DISTRIBUTION U64 MOD P***********\n";
    rngongpu::UniformRNG* urng = new rngongpu::UniformRNG();


    Data64* d_res_u64, *h_res_u64;
    cudaMalloc(&d_res_u64, 4096 * sizeof(Data64));
    Modulus64* p = new Modulus64(1265904160645881121UL);

    urng -> gen_random_u64_mod_p(4096, p, d_res_u64);

    h_res_u64 = new Data64[4096];
    cudaMemcpy(h_res_u64, d_res_u64, 4096 * sizeof(Data64), cudaMemcpyDeviceToHost);

    for (int i = 0 ; i<= 4092; i+=4) printf("%lld %lld %lld %lld\n", h_res_u64[i],  h_res_u64[i+1], h_res_u64[i+2], h_res_u64[i+3]);

    return EXIT_SUCCESS;
}