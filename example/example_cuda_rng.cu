// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "cuda_rng.cuh"

using namespace std;
using namespace rngongpu;

int main()
{
    RNG<Mode::CUDA, curandStateXORWOW> gen(1);

    int size = 65536;
    f64* d_results;
    cudaMalloc(&d_results, size * sizeof(f64));

    gen.normal_random_number(1.0, d_results, size);

    f64* h_results = new f64[size];
    cudaMemcpy(h_results, d_results, size * sizeof(f64),
               cudaMemcpyDeviceToHost);

    std::cout << "NORMAL DISTRIBUTION F64:" << std::endl;
    for (int i = 0; i <= 32; i += 4)
    {
        std::cout << h_results[i] << ", " << h_results[i + 1] << ", "
                  << h_results[i + 1] << ", " << h_results[i + 1] << ", "
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
