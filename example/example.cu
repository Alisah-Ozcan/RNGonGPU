// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "cuda_rng.cuh"
#include "aes.cuh"
#include "base_rng.cuh"
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
    std::vector<unsigned char> entropy = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
        0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
        0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
    std::vector<unsigned char> nonce = {0x20, 0x21, 0x22, 0x23,
                                        0x24, 0x25, 0x26, 0x27};
    std::vector<unsigned char> personalization = {};

    rngongpu::RNG<rngongpu::Mode::AES> drbg(entropy, nonce, personalization,
                                            rngongpu::SecurityLevel::AES128);
    std::cout << "Instantiate: " << std::endl;
    drbg.print_params();

    int size = 65536;
    f64* d_results;
    cudaMalloc(&d_results, size * sizeof(f64));

    drbg.normal_random_number(1.0, d_results, size);

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