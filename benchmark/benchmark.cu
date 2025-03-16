// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <nvbench/nvbench.cuh>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include "aes_rng.cuh"
#include "cuda_rng.cuh"
#include "aes.cuh"
#include "base_rng.cuh"

using namespace std;
using namespace rngongpu;

void CTR_DRBG_with_AES_Benchmark_32bit_Data(nvbench::state& state)
{
    const auto size_logN = state.get_int64("Data Size LogN");
    const auto sev_level_int = state.get_int64("Security Level");

    int entropy_input_len;
    int nonce_len;
    rngongpu::SecurityLevel sev_level;
    switch (sev_level_int)
    {
        case 128:
            entropy_input_len = 128;
            nonce_len = 64;
            sev_level = rngongpu::SecurityLevel::AES128;
            break;
        case 192:
            entropy_input_len = 192;
            nonce_len = 128;
            sev_level = rngongpu::SecurityLevel::AES192;
            break;
        case 256:
            entropy_input_len = 256;
            nonce_len = 128;
            sev_level = rngongpu::SecurityLevel::AES256;
            break;
        default:
            throw std::invalid_argument("Invalid security level!");
    }

    std::vector<unsigned char> entropy(entropy_input_len);
    if (1 != RAND_bytes(entropy.data(), entropy.size()))
        throw std::runtime_error("RAND_bytes failed during reseed");
    std::vector<unsigned char> nonce(nonce_len);
    if (1 != RAND_bytes(nonce.data(), nonce.size()))
        throw std::runtime_error("RAND_bytes failed during reseed");
    std::vector<unsigned char> personalization = {};

    rngongpu::RNG<rngongpu::Mode::AES> drbg(entropy, nonce, personalization,
                                            sev_level, false);

    Data64 size = 1ULL << size_logN;
    Data32* d_results;
    cudaMalloc(&d_results, size * sizeof(Data32));

    state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    state.exec(
        [&](nvbench::launch& launch)
        {
            std::vector<unsigned char> additional_input = {};
            drbg.uniform_random_number(d_results, size, additional_input,
                                       stream);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(CTR_DRBG_with_AES_Benchmark_32bit_Data)
    .add_int64_axis("Data Size LogN", {16, 17, 18, 19, 20, 21, 22, 23, 24})
    .add_int64_axis("Security Level", {128, 192, 256})
    .set_timeout(1);

void CTR_DRBG_with_AES_Benchmark_64bit_Data(nvbench::state& state)
{
    const auto size_logN = state.get_int64("Data Size LogN");
    const auto sev_level_int = state.get_int64("Security Level");

    int entropy_input_len;
    int nonce_len;
    rngongpu::SecurityLevel sev_level;
    switch (sev_level_int)
    {
        case 128:
            entropy_input_len = 128;
            nonce_len = 64;
            sev_level = rngongpu::SecurityLevel::AES128;
            break;
        case 192:
            entropy_input_len = 192;
            nonce_len = 128;
            sev_level = rngongpu::SecurityLevel::AES192;
            break;
        case 256:
            entropy_input_len = 256;
            nonce_len = 128;
            sev_level = rngongpu::SecurityLevel::AES256;
            break;
        default:
            throw std::invalid_argument("Invalid security level!");
    }

    std::vector<unsigned char> entropy(entropy_input_len);
    if (1 != RAND_bytes(entropy.data(), entropy.size()))
        throw std::runtime_error("RAND_bytes failed during reseed");
    std::vector<unsigned char> nonce(nonce_len);
    if (1 != RAND_bytes(nonce.data(), nonce.size()))
        throw std::runtime_error("RAND_bytes failed during reseed");
    std::vector<unsigned char> personalization = {};

    rngongpu::RNG<rngongpu::Mode::AES> drbg(entropy, nonce, personalization,
                                            sev_level, false);

    Data64 size = 1ULL << size_logN;
    Data64* d_results;
    cudaMalloc(&d_results, size * sizeof(Data64));

    state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    state.exec(
        [&](nvbench::launch& launch)
        {
            std::vector<unsigned char> additional_input = {};
            drbg.uniform_random_number(d_results, size, additional_input,
                                       stream);
        });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(CTR_DRBG_with_AES_Benchmark_64bit_Data)
    .add_int64_axis("Data Size LogN", {16, 17, 18, 19, 20, 21, 22, 23, 24})
    .add_int64_axis("Security Level", {128, 192, 256})
    .set_timeout(1);

// -

void Curand_Benchmark_32bit_Data(nvbench::state& state)
{
    const auto size_logN = state.get_int64("Data Size LogN");

    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
    uint64_t seed = dis(generator);

    RNG<Mode::CUDA, curandStateXORWOW> gen(seed);

    Data64 size = 1ULL << size_logN;
    Data32* d_results;
    cudaMalloc(&d_results, size * sizeof(Data32));

    state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    state.exec([&](nvbench::launch& launch)
               { gen.uniform_random_number(d_results, size, stream); });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(Curand_Benchmark_32bit_Data)
    .add_int64_axis("Data Size LogN", {16, 17, 18, 19, 20, 21, 22, 23, 24})
    .set_timeout(1);

void Curand_Benchmark_64bit_Data(nvbench::state& state)
{
    const auto size_logN = state.get_int64("Data Size LogN");

    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
    uint64_t seed = dis(generator);

    RNG<Mode::CUDA, curandStateXORWOW> gen(seed);

    Data64 size = 1ULL << size_logN;
    Data64* d_results;
    cudaMalloc(&d_results, size * sizeof(Data64));

    state.collect_dram_throughput();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    state.exec([&](nvbench::launch& launch)
               { gen.uniform_random_number(d_results, size, stream); });

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

NVBENCH_BENCH(Curand_Benchmark_64bit_Data)
    .add_int64_axis("Data Size LogN", {16, 17, 18, 19, 20, 21, 22, 23, 24})
    .set_timeout(1);
