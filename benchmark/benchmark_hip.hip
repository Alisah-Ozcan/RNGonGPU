// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <nvbench/nvbench.hip>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include "rngonAMDGPU/rand_aes/aes_rng.hip"
#include "rngonAMDGPU/rand_hip/hip_rng.hip"
#include "rngonAMDGPU/common/aes.hip"
#include "rngonAMDGPU/common/base_rng.hip"

using namespace std;
using namespace rngonAMDGPU;

void Curand_Benchmark_32bit_Data(nvbench::state& state)
{
    const auto size_logN = state.get_int64("Data Size LogN");

    std::random_device rd;
    std::mt19937_64 generator(rd());
    std::uniform_int_distribution<uint64_t> dis(0, UINT64_MAX);
    uint64_t seed = dis(generator);

    RNG<Mode::CUDA, hiprandStateXORWOW_t> gen(seed);

    Data64 size = 1ULL << size_logN;
    Data32* d_results;
    hipMalloc(&d_results, size * sizeof(Data32));

    state.collect_dram_throughput();

    hipStream_t stream;
    hipStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    state.exec([&](nvbench::launch& launch)
               { gen.uniform_random_number(d_results, size, stream); });

    hipStreamSynchronize(stream);
    hipStreamDestroy(stream);
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

    RNG<Mode::CUDA, hiprandStateXORWOW_t> gen(seed);

    Data64 size = 1ULL << size_logN;
    Data64* d_results;
    hipMalloc(&d_results, size * sizeof(Data64));

    state.collect_dram_throughput();

    hipStream_t stream;
    hipStreamCreate(&stream);
    state.set_cuda_stream(nvbench::make_cuda_stream_view(stream));

    state.exec([&](nvbench::launch& launch)
               { gen.uniform_random_number(d_results, size, stream); });

    hipStreamSynchronize(stream);
    hipStreamDestroy(stream);
}

NVBENCH_BENCH(Curand_Benchmark_64bit_Data)
    .add_int64_axis("Data Size LogN", {16, 17, 18, 19, 20, 21, 22, 23, 24})
    .set_timeout(1);
