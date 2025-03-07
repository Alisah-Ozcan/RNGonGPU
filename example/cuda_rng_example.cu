#include <iostream>
#include "cuda_rng.cuh"
#include <cuda_runtime.h>

using namespace rngongpu;

void run_rng_example_XORWOW()
{
    std::cout << "=== Testing XORWOW Generator ===\n";
    unsigned long long seed = 12345ULL;
    CudarandRNG<curandStateXORWOW_t> rng(seed);
    const int N = 10;

    // Allocate device memory.
    Data32* d_randomInts = nullptr;
    Data64* d_randomInts64 = nullptr;
    f32* d_randomFloats = nullptr;
    f64* d_randomDoubles = nullptr;
    cudaMalloc(&d_randomInts, N * sizeof(Data32));
    cudaMalloc(&d_randomInts64, N * sizeof(Data64));
    cudaMalloc(&d_randomFloats, N * sizeof(f32));
    cudaMalloc(&d_randomDoubles, N * sizeof(f64));

    // Generate non-modulo random numbers.
    rng.gen_random_u32(N, d_randomInts);
    rng.gen_random_u64(N, d_randomInts64);
    rng.gen_random_f32(N, d_randomFloats);
    rng.gen_random_f64(N, d_randomDoubles);

    // Copy results back to host.
    Data32 h_randomInts[N];
    Data64 h_randomInts64[N];
    f32 h_randomFloats[N];
    f64 h_randomDoubles[N];
    cudaMemcpy(h_randomInts, d_randomInts, N * sizeof(Data32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomInts64, d_randomInts64, N * sizeof(Data64),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomFloats, d_randomFloats, N * sizeof(f32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomDoubles, d_randomDoubles, N * sizeof(f64),
               cudaMemcpyDeviceToHost);

    // Print non-modulo random numbers.
    std::cout << "Non-modulo random 32-bit integers:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts[i] << " ";
    std::cout << "\nNon-modulo random 64-bit integers:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts64[i] << " ";
    std::cout << "\nNon-modulo random 32-bit floats (f32):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomFloats[i] << " ";
    std::cout << "\nNon-modulo random 64-bit doubles (f64):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomDoubles[i] << " ";
    std::cout << "\n";

    // Test modulo versions.
    Modulus32 mod32(100);
    Modulus64 mod64(1000);
    Data32* d_randomInts_mod = nullptr;
    Data64* d_randomInts64_mod = nullptr;
    cudaMalloc(&d_randomInts_mod, N * sizeof(Data32));
    cudaMalloc(&d_randomInts64_mod, N * sizeof(Data64));

    rng.gen_random_u32_mod_p(N, &mod32, d_randomInts_mod);
    rng.gen_random_u64_mod_p(N, &mod64, d_randomInts64_mod);

    Data32 h_randomInts_mod[N];
    Data64 h_randomInts64_mod[N];
    cudaMemcpy(h_randomInts_mod, d_randomInts_mod, N * sizeof(Data32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomInts64_mod, d_randomInts64_mod, N * sizeof(Data64),
               cudaMemcpyDeviceToHost);

    std::cout << "Modulo random 32-bit integers (mod 100):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts_mod[i] << " ";
    std::cout << "\nModulo random 64-bit integers (mod 1000):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts64_mod[i] << " ";
    std::cout << "\n";

    // Test modulo version with an array of moduli.
    const int numMods = 3;
    Modulus32 mod32_arr[numMods] = {Modulus32(100), Modulus32(200),
                                    Modulus32(300)};
    Data32* d_randomInts_mod_arr = nullptr;
    cudaMalloc(&d_randomInts_mod_arr, N * sizeof(Data32));
    rng.gen_random_u32_mod_p(N, mod32_arr, numMods, d_randomInts_mod_arr);

    Data32 h_randomInts_mod_arr[N];
    cudaMemcpy(h_randomInts_mod_arr, d_randomInts_mod_arr, N * sizeof(Data32),
               cudaMemcpyDeviceToHost);
    std::cout << "Modulo array random 32-bit integers (mod 100, 200, 300):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts_mod_arr[i] << " ";
    std::cout << "\n";

    // Free device memory.
    cudaFree(d_randomInts);
    cudaFree(d_randomInts64);
    cudaFree(d_randomFloats);
    cudaFree(d_randomDoubles);
    cudaFree(d_randomInts_mod);
    cudaFree(d_randomInts64_mod);
    cudaFree(d_randomInts_mod_arr);
}

void run_rng_example_MRG32k3a()
{
    std::cout << "\n=== Testing MRG32k3a Generator ===\n";
    unsigned long long seed = 54321ULL;
    CudarandRNG<curandStateMRG32k3a_t> rng(seed);
    const int N = 10;

    Data32* d_randomInts = nullptr;
    Data64* d_randomInts64 = nullptr;
    f32* d_randomFloats = nullptr;
    f64* d_randomDoubles = nullptr;
    cudaMalloc(&d_randomInts, N * sizeof(Data32));
    cudaMalloc(&d_randomInts64, N * sizeof(Data64));
    cudaMalloc(&d_randomFloats, N * sizeof(f32));
    cudaMalloc(&d_randomDoubles, N * sizeof(f64));

    rng.gen_random_u32(N, d_randomInts);
    rng.gen_random_u64(N, d_randomInts64);
    rng.gen_random_f32(N, d_randomFloats);
    rng.gen_random_f64(N, d_randomDoubles);

    Data32 h_randomInts[N];
    Data64 h_randomInts64[N];
    f32 h_randomFloats[N];
    f64 h_randomDoubles[N];
    cudaMemcpy(h_randomInts, d_randomInts, N * sizeof(Data32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomInts64, d_randomInts64, N * sizeof(Data64),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomFloats, d_randomFloats, N * sizeof(f32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomDoubles, d_randomDoubles, N * sizeof(f64),
               cudaMemcpyDeviceToHost);

    std::cout << "Non-modulo random 32-bit integers:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts[i] << " ";
    std::cout << "\nNon-modulo random 64-bit integers:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts64[i] << " ";
    std::cout << "\nNon-modulo random 32-bit floats (f32):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomFloats[i] << " ";
    std::cout << "\nNon-modulo random 64-bit doubles (f64):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomDoubles[i] << " ";
    std::cout << "\n";

    cudaFree(d_randomInts);
    cudaFree(d_randomInts64);
    cudaFree(d_randomFloats);
    cudaFree(d_randomDoubles);
}

void run_rng_example_Philox()
{
    std::cout << "\n=== Testing Philox Generator ===\n";
    unsigned long long seed = 98765ULL;
    CudarandRNG<curandStatePhilox4_32_10_t> rng(seed);
    const int N = 10;

    Data32* d_randomInts = nullptr;
    Data64* d_randomInts64 = nullptr;
    f32* d_randomFloats = nullptr;
    f64* d_randomDoubles = nullptr;
    cudaMalloc(&d_randomInts, N * sizeof(Data32));
    cudaMalloc(&d_randomInts64, N * sizeof(Data64));
    cudaMalloc(&d_randomFloats, N * sizeof(f32));
    cudaMalloc(&d_randomDoubles, N * sizeof(f64));

    rng.gen_random_u32(N, d_randomInts);
    rng.gen_random_u64(N, d_randomInts64);
    rng.gen_random_f32(N, d_randomFloats);
    rng.gen_random_f64(N, d_randomDoubles);

    Data32 h_randomInts[N];
    Data64 h_randomInts64[N];
    f32 h_randomFloats[N];
    f64 h_randomDoubles[N];
    cudaMemcpy(h_randomInts, d_randomInts, N * sizeof(Data32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomInts64, d_randomInts64, N * sizeof(Data64),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomFloats, d_randomFloats, N * sizeof(f32),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomDoubles, d_randomDoubles, N * sizeof(f64),
               cudaMemcpyDeviceToHost);

    std::cout << "Non-modulo random 32-bit integers:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts[i] << " ";
    std::cout << "\nNon-modulo random 64-bit integers:\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomInts64[i] << " ";
    std::cout << "\nNon-modulo random 32-bit floats (f32):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomFloats[i] << " ";
    std::cout << "\nNon-modulo random 64-bit doubles (f64):\n";
    for (int i = 0; i < N; ++i)
        std::cout << h_randomDoubles[i] << " ";
    std::cout << "\n";

    cudaFree(d_randomInts);
    cudaFree(d_randomInts64);
    cudaFree(d_randomFloats);
    cudaFree(d_randomDoubles);
}

int main()
{
    run_rng_example_XORWOW();
    run_rng_example_MRG32k3a();
    run_rng_example_Philox();
    return 0;
}
