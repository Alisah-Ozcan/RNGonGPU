#include <iostream>
#include <string>
#include "cuda_rng.cuh"
#include <cuda_runtime.h>

using namespace rngongpu;

int main() {
    
    unsigned long long seed = 12345ULL;
    std::string generator = "XORWOW";  
    CudarandRNG rng(seed, generator);

    const int N = 10;  

   
    Data32* d_randomInts;
    Data64* d_randomInts64;
    f32* d_randomFloats;
    f64* d_randomDoubles;
    cudaMalloc(&d_randomInts, N * sizeof(Data32));
    cudaMalloc(&d_randomInts64, N * sizeof(Data64));
    cudaMalloc(&d_randomFloats, N * sizeof(f32));
    cudaMalloc(&d_randomDoubles, N * sizeof(f64));

    // Generate non-modulo random numbers.
    rng.gen_random_u32(N, d_randomInts);
    rng.gen_random_u64(N, d_randomInts64);
    rng.gen_random_f32(N, d_randomFloats);
    rng.gen_random_f64(N, d_randomDoubles);

    Data32 h_randomInts[N];
    Data64 h_randomInts64[N];
    f32 h_randomFloats[N];
    f64 h_randomDoubles[N];

    cudaMemcpy(h_randomInts, d_randomInts, N * sizeof(Data32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomInts64, d_randomInts64, N * sizeof(Data64), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomFloats, d_randomFloats, N * sizeof(f32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomDoubles, d_randomDoubles, N * sizeof(f64), cudaMemcpyDeviceToHost);

    std::cout << "Non-modulo random 32-bit integers:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomInts[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Non-modulo random 64-bit integers:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomInts64[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Non-modulo random 32-bit floats (f32):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomFloats[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Non-modulo random 64-bit doubles (f64):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomDoubles[i] << " ";
    }
    std::cout << "\n\n";

    // Test modulo versions

    
    Modulus32 mod32(100);
    Modulus64 mod64(1000);

    
    Data32* d_randomInts_mod;
    Data64* d_randomInts64_mod;
    cudaMalloc(&d_randomInts_mod, N * sizeof(Data32));
    cudaMalloc(&d_randomInts64_mod, N * sizeof(Data64));

    
    rng.gen_random_u32_mod_p(N, &mod32, d_randomInts_mod);
    rng.gen_random_u64_mod_p(N, &mod64, d_randomInts64_mod);

   
    Data32 h_randomInts_mod[N];
    Data64 h_randomInts64_mod[N];
    cudaMemcpy(h_randomInts_mod, d_randomInts_mod, N * sizeof(Data32), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_randomInts64_mod, d_randomInts64_mod, N * sizeof(Data64), cudaMemcpyDeviceToHost);

    std::cout << "Modulo random 32-bit integers (mod 100):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomInts_mod[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Modulo random 64-bit integers (mod 1000):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomInts64_mod[i] << " ";
    }
    std::cout << "\n\n";

    
    const int numMods = 3;
    Modulus32 mod32_arr[numMods] = { Modulus32(100), Modulus32(200), Modulus32(300) };
    Data32* d_randomInts_mod_arr;
    cudaMalloc(&d_randomInts_mod_arr, N * sizeof(Data32));
    rng.gen_random_u32_mod_p(N, mod32_arr, numMods, d_randomInts_mod_arr);

    Data32 h_randomInts_mod_arr[N];
    cudaMemcpy(h_randomInts_mod_arr, d_randomInts_mod_arr, N * sizeof(Data32), cudaMemcpyDeviceToHost);
    std::cout << "Modulo array random 32-bit integers (mod 100, 200, 300):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_randomInts_mod_arr[i] << " ";
    }
    std::cout << "\n";

    
    cudaFree(d_randomInts);
    cudaFree(d_randomInts64);
    cudaFree(d_randomFloats);
    cudaFree(d_randomDoubles);
    cudaFree(d_randomInts_mod);
    cudaFree(d_randomInts64_mod);
    cudaFree(d_randomInts_mod_arr);

    return 0;
}
