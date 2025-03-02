// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AES_RNG_H
#define AES_RNG_H

#include "aes.cuh"
#include <openssl/evp.h>
#include <vector>

namespace rngongpu
{
    class AES_RNG
    {
      private:
        // Working state fields
        std::vector<unsigned char> seed;
        std::vector<unsigned char> key;
        std::vector<unsigned char> nonce; // The V in SP800-90.

        bool isPredictionResistanceEnabled;

        // NIST SP 800‑90A recommends that the number of blocks generated before
        // a reseed be limited.
        const Data64 RESEED_INTERVAL = (1ULL << 48);
        const Data32 MAX_BYTES_PER_REQUEST = 1 << 19;
        const std::size_t securityLevel = 128;
        Data64 reseedCounter;

        // AES-128 relevant fields
        Data32 *t0, *t1, *t2, *t3, *t4, *t4_0, *t4_1, *t4_2, *t4_3;
        Data8* SAES_d;
        Data32* rcon;
        Data32* roundKeys;
        Data32* d_nonce;

        void init();
        void increment_nonce(Data32 N);
        void update(std::vector<unsigned char> additionalInput);
        void resetReseedCounter();
        std::vector<unsigned char> DF(const std::vector<unsigned char>& input,
                                      std::size_t outputLen);

        // generate random bits on the device. Write N bytes to res
        // using BLOCKS blocks with THREADS threads each.
        void gen_random_bytes(int N, int nBLOCKS, int nTHREADS, Data64* res,
                              std::vector<unsigned char> additionalInput);

      public:
        // Potential additional input?
        void reseed(std::vector<unsigned char>);
        void printWorkingState();
        AES_RNG(bool _isPredictionResistanceEnabled);

        void gen_random_u32(int N, Data32* res);
        void gen_random_u32_mod_p(int N, Modulus32* p, Data32* res);
        void gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num,
                                  Data32* res);
        void gen_random_u64(int N, Data64* res);
        void gen_random_u64_mod_p(int N, Modulus64* p, Data64* res);
        void gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num,
                                  Data64* res);
        void gen_random_f32(int N, f32* res);
        void gen_random_f64(int N, f64* res);

        ~AES_RNG();
    };
} // namespace rngongpu

#endif // AES_RNG_H