// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AES_RNG_H
#define AES_RNG_H

#include "aes.cuh"

namespace rngongpu
{
    class BaseRNG_AES {
        private:
            // Working state fields
            Data32* seed;
            Data32* key;
            Data32* nonce; //The V in SP800-90.
            
            //AES-128 relevant fields
            Data32 *t0, *t1, *t2, *t3, *t4, *t4_0, *t4_1, *t4_2, *t4_3;
            Data8* SAES_d;
            Data32* rcon;
            Data32* roundKeys;
            Data32* d_nonce;
    
            void init();
    
        protected:
            virtual void initState();
            void increment_nonce(Data32 N);
    
            // generate random bits on the device. Write N bytes to res 
            // using BLOCKS blocks with THREADS threads each.
            void gen_random_bytes(int N, int BLOCKS, int THREADS, Data64* res);
        public:
            BaseRNG_AES();
    
            // tune the object for desired output in the next function call 
            // ex: set the stddev and mean for Normal distribution objects
            // virtual void set_state() = 0;
    
            virtual void gen_random_u32(int N, Data32* res) = 0;
            virtual void gen_random_u32_mod_p(int N, Modulus32* p, Data32* res) = 0;
            virtual void gen_random_u32_mod_p(int N, Modulus32* p, Data32 p_num, Data32* res) = 0;
            virtual void gen_random_u64(int N, Data64* res) = 0;
            virtual void gen_random_u64_mod_p(int N, Modulus64* p, Data64* res) = 0;
            virtual void gen_random_u64_mod_p(int N, Modulus64* p, Data32 p_num, Data64* res) = 0;
            virtual void gen_random_f32(int N, f32* res) = 0;
            virtual void gen_random_f64(int N, f64* res) = 0;
    
            ~BaseRNG_AES();
    };
} // namespace rngongpu

#endif // AES_RNG_H