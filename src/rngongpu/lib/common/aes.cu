// Original code by Cihangir Tezcan.
// Licensed under the Apache License, Version 2.0
// Original repository: https://github.com/cihangirtezcan/CUDA_AES
// Paper: https://ieeexplore.ieee.org/document/9422754
//
// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Modifications by Alişah Özcan, 2025.

#include "aes.cuh"
#include <cmath>
#include <iomanip>

namespace rngongpu
{
    __device__ Data64 reverseBytesULL(Data64 x) {
        int2 t = *reinterpret_cast<int2*>(&x);
        int2 r;
        
        r.x = __byte_perm(t.y, 0, 0x0123);
        r.y = __byte_perm(t.x, 0, 0x0123);
        
        return *reinterpret_cast<Data64*>(&r);
    }
    __device__ Data32 arithmeticRightShift(Data32 x, Data32 n)
    {
        return (x >> n) | (x << (-n & 31));
    }
    __device__ Data32 arithmetic16bitRightShift(Data32 x, Data32 n,
                                                Data32 n2Power)
    {
        return (x >> n) | ((x & n2Power) << (-n & 15));
    }
    __device__ Data32 arithmeticRightShiftBytePerm(Data32 x, Data32 n)
    {
        return __byte_perm(x, x, n);
    }

    // Key expansion from given key set, populate rk[44]
    __host__ void keyExpansion(std::vector<unsigned char> key, Data32* rk)
    {
        Data32 rk0, rk1, rk2, rk3;
        rk0 = (key[0] << 24) | (key[1] << 16) | (key[2] << 8) | key[3];
        rk1 = (key[4] << 24) | (key[5] << 16) | (key[6] << 8) | key[7];
        rk2 = (key[8] << 24) | (key[9] << 16) | (key[10] << 8) | key[11];
        rk3 = (key[12] << 24) | (key[13] << 16) | (key[14] << 8) | key[15];

        rk[0] = rk0;
        rk[1] = rk1;
        rk[2] = rk2;
        rk[3] = rk3;
        for (Data8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++)
        {
            Data32 temp = rk3;
            rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^
                  T4_1[(temp) &0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
            rk1 = rk1 ^ rk0;
            rk2 = rk2 ^ rk1;
            rk3 = rk2 ^ rk3;

            rk[roundCount * 4 + 4] = rk0;
            rk[roundCount * 4 + 5] = rk1;
            rk[roundCount * 4 + 6] = rk2;
            rk[roundCount * 4 + 7] = rk3;
        }
    }

    __host__ void keyExpansion192(std::vector<unsigned char> key, Data32* rk)
    {
        Data32 rk0, rk1, rk2, rk3, rk4, rk5;
        rk0 = (key[0] << 24) | (key[1] << 16) | (key[2] << 8) | key[3];
        rk1 = (key[4] << 24) | (key[5] << 16) | (key[6] << 8) | key[7];
        rk2 = (key[8] << 24) | (key[9] << 16) | (key[10] << 8) | key[11];
        rk3 = (key[12] << 24) | (key[13] << 16) | (key[14] << 8) | key[15];
        rk4 = (key[16] << 24) | (key[17] << 16) | (key[18] << 8) | key[19];
        rk5 = (key[20] << 24) | (key[21] << 16) | (key[22] << 8) | key[23];

        rk[0] = rk0;
        rk[1] = rk1;
        rk[2] = rk2;
        rk[3] = rk3;
        rk[4] = rk4;
        rk[5] = rk5;

        for (Data8 roundCount = 0; roundCount < ROUND_COUNT_192; roundCount++)
        {
            Data32 temp = rk5;
            rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^
                  T4_1[(temp) &0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
            rk1 = rk1 ^ rk0;
            rk2 = rk2 ^ rk1;
            rk3 = rk3 ^ rk2;
            rk4 = rk4 ^ rk3;
            rk5 = rk5 ^ rk4;

            rk[roundCount * 6 + 6] = rk0;
            rk[roundCount * 6 + 7] = rk1;
            rk[roundCount * 6 + 8] = rk2;
            rk[roundCount * 6 + 9] = rk3;
            if (roundCount == 7)
            {
                break;
            }
            rk[roundCount * 6 + 10] = rk4;
            rk[roundCount * 6 + 11] = rk5;
        }
    }

    __host__ void keyExpansion256(std::vector<unsigned char> key, Data32* rk)
    {
        Data32 rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7;
        rk0 = (key[0] << 24) | (key[1] << 16) | (key[2] << 8) | key[3];
        rk1 = (key[4] << 24) | (key[5] << 16) | (key[6] << 8) | key[7];
        rk2 = (key[8] << 24) | (key[9] << 16) | (key[10] << 8) | key[11];
        rk3 = (key[12] << 24) | (key[13] << 16) | (key[14] << 8) | key[15];
        rk4 = (key[16] << 24) | (key[17] << 16) | (key[18] << 8) | key[19];
        rk5 = (key[20] << 24) | (key[21] << 16) | (key[22] << 8) | key[23];
        rk6 = (key[24] << 24) | (key[25] << 16) | (key[26] << 8) | key[27];
        rk7 = (key[28] << 24) | (key[29] << 16) | (key[30] << 8) | key[31];

        rk[0] = rk0;
        rk[1] = rk1;
        rk[2] = rk2;
        rk[3] = rk3;
        rk[4] = rk4;
        rk[5] = rk5;
        rk[6] = rk6;
        rk[7] = rk7;

        for (Data8 roundCount = 0; roundCount < ROUND_COUNT_256; roundCount++)
        {
            Data32 temp = rk7;
            rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^
                  T4_1[(temp) &0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
            rk1 = rk1 ^ rk0;
            rk2 = rk2 ^ rk1;
            rk3 = rk3 ^ rk2;
            rk4 = rk4 ^ T4_3[(rk3 >> 24) & 0xff] ^ T4_2[(rk3 >> 16) & 0xff] ^
                  T4_1[(rk3 >> 8) & 0xff] ^ T4_0[rk3 & 0xff];
            rk5 = rk5 ^ rk4;
            rk6 = rk6 ^ rk5;
            rk7 = rk7 ^ rk6;

            rk[roundCount * 8 + 8] = rk0;
            rk[roundCount * 8 + 9] = rk1;
            rk[roundCount * 8 + 10] = rk2;
            rk[roundCount * 8 + 11] = rk3;
            if (roundCount == 6)
            {
                break;
            }
            rk[roundCount * 8 + 12] = rk4;
            rk[roundCount * 8 + 13] = rk5;
            rk[roundCount * 8 + 14] = rk6;
            rk[roundCount * 8 + 15] = rk7;
        }
    }

    __global__ void
    counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir(
        Data32* pt, Data32* rk, Data32* t0G, Data32* t4G, Data64* range,
        Data8* SAES, Data32 totalThreadCount, Data64* rng_res, Data32 N)
    {
        Data64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int warpThreadIndex = threadIdx.x & 31;

        __shared__ Data32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
        __shared__ Data8 Sbox[64][32][4];
        __shared__ Data32 rkS[AES_128_KEY_SIZE_INT];

        if (threadIdx.x < TABLE_SIZE)
        {
            for (Data8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE;
                 bankIndex++)
            {
                t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
                Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] =
                    SAES[threadIdx.x];
            }
            if (threadIdx.x < AES_128_KEY_SIZE_INT)
            {
                rkS[threadIdx.x] = rk[threadIdx.x];
            }
        }

        __syncthreads();
        Data32 pt0Init, pt1Init, pt2Init, pt3Init;
        Data32 s0, s1, s2, s3;
        pt0Init = pt[3];
        pt1Init = pt[2];
        pt2Init = pt[1];
        pt3Init = pt[0];
        Data64 threadRange = *range;
        Data64 threadRangeStart = pt2Init;
        threadRangeStart = threadRangeStart << 32;
        threadRangeStart ^= pt3Init;
        threadRangeStart += threadIndex * threadRange;
        pt2Init = threadRangeStart >> 32;
        pt3Init = threadRangeStart & 0xFFFFFFFF;
        // Overflow
        if (pt3Init == MAX_U32)
        {
            pt2Init++;
        }
        pt3Init++;
        for (Data32 rangeCount = 0; rangeCount < threadRange; rangeCount++)
        {
            // Create plaintext as 32 bit unsigned integers
            s0 = pt0Init;
            s1 = pt1Init;
            s2 = pt2Init;
            s3 = pt3Init;

            // First round just XORs input with key.
            s0 = s0 ^ rkS[0];
            s1 = s1 ^ rkS[1];
            s2 = s2 ^ rkS[2];
            s3 = s3 ^ rkS[3];

            Data32 t0, t1, t2, t3;
            for (Data8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1;
                 roundCount++)
            {
                // Table based round function
                Data32 rkStart = roundCount * 4 + 4;
                t0 =
                    t0S[s0 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s1 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart];
                t1 =
                    t0S[s1 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s2 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 1];
                t2 =
                    t0S[s2 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s3 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 2];
                t3 =
                    t0S[s3 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s0 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 3];
                s0 = t0;
                s1 = t1;
                s2 = t2;
                s3 = t3;
            }

            // Calculate the last round key
            // Last round uses s-box directly and XORs to produce output.
            s0 = arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t0 >> 24)) / 4][warpThreadIndex]
                                  [((t0 >> 24)) % 4],
                     SHIFT_1_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex]
                                  [((t1 >> 16)) % 4],
                     SHIFT_2_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex]
                                  [((t2 >> 8)) % 4],
                     SHIFT_3_RIGHT) ^
                 ((Data32) Sbox[((t3 & 0xFF) / 4)][warpThreadIndex]
                               [((t3 & 0xFF) % 4)]) ^
                 rkS[40];
            s1 = arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t1 >> 24)) / 4][warpThreadIndex]
                                  [((t1 >> 24)) % 4],
                     SHIFT_1_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex]
                                  [((t2 >> 16)) % 4],
                     SHIFT_2_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex]
                                  [((t3 >> 8)) % 4],
                     SHIFT_3_RIGHT) ^
                 ((Data32) Sbox[((t0 & 0xFF) / 4)][warpThreadIndex]
                               [((t0 & 0xFF) % 4)]) ^
                 rkS[41];
            s2 = arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t2 >> 24)) / 4][warpThreadIndex]
                                  [((t2 >> 24)) % 4],
                     SHIFT_1_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex]
                                  [((t3 >> 16)) % 4],
                     SHIFT_2_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex]
                                  [((t0 >> 8)) % 4],
                     SHIFT_3_RIGHT) ^
                 ((Data32) Sbox[((t1 & 0xFF) / 4)][warpThreadIndex]
                               [((t1 & 0xFF) % 4)]) ^
                 rkS[42];
            s3 = arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t3 >> 24)) / 4][warpThreadIndex]
                                  [((t3 >> 24)) % 4],
                     SHIFT_1_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex]
                                  [((t0 >> 16)) % 4],
                     SHIFT_2_RIGHT) ^
                 arithmeticRightShiftBytePerm(
                     (Data32) Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex]
                                  [((t1 >> 8)) % 4],
                     SHIFT_3_RIGHT) ^
                 ((Data32) Sbox[((t2 & 0xFF) / 4)][warpThreadIndex]
                               [((t2 & 0xFF) % 4)]) ^
                 rkS[43];
                 
            // Overflow
            if (pt3Init == MAX_U32)
            {
                pt2Init++;
            }
            pt3Init++;

            Data64 res_num1, res_num2;

            res_num1 = s0;
            res_num1 <<= 32;
            res_num1 ^= s1;

            res_num2 = s2;
            res_num2 <<= 32;
            res_num2 ^= s3;
            if (2 * rangeCount * totalThreadCount + 2 * threadIndex + 1 < N)
            {
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex] =
                    reverseBytesULL(res_num1);
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex + 1] =
                    reverseBytesULL(res_num2);
            }
            else if (2 * rangeCount * totalThreadCount + 2 * threadIndex < N)
            {
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex] =
                    reverseBytesULL(res_num1);
            } 
        }
    }

    __global__ void
    counter192WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(
        Data32* pt, Data32* rk, Data32* t0G, Data32* t4G, Data64* range,
        Data32 totalThreadCount, Data64* rng_res, Data32 N)
    {
        int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int warpThreadIndex = threadIdx.x & 31;
        int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

        // <SHARED MEMORY>
        __shared__ Data32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
        __shared__ Data32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
        __shared__ Data32 rkS[AES_192_KEY_SIZE_INT];

        if (threadIdx.x < TABLE_SIZE)
        {
            for (Data8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE;
                 bankIndex++)
            {
                t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
            }

            for (Data8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++)
            {
                t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
            }

            if (threadIdx.x < AES_192_KEY_SIZE_INT)
            {
                rkS[threadIdx.x] = rk[threadIdx.x];
            }
        }
        // </SHARED MEMORY>

        // Wait until every thread is ready
        __syncthreads();

        Data32 pt0Init, pt1Init, pt2Init, pt3Init;
        Data32 s0, s1, s2, s3;
        pt0Init = pt[0];
        pt1Init = pt[1];
        pt2Init = pt[2];
        pt3Init = pt[3];

        Data32 threadRange = *range;
        Data64 threadRangeStart = pt2Init;
        threadRangeStart = threadRangeStart << 32;
        threadRangeStart ^= pt3Init;
        threadRangeStart += (Data64) threadIndex * threadRange;
        pt2Init = threadRangeStart >> 32;
        pt3Init = threadRangeStart & 0xFFFFFFFF;
        // Overflow
        if (pt3Init == MAX_U32)
        {
            pt2Init++;
        }
        pt3Init++;
        for (Data32 rangeCount = 0; rangeCount < threadRange; rangeCount++)
        {
            // Create plaintext as 32 bit unsigned integers
            s0 = pt0Init;
            s1 = pt1Init;
            s2 = pt2Init;
            s3 = pt3Init;

            // First round just XORs input with key.
            s0 = s0 ^ rkS[0];
            s1 = s1 ^ rkS[1];
            s2 = s2 ^ rkS[2];
            s3 = s3 ^ rkS[3];

            Data32 t0, t1, t2, t3;
            for (Data8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_192;
                 roundCount++)
            {
                // Table based round function
                Data32 rkStart = roundCount * 4 + 4;
                t0 =
                    t0S[s0 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s1 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart];
                t1 =
                    t0S[s1 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s2 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 1];
                t2 =
                    t0S[s2 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s3 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 2];
                t3 =
                    t0S[s3 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s0 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 3];

                s0 = t0;
                s1 = t1;
                s2 = t2;
                s3 = t3;
            }

            // Calculate the last round key
            // Last round uses s-box directly and XORs to produce output.
            s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t3) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[48];
            s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t0) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[49];
            s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t1) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[50];
            s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t2) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[51];

            /*if (threadIndex == 0 && rangeCount == 0) {
                printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
            }*/

            // Overflow
            if (pt3Init == MAX_U32)
            {
                pt2Init++;
            }

            // Create key as 32 bit unsigned integers
            pt3Init++;
            Data64 res_num1, res_num2;

            res_num1 = s0;
            res_num1 <<= 32;
            res_num1 ^= s1;

            res_num2 = s2;
            res_num2 <<= 32;
            res_num2 ^= s3;
            if (2 * rangeCount * totalThreadCount + 2 * threadIndex + 1 < N)
            {
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex] =
                    reverseBytesULL(res_num1);
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex + 1] =
                    reverseBytesULL(res_num2);
            }
            else if (2 * rangeCount * totalThreadCount + 2 * threadIndex < N)
            {
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex] =
                    reverseBytesULL(res_num1);
            } 
        }
    }

    __global__ void
    counter256WithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox(
        Data32* pt, Data32* rk, Data32* t0G, Data32* t4G, Data64* range,
        Data32 totalThreadCount, Data64* rng_res, Data32 N)
    {
        int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
        int warpThreadIndex = threadIdx.x & 31;
        int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

        // <SHARED MEMORY>
        __shared__ Data32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
        __shared__ Data32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
        __shared__ Data32 rkS[AES_256_KEY_SIZE_INT];

        if (threadIdx.x < TABLE_SIZE)
        {
            for (Data8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE;
                 bankIndex++)
            {
                t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
            }

            for (Data8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++)
            {
                t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
            }

            if (threadIdx.x < AES_256_KEY_SIZE_INT)
            {
                rkS[threadIdx.x] = rk[threadIdx.x];
            }
        }
        // </SHARED MEMORY>

        // Wait until every thread is ready
        __syncthreads();

        Data32 pt0Init, pt1Init, pt2Init, pt3Init;
        Data32 s0, s1, s2, s3;
        pt0Init = pt[0];
        pt1Init = pt[1];
        pt2Init = pt[2];
        pt3Init = pt[3];

        Data32 threadRange = *range;
        Data64 threadRangeStart = pt2Init;
        threadRangeStart = threadRangeStart << 32;
        threadRangeStart ^= pt3Init;
        threadRangeStart += (Data64) threadIndex * threadRange;
        pt2Init = threadRangeStart >> 32;
        pt3Init = threadRangeStart & 0xFFFFFFFF;
        // Overflow
        if (pt3Init == MAX_U32)
        {
            pt2Init++;
        }
        pt3Init++;
        for (Data32 rangeCount = 0; rangeCount < threadRange; rangeCount++)
        {
            // Create plaintext as 32 bit unsigned integers
            s0 = pt0Init;
            s1 = pt1Init;
            s2 = pt2Init;
            s3 = pt3Init;

            // First round just XORs input with key.
            s0 = s0 ^ rkS[0];
            s1 = s1 ^ rkS[1];
            s2 = s2 ^ rkS[2];
            s3 = s3 ^ rkS[3];

            Data32 t0, t1, t2, t3;
            for (Data8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_256;
                 roundCount++)
            {
                // Table based round function
                Data32 rkStart = roundCount * 4 + 4;
                t0 =
                    t0S[s0 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s1 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart];
                t1 =
                    t0S[s1 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s2 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 1];
                t2 =
                    t0S[s2 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s3 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 2];
                t3 =
                    t0S[s3 >> 24][warpThreadIndex] ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s0 >> 16) & 0xFF][warpThreadIndex],
                        SHIFT_1_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^
                    arithmeticRightShiftBytePerm(
                        t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^
                    rkS[rkStart + 3];

                s0 = t0;
                s1 = t1;
                s2 = t2;
                s3 = t3;
            }

            // Calculate the last round key
            // Last round uses s-box directly and XORs to produce output.
            s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t3) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[56];
            s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t0) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[57];
            s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t1) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[58];
            s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^
                 (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^
                 (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^
                 (t4S[(t2) &0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[59];

            // if (threadIndex == 0 && rangeCount == 0) {
            // printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
            // }

            // Overflow
            if (pt3Init == MAX_U32)
            {
                pt2Init++;
            }

            // Create key as 32 bit unsigned integers
            pt3Init++;

            Data64 res_num1, res_num2;

            res_num1 = s0;
            res_num1 <<= 32;
            res_num1 ^= s1;

            res_num2 = s2;
            res_num2 <<= 32;
            res_num2 ^= s3;
            if (2 * rangeCount * totalThreadCount + 2 * threadIndex + 1 < N)
            {
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex] =
                    reverseBytesULL(res_num1);
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex + 1] =
                    reverseBytesULL(res_num2);
            }
            else if (2 * rangeCount * totalThreadCount + 2 * threadIndex < N)
            {
                rng_res[2 * rangeCount * totalThreadCount + 2 * threadIndex] =
                    reverseBytesULL(res_num1);
            } 
        }
    }

} // namespace rngongpu