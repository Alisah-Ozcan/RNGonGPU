// Original code by Cihangir Tezcan.
// (No license specified in the original repository.)
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

namespace rngongpu
{
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
    __global__ void
    counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBoxCihangir(
        Data32* pt, Data32* rk, Data32* t0G, Data32* t4G, Data64* range,
        Data8* SAES, Data64* rng_res, Data32 N)
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
        pt0Init = pt[0];
        pt1Init = pt[1];
        pt2Init = pt[2];
        pt3Init = pt[3];
        Data64 threadRange = *range;
        Data64 threadRangeStart = pt2Init;
        threadRangeStart = threadRangeStart << 32;
        threadRangeStart ^= pt3Init;
        threadRangeStart += threadIndex * threadRange;
        pt2Init = threadRangeStart >> 32;
        pt3Init = threadRangeStart & 0xFFFFFFFF;

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
            if (2 * threadRange * threadIndex + 2 * rangeCount + 1 < N)
            {
                rng_res[2 * threadRange * threadIndex + 2 * rangeCount] =
                    res_num1;
                rng_res[2 * threadRange * threadIndex + 2 * rangeCount + 1] =
                    res_num2;
            }
            else if (2 * threadRange * threadIndex + 2 * rangeCount)
            {
                rng_res[2 * threadRange * threadIndex + 2 * rangeCount] =
                    res_num1;
            }

            // if (2 * threadRange * threadIndex + 2 * rangeCount + 1 > N)
            // printf("I exceeded the limit with %llu %d\n", threadIndex,
            // rangeCount);
        }
    }

    __global__ void box_muller_u32(Data32* nums, f32* res, Data32 N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (2 * tid + 1 < N) {
            f32 u1 = (float) nums[2 * tid] / MAX_U32;
            f32 u2 = (float) nums[2 * tid + 1] / MAX_U32;
    
            double radius = std::sqrt(-2.0 * std::log(u1));
            double theta = 2.0 * M_PI * u2;
    
            res[2 * tid] = radius * std::cos(theta);
            res[2 * tid + 1] = radius * std::sin(theta);
        }
    }
    
    __global__ void box_muller_u64(Data64* nums, f64* res, Data32 N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (2 * tid + 1 < N) {
            f64 u1 = (double) nums[2 * tid] / MAX_U64;
            f64 u2 = (double) nums[2 * tid + 1] / MAX_U64;
    
            double radius = std::sqrt(-2.0 * std::log(u1));
            double theta = 2.0 * M_PI * u2;
    
            res[2 * tid] = radius * std::cos(theta);
            res[2 * tid + 1] = radius * std::sin(theta);
        }
    }


    __global__ void mod_reduce_u64(Data64* nums, Modulus64* p, Data32 N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < N) {
            nums[tid] = OPERATOR_GPU_64::reduce(nums[tid], *p);
        }
    }

    __global__ void mod_reduce_u64(Data64* nums, Modulus64* p, Data32 p_N, Data32 N) {
        int local_tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y_id = blockIdx.y;
        int global_tid = y_id * (blockDim.x * gridDim.x) + local_tid;

        if (global_tid < N && y_id < p_N) {
            nums[global_tid] = OPERATOR_GPU_64::reduce(nums[global_tid], p[y_id]);
        }
    }

    __global__ void mod_reduce_u32(Data32* nums, Modulus32* p, Data32 p_N, Data32 N) {
        int local_tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y_id = blockIdx.y;
        int global_tid = y_id * (blockDim.x * gridDim.x) + local_tid;

        if (global_tid < N && y_id < p_N) {
            nums[global_tid] = OPERATOR_GPU_32::reduce(nums[global_tid], p[y_id]);
        }
    }

    __global__ void mod_reduce_u32(Data32* nums, Modulus32* p, Data32 N) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x; 

        if (tid < N) {
            nums[tid] = OPERATOR_GPU_32::reduce(nums[tid], *p);
        }
    }
} // namespace rngongpu