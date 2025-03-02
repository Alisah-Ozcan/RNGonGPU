// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "base_rng.cuh"

namespace rngongpu
{
    __global__ void box_muller_u32(Data32* nums, f32* res, Data32 N)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (2 * tid + 1 < N)
        {
            f32 u1 = (float) nums[2 * tid] / MAX_U32;
            f32 u2 = (float) nums[2 * tid + 1] / MAX_U32;

            double radius = std::sqrt(-2.0 * std::log(u1));
            double theta = 2.0 * M_PI * u2;

            res[2 * tid] = radius * std::cos(theta);
            res[2 * tid + 1] = radius * std::sin(theta);
        }
    }

    __global__ void box_muller_u64(Data64* nums, f64* res, Data32 N)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (2 * tid + 1 < N)
        {
            f64 u1 = (double) nums[2 * tid] / MAX_U64;
            f64 u2 = (double) nums[2 * tid + 1] / MAX_U64;

            double radius = std::sqrt(-2.0 * std::log(u1));
            double theta = 2.0 * M_PI * u2;

            res[2 * tid] = radius * std::cos(theta);
            res[2 * tid + 1] = radius * std::sin(theta);
        }
    }

    __global__ void mod_reduce_u64(Data64* nums, Modulus64* p, Data32 N)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < N)
        {
            nums[tid] = OPERATOR_GPU_64::reduce(nums[tid], *p);
        }
    }

    __global__ void mod_reduce_u64(Data64* nums, Modulus64* p, Data32 p_N,
                                   Data32 N)
    {
        int local_tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y_id = blockIdx.y;
        int global_tid = y_id * (blockDim.x * gridDim.x) + local_tid;

        if (global_tid < N && y_id < p_N)
        {
            nums[global_tid] =
                OPERATOR_GPU_64::reduce(nums[global_tid], p[y_id]);
        }
    }

    __global__ void mod_reduce_u32(Data32* nums, Modulus32* p, Data32 p_N,
                                   Data32 N)
    {
        int local_tid = blockDim.x * blockIdx.x + threadIdx.x;
        int y_id = blockIdx.y;
        int global_tid = y_id * (blockDim.x * gridDim.x) + local_tid;

        if (global_tid < N && y_id < p_N)
        {
            nums[global_tid] =
                OPERATOR_GPU_32::reduce(nums[global_tid], p[y_id]);
        }
    }

    __global__ void mod_reduce_u32(Data32* nums, Modulus32* p, Data32 N)
    {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < N)
        {
            nums[tid] = OPERATOR_GPU_32::reduce(nums[tid], *p);
        }
    }

} // namespace rngongpu