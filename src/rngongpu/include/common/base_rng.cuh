// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef BASE_RNG_H
#define BASE_RNG_H

#include <cassert>
#include <exception>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "modular_arith.cuh"

class CudaException___ : public std::exception
{
  public:
    CudaException___(const std::string& file, int line, cudaError_t error)
        : file_(file), line_(line), error_(error)
    {
    }

    const char* what() const noexcept override
    {
        return m_error_string.c_str();
    }

  private:
    std::string file_;
    int line_;
    cudaError_t error_;
    std::string m_error_string = "CUDA Error in " + file_ + " at line " +
                                 std::to_string(line_) + ": " +
                                 cudaGetErrorString(error_);
};

#define RNGONGPU_CUDA_CHECK(err)                                               \
    do                                                                         \
    {                                                                          \
        cudaError_t error = err;                                               \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            throw CudaException___(__FILE__, __LINE__, error);                 \
        }                                                                      \
    } while (0)

namespace rngongpu
{

} // namespace rngongpu
#endif // BASE_RNG_H
