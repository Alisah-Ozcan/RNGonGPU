// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RNGONGPU_BENCHMARK_COMMON_CUH
#define RNGONGPU_BENCHMARK_COMMON_CUH

#include <cuda_runtime.h>
#include <openssl/rand.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rngongpu/common/base_rng.cuh"
#include "rngongpu/rand_aes/aes_rng.cuh"
#include "rngongpu/rand_cuda/cuda_rng.cuh"

namespace rngongpu::benchmark
{
    struct RuntimeConfig
    {
        std::vector<int> size_logs{16, 17, 18, 19, 20, 21, 22, 23, 24};
        std::vector<int> security_levels{128, 192, 256};
        std::vector<int> stddevs{3};
        std::vector<std::string> distributions{"uniform"};
        std::vector<std::string> data_types{"u32", "u64"};
        std::vector<std::string> curand_states{"xorwow"};
        int warmup = 3;
        int iterations = 10;
        bool csv = false;
        bool help = false;
        std::string output_dir;
        std::string output_file;
    };

    class CudaStream
    {
      public:
        CudaStream() { RNGONGPU_CUDA_CHECK(cudaStreamCreate(&stream_)); }
        ~CudaStream() { cudaStreamDestroy(stream_); }

        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;

        cudaStream_t get() const { return stream_; }

      private:
        cudaStream_t stream_{};
    };

    class CudaEventTimer
    {
      public:
        CudaEventTimer()
        {
            RNGONGPU_CUDA_CHECK(cudaEventCreate(&start_));
            RNGONGPU_CUDA_CHECK(cudaEventCreate(&stop_));
        }

        ~CudaEventTimer()
        {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        template <typename F> float measure(cudaStream_t stream, F&& fn)
        {
            RNGONGPU_CUDA_CHECK(cudaEventRecord(start_, stream));
            fn();
            RNGONGPU_CUDA_CHECK(cudaEventRecord(stop_, stream));
            RNGONGPU_CUDA_CHECK(cudaEventSynchronize(stop_));

            float milliseconds = 0.0F;
            RNGONGPU_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
            return milliseconds;
        }

      private:
        cudaEvent_t start_{};
        cudaEvent_t stop_{};
    };

    template <typename T> class DeviceBuffer
    {
      public:
        explicit DeviceBuffer(std::size_t count) : count_(count)
        {
            RNGONGPU_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }

        ~DeviceBuffer() { cudaFree(ptr_); }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* get() const { return ptr_; }
        std::size_t count() const { return count_; }
        std::size_t bytes() const { return count_ * sizeof(T); }

      private:
        T* ptr_{};
        std::size_t count_{};
    };

    struct Measurement
    {
        std::string backend;
        std::string distribution;
        std::string data_type;
        std::string variant;
        int size_log = 0;
        int security_level = 0;
        int stddev = 0;
        std::size_t elements = 0;
        std::size_t bytes = 0;
        float avg_ms = 0.0F;
    };

    class ReportSink
    {
      public:
        ReportSink(const RuntimeConfig& config, std::string default_filename)
            : csv_(config.csv)
        {
            if (!config.output_dir.empty())
            {
                std::filesystem::create_directories(config.output_dir);
                const auto filename = config.output_file.empty()
                                          ? std::move(default_filename)
                                          : config.output_file;
                file_.open(std::filesystem::path(config.output_dir) / filename);
                if (!file_)
                {
                    throw std::runtime_error("failed to open benchmark output file");
                }
            }
            else if (!config.output_file.empty())
            {
                file_.open(config.output_file);
                if (!file_)
                {
                    throw std::runtime_error("failed to open benchmark output file");
                }
            }
        }

        bool csv() const { return csv_; }
        bool has_file() const { return file_.is_open(); }
        std::ofstream& file() { return file_; }

      private:
        bool csv_ = false;
        std::ofstream file_;
    };

    inline std::vector<unsigned char> random_bytes(std::size_t size)
    {
        std::vector<unsigned char> bytes(size);
        if (size != 0 &&
            RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
        {
            throw std::runtime_error("RAND_bytes failed while preparing benchmark input");
        }
        return bytes;
    }

    inline std::uint64_t random_seed()
    {
        std::random_device rd;
        std::mt19937_64 generator(rd());
        std::uniform_int_distribution<std::uint64_t> distribution;
        return distribution(generator);
    }

    inline SecurityLevel security_level_from_bits(int bits)
    {
        switch (bits)
        {
            case 128:
                return SecurityLevel::AES128;
            case 192:
                return SecurityLevel::AES192;
            case 256:
                return SecurityLevel::AES256;
            default:
                throw std::invalid_argument("security levels must be 128, 192, or 256");
        }
    }

    inline std::size_t entropy_bytes_for_security_level(int bits)
    {
        return static_cast<std::size_t>(bits / 8);
    }

    inline std::size_t nonce_bytes_for_security_level(int bits)
    {
        return bits == 128 ? 8 : 16;
    }

    inline std::vector<std::string> split(std::string value, char separator)
    {
        std::replace(value.begin(), value.end(), ';', separator);

        std::vector<std::string> out;
        std::stringstream ss(value);
        std::string item;
        while (std::getline(ss, item, separator))
        {
            if (!item.empty())
            {
                out.push_back(item);
            }
        }
        return out;
    }

    inline std::vector<int> parse_int_list(const std::string& value)
    {
        std::vector<int> out;
        for (const auto& item : split(value, ','))
        {
            out.push_back(std::stoi(item));
        }
        return out;
    }

    inline bool contains(const std::vector<std::string>& values,
                         const std::string& value)
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    }

    inline void print_common_help(const char* executable, const char* backend)
    {
        std::cout
            << "Usage: " << executable << " [options]\n\n"
            << "Runtime options for " << backend << " benchmarks:\n"
            << "  --sizes 16,20,24                log2 element counts\n"
            << "  --distributions uniform,normal,ternary\n"
            << "  --data-types u32,u64,f32,f64    data types; valid values depend on distribution\n"
            << "  --warmup 3                      warmup iterations\n"
            << "  --iterations 10                 measured iterations\n"
            << "  --csv                           print CSV output\n"
            << "  --output-dir benchmark/csv      also write CSV results to directory\n"
            << "  --output-file results.csv       also write CSV results to file\n"
            << "  --help                          show this message\n\n";
    }

    inline RuntimeConfig parse_common_args(int argc, char** argv,
                                           const char* backend)
    {
        RuntimeConfig config;

        for (int i = 1; i < argc; i++)
        {
            const std::string arg = argv[i];
            auto require_value = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc)
                {
                    throw std::invalid_argument(name + " requires a value");
                }
                return argv[++i];
            };

            if (arg == "--sizes")
            {
                config.size_logs = parse_int_list(require_value(arg));
            }
            else if (arg == "--distributions")
            {
                config.distributions = split(require_value(arg), ',');
            }
            else if (arg == "--data-types" || arg == "--types")
            {
                config.data_types = split(require_value(arg), ',');
            }
            else if (arg == "--warmup")
            {
                config.warmup = std::stoi(require_value(arg));
            }
            else if (arg == "--iterations")
            {
                config.iterations = std::stoi(require_value(arg));
            }
            else if (arg == "--csv")
            {
                config.csv = true;
            }
            else if (arg == "--output-dir")
            {
                config.output_dir = require_value(arg);
            }
            else if (arg == "--output-file")
            {
                config.output_file = require_value(arg);
            }
            else if (arg == "--help")
            {
                config.help = true;
            }
        }

        if (config.help)
        {
            return config;
        }

        if (config.iterations <= 0)
        {
            throw std::invalid_argument("--iterations must be greater than zero");
        }
        if (config.warmup < 0)
        {
            throw std::invalid_argument("--warmup must be zero or greater");
        }

        return config;
    }

    inline void print_header(std::ostream& out, bool csv)
    {
        if (csv)
        {
            out << "backend,distribution,data_type,variant,size_log,security_level,"
                   "stddev,elements,bytes,avg_ms,throughput_gib_s\n";
            return;
        }

        out << std::left << std::setw(10) << "backend" << std::setw(10)
            << "dist" << std::setw(10) << "data_type" << std::setw(16)
            << "variant" << std::setw(8) << "logN" << std::setw(8)
            << "sec" << std::setw(8) << "stddev" << std::setw(12) << "avg_ms"
            << "GiB/s\n";
    }

    inline void print_header(ReportSink& sink)
    {
        print_header(std::cout, sink.csv());
        if (sink.has_file())
        {
            print_header(sink.file(), true);
        }
    }

    inline void print_measurement(std::ostream& out, const Measurement& m,
                                  bool csv)
    {
        const double seconds = static_cast<double>(m.avg_ms) / 1000.0;
        const double gib = static_cast<double>(m.bytes) / (1024.0 * 1024.0 * 1024.0);
        const double throughput = seconds > 0.0 ? gib / seconds : 0.0;

        if (csv)
        {
            out << m.backend << ',' << m.distribution << ',' << m.data_type
                << ',' << m.variant << ',' << m.size_log << ','
                << m.security_level << ',' << m.stddev << ','
                << m.elements << ',' << m.bytes << ',' << m.avg_ms << ','
                << throughput << '\n';
            return;
        }

        out << std::left << std::setw(10) << m.backend << std::setw(10)
            << m.distribution << std::setw(10) << m.data_type
            << std::setw(16) << m.variant << std::setw(8) << m.size_log
            << std::setw(8) << m.security_level << std::setw(8) << m.stddev
            << std::setw(12) << m.avg_ms << throughput << '\n';
    }

    inline void print_measurement(ReportSink& sink, const Measurement& m)
    {
        print_measurement(std::cout, m, sink.csv());
        if (sink.has_file())
        {
            print_measurement(sink.file(), m, true);
        }
    }

    template <typename F>
    Measurement measure(RuntimeConfig const& config, Measurement measurement,
                        cudaStream_t stream, F&& fn)
    {
        CudaEventTimer timer;

        for (int i = 0; i < config.warmup; i++)
        {
            fn();
            RNGONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        float total = 0.0F;
        for (int i = 0; i < config.iterations; i++)
        {
            total += timer.measure(stream, fn);
        }
        measurement.avg_ms = total / static_cast<float>(config.iterations);

        return measurement;
    }
} // namespace rngongpu::benchmark

#endif // RNGONGPU_BENCHMARK_COMMON_CUH
