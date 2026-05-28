// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_common.cuh"

namespace bench = rngongpu::benchmark;
using namespace rngongpu;

namespace
{
    struct AesConfig
    {
        bench::RuntimeConfig common;
    };

    AesConfig parse_args(int argc, char** argv)
    {
        AesConfig config;
        config.common = bench::parse_common_args(argc, argv, "AES CTR-DRBG");

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

            if (arg == "--security-levels")
            {
                config.common.security_levels =
                    bench::parse_int_list(require_value(arg));
            }
            else if (arg == "--stddevs")
            {
                config.common.stddevs = bench::parse_int_list(require_value(arg));
            }
            else if (arg == "--help")
            {
                bench::print_common_help(argv[0], "AES CTR-DRBG");
                std::cout
                    << "AES-specific options:\n"
                    << "  --security-levels 128,192,256\n"
                    << "  --stddevs 3,8\n";
                std::exit(0);
            }
        }

        return config;
    }

    RNG<Mode::AES> make_aes_rng(int security_bits)
    {
        auto entropy =
            bench::random_bytes(bench::entropy_bytes_for_security_level(security_bits));
        auto nonce =
            bench::random_bytes(bench::nonce_bytes_for_security_level(security_bits));
        std::vector<unsigned char> personalization;

        return RNG<Mode::AES>(entropy, nonce, personalization,
                              bench::security_level_from_bits(security_bits),
                              false);
    }

    template <typename T>
    void run_uniform(const bench::RuntimeConfig& config, bench::ReportSink& sink,
                     int size_log,
                     int security_bits, const std::string& type_name)
    {
        const auto size = Data64{1} << size_log;
        auto drbg = make_aes_rng(security_bits);
        bench::DeviceBuffer<T> results(size);
        bench::CudaStream stream;
        std::vector<unsigned char> additional_input;

        bench::Measurement measurement;
        measurement.backend = "aes";
        measurement.distribution = "uniform";
        measurement.data_type = type_name;
        measurement.variant = "ctr_drbg";
        measurement.size_log = size_log;
        measurement.security_level = security_bits;
        measurement.elements = results.count();
        measurement.bytes = results.bytes();

        auto result = bench::measure(
            config, measurement, stream.get(),
            [&]
            {
                drbg.uniform_random_number(results.get(), size, additional_input,
                                           stream.get());
            });
        bench::print_measurement(sink, result);
    }

    template <typename T>
    void run_normal(const bench::RuntimeConfig& config, bench::ReportSink& sink,
                    int size_log,
                    int security_bits, int stddev,
                    const std::string& type_name)
    {
        const auto size = Data64{1} << size_log;
        auto drbg = make_aes_rng(security_bits);
        bench::DeviceBuffer<T> results(size);
        bench::CudaStream stream;
        std::vector<unsigned char> additional_input;

        bench::Measurement measurement;
        measurement.backend = "aes";
        measurement.distribution = "normal";
        measurement.data_type = type_name;
        measurement.variant = "ctr_drbg";
        measurement.size_log = size_log;
        measurement.security_level = security_bits;
        measurement.stddev = stddev;
        measurement.elements = results.count();
        measurement.bytes = results.bytes();

        auto result = bench::measure(
            config, measurement, stream.get(),
            [&]
            {
                drbg.normal_random_number(static_cast<T>(stddev), results.get(), size,
                                          additional_input, stream.get());
            });
        bench::print_measurement(sink, result);
    }

    template <typename T>
    void run_ternary(const bench::RuntimeConfig& config, bench::ReportSink& sink,
                     int size_log,
                     int security_bits, const std::string& type_name)
    {
        const auto size = Data64{1} << size_log;
        auto drbg = make_aes_rng(security_bits);
        bench::DeviceBuffer<T> results(size);
        bench::CudaStream stream;
        std::vector<unsigned char> additional_input;

        bench::Measurement measurement;
        measurement.backend = "aes";
        measurement.distribution = "ternary";
        measurement.data_type = type_name;
        measurement.variant = "ctr_drbg";
        measurement.size_log = size_log;
        measurement.security_level = security_bits;
        measurement.elements = results.count();
        measurement.bytes = results.bytes();

        auto result = bench::measure(
            config, measurement, stream.get(),
            [&]
            {
                drbg.ternary_random_number(results.get(), size, additional_input,
                                           stream.get());
            });
        bench::print_measurement(sink, result);
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto config = parse_args(argc, argv).common;
        if (config.help)
        {
            return 0;
        }
        bench::ReportSink sink(config, "aes_benchmark.csv");
        bench::print_header(sink);

        for (const auto size_log : config.size_logs)
        {
            for (const auto security_bits : config.security_levels)
            {
                if (bench::contains(config.distributions, "uniform"))
                {
                    if (bench::contains(config.data_types, "u32"))
                        run_uniform<Data32>(config, sink, size_log, security_bits, "u32");
                    if (bench::contains(config.data_types, "u64"))
                        run_uniform<Data64>(config, sink, size_log, security_bits, "u64");
                }

                if (bench::contains(config.distributions, "normal"))
                {
                    for (const auto stddev : config.stddevs)
                    {
                        if (bench::contains(config.data_types, "f32"))
                            run_normal<f32>(config, sink, size_log, security_bits,
                                            stddev, "f32");
                        if (bench::contains(config.data_types, "f64"))
                            run_normal<f64>(config, sink, size_log, security_bits,
                                            stddev, "f64");
                    }
                }

                if (bench::contains(config.distributions, "ternary"))
                {
                    if (bench::contains(config.data_types, "u32"))
                        run_ternary<Data32>(config, sink, size_log, security_bits,
                                            "u32");
                    if (bench::contains(config.data_types, "u64"))
                        run_ternary<Data64>(config, sink, size_log, security_bits,
                                            "u64");
                }
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "aes_benchmark error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
