// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <curand_kernel.h>

#include <cstdlib>
#include <iostream>
#include <string>

#include "benchmark_common.cuh"

namespace bench = rngongpu::benchmark;
using namespace rngongpu;

namespace
{
    struct CudaConfig
    {
        bench::RuntimeConfig common;
    };

    CudaConfig parse_args(int argc, char** argv)
    {
        CudaConfig config;
        config.common = bench::parse_common_args(argc, argv, "cuRAND");

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

            if (arg == "--curand-states")
            {
                config.common.curand_states = bench::split(require_value(arg), ',');
            }
            else if (arg == "--stddevs")
            {
                config.common.stddevs = bench::parse_int_list(require_value(arg));
            }
            else if (arg == "--help")
            {
                bench::print_common_help(argv[0], "cuRAND");
                std::cout << "cuRAND-specific options:\n"
                          << "  --curand-states xorwow,mrg32k3a,philox\n"
                          << "  --stddevs 3,8\n";
                std::exit(0);
            }
        }

        return config;
    }

    template <typename State, typename T>
    void run_uniform(const bench::RuntimeConfig& config, bench::ReportSink& sink,
                     int size_log,
                     const std::string& state_name,
                     const std::string& type_name)
    {
        const auto size = Data64{1} << size_log;
        RNG<Mode::CUDA, State> gen(bench::random_seed());
        bench::DeviceBuffer<T> results(size);
        bench::CudaStream stream;

        bench::Measurement measurement;
        measurement.backend = "curand";
        measurement.distribution = "uniform";
        measurement.data_type = type_name;
        measurement.variant = state_name;
        measurement.size_log = size_log;
        measurement.elements = results.count();
        measurement.bytes = results.bytes();

        auto result = bench::measure(
            config, measurement, stream.get(),
            [&] { gen.uniform_random_number(results.get(), size, stream.get()); });
        bench::print_measurement(sink, result);
    }

    template <typename State, typename T>
    void run_normal(const bench::RuntimeConfig& config, bench::ReportSink& sink,
                    int size_log,
                    const std::string& state_name, int stddev,
                    const std::string& type_name)
    {
        const auto size = Data64{1} << size_log;
        RNG<Mode::CUDA, State> gen(bench::random_seed());
        bench::DeviceBuffer<T> results(size);
        bench::CudaStream stream;

        bench::Measurement measurement;
        measurement.backend = "curand";
        measurement.distribution = "normal";
        measurement.data_type = type_name;
        measurement.variant = state_name;
        measurement.size_log = size_log;
        measurement.stddev = stddev;
        measurement.elements = results.count();
        measurement.bytes = results.bytes();

        auto result = bench::measure(
            config, measurement, stream.get(),
            [&]
            {
                gen.normal_random_number(static_cast<T>(stddev), results.get(), size,
                                         stream.get());
            });
        bench::print_measurement(sink, result);
    }

    template <typename State, typename T>
    void run_ternary(const bench::RuntimeConfig& config, bench::ReportSink& sink,
                     int size_log,
                     const std::string& state_name,
                     const std::string& type_name)
    {
        const auto size = Data64{1} << size_log;
        RNG<Mode::CUDA, State> gen(bench::random_seed());
        bench::DeviceBuffer<T> results(size);
        bench::CudaStream stream;

        bench::Measurement measurement;
        measurement.backend = "curand";
        measurement.distribution = "ternary";
        measurement.data_type = type_name;
        measurement.variant = state_name;
        measurement.size_log = size_log;
        measurement.elements = results.count();
        measurement.bytes = results.bytes();

        auto result = bench::measure(
            config, measurement, stream.get(),
            [&] { gen.ternary_random_number(results.get(), size, stream.get()); });
        bench::print_measurement(sink, result);
    }

    template <typename State>
    void run_for_state(const bench::RuntimeConfig& config,
                       bench::ReportSink& sink,
                       const std::string& state_name)
    {
        for (const auto size_log : config.size_logs)
        {
            if (bench::contains(config.distributions, "uniform"))
            {
                if (bench::contains(config.data_types, "u32"))
                    run_uniform<State, Data32>(config, sink, size_log, state_name, "u32");
                if (bench::contains(config.data_types, "u64"))
                    run_uniform<State, Data64>(config, sink, size_log, state_name, "u64");
            }

            if (bench::contains(config.distributions, "normal"))
            {
                for (const auto stddev : config.stddevs)
                {
                    if (bench::contains(config.data_types, "f32"))
                        run_normal<State, f32>(config, sink, size_log, state_name,
                                               stddev, "f32");
                    if (bench::contains(config.data_types, "f64"))
                        run_normal<State, f64>(config, sink, size_log, state_name,
                                               stddev, "f64");
                }
            }

            if (bench::contains(config.distributions, "ternary"))
            {
                if (bench::contains(config.data_types, "u32"))
                    run_ternary<State, Data32>(config, sink, size_log, state_name, "u32");
                if (bench::contains(config.data_types, "u64"))
                    run_ternary<State, Data64>(config, sink, size_log, state_name, "u64");
            }
        }
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
        bench::ReportSink sink(config, "cuda_benchmark.csv");
        bench::print_header(sink);

        if (bench::contains(config.curand_states, "xorwow"))
            run_for_state<curandStateXORWOW>(config, sink, "xorwow");
        if (bench::contains(config.curand_states, "mrg32k3a"))
            run_for_state<curandStateMRG32k3a>(config, sink, "mrg32k3a");
        if (bench::contains(config.curand_states, "philox"))
            run_for_state<curandStatePhilox4_32_10>(config, sink, "philox");
    }
    catch (const std::exception& error)
    {
        std::cerr << "cuda_benchmark error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
