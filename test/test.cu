// Copyright 2025-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "test_util.h"

using namespace rngongputestcase;

class RNGONGPU_TEST_DRBGVECTORS_PR_FALSE
    : public ::testing::TestWithParam<std::string>
{
  protected:
    std::vector<FeatureBlock> cases;
    std::string case_file_location;

    Data64 size = 16;
    Data32* d_results;

    void SetUp() override
    {
        case_file_location = get_full_file_path(GetParam());
        cases = load_test_cases(case_file_location);

        cudaMalloc(&d_results, size * sizeof(Data32)); // 512 bit
    }

    void TearDown() override { cudaFree(d_results); }
};

TEST_P(RNGONGPU_TEST_DRBGVECTORS_PR_FALSE, DRBGVECTORS_PR_FALSE)
{
    for (const auto& block : cases)
    {
        for (const auto& test : block.test_vectors)
        {
            rngongpu::SecurityLevel sec_level =
                get_security_level(block.feature_info.entropy_input_len);
            std::vector<unsigned char> entropy =
                hex_string_to_bytes(test.entropy_input);
            std::vector<unsigned char> nonce = hex_string_to_bytes(test.nonce);
            std::vector<unsigned char> personalization =
                hex_string_to_bytes(test.personalization_string);

            rngongpu::RNG<rngongpu::Mode::AES> drbg(
                entropy, nonce, personalization, sec_level,
                block.feature_info.prediction_resistance);
            // std::cout << "Instantiate: " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_instantiate));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_instantiate));

            std::vector<unsigned char> entropy_input_reseed =
                hex_string_to_bytes(test.entropy_input_reseed);
            std::vector<unsigned char> additional_input_reseed =
                hex_string_to_bytes(test.additional_input_reseed);
            drbg.reseed(entropy_input_reseed, additional_input_reseed);
            // std::cout << "Reseed: " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(), hex_string_to_bytes(test.key_reseed));
            EXPECT_EQ(drbg.get_nonce(), hex_string_to_bytes(test.v_reseed));

            std::vector<unsigned char> additional_input1 =
                hex_string_to_bytes(test.additional_input_generate_first);
            std::vector<unsigned char> entropy_PR1 =
                hex_string_to_bytes(test.entropy_inputPR_generate_first);
            drbg.uniform_random_number(d_results, size, entropy_PR1,
                                       additional_input1);
            // std::cout << "Random Bytes (First Call): " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_generate_first));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_generate_first));

            std::vector<unsigned char> additional_input2 =
                hex_string_to_bytes(test.additional_input_generate_second);
            std::vector<unsigned char> entropy_PR2 =
                hex_string_to_bytes(test.entropy_inputPR_generate_second);
            drbg.uniform_random_number(d_results, size, entropy_PR2,
                                       additional_input2);
            // std::cout << "Random Bytes (Second Call): " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_generate_second));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_generate_second));

            std::vector<unsigned char> vec_byte(size * sizeof(Data32));
            cudaMemcpy(vec_byte.data(), d_results, size * sizeof(Data32),
                       cudaMemcpyDeviceToHost);

            EXPECT_EQ(vec_byte, hex_string_to_bytes(test.returned_bits));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    RNGONGPU_TEST1, RNGONGPU_TEST_DRBGVECTORS_PR_FALSE,
    ::testing::Values("drbgvectors_pr_false/CTR_DRBG.txt"));

class RNGONGPU_TEST_DRBGVECTORS_PR_TRUE
    : public ::testing::TestWithParam<std::string>
{
  protected:
    std::vector<FeatureBlock> cases;
    std::string case_file_location;

    Data64 size = 16;
    Data32* d_results;

    void SetUp() override
    {
        case_file_location = get_full_file_path(GetParam());
        cases = load_test_cases(case_file_location);

        cudaMalloc(&d_results, size * sizeof(Data32)); // 512 bit
    }

    void TearDown() override { cudaFree(d_results); }
};

TEST_P(RNGONGPU_TEST_DRBGVECTORS_PR_TRUE, DRBGVECTORS_PR_TRUE)
{
    for (const auto& block : cases)
    {
        for (const auto& test : block.test_vectors)
        {
            rngongpu::SecurityLevel sec_level =
                get_security_level(block.feature_info.entropy_input_len);
            std::vector<unsigned char> entropy =
                hex_string_to_bytes(test.entropy_input);
            std::vector<unsigned char> nonce = hex_string_to_bytes(test.nonce);
            std::vector<unsigned char> personalization =
                hex_string_to_bytes(test.personalization_string);

            rngongpu::RNG<rngongpu::Mode::AES> drbg(
                entropy, nonce, personalization, sec_level,
                block.feature_info.prediction_resistance);
            // std::cout << "Instantiate: " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_instantiate));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_instantiate));

            std::vector<unsigned char> additional_input1 =
                hex_string_to_bytes(test.additional_input_generate_first);
            std::vector<unsigned char> entropy_PR1 =
                hex_string_to_bytes(test.entropy_inputPR_generate_first);
            drbg.uniform_random_number(d_results, size, entropy_PR1,
                                       additional_input1);
            // std::cout << "Random Bytes (First Call): " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_generate_first));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_generate_first));

            std::vector<unsigned char> additional_input2 =
                hex_string_to_bytes(test.additional_input_generate_second);
            std::vector<unsigned char> entropy_PR2 =
                hex_string_to_bytes(test.entropy_inputPR_generate_second);
            drbg.uniform_random_number(d_results, size, entropy_PR2,
                                       additional_input2);
            // std::cout << "Random Bytes (Second Call): " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_generate_second));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_generate_second));

            std::vector<unsigned char> vec_byte(size * sizeof(Data32));
            cudaMemcpy(vec_byte.data(), d_results, size * sizeof(Data32),
                       cudaMemcpyDeviceToHost);

            EXPECT_EQ(vec_byte, hex_string_to_bytes(test.returned_bits));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(RNGONGPU_TEST2, RNGONGPU_TEST_DRBGVECTORS_PR_TRUE,
                         ::testing::Values("drbgvectors_pr_true/CTR_DRBG.txt"));

class RNGONGPU_TEST_DRBGVECTORS_NO_RESEED
    : public ::testing::TestWithParam<std::string>
{
  protected:
    std::vector<FeatureBlock> cases;
    std::string case_file_location;

    Data64 size = 16;
    Data32* d_results;

    void SetUp() override
    {
        case_file_location = get_full_file_path(GetParam());
        cases = load_test_cases(case_file_location);

        cudaMalloc(&d_results, size * sizeof(Data32)); // 512 bit
    }

    void TearDown() override { cudaFree(d_results); }
};

TEST_P(RNGONGPU_TEST_DRBGVECTORS_NO_RESEED, DRBGVECTORS_NO_RESEED)
{
    for (const auto& block : cases)
    {
        for (const auto& test : block.test_vectors)
        {
            rngongpu::SecurityLevel sec_level =
                get_security_level(block.feature_info.entropy_input_len);
            std::vector<unsigned char> entropy =
                hex_string_to_bytes(test.entropy_input);
            std::vector<unsigned char> nonce = hex_string_to_bytes(test.nonce);
            std::vector<unsigned char> personalization =
                hex_string_to_bytes(test.personalization_string);

            rngongpu::RNG<rngongpu::Mode::AES> drbg(
                entropy, nonce, personalization, sec_level,
                block.feature_info.prediction_resistance);
            // std::cout << "Instantiate: " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_instantiate));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_instantiate));

            std::vector<unsigned char> additional_input1 =
                hex_string_to_bytes(test.additional_input_generate_first);
            std::vector<unsigned char> entropy_PR1 =
                hex_string_to_bytes(test.entropy_inputPR_generate_first);
            drbg.uniform_random_number(d_results, size, entropy_PR1,
                                       additional_input1);
            // std::cout << "Random Bytes (First Call): " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_generate_first));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_generate_first));

            std::vector<unsigned char> additional_input2 =
                hex_string_to_bytes(test.additional_input_generate_second);
            std::vector<unsigned char> entropy_PR2 =
                hex_string_to_bytes(test.entropy_inputPR_generate_second);
            drbg.uniform_random_number(d_results, size, entropy_PR2,
                                       additional_input2);
            // std::cout << "Random Bytes (Second Call): " << std::endl;
            // drbg.print_params();

            EXPECT_EQ(drbg.get_key(),
                      hex_string_to_bytes(test.key_generate_second));
            EXPECT_EQ(drbg.get_nonce(),
                      hex_string_to_bytes(test.v_generate_second));

            std::vector<unsigned char> vec_byte(size * sizeof(Data32));
            cudaMemcpy(vec_byte.data(), d_results, size * sizeof(Data32),
                       cudaMemcpyDeviceToHost);

            EXPECT_EQ(vec_byte, hex_string_to_bytes(test.returned_bits));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    RNGONGPU_TEST3, RNGONGPU_TEST_DRBGVECTORS_NO_RESEED,
    ::testing::Values("drbgvectors_no_reseed/CTR_DRBG.txt"));

// The same seed gives the same numbers on every kind of stream: each request
// copies its round keys and counter on its own stream.
class RNGONGPU_TEST_STREAMS
    : public ::testing::TestWithParam<rngongpu::SecurityLevel>
{
  protected:
    std::vector<Data64> generate(cudaStream_t stream)
    {
        const Data64 size = 1 << 21;
        const int requests = 4;
        std::vector<unsigned char> entropy(32, 0x11);
        std::vector<unsigned char> nonce(16, 0x22);
        std::vector<unsigned char> personalization = {0x33};
        rngongpu::RNG<rngongpu::Mode::AES> drbg(entropy, nonce,
                                                personalization, GetParam(),
                                                false);

        Data64* d_results;
        cudaMalloc(&d_results, size * sizeof(Data64));
        std::vector<Data64> results(size * requests);
        for (int i = 0; i < requests; i++)
        {
            drbg.uniform_random_number(d_results, size, {}, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpy(results.data() + i * size, d_results,
                       size * sizeof(Data64), cudaMemcpyDeviceToHost);
        }
        cudaFree(d_results);
        return results;
    }
};

TEST_P(RNGONGPU_TEST_STREAMS, SAME_NUMBERS_ON_EVERY_STREAM)
{
    cudaStream_t blocking, non_blocking;
    cudaStreamCreate(&blocking);
    cudaStreamCreateWithFlags(&non_blocking, cudaStreamNonBlocking);

    std::vector<Data64> reference = generate(cudaStreamDefault);
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(generate(blocking), reference);
        EXPECT_EQ(generate(non_blocking), reference);
    }

    cudaStreamDestroy(blocking);
    cudaStreamDestroy(non_blocking);
}

INSTANTIATE_TEST_SUITE_P(RNGONGPU_TEST4, RNGONGPU_TEST_STREAMS,
                         ::testing::Values(rngongpu::SecurityLevel::AES128,
                                           rngongpu::SecurityLevel::AES192,
                                           rngongpu::SecurityLevel::AES256));

// Requests issued on several streams without waiting in between give the same
// numbers as the same requests issued one after another on one stream: a
// request does not see the device state of a later one, and temporary buffers
// of different streams do not overlap.
class RNGONGPU_TEST_CONCURRENT_STREAMS
    : public ::testing::TestWithParam<rngongpu::SecurityLevel>
{
  protected:
    static constexpr Data64 size = 1 << 20;
    static constexpr int requests = 16;

    std::vector<Data64> generate(const std::vector<cudaStream_t>& streams)
    {
        std::vector<unsigned char> entropy(32, 0x44);
        std::vector<unsigned char> nonce(16, 0x55);
        std::vector<unsigned char> personalization = {0x66};
        rngongpu::RNG<rngongpu::Mode::AES> drbg(entropy, nonce,
                                                personalization, GetParam(),
                                                false);

        Data64* d_results;
        cudaMalloc(&d_results, requests * size * sizeof(Data64));
        for (int i = 0; i < requests; i++)
        {
            cudaStream_t stream = streams[i % streams.size()];
            if (i % 2 == 0)
            {
                drbg.uniform_random_number(d_results + i * size, size, {},
                                           stream);
            }
            else
            {
                drbg.normal_random_number(
                    3.2, reinterpret_cast<f64*>(d_results + i * size), size,
                    {}, stream);
            }
        }
        for (cudaStream_t stream : streams)
            cudaStreamSynchronize(stream);

        std::vector<Data64> results(requests * size);
        cudaMemcpy(results.data(), d_results,
                   requests * size * sizeof(Data64), cudaMemcpyDeviceToHost);
        cudaFree(d_results);
        return results;
    }
};

TEST_P(RNGONGPU_TEST_CONCURRENT_STREAMS, SAME_NUMBERS_AS_ONE_STREAM)
{
    std::vector<Data64> reference = generate({cudaStreamDefault});

    std::vector<cudaStream_t> streams(4);
    for (cudaStream_t& stream : streams)
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(generate(streams), reference);
    }
    for (cudaStream_t stream : streams)
        cudaStreamDestroy(stream);
}

INSTANTIATE_TEST_SUITE_P(RNGONGPU_TEST5, RNGONGPU_TEST_CONCURRENT_STREAMS,
                         ::testing::Values(rngongpu::SecurityLevel::AES128,
                                           rngongpu::SecurityLevel::AES192,
                                           rngongpu::SecurityLevel::AES256));

// A large request is the AES-CTR key stream E(K, V + 1), E(K, V + 2), ... of
// SP 800-90A in this order, so the numbers do not depend on the number of
// multiprocessors of the GPU or on the launch size.
class RNGONGPU_TEST_LARGE_REQUEST
    : public ::testing::TestWithParam<rngongpu::SecurityLevel>
{
  protected:
    static const EVP_CIPHER* cipher()
    {
        switch (GetParam())
        {
            case rngongpu::SecurityLevel::AES128:
                return EVP_aes_128_ctr();
            case rngongpu::SecurityLevel::AES192:
                return EVP_aes_192_ctr();
            default:
                return EVP_aes_256_ctr();
        }
    }
};

TEST_P(RNGONGPU_TEST_LARGE_REQUEST, CTR_KEY_STREAM_ORDER)
{
    std::vector<unsigned char> entropy(32, 0x77);
    std::vector<unsigned char> nonce(16, 0x88);
    std::vector<unsigned char> personalization = {0x99};
    rngongpu::RNG<rngongpu::Mode::AES> drbg(entropy, nonce, personalization,
                                            GetParam(), false);

    for (Data64 size : {Data64(1) << 10, Data64(1) << 21, Data64(3) << 19})
    {
        std::vector<unsigned char> key = drbg.get_key();
        std::vector<unsigned char> v = drbg.get_nonce();
        for (int i = 15; i >= 0 && ++v[i] == 0; i--)
        {
        }

        Data64* d_results;
        cudaMalloc(&d_results, size * sizeof(Data64));
        drbg.uniform_random_number(d_results, size, {});
        std::vector<unsigned char> results(size * sizeof(Data64));
        cudaMemcpy(results.data(), d_results, results.size(),
                   cudaMemcpyDeviceToHost);
        cudaFree(d_results);

        std::vector<unsigned char> zeros(results.size(), 0);
        std::vector<unsigned char> expected(results.size());
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        int len = 0;
        ASSERT_EQ(EVP_EncryptInit_ex(ctx, cipher(), nullptr, key.data(),
                                     v.data()),
                  1);
        ASSERT_EQ(EVP_EncryptUpdate(ctx, expected.data(), &len, zeros.data(),
                                    static_cast<int>(zeros.size())),
                  1);
        EVP_CIPHER_CTX_free(ctx);

        EXPECT_EQ(results, expected) << "size " << size;
    }
}

INSTANTIATE_TEST_SUITE_P(RNGONGPU_TEST6, RNGONGPU_TEST_LARGE_REQUEST,
                         ::testing::Values(rngongpu::SecurityLevel::AES128,
                                           rngongpu::SecurityLevel::AES192,
                                           rngongpu::SecurityLevel::AES256));

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}