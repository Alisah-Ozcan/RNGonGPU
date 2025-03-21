// Copyright 2025 Alişah Özcan
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

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}