// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RNGONGPU_TEST_UTIL_H
#define RNGONGPU_TEST_UTIL_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include "aes_rng.cuh"
#include "cuda_rng.cuh"
#include "aes.cuh"
#include "base_rng.cuh"

#ifndef PROJECT_DIR
#define PROJECT_DIR "."
#endif

namespace rngongputestcase
{

    std::string get_full_file_path(const std::string& filename)
    {
        return std::string(PROJECT_DIR) + "/test/tests_input/" + filename;
    }

    enum class Step
    {
        None,
        Instantiate,
        Reseed,
        GenerateFirst,
        GenerateSecond
    };

    struct FeatureInfo
    {
        std::string algorithm;
        bool prediction_resistance = false;
        int entropy_input_len = 0;
        int nonce_len = 0;
        int personalization_string_len = 0;
        int additional_input_len = 0;
        int returned_bits_len = 0;
    };

    struct TestVector
    {
        int count = -1;
        // INSTANTIATE:
        std::string entropy_input;
        std::string nonce;
        std::string personalization_string;
        std::string key_instantiate;
        std::string v_instantiate;

        // RESEED:
        std::string entropy_input_reseed;
        std::string additional_input_reseed;
        std::string key_reseed;
        std::string v_reseed;

        // GENERATE (FIRST CALL):
        std::string additional_input_generate_first;
        std::string entropy_inputPR_generate_first;
        std::string key_generate_first;
        std::string v_generate_first;

        // GENERATE (SECOND CALL):
        std::string additional_input_generate_second;
        std::string entropy_inputPR_generate_second;
        std::string returned_bits;
        std::string key_generate_second;
        std::string v_generate_second;
    };

    struct FeatureBlock
    {
        FeatureInfo feature_info;
        std::vector<TestVector> test_vectors;
    };

    std::string trim(const std::string& s)
    {
        std::string result = s;
        result.erase(result.begin(),
                     std::find_if(result.begin(), result.end(),
                                  [](int ch) { return !std::isspace(ch); }));
        result.erase(std::find_if(result.rbegin(), result.rend(),
                                  [](int ch) { return !std::isspace(ch); })
                         .base(),
                     result.end());
        return result;
    }

    std::string extract_value(const std::string& line)
    {
        size_t start = (line.front() == '[') ? 1 : 0;
        size_t end = (line.back() == ']') ? line.size() - 1 : line.size();
        std::string inner = line.substr(start, end - start);

        size_t pos = inner.find('=');
        if (pos != std::string::npos)
        {
            std::string value = inner.substr(pos + 1);
            value.erase(value.begin(),
                        std::find_if(value.begin(), value.end(),
                                     [](int ch) { return !std::isspace(ch); }));
            value.erase(std::find_if(value.rbegin(), value.rend(),
                                     [](int ch) { return !std::isspace(ch); })
                            .base(),
                        value.end());
            if (!value.empty() && value.back() == ']')
            {
                value.pop_back();
                value = trim(value);
            }
            return value;
        }
        return "";
    }

    std::vector<FeatureBlock> load_test_cases(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "File could not be opened: " << filename << std::endl;
            return {};
        }

        std::vector<FeatureBlock> feature_blocks;
        FeatureBlock current_block;
        TestVector current_test;
        std::string line;
        bool reading_features = false;
        bool inside_test_vectors = false;
        Step current_step = Step::None;

        while (std::getline(file, line))
        {
            if (line.empty())
                continue;

            if (line.find("[AES-") != std::string::npos)
            {
                if (inside_test_vectors || reading_features)
                {
                    if (current_test.count != -1)
                    {
                        current_block.test_vectors.push_back(current_test);
                        current_test = TestVector{};
                    }
                    feature_blocks.push_back(current_block);
                    current_block = FeatureBlock{};
                }
                reading_features = true;
                inside_test_vectors = false;
                current_step = Step::None;
                current_block.feature_info.algorithm = trim(line);
                continue;
            }
            else if (line.find("[PredictionResistance = ") != std::string::npos)
            {
                current_block.feature_info.prediction_resistance =
                    (extract_value(line) == "True");
            }
            else if (line.find("[EntropyInputLen = ") != std::string::npos)
            {
                current_block.feature_info.entropy_input_len =
                    std::stoi(extract_value(line));
            }
            else if (line.find("[NonceLen = ") != std::string::npos)
            {
                current_block.feature_info.nonce_len =
                    std::stoi(extract_value(line));
            }
            else if (line.find("[PersonalizationStringLen = ") !=
                     std::string::npos)
            {
                current_block.feature_info.personalization_string_len =
                    std::stoi(extract_value(line));
            }
            else if (line.find("[AdditionalInputLen = ") != std::string::npos)
            {
                current_block.feature_info.additional_input_len =
                    std::stoi(extract_value(line));
            }
            else if (line.find("[ReturnedBitsLen = ") != std::string::npos)
            {
                current_block.feature_info.returned_bits_len =
                    std::stoi(extract_value(line));
            }

            if (line.find("COUNT") != std::string::npos)
            {
                reading_features = false;
                if (current_test.count != -1)
                {
                    current_block.test_vectors.push_back(current_test);
                    current_test = TestVector{};
                }
                size_t pos = line.find('=');
                current_test.count = std::stoi(trim(line.substr(pos + 1)));
                inside_test_vectors = true;
                current_step = Step::None;
                continue;
            }

            size_t pos = line.find('=');
            if (pos != std::string::npos)
            {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                if (key == "EntropyInput")
                {
                    current_test.entropy_input = value;
                }
                else if (key == "Nonce")
                {
                    current_test.nonce = value;
                }
                else if (key == "PersonalizationString")
                {
                    current_test.personalization_string = value;
                }
                else if (key == "EntropyInputReseed")
                {
                    current_test.entropy_input_reseed = value;
                }
                else if (key == "AdditionalInputReseed")
                {
                    current_test.additional_input_reseed = value;
                }
                else if (key == "ReturnedBits")
                {
                    current_test.returned_bits = value;
                }
                else if (key == "AdditionalInput")
                {
                    if (current_step == Step::None)
                        current_test.additional_input_generate_first = value;
                    else if (current_step == Step::Instantiate)
                        current_test.additional_input_generate_first = value;
                    else if (current_step == Step::GenerateFirst)
                        current_test.additional_input_generate_second = value;
                }
                else if (key == "EntropyInputPR")
                {
                    if (current_step == Step::Instantiate)
                        current_test.entropy_inputPR_generate_first = value;
                    else if (current_step == Step::GenerateFirst)
                        current_test.entropy_inputPR_generate_second = value;
                }

                continue;
            }

            if (line.find("** INSTANTIATE:") != std::string::npos)
            {
                current_step = Step::Instantiate;
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.key_instantiate = trim(line.substr(pos + 1));
                }
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.v_instantiate = trim(line.substr(pos + 1));
                }

                while (file.good())
                {
                    std::streampos curPos = file.tellg();
                    std::string peekLine;
                    std::getline(file, peekLine);
                    peekLine = trim(peekLine);
                    if (peekLine.find("AdditionalInput") == 0)
                    {
                        current_test.additional_input_generate_first =
                            trim(peekLine.substr(peekLine.find('=') + 1));
                    }
                    else if (peekLine.find("EntropyInputPR") == 0)
                    {
                        current_test.entropy_inputPR_generate_first =
                            trim(peekLine.substr(peekLine.find('=') + 1));
                    }
                    else
                    {
                        file.seekg(curPos);
                        break;
                    }
                }
                continue;
            }
            else if (line.find("** RESEED:") != std::string::npos)
            {
                current_step = Step::Reseed;
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.key_reseed = trim(line.substr(pos + 1));
                }
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.v_reseed = trim(line.substr(pos + 1));
                }
                {
                    std::streampos curPos = file.tellg();
                    std::string peekLine;
                    while (std::getline(file, peekLine))
                    {
                        peekLine = trim(peekLine);
                        if (peekLine.empty())
                            continue;
                        if (peekLine.find("AdditionalInput") == 0)
                        {
                            current_test.additional_input_generate_first =
                                trim(peekLine.substr(peekLine.find('=') + 1));
                            break;
                        }
                        else
                        {
                            file.seekg(curPos);
                            break;
                        }
                        curPos = file.tellg();
                    }
                }
                continue;
            }
            else if (line.find("** GENERATE (FIRST CALL):") !=
                     std::string::npos)
            {
                current_step = Step::GenerateFirst;
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.key_generate_first =
                        trim(line.substr(pos + 1));
                }
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.v_generate_first = trim(line.substr(pos + 1));
                }
                while (file.good())
                {
                    std::streampos curPos = file.tellg();
                    std::string peekLine;
                    std::getline(file, peekLine);
                    peekLine = trim(peekLine);
                    if (peekLine.find("AdditionalInput") == 0)
                    {
                        current_test.additional_input_generate_second =
                            trim(peekLine.substr(peekLine.find('=') + 1));
                    }
                    else if (peekLine.find("EntropyInputPR") == 0)
                    {
                        current_test.entropy_inputPR_generate_second =
                            trim(peekLine.substr(peekLine.find('=') + 1));
                    }
                    else
                    {
                        file.seekg(curPos);
                        break;
                    }
                }
                continue;
            }
            else if (line.find("** GENERATE (SECOND CALL):") !=
                     std::string::npos)
            {
                current_step = Step::GenerateSecond;
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.key_generate_second =
                        trim(line.substr(pos + 1));
                }
                if (std::getline(file, line))
                {
                    pos = line.find('=');
                    current_test.v_generate_second = trim(line.substr(pos + 1));
                }
                continue;
            }
        }

        if (current_test.count != -1)
        {
            current_block.test_vectors.push_back(current_test);
        }
        if (!current_block.feature_info.algorithm.empty())
        {
            feature_blocks.push_back(current_block);
        }

        return feature_blocks;
    }

    void print_hex(const std::vector<unsigned char>& data)
    {
        for (unsigned char byte : data)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(byte);
        }
        std::cout << std::dec << std::endl;
    }

    std::vector<unsigned char> hex_string_to_bytes(const std::string hex)
    {
        std::vector<unsigned char> bytes;
        for (size_t i = 0; i < hex.length(); i += 2)
        {
            std::string byteString = hex.substr(i, 2); // İkili karakter al
            unsigned char byte =
                static_cast<unsigned char>(std::stoul(byteString, nullptr, 16));
            bytes.push_back(byte);
        }
        return bytes;
    }

    rngongpu::SecurityLevel get_security_level(int sec_level)
    {
        switch (sec_level)
        {
            case 128:
                return rngongpu::SecurityLevel::AES128;
            case 192:
                return rngongpu::SecurityLevel::AES192;
            case 256:
                return rngongpu::SecurityLevel::AES256;
            default:
                throw std::invalid_argument("Invalid security level!");
        }
    }

} // namespace rngongputestcase

#endif // RNGONGPU_TEST_UTIL_H