// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include "aes_rng.cuh"

bool determinePRFromString(std::string str) {
    return str == "True";
}

std::string boolToStr(bool b) {
    if (b) return "True"; 
    else return "False";
}

void convertHexStringToBytes(const std::string& str, std::vector<unsigned char>& res) 
{
    for (size_t i = 0; i < str.length(); i += 2) {
        std::string byteString = str.substr(i, 2);
        unsigned char byte = static_cast<unsigned char>(std::stoi(byteString, nullptr, 16));
        res[i/2] = byte;
    }
}

void printVec(std::vector<unsigned char> vec, std::ostream& out = std::cout) {
    for (unsigned char byte : vec)
        {
            out << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(byte);
        }
        out << std::dec << std::endl;
}

std::string extractValueStr(std::string str) {
    size_t pos = str.find("=") + 2;
    size_t last_pos = str.find("]");
    if (last_pos == std::string::npos) last_pos = str.length() - 1;
    return str.substr(pos, last_pos - pos);
}

void readParamsPr(
    std::ifstream& file, std::ofstream& out, std::string& line, 
    size_t& entropyInputLength, size_t& nonceLength, 
    size_t& personalizationStringLen, 
    size_t& additionalInputLen, size_t& returnedBitsLen,
    rngongpu::SecurityLevel& securityLevel, 
    bool& isPredictionResistanceEnabled) 
{
    std::string mode = line.substr(1, line.length() - 3);
    if (mode == "AES-128 use df") 
    {     
        securityLevel = rngongpu::SecurityLevel::AES128;        
    } else if (mode == "AES-192 use df")
    {
        securityLevel = rngongpu::SecurityLevel::AES192;
    } else if (mode == "AES-256 use df")
    {
        securityLevel = rngongpu::SecurityLevel::AES256;
    } else 
    {
        std::cout << "This mode is not supported. Skipping.. . .. \n";
    }
    
    std::getline(file, line);
    isPredictionResistanceEnabled = determinePRFromString(extractValueStr(line));

    std::getline(file, line);
    entropyInputLength = stoi(extractValueStr(line)) / 8;

    std::getline(file, line);
    nonceLength = stoi(extractValueStr(line)) / 8;

    std::getline(file, line);
    personalizationStringLen = stoi(extractValueStr(line)) / 8;

    std::getline(file, line);
    additionalInputLen = stoi(extractValueStr(line)) / 8;

    std::getline(file, line);
    returnedBitsLen = stoi(extractValueStr(line)) / 8;   
    
    out << "[" << mode << "]\n" << "[PredictionResistance = " << boolToStr(isPredictionResistanceEnabled) << 
        "]\n" << "[EntropyInputLen = " << entropyInputLength * 8 << "]\n" << "[NonceLen = " << nonceLength * 8 << 
        "]\n" << "[PersonalizationStringLen = " << personalizationStringLen * 8 << "]\n" << "[AdditionalInputLen = " <<
        additionalInputLen * 8 << "]\n" << "[ReturnedBitsLen = " << returnedBitsLen * 8<< "]\n\n";

    //skip an empty line
    std::getline(file, line);
}

void runPrTests(
    std::ifstream& file, std::ofstream& out, int entropyInputLength, 
    int nonceLength, int personalizationStringLen, 
    bool isPredictionResistanceEnabled, 
    int additionalInputLen, int returnedBitsLen,
    rngongpu::SecurityLevel securityLevel) 
{
    std::vector<unsigned char> entropyInput(entropyInputLength, 0);
    std::vector<unsigned char> entropyInputPR1(entropyInputLength, 0);
    std::vector<unsigned char> entropyInputPR2(entropyInputLength, 0);
    std::vector<unsigned char> nonce(nonceLength, 0);
    std::vector<unsigned char> personalizationString(personalizationStringLen, 0);
    std::vector<unsigned char> additionalInput1(additionalInputLen, 0);
    std::vector<unsigned char> additionalInput2(additionalInputLen, 0);
    std::vector<unsigned char> returnedBits(returnedBitsLen, 0);

    std::string line;
    int numTests = 15;
    while (numTests > 0) {
        std::getline(file, line);
        int numTrial = stoi(extractValueStr(line));
        out << "COUNT = " << numTrial << std::endl;
        
        std::getline(file, line);
        if (entropyInputLength > 0) {
            std::string entropyInputStr = extractValueStr(line);
            convertHexStringToBytes(entropyInputStr, entropyInput);
        }
        
        std::getline(file, line);
        if (nonceLength > 0) {
            std::string nonceStr = extractValueStr(line);
            convertHexStringToBytes(nonceStr, nonce);
        }
        
        std::getline(file, line);
        if (personalizationStringLen > 0) {
            std::string personalizationStringStr = extractValueStr(line);
            convertHexStringToBytes(personalizationStringStr, personalizationString);
        }
        
        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInput1);
        }

        std::getline(file, line);
        if (entropyInputLength > 0) {
            std::string entropyInputStr = extractValueStr(line);
            convertHexStringToBytes(entropyInputStr, entropyInputPR1);
        }
        
        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInput2);
        }
        
        std::getline(file, line);
        if (entropyInputLength > 0) {
            std::string entropyInputStr = extractValueStr(line);
            convertHexStringToBytes(entropyInputStr, entropyInputPR2);
        }
        
        std::getline(file, line);
        if (returnedBitsLen > 0) {
            std::string returnedBitsStr = extractValueStr(line);
            convertHexStringToBytes(returnedBitsStr, returnedBits);
        }

        rngongpu::RNG<rngongpu::Mode::AES> drbg(entropyInput, nonce, personalizationString, securityLevel, isPredictionResistanceEnabled);
        const int N = returnedBitsLen / 8;
        std::vector<unsigned char> res1(returnedBitsLen, 0), res2(returnedBitsLen, 0);
        Data64* d_res1, *d_res2;
        cudaMalloc(&d_res1, returnedBitsLen);
        cudaMalloc(&d_res2, returnedBitsLen);

        out << "EntropyInput = ";
        printVec(entropyInput, out);
        out << "Nonce = ";
        printVec(nonce, out);
        out << "PersonalizationString = ";
        printVec(personalizationString, out);
        out << "** INSTANTIATE:\n"; 
        drbg.print_params(out);

        out <<"AdditionalInput = ";
        printVec(additionalInput1, out);
        out << "EntropyInputPR = ";
        printVec(entropyInputPR1, out);
        drbg.uniform_random_number(d_res1, N, entropyInputPR1, additionalInput1);
        cudaMemcpy(res1.data(), d_res1, N, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        out << "** GENERATE (FIRST CALL):\n"; 
        drbg.print_params(out);

        out <<"AdditionalInput = ";
        printVec(additionalInput2, out);
        out << "EntropyInputPR = ";
        printVec(entropyInputPR2, out);

        drbg.uniform_random_number(d_res2, N, entropyInputPR2, additionalInput2);
        cudaMemcpy(res2.data(), d_res2, returnedBitsLen, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        out << "ReturnedBits = ";
        printVec(res2, out);
        out << "** GENERATE (SECOND CALL):\n"; 
        drbg.print_params(out);

        std::getline(file, line);
        out << "\n";

        numTests--;
    }
}

void runNoPrTests(
    std::ifstream& file, std::ofstream& out, int entropyInputLength, 
    int nonceLength, int personalizationStringLen, 
    bool isPredictionResistanceEnabled, 
    int additionalInputLen, int returnedBitsLen,
    rngongpu::SecurityLevel securityLevel) 
{
    std::vector<unsigned char> entropyInput(entropyInputLength, 0);
    std::vector<unsigned char> nonce(nonceLength, 0);
    std::vector<unsigned char> personalizationString(personalizationStringLen, 0);
    std::vector<unsigned char> entropyInputReseed(entropyInputLength, 0);
    std::vector<unsigned char> additionalInputReseed(additionalInputLen, 0);
    std::vector<unsigned char> additionalInput1(additionalInputLen, 0);
    std::vector<unsigned char> additionalInput2(additionalInputLen, 0);
    std::vector<unsigned char> returnedBits(returnedBitsLen, 0);
    std::vector<unsigned char> entropyReseed;

    std::string line;
    int numTests = 15;
    while (numTests > 0) {
        std::getline(file, line);
        int numTrial = stoi(extractValueStr(line));
        out << "COUNT = " << numTrial << std::endl;
        
        std::getline(file, line);
        if (entropyInputLength > 0) {
            std::string entropyInputStr = extractValueStr(line);
            convertHexStringToBytes(entropyInputStr, entropyInput);
        }
        
        std::getline(file, line);
        if (nonceLength > 0) {
            std::string nonceStr = extractValueStr(line);
            convertHexStringToBytes(nonceStr, nonce);
        }
        
        std::getline(file, line);
        if (personalizationStringLen > 0) {
            std::string personalizationStringStr = extractValueStr(line);
            convertHexStringToBytes(personalizationStringStr, personalizationString);
        }

        std::getline(file, line);
        if (entropyInputLength > 0) {
            std::string entropyInputStr = extractValueStr(line);
            convertHexStringToBytes(entropyInputStr, entropyInputReseed);
        }

        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInputReseed);
        }

        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInput1);
        }
        
        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInput2);
        }
        
        std::getline(file, line);
        if (returnedBitsLen > 0) {
            std::string returnedBitsStr = extractValueStr(line);
            convertHexStringToBytes(returnedBitsStr, returnedBits);
        }

        rngongpu::RNG<rngongpu::Mode::AES> drbg(entropyInput, nonce, personalizationString, securityLevel, isPredictionResistanceEnabled);
        const int N = returnedBitsLen / 8;
        std::vector<unsigned char> res1(returnedBitsLen, 0), res2(returnedBitsLen, 0);
        Data64* d_res1, *d_res2;
        cudaMalloc(&d_res1, returnedBitsLen);
        cudaMalloc(&d_res2, returnedBitsLen);

        out << "EntropyInput = ";
        printVec(entropyInput, out);
        out << "Nonce = ";
        printVec(nonce, out);
        out << "PersonalizationString = ";
        printVec(personalizationString, out);
        out << "** INSTANTIATE:\n"; 
        drbg.print_params(out);

        out << "EntropyInputReseed = ";
        printVec(entropyInputReseed, out);
        out <<"AdditionalInputReseed = ";
        printVec(additionalInputReseed, out);
        drbg.reseed();
        out << "** RESEED:\n"; 
        drbg.print_params(out);

        out <<"AdditionalInput = ";
        printVec(additionalInput1, out);
        drbg.uniform_random_number(d_res1, N, entropyReseed, additionalInput1);
        cudaMemcpy(res1.data(), d_res1, returnedBitsLen, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        out << "ReturnedBits = ";
        printVec(res1, out);
        out << "** GENERATE (SECOND CALL):\n"; 
        drbg.print_params(out);

        out <<"AdditionalInput = ";
        printVec(additionalInput2, out);
        drbg.uniform_random_number(d_res2, N, entropyReseed, additionalInput2);
        cudaMemcpy(res2.data(), d_res2, returnedBitsLen, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        out << "ReturnedBits = ";
        printVec(res2, out);
        out << "** GENERATE (SECOND CALL):\n"; 
        drbg.print_params(out);

        std::getline(file, line);
        out << "\n";

        numTests--;
    }
}

void runNoReseedTests(
    std::ifstream& file, std::ofstream& out, int entropyInputLength, 
    int nonceLength, int personalizationStringLen, 
    bool isPredictionResistanceEnabled, 
    int additionalInputLen, int returnedBitsLen,
    rngongpu::SecurityLevel securityLevel) 
{
    std::vector<unsigned char> entropyInput(entropyInputLength, 0);
    std::vector<unsigned char> nonce(nonceLength, 0);
    std::vector<unsigned char> personalizationString(personalizationStringLen, 0);
    std::vector<unsigned char> additionalInput1(additionalInputLen, 0);
    std::vector<unsigned char> additionalInput2(additionalInputLen, 0);
    std::vector<unsigned char> returnedBits(returnedBitsLen, 0);
    std::vector<unsigned char> entropyReseed;

    std::string line;
    int numTests = 15;
    while (numTests > 0) {
        std::getline(file, line);
        int numTrial = stoi(extractValueStr(line));
        out << "COUNT = " << numTrial << std::endl;
        
        std::getline(file, line);
        if (entropyInputLength > 0) {
            std::string entropyInputStr = extractValueStr(line);
            convertHexStringToBytes(entropyInputStr, entropyInput);
        }
        
        std::getline(file, line);
        if (nonceLength > 0) {
            std::string nonceStr = extractValueStr(line);
            convertHexStringToBytes(nonceStr, nonce);
        }
        
        std::getline(file, line);
        if (personalizationStringLen > 0) {
            std::string personalizationStringStr = extractValueStr(line);
            convertHexStringToBytes(personalizationStringStr, personalizationString);
        }

        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInput1);
        }
        
        std::getline(file, line);
        if (additionalInputLen > 0) {
            std::string additionalInputStr = extractValueStr(line);
            convertHexStringToBytes(additionalInputStr, additionalInput2);
        }
        
        std::getline(file, line);
        if (returnedBitsLen > 0) {
            std::string returnedBitsStr = extractValueStr(line);
            convertHexStringToBytes(returnedBitsStr, returnedBits);
        }

        rngongpu::RNG<rngongpu::Mode::AES> drbg(entropyInput, nonce, personalizationString, securityLevel, isPredictionResistanceEnabled);
        const int N = returnedBitsLen / 8;
        std::vector<unsigned char> res1(returnedBitsLen, 0), res2(returnedBitsLen, 0);
        Data64* d_res1, *d_res2;
        cudaMalloc(&d_res1, returnedBitsLen);
        cudaMalloc(&d_res2, returnedBitsLen);

        out << "EntropyInput = ";
        printVec(entropyInput, out);
        out << "Nonce = ";
        printVec(nonce, out);
        out << "PersonalizationString = ";
        printVec(personalizationString, out);
        out << "** INSTANTIATE:\n"; 
        drbg.print_params(out);

        out <<"AdditionalInput = ";
        printVec(additionalInput1, out);
        drbg.uniform_random_number(d_res1, N, entropyReseed, additionalInput1);
        cudaMemcpy(res1.data(), d_res1, returnedBitsLen, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        out << "ReturnedBits = ";
        printVec(res1, out);
        out << "** GENERATE (SECOND CALL):\n"; 
        drbg.print_params(out);

        out <<"AdditionalInput = ";
        printVec(additionalInput2, out);
        drbg.uniform_random_number(d_res2, N, entropyReseed, additionalInput2);
        cudaMemcpy(res2.data(), d_res2, returnedBitsLen, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        out << "ReturnedBits = ";
        printVec(res2, out);
        out << "** GENERATE (SECOND CALL):\n"; 
        drbg.print_params(out);

        std::getline(file, line);
        out << "\n";

        numTests--;
    }
}

int main() {
    std::filesystem::path prPath = "./tests_input/drbgvectors_pr_true/CTR_DRBG.rsp";
    std::filesystem::path noPrPath = "./tests_input/drbgvectors_pr_false/CTR_DRBG.rsp";
    std::filesystem::path noReseedPath = "./tests_input/drbgvectors_no_reseed/CTR_DRBG.rsp";
    
    std::filesystem::path outPrPath = "./out/pr_true.txt";
    std::filesystem::path outNoPrPath = "./out/pr_false.txt";
    std::filesystem::path outNoReseedPath = "./out/no_reseed.txt";

    std::ifstream file(prPath);
    if (!file) {
        std::cerr << "Error opening file: " << prPath << std::endl;
        return 1;
    }
    std::ofstream outPr(outPrPath);
    if (!outPr) {
        std::cerr << "Error opening file " << outPrPath << std::endl;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '[')
        {
            size_t entropyInputLength;
            size_t nonceLength;
            size_t personalizationStringLen;
            size_t additionalInputLen;
            size_t returnedBitsLen;
            rngongpu::SecurityLevel securityLevel;
            bool isPredictionResistanceEnabled;
            readParamsPr(file, outPr, line, entropyInputLength, nonceLength, personalizationStringLen, additionalInputLen, returnedBitsLen, securityLevel, isPredictionResistanceEnabled);
            runPrTests(file, outPr, entropyInputLength, nonceLength, personalizationStringLen, isPredictionResistanceEnabled, additionalInputLen, returnedBitsLen, securityLevel);
        }
    }

    file.close();
    outPr.close();

    std::ifstream file2(noPrPath);
    if (!file2) {
        std::cerr << "Error opening file: " << noPrPath << std::endl;
        return 1;
    }
    std::ofstream outNoPr(outNoPrPath);
    if (!outNoPr) {
        std::cerr << "Error opening file " << outNoPrPath << std::endl;
    }
    while (std::getline(file2, line)) {
        if (line[0] == '[')
        {
            size_t entropyInputLength;
            size_t nonceLength;
            size_t personalizationStringLen;
            size_t additionalInputLen;
            size_t returnedBitsLen;
            rngongpu::SecurityLevel securityLevel;
            bool isPredictionResistanceEnabled;
            readParamsPr(file2, outNoPr, line, entropyInputLength, nonceLength, personalizationStringLen, additionalInputLen, returnedBitsLen, securityLevel, isPredictionResistanceEnabled);
            runNoPrTests(file2, outNoPr, entropyInputLength, nonceLength, personalizationStringLen, isPredictionResistanceEnabled, additionalInputLen, returnedBitsLen, securityLevel);
        }
    }

    file2.close();
    outNoPr.close();

    std::ifstream file3(noReseedPath);
    if (!file3) {
        std::cerr << "Error opening file: " << noReseedPath << std::endl;
        return 1;
    }
    std::ofstream outNoReseedPr(outNoReseedPath);
    if (!outNoReseedPr) {
        std::cerr << "Error opening file " << outNoReseedPath << std::endl;
    }
    while (std::getline(file3, line)) {
        if (line[0] == '[')
        {
            size_t entropyInputLength;
            size_t nonceLength;
            size_t personalizationStringLen;
            size_t additionalInputLen;
            size_t returnedBitsLen;
            rngongpu::SecurityLevel securityLevel;
            bool isPredictionResistanceEnabled;
            readParamsPr(file3, outNoReseedPr, line, entropyInputLength, nonceLength, personalizationStringLen, additionalInputLen, returnedBitsLen, securityLevel, isPredictionResistanceEnabled);
            runNoReseedTests(file3, outNoReseedPr, entropyInputLength, nonceLength, personalizationStringLen, isPredictionResistanceEnabled, additionalInputLen, returnedBitsLen, securityLevel);
        }
    }

    file3.close();
    outNoPr.close();



    return 0;
}

