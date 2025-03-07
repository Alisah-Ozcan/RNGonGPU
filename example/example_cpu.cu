// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <iomanip>
#include <vector>
#include "aes_cpu_rng.h"
#include <openssl/rand.h>
#include <stdexcept>

using namespace rngongpu;

void print_hex(const std::vector<unsigned char>& data)
{
    for (unsigned char byte : data)
    {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(byte);
    }
    std::cout << std::dec << std::endl;
}

std::vector<unsigned char> generate_random_bytes(std::size_t n)
{
    std::vector<unsigned char> buf(n);
    if (1 != RAND_bytes(buf.data(), buf.size()))
        throw std::runtime_error("RAND_bytes failed");
    return buf;
}

void case1()
{
    // PAGE 210
    // https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/CTR_DRBG_withDF.pdf
    std::vector<unsigned char> entropy = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
        0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15,
        0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
    std::vector<unsigned char> nonce = {0x20, 0x21, 0x22, 0x23,
                                        0x24, 0x25, 0x26, 0x27};
    std::vector<unsigned char> personalization = {};

    AESCTRRNG drbg(entropy, nonce, personalization, SecurityLevel::AES128,
                   false);
    std::cout << "Instantiate: " << std::endl;
    drbg.print_params();

    std::cout << "Random Bytes (First Call): " << std::endl;
    std::vector<unsigned char> randomBytes1 =
        drbg.generate_bytes(32); // 64 // page 211
    drbg.print_params();
    print_hex(randomBytes1);

    std::cout << "Random Bytes (Second Call): " << std::endl;
    std::vector<unsigned char> randomBytes2 =
        drbg.generate_bytes(32); // 64 // page 211
    drbg.print_params();
    print_hex(randomBytes2);
}

void case2()
{
    //[AES-128 use df]
    //[PredictionResistance = False]
    //[EntropyInputLen = 128]
    //[NonceLen = 64]
    //[PersonalizationStringLen = 0]
    //[AdditionalInputLen = 0]
    //[ReturnedBitsLen = 512]
    //
    // COUNT = 0
    // EntropyInput = 0f65da13dca407999d4773c2b4a11d85
    // Nonce = 5209e5b4ed82a234
    // PersonalizationString =
    //** INSTANTIATE:
    //    Key = 0c42ea6804303954deb197a07e6dbdd2
    //    V   = 80941680713df715056fb2a3d2e998b2
    //
    // EntropyInputReseed = 1dea0a12c52bf64339dd291c80d8ca89
    // AdditionalInputReseed =
    //** RESEED:
    //    Key = 32fbfd0109f364ed21ef21a6e5c763e7
    //    V   = f2bacbb233252fba35fb0582f9286179
    //
    // AdditionalInput =
    //** GENERATE (FIRST CALL):
    //    Key = 757c8eb766f9aaa4650d6500b58624a3
    //    V   = 99003d630bba500fe17c37f8c7331bf6
    //
    // AdditionalInput =
    // ReturnedBits =
    // 2859cc468a76b08661ffd23b28547ffd0997ad526a0f51261b99ed3a37bd407bf418dbe6c6c3e26ed0ddefcb7474d899bd99f3655427519fc5b4057bcaf306d4
    //** GENERATE (SECOND CALL):
    //    Key = e421ff2445e04992faf36cf9a5eaf1f9
    //    V   = 5907ab447a88e5106753507cc97e0fd5

    std::vector<unsigned char> entropy = {0x0f, 0x65, 0xda, 0x13, 0xdc, 0xa4,
                                          0x07, 0x99, 0x9d, 0x47, 0x73, 0xc2,
                                          0xb4, 0xa1, 0x1d, 0x85};
    std::vector<unsigned char> nonce = {0x52, 0x09, 0xe5, 0xb4,
                                        0xed, 0x82, 0xa2, 0x34};
    std::vector<unsigned char> personalization = {};

    AESCTRRNG drbg(entropy, nonce, personalization, SecurityLevel::AES128,
                   false);
    std::cout << "Instantiate: " << std::endl;
    drbg.print_params();

    std::vector<unsigned char> entropy_input_reseed = {
        0x1d, 0xea, 0x0a, 0x12, 0xc5, 0x2b, 0xf6, 0x43,
        0x39, 0xdd, 0x29, 0x1c, 0x80, 0xd8, 0xca, 0x89};
    std::vector<unsigned char> additional_input_reseed = {};
    std::cout << "Reseed: " << std::endl;
    drbg.reseed(entropy_input_reseed, additional_input_reseed);
    drbg.print_params();

    std::cout << "Random Bytes (First Call): " << std::endl;
    std::vector<unsigned char> randomBytes1 = drbg.generate_bytes(64);
    drbg.print_params();
    print_hex(randomBytes1);

    std::cout << "Random Bytes (Second Call): " << std::endl;
    std::vector<unsigned char> randomBytes2 = drbg.generate_bytes(64);
    drbg.print_params();
    print_hex(randomBytes2);
}

int main()
{
    std::cout << " -- CASE 1 -- " << std::endl;
    case1();
    std::cout << std::endl << std::endl << std::endl;

    std::cout << " -- CASE 2 -- " << std::endl;
    case2();

    return EXIT_SUCCESS;
}
