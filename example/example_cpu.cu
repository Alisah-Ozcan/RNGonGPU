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

// Helper function: print a vector of bytes in hexadecimal.
void printHex(const std::vector<unsigned char>& data) {
    for (unsigned char byte : data) {
        std::cout << std::hex << std::setw(2) << std::setfill('0')
                  << static_cast<int>(byte);
    }
    std::cout << std::dec << std::endl;
}

// Helper function: generate n random bytes using OpenSSL RAND_bytes.
std::vector<unsigned char> generateRandomBytes(std::size_t n) {
    std::vector<unsigned char> buf(n);
    if (1 != RAND_bytes(buf.data(), buf.size()))
        throw std::runtime_error("RAND_bytes failed");
    return buf;
}

// Helper: Get expected seed length based on security level.
std::size_t getExpectedSeedLen(SecurityLevel level) {
    switch (level) {
        case SecurityLevel::AES128:
            return 16 + 16; // 32 bytes
        case SecurityLevel::AES192:
            return 24 + 16; // 40 bytes
        case SecurityLevel::AES256:
            return 32 + 16; // 48 bytes
        default:
            throw std::runtime_error("Unsupported security level");
    }
}

/*
int main() {
    try {
        // Define the security levels to test.
        std::vector<SecurityLevel> levels = { SecurityLevel::AES128 };

        // Optional personalization string (same for all tests).
        std::vector<unsigned char> personalization = { 'E','x','a','m','p','l','e','D','R','B','G' };

        for (auto secLevel : levels) {
            std::cout << "-----------------------------------" << std::endl;
            std::cout << "Testing DRBG with security level: ";
            if (secLevel == SecurityLevel::AES128)
                std::cout << "AES-128" << std::endl;
            else if (secLevel == SecurityLevel::AES192)
                std::cout << "AES-192" << std::endl;
            else if (secLevel == SecurityLevel::AES256)
                std::cout << "AES-256" << std::endl;

            // Set entropy and nonce lengths based on the security level.
            std::size_t entropyLen = 16;
            std::size_t nonceLen   = 16;
            if (secLevel == SecurityLevel::AES192) {
                entropyLen = 24;
                nonceLen   = 16; // For higher security, nonce length can be increased.
            } else if (secLevel == SecurityLevel::AES256) {
                entropyLen = 32;
                nonceLen   = 16;
            }
            std::vector<unsigned char> entropy = generateRandomBytes(entropyLen);
            std::vector<unsigned char> nonce   = generateRandomBytes(nonceLen);

            // Instantiate DRBG with prediction resistance disabled for initial tests.
            AESCTRRNG drbg(entropy, nonce, personalization, secLevel, false);

            // Generate 64 random bytes.
            std::cout << "Random Bytes (initial): ";
            std::vector<unsigned char> randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);
            randomBytes = drbg.getRandomBytes(64); // 64
            printHex(randomBytes);

        }
        std::cout << "-----------------------------------" << std::endl;
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
*/


int main() {
  
    std::vector<unsigned char> entropy = {0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 
        0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
        0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F};
    std::vector<unsigned char> nonce = {
        0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27
    };
    std::vector<unsigned char> personalization = { };

    AESCTRRNG drbg(entropy, nonce, personalization, SecurityLevel::AES128, false);

    // Generate 64 random bytes.
    std::cout << "Random Bytes (initial): ";
    std::vector<unsigned char> randomBytes = drbg.getRandomBytes(64); // 64
    printHex(randomBytes);

    return 0;
}
