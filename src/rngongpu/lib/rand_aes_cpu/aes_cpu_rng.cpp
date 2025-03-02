// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_cpu_rng.h"

namespace rngongpu
{
    // Helper: Convert a 32-bit unsigned integer to 4-byte big-endian.
    static std::vector<unsigned char> uint32ToBytes(unsigned int x)
    {
        std::vector<unsigned char> bytes(4);
        bytes[0] = (x >> 24) & 0xFF;
        bytes[1] = (x >> 16) & 0xFF;
        bytes[2] = (x >> 8) & 0xFF;
        bytes[3] = x & 0xFF;
        return bytes;
    }

    const EVP_CIPHER* AESCTRRNG::getEVPCipherCBC() const
    {
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                return EVP_aes_128_cbc();
            case SecurityLevel::AES192:
                return EVP_aes_192_cbc();
            case SecurityLevel::AES256:
                return EVP_aes_256_cbc();
            default:
                throw std::runtime_error("Unsupported security level in CBC");
        }
    }

    const EVP_CIPHER* AESCTRRNG::getEVPCipherECB() const
    {
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                return EVP_aes_128_ecb();
            case SecurityLevel::AES192:
                return EVP_aes_192_ecb();
            case SecurityLevel::AES256:
                return EVP_aes_256_ecb();
            default:
                throw std::runtime_error("Unsupported security level in ECB");
        }
    }

    // DF (Derivation Function) per NIST SP 800‑90A.
    // According to NIST, the DF input should be constructed as follows:
    // [requestedOutputBits (4 bytes) || inputLengthBits (4 bytes) || input ||
    // 0x80 || padding] Then, encrypt S using AES-CBC with a zero key and zero
    // IV to produce the seed.
    std::vector<unsigned char>
    AESCTRRNG::DF(const std::vector<unsigned char>& input,
                  std::size_t outputLen)
    {
        unsigned int requestedBits = static_cast<unsigned int>(outputLen * 8);
        std::vector<unsigned char> S = uint32ToBytes(requestedBits);

        unsigned int inputBits = static_cast<unsigned int>(input.size() * 8);
        std::vector<unsigned char> lenBytes = uint32ToBytes(inputBits);
        S.insert(S.end(), lenBytes.begin(), lenBytes.end());
        S.insert(S.end(), input.begin(), input.end());
        S.push_back(0x80);
        while (S.size() % 16 != 0)
            S.push_back(0x00);

        // Use a zero key whose length is equal to keyLen.
        std::vector<unsigned char> zeroKey(keyLen, 0x00);
        unsigned char zeroIV[16] = {0};

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error("DF: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherCBC(), nullptr,
                                    zeroKey.data(), zeroIV))
            throw std::runtime_error("DF: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> cipher(S.size() + 16);
        int outlen1 = 0, outlen2 = 0;
        if (1 !=
            EVP_EncryptUpdate(ctx, cipher.data(), &outlen1, S.data(), S.size()))
            throw std::runtime_error("DF: EVP_EncryptUpdate failed");
        if (1 != EVP_EncryptFinal_ex(ctx, cipher.data() + outlen1, &outlen2))
            throw std::runtime_error("DF: EVP_EncryptFinal_ex failed");
        EVP_CIPHER_CTX_free(ctx);

        cipher.resize(outlen1 + outlen2);
        if (cipher.size() > outputLen)
            cipher.resize(outputLen);
        return cipher;
    }

    // CTR_DRBG_Update updates the internal state as per NIST SP 800‑90A:
    // It encrypts successive increments of the counter V in ECB mode to produce
    // a temporary value of length seedLen. If providedData is non-empty, it
    // must be exactly seedLen bytes and is XORed with the temporary value
    // before updating the internal state.
    void
    AESCTRRNG::CTR_DRBG_Update(const std::vector<unsigned char>& providedData)
    {
        const std::size_t seedlen = seedLen;
        std::vector<unsigned char> temp;
        temp.reserve(seedlen);

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error(
                "CTR_DRBG_Update: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error(
                "CTR_DRBG_Update: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        const std::size_t blockSize = 16;
        std::vector<unsigned char> outputBlock(blockSize);
        std::vector<unsigned char> Vtemp = V;
        for (std::size_t i = 0; i < seedlen / blockSize; i++)
        {
            // Increment Vtemp in big-endian order.
            for (int j = blockSize - 1; j >= 0; j--)
            {
                if (++Vtemp[j] != 0)
                    break;
            }
            int outlen = 0;
            if (1 != EVP_EncryptUpdate(ctx, outputBlock.data(), &outlen,
                                       Vtemp.data(), blockSize))
                throw std::runtime_error(
                    "CTR_DRBG_Update: EVP_EncryptUpdate failed");
            if (outlen != static_cast<int>(blockSize))
                throw std::runtime_error(
                    "CTR_DRBG_Update: Unexpected block size");
            temp.insert(temp.end(), outputBlock.begin(), outputBlock.end());
        }
        EVP_CIPHER_CTX_free(ctx);

        if (!providedData.empty())
        {
            if (providedData.size() != seedlen)
                throw std::runtime_error("CTR_DRBG_Update: additional input "
                                         "must be of length seedLen");
            for (std::size_t i = 0; i < seedlen; i++)
            {
                temp[i] ^= providedData[i];
            }
        }
        // Update internal state: new key is first keyLen bytes; new V is the
        // remaining 16 bytes.
        key.assign(temp.begin(), temp.begin() + keyLen);
        V.assign(temp.begin() + keyLen, temp.end());
    }

    void AESCTRRNG::incrementV()
    {
        const std::size_t blockSize = 16;
        for (int i = blockSize - 1; i >= 0; i--)
        {
            if (++V[i] != 0)
                break;
        }
    }

    void AESCTRRNG::validateEntropyInput(
        const std::vector<unsigned char>& entropyInput)
    {
        // For all levels, at least 16 bytes are required; higher levels should
        // provide more.
        if (entropyInput.size() < 16)
            throw std::runtime_error(
                "Insufficient entropy: minimum 16 bytes required");
    }

    void AESCTRRNG::validateAdditionalInput(
        const std::vector<unsigned char>& additionalInput)
    {
        if (!additionalInput.empty() && additionalInput.size() != seedLen)
            throw std::runtime_error(
                "Additional input must be exactly seedLen bytes if provided");
    }

    // Constructor (Instantiation)
    // It assembles seed material = entropyInput || nonce ||
    // personalizationString, and then uses DF to produce a seed.
    AESCTRRNG::AESCTRRNG(
        const std::vector<unsigned char>& entropyInput,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalizationString,
        SecurityLevel secLevel, bool predictionResistance)
        : securityLevel(secLevel),
          predictionResistanceEnabled(predictionResistance), reseedCounter(1)
    {
        validateEntropyInput(entropyInput);
        // Set keyLen based on security level.
        switch (securityLevel)
        {
            case SecurityLevel::AES128:
                keyLen = 16;
                break;
            case SecurityLevel::AES192:
                keyLen = 24;
                break;
            case SecurityLevel::AES256:
                keyLen = 32;
                break;
            default:
                throw std::runtime_error("Unsupported security level");
        }
        seedLen = keyLen + 16; // 16 bytes for the block size (V)

        // Assemble seed material.
        std::vector<unsigned char> seed_material = entropyInput;
        seed_material.insert(seed_material.end(), nonce.begin(), nonce.end());
        seed_material.insert(seed_material.end(), personalizationString.begin(),
                             personalizationString.end());

        // Derive seed using DF.
        std::vector<unsigned char> seed = DF(seed_material, seedLen);
        key.assign(seed.begin(), seed.begin() + keyLen);
        V.assign(seed.begin() + keyLen, seed.end());
    }

    // Reseed: Gathers new entropy and optional additional input.
    // The new entropy length is now chosen to be keyLen (matching the security
    // level).
    void AESCTRRNG::reseed(const std::vector<unsigned char>& additionalInput)
    {
        validateAdditionalInput(additionalInput);
        // Generate new entropy of length equal to keyLen.
        std::vector<unsigned char> newEntropy(keyLen);
        if (1 != RAND_bytes(newEntropy.data(), newEntropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        std::vector<unsigned char> seed_material = newEntropy;
        seed_material.insert(seed_material.end(), additionalInput.begin(),
                             additionalInput.end());

        std::vector<unsigned char> seed = DF(seed_material, seedLen);
        CTR_DRBG_Update(seed);
        reseedCounter = 1;
    }

    // getRandomBytes: Generates n random bytes using AES-ECB encryption of the
    // incremented counter V. If prediction resistance is enabled, a reseed is
    // performed before generation. Also, if the reseedCounter reaches
    // RESEED_INTERVAL, a reseed is triggered.
    std::vector<unsigned char> AESCTRRNG::getRandomBytes(std::size_t n)
    {
        if (predictionResistanceEnabled)
        {
            reseed(std::vector<unsigned char>());
        }

        std::vector<unsigned char> output;
        output.reserve(n);

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error(
                "getRandomBytes: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error(
                "getRandomBytes: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        const std::size_t blockSize = 16;
        std::vector<unsigned char> block(blockSize);
        int outlen = 0;
        while (output.size() < n)
        {
            incrementV();

            ///////////////////////////
            std::cout << output.size() << " -> ";
            for (unsigned char byte : key)
            {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(byte);
            }
            std::cout << std::dec << std::endl;

            ///////////////////////////

            if (1 != EVP_EncryptUpdate(ctx, block.data(), &outlen, V.data(),
                                       blockSize))
                throw std::runtime_error(
                    "getRandomBytes: EVP_EncryptUpdate failed");
            if (outlen != static_cast<int>(blockSize))
                throw std::runtime_error(
                    "getRandomBytes: Unexpected block size");
            output.insert(output.end(), block.begin(), block.end());
            reseedCounter++;
            if (reseedCounter >= RESEED_INTERVAL)
                reseed(std::vector<unsigned char>());
        }
        EVP_CIPHER_CTX_free(ctx);

        CTR_DRBG_Update(std::vector<unsigned char>());
        output.resize(n);
        return output;
    }

} // namespace rngongpu
