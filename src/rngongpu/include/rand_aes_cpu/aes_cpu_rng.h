// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AES_CPU_RNG_H
#define AES_CPU_RNG_H

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace rngongpu
{
    // Supported security levels in bits.
    enum class SecurityLevel
    {
        AES128 = 128,
        AES192 = 192,
        AES256 = 256
    };

    // CTR_DRBG class compliant with NIST SP 800‑90A.
    // The design follows the structure described in NIST SP 800‑90A
    // Section 10.2.1: "CTR_DRBG Instantiation" uses a derivation function (DF)
    // that takes an input string of the form: [requestedOutputBits (4 bytes) ||
    // inputLengthBits (4 bytes) || input || 0x80 || padding]. The DRBG uses AES
    // in counter mode (with an internal state of key and V) and supports
    // prediction resistance and reseed as described in the standard.
    class AESCTRRNG
    {
      public:
        // Constructor (Instantiation)
        // entropyInput: Entropy input. Recommended length is equal to the
        // security strength:
        //   AES-128: 16 bytes, AES-192: 24 bytes, AES-256: 32 bytes.
        // nonce: A nonce value (we choose the same length as entropy for higher
        // security). personalizationString: Optional string; can be empty.
        // secLevel: Security level (AES128, AES192, or AES256).
        // predictionResistance: If true, reseed is performed on each
        // getRandomBytes() call.
        AESCTRRNG(const std::vector<unsigned char>& entropyInput,
                  const std::vector<unsigned char>& nonce,
                  const std::vector<unsigned char>& personalizationString,
                  SecurityLevel secLevel, bool predictionResistance = false);

        // getRandomBytes: Generates n random bytes.
        std::vector<unsigned char> getRandomBytes(std::size_t n);

        // reseed: Reseeds the DRBG with new entropy and optional additional
        // input. If additionalInput is provided, its length must be exactly
        // seedLen (keyLen + 16 bytes).
        void reseed(const std::vector<unsigned char>& additionalInput =
                        std::vector<unsigned char>());

      private:
        SecurityLevel securityLevel; // Security level
        std::vector<unsigned char>
            key; // Internal AES key (16, 24, or 32 bytes)
        std::vector<unsigned char>
            V; // Internal state (counter, always 16 bytes)
        unsigned long long reseedCounter; // Reseed counter
        bool predictionResistanceEnabled; // Prediction resistance flag

        // Derived parameters:
        std::size_t keyLen; // Key length in bytes (16 for AES-128, 24 for
                            // AES-192, 32 for AES-256)
        std::size_t
            seedLen; // seedLen = keyLen + 16 (16 bytes for the block size)

        // NIST SP 800‑90A recommends that the number of blocks generated before
        // a reseed be limited.
        const unsigned long long RESEED_INTERVAL = (1ULL << 48);

        // DF: Derivation Function per NIST SP 800‑90A.
        // It takes an input string and produces outputLen bytes.
        std::vector<unsigned char> DF(const std::vector<unsigned char>& input,
                                      std::size_t outputLen);

        // CTR_DRBG_Update: Updates the internal state (key and V) using the
        // provided data. If providedData is empty, it is treated as a zero
        // string of length seedLen.
        void CTR_DRBG_Update(const std::vector<unsigned char>& providedData);

        // incrementV: Increments the internal counter V in big-endian order.
        void incrementV();

        // Validation functions.
        void
        validateEntropyInput(const std::vector<unsigned char>& entropyInput);
        void validateAdditionalInput(
            const std::vector<unsigned char>& additionalInput);

        // Helper functions to select the appropriate EVP_CIPHER for the chosen
        // security level.
        const EVP_CIPHER* getEVPCipherCBC() const;
        const EVP_CIPHER* getEVPCipherECB() const;

        
        // Blok cipher: AES-128 ECB mode, no padding
        std::vector<unsigned char> Block_Encrypt(const std::vector<unsigned char>& key, const std::vector<unsigned char>& plaintext);

        std::vector<unsigned char> BCC(const std::vector<unsigned char>& K, const std::vector<unsigned char>& data);
            
        // Helper TODO: remove it!
        void appendBytes(std::vector<unsigned char>& dest, const std::vector<unsigned char>& src);

    };

} // namespace rngongpu
#endif // AES_CPU_RNG_H
