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

    /**
     * Random Number Generation Using Deterministic Random Bit Generators:
     * https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf
     */
    class AESCTRRNG
    {
      public:
        /**
         * NIST Special Publication 800-90A:
         * Instantiation When a Derivation Function is Used
         * When instantiation is performed using this method, the entropy input may or may not have full
         * entropy; in either case, a nonce is required. 
         * Let df be the derivation function specified in Section 10.3.2. When instantiation is performed
         * using this method, a nonce is required, whereas using the method in Section 10.2.1.3.1 does not
         * require a nonce, since full entropy is provided when using that method.
         * The following process or its equivalent shall be used as the instantiate algorithm for this DRBG
         * mechanism: 
         * 
         * entropy_input: The string of bits obtained from the randomness source.
         * nonce: A string of bits as specified in Section 8.6.7.
         * personalization_string: The personalization string received from the consuming
         * application. Note that the length of the personalization_string may be zero. 
         * sec_level: The security strength for the instantiation. This parameter is
         * optional for CTR_DRBG, since it is not used. 
         * prediction_resistance: If true, reseed is performed on each generate_bytes() call.
         */
        AESCTRRNG(const std::vector<unsigned char>& entropy_input,
                  const std::vector<unsigned char>& nonce,
                  const std::vector<unsigned char>& personalization_string,
                  SecurityLevel sec_level, bool prediction_resistance = false);

        /**
         * NIST Special Publication 800-90A:
         * This method of generating bits is used when a derivation function is used by an implementation.
         * Let df be the derivation function specified in Section 10.3.2.
         * The following process or its equivalent shall be used as the generate algorithm for this DRBG
         * mechanism (see step 8 of the generate process in Section 9.3.3)
         * 
         * requested_number_of_bits: The number of pseudorandom bits to be returned to the
         * generate function.
         * additional_input: None
         */
        std::vector<unsigned char> generate_bytes(std::size_t requested_number_of_bytes);

        /**
         * NIST Special Publication 800-90A:
         * This method of generating bits is used when a derivation function is used by an implementation.
         * Let df be the derivation function specified in Section 10.3.2.
         * The following process or its equivalent shall be used as the generate algorithm for this DRBG
         * mechanism (see step 8 of the generate process in Section 9.3.3)
         * 
         * requested_number_of_bits: The number of pseudorandom bits to be returned to the
         * generate function.
         * additional_input: The additional input string received from the consuming
         * application. Note that the length of the additional_input string may be zero. 
         */
        std::vector<unsigned char> generate_bytes(std::size_t requested_number_of_bytes, const std::vector<unsigned char>& additional_input);

        /**
         * NIST Special Publication 800-90A:
         * This method of generating bits is used when a derivation function is used by an implementation.
         * Let df be the derivation function specified in Section 10.3.2.
         * The following process or its equivalent shall be used as the generate algorithm for this DRBG
         * mechanism (see step 8 of the generate process in Section 9.3.3)
         * 
         * requested_number_of_bits: The number of pseudorandom bits to be returned to the
         * generate function.
         * entropy_input: Custom entropy_input for reseed.
         * additional_input: The additional input string received from the consuming
         * application. Note that the length of the additional_input string may be zero. 
         */
        std::vector<unsigned char> generate_bytes(std::size_t requested_number_of_bytes, const std::vector<unsigned char>& entropy_input, const std::vector<unsigned char>& additional_input);

        /**
         * NIST Special Publication 800-90A:
         * Reseeding When a Derivation Function is Used.
         * Let df be the derivation function specified in Section 10.3.2. 
         * The following process or its equivalent shall be used as the reseed algorithm for this DRBG
         * mechanism (see reseed process step 6 of Section 9.2): 
         * 
         * entropy_input: Entropy input is automatically generated by RAND_bytes(Open SSL).
         * additional_input: The additional input string received from the consuming
         * application. Note that the length of the additional_input string may be zero. 
         */
        void reseed(const std::vector<unsigned char>& additional_input = std::vector<unsigned char>());

        /**
         * NIST Special Publication 800-90A:
         * Reseeding When a Derivation Function is Used.
         * Let df be the derivation function specified in Section 10.3.2. 
         * The following process or its equivalent shall be used as the reseed algorithm for this DRBG
         * mechanism (see reseed process step 6 of Section 9.2): 
         * 
         * entropy_input: The string of bits obtained from the randomness source. 
         * additional_input: The additional input string received from the consuming
         * application. Note that the length of the additional_input string may be zero. 
         */
        void reseed(const std::vector<unsigned char>& entropy_input, const std::vector<unsigned char>& additional_input);

        void print_params();

      private:
        SecurityLevel security_level; // Security level
        std::vector<unsigned char> key; // Internal AES key (16, 24, or 32 bytes)
        std::vector<unsigned char> V; // Internal state (counter, always 16 bytes)
        unsigned long long reseed_counter; // Reseed counter
        bool prediction_resistance_enabled; // Prediction resistance flag

        // Derived parameters:
        std::size_t key_len; // Key length in bytes (16 for AES-128, 24 for AES-192, 32 for AES-256)
        std::size_t seed_len; // seedLen = keyLen + 16 (16 bytes for the block size)
        std::size_t out_len = 16; // for AES
        std::size_t block_len = 16; // for AES
        
        // NIST SP 800‑90A recommends that the number of blocks generated before
        // a reseed be limited.
        const unsigned long long RESEED_INTERVAL = (1ULL << 48);

        
        /**
         * NIST Special Publication 800-90A:
         * This derivation function is used by the CTR_DRBG that is specified in Section 10.2. BCC and
         * Block_Encrypt are discussed in Section 10.3.3. Let outlen be its output block length, which is a
         * multiple of eight bits for the approved block cipher algorithms, and let keylen be the key length.
         *
         * input_string: The string to be operated on. This string shall be a multiple of eight bits. 
         * no_of_bits_to_return: The number of bits to be returned by Block_Cipher_df. The
         * maximum length (max_number_of_bits) is 512 bits for the currently approved block
         * cipher algorithms. 
         */
        std::vector<unsigned char> derivation_function(const std::vector<unsigned char>& input_string, std::size_t no_of_bits_to_return);

        /**
         * NIST Special Publication 800-90A:
         * The update function updates the internal state of the CTR_DRBG using the
         * provided_data. The values for blocklen, keylen and seedlen are provided in Table 3 of Section
         * 10.2.1. The value of ctr_len is known by an implementation. The block cipher operation in step
         * 2.2 of the CTR_DRBG_UPDATE process uses the selected block cipher algorithm. The
         * specification of Block_Encrypt is discussed in Section 10.3.3.
         * 
         * provided_data: The data to be used. This must be exactly seedlen bits in length; this
         * length is guaranteed by the construction of the provided_data in the instantiate,
         * reseed and generate functions.
         */
        void update(const std::vector<unsigned char>& provided_data);

        // incrementV: Increments the internal counter V in big-endian order.
        void incrementV();

        // Validation functions.
        void
        validate_entropy(const std::vector<unsigned char>& entropy_input);

        // Helper functions to select the appropriate EVP_CIPHER for the chosen
        // security level.
        const EVP_CIPHER* getEVPCipherCBC() const;
        const EVP_CIPHER* getEVPCipherECB() const;

        std::vector<unsigned char> block_encrypt(const std::vector<unsigned char>& key, const std::vector<unsigned char>& plaintext);

        std::vector<unsigned char> BCC(const std::vector<unsigned char>& key, const std::vector<unsigned char>& data);
    };

} // namespace rngongpu
#endif // AES_CPU_RNG_H
