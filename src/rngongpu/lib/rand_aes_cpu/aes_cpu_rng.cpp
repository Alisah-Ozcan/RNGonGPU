// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_cpu_rng.h"

namespace rngongpu
{
    static std::vector<unsigned char> uint32ToBytes(unsigned int x)
    {
        std::vector<unsigned char> bytes(4);
        bytes[0] = (x >> 24) & 0xFF;
        bytes[1] = (x >> 16) & 0xFF;
        bytes[2] = (x >> 8) & 0xFF;
        bytes[3] = x & 0xFF;
        return bytes;
    }

    void AESCTRRNG::print_params()
    {
        std::cout << "key: " << std::endl;
        for (unsigned char byte : key)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(byte);
        }
        std::cout << std::dec << std::endl << std::endl;

        std::cout << "V: " << std::endl;
        for (unsigned char byte : V)
        {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<int>(byte);
        }
        std::cout << std::dec << std::endl << std::endl;
    }

    AESCTRRNG::AESCTRRNG(
        const std::vector<unsigned char>& entropy_input,
        const std::vector<unsigned char>& nonce,
        const std::vector<unsigned char>& personalization_string,
        SecurityLevel sec_level, bool prediction_resistance)
        : security_level(sec_level),
          prediction_resistance_enabled(prediction_resistance),
          reseed_counter(1)
    {
        validate_entropy(entropy_input);
        // Set key_len based on security level.
        switch (security_level)
        {
            case SecurityLevel::AES128:
                key_len = 16;
                break;
            case SecurityLevel::AES192:
                key_len = 24;
                break;
            case SecurityLevel::AES256:
                key_len = 32;
                break;
            default:
                throw std::runtime_error("Unsupported security level");
        }
        seed_len = key_len + 16;

        std::vector<unsigned char> seed_material = entropy_input;
        seed_material.insert(seed_material.end(), nonce.begin(), nonce.end());
        seed_material.insert(seed_material.end(),
                             personalization_string.begin(),
                             personalization_string.end()); // It's OK!

        std::vector<unsigned char> seed =
            derivation_function(seed_material, seed_len);

        key = std::vector<unsigned char>(key_len, 0);
        V = std::vector<unsigned char>(16, 0);

        update(seed);
    }

    void AESCTRRNG::reseed(const std::vector<unsigned char>& additional_input)
    {
        std::vector<unsigned char> newEntropy(key_len);
        if (1 != RAND_bytes(newEntropy.data(), newEntropy.size()))
            throw std::runtime_error("RAND_bytes failed during reseed");

        std::vector<unsigned char> seed_material = newEntropy;
        seed_material.insert(seed_material.end(), additional_input.begin(),
                             additional_input.end());

        std::vector<unsigned char> seed =
            derivation_function(seed_material, seed_len);
        update(seed);
        reseed_counter = 1;
    }

    void AESCTRRNG::reseed(const std::vector<unsigned char>& entropy_input,
                           const std::vector<unsigned char>& additional_input)
    {
        std::vector<unsigned char> seed_material = entropy_input;
        seed_material.insert(seed_material.end(), additional_input.begin(),
                             additional_input.end());

        std::vector<unsigned char> seed =
            derivation_function(seed_material, seed_len);
        update(seed);
        reseed_counter = 1;
    }

    std::vector<unsigned char>
    AESCTRRNG::generate_bytes(std::size_t requested_number_of_bytes)
    {
        if (prediction_resistance_enabled || reseed_counter >= RESEED_INTERVAL)
        {
            reseed();
        }

        std::vector<unsigned char> output;
        output.reserve(requested_number_of_bytes);

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error(
                "getRandomBytes: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error(
                "getRandomBytes: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> block(block_len);
        int outlen = 0;
        while (output.size() < requested_number_of_bytes)
        {
            incrementV();

            if (1 != EVP_EncryptUpdate(ctx, block.data(), &outlen, V.data(),
                                       block_len))
            {
                throw std::runtime_error(
                    "getRandomBytes: EVP_EncryptUpdate failed");
            }

            if (outlen != static_cast<int>(block_len))
            {
                throw std::runtime_error(
                    "getRandomBytes: Unexpected block size");
            }

            output.insert(output.end(), block.begin(), block.end());
        }
        EVP_CIPHER_CTX_free(ctx);

        output.resize(requested_number_of_bytes);

        update(std::vector<unsigned char>());

        reseed_counter++;

        return output;
    }

    std::vector<unsigned char> AESCTRRNG::generate_bytes(
        std::size_t requested_number_of_bytes,
        const std::vector<unsigned char>& additional_input)
    {
        if (prediction_resistance_enabled || reseed_counter >= RESEED_INTERVAL)
        {
            reseed(additional_input);
        }

        std::vector<unsigned char> additional_input_in;
        if (additional_input.size() != 0)
        {
            additional_input_in =
                derivation_function(additional_input, seed_len);
            update(additional_input_in);
        }
        else
        {
            additional_input_in = std::vector<unsigned char>(seed_len, 0);
        }

        std::vector<unsigned char> temp;
        temp.reserve(requested_number_of_bytes);

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error("Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error("EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> block(block_len);
        int outlen = 0;
        while (temp.size() < requested_number_of_bytes)
        {
            incrementV();

            if (1 != EVP_EncryptUpdate(ctx, block.data(), &outlen, V.data(),
                                       block_len))
                throw std::runtime_error("EVP_EncryptUpdate failed");

            if (outlen != static_cast<int>(block_len))
                throw std::runtime_error("Unexpected block size");

            temp.insert(temp.end(), block.begin(), block.end());
        }
        EVP_CIPHER_CTX_free(ctx);
        temp.resize(requested_number_of_bytes);

        update(additional_input_in);

        reseed_counter++;

        return temp;
    }

    std::vector<unsigned char> AESCTRRNG::generate_bytes(
        std::size_t requested_number_of_bytes,
        const std::vector<unsigned char>& entropy_input,
        const std::vector<unsigned char>& additional_input)
    {
        if (prediction_resistance_enabled || reseed_counter >= RESEED_INTERVAL)
        {
            reseed(entropy_input, additional_input);
        }

        std::vector<unsigned char> additional_input_in;
        if (additional_input.size() != 0)
        {
            additional_input_in =
                derivation_function(additional_input, seed_len);
            update(additional_input_in);
        }
        else
        {
            additional_input_in = std::vector<unsigned char>(seed_len, 0);
        }

        std::vector<unsigned char> temp;
        temp.reserve(requested_number_of_bytes);

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error(
                "getRandomBytes: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error(
                "getRandomBytes: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> block(block_len);
        int outlen = 0;
        while (temp.size() < requested_number_of_bytes)
        {
            incrementV();

            if (1 != EVP_EncryptUpdate(ctx, block.data(), &outlen, V.data(),
                                       block_len))
                throw std::runtime_error(
                    "getRandomBytes: EVP_EncryptUpdate failed");

            if (outlen != static_cast<int>(block_len))
                throw std::runtime_error(
                    "getRandomBytes: Unexpected block size");

            temp.insert(temp.end(), block.begin(), block.end());
        }
        EVP_CIPHER_CTX_free(ctx);
        temp.resize(requested_number_of_bytes);

        update(additional_input_in);

        reseed_counter++;

        return temp;
    }

    std::vector<unsigned char> AESCTRRNG::derivation_function(
        const std::vector<unsigned char>& input_string,
        std::size_t no_of_bits_to_return)
    {
        unsigned int input_bits =
            static_cast<unsigned int>(input_string.size());
        std::vector<unsigned char> S = uint32ToBytes(input_bits);

        unsigned int requested_bit =
            static_cast<unsigned int>(no_of_bits_to_return);
        std::vector<unsigned char> len_bytes = uint32ToBytes(requested_bit);

        S.insert(S.end(), len_bytes.begin(), len_bytes.end());
        S.insert(S.end(), input_string.begin(), input_string.end());

        S.push_back(0x80);
        while (S.size() % out_len != 0)
            S.push_back(0x00);

        std::vector<unsigned char> temp;

        uint32_t i = 0;

        std::vector<unsigned char> K;
        for (uint32_t j = 0; j < key_len; j++)
        {
            K.push_back(static_cast<unsigned char>(j));
        }

        while (temp.size() < (key_len + out_len))
        {
            std::vector<unsigned char> IV = uint32ToBytes(i);
            while (IV.size() < out_len)
            {
                IV.push_back(0x00);
            }

            std::vector<unsigned char> dataForBCC;
            dataForBCC.insert(dataForBCC.end(), IV.begin(), IV.end());
            dataForBCC.insert(dataForBCC.end(), S.begin(), S.end());

            std::vector<unsigned char> bccResult = BCC(K, dataForBCC);
            temp.insert(temp.end(), bccResult.begin(), bccResult.end());
            i++;
        }

        std::vector<unsigned char> newK(temp.begin(), temp.begin() + key_len);
        K = newK;

        std::vector<unsigned char> X(temp.begin() + key_len,
                                     temp.begin() + key_len + out_len);

        temp.clear();

        while (temp.size() < no_of_bits_to_return)
        {
            X = block_encrypt(K, X);
            temp.insert(temp.end(), X.begin(), X.end());
        }

        std::vector<unsigned char> requested_bits(
            temp.begin(), temp.begin() + no_of_bits_to_return);

        return requested_bits;
    }

    void AESCTRRNG::update(const std::vector<unsigned char>& provided_data)
    {
        const std::size_t seedlen = seed_len;
        std::vector<unsigned char> temp;
        temp.reserve(seedlen);

        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
            throw std::runtime_error("update: Failed to create EVP_CIPHER_CTX");

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error("update: EVP_EncryptInit_ex failed");
        EVP_CIPHER_CTX_set_padding(ctx, 0);

        std::vector<unsigned char> outputBlock(block_len);
        std::vector<unsigned char> Vtemp = V;
        for (std::size_t i = 0; i < seedlen / block_len; i++)
        {
            for (int j = block_len - 1; j >= 0; j--)
            {
                if (++Vtemp[j] != 0)
                    break;
            }
            int outlen = 0;
            if (1 != EVP_EncryptUpdate(ctx, outputBlock.data(), &outlen,
                                       Vtemp.data(), block_len))
                throw std::runtime_error("update: EVP_EncryptUpdate failed");
            if (outlen != static_cast<int>(block_len))
                throw std::runtime_error("update: Unexpected block size");
            temp.insert(temp.end(), outputBlock.begin(), outputBlock.end());
        }
        EVP_CIPHER_CTX_free(ctx);

        if (!provided_data.empty())
        {
            if (provided_data.size() != seedlen)
                throw std::runtime_error("update: additional input "
                                         "must be of length seedLen");
            for (std::size_t i = 0; i < seedlen; i++)
            {
                temp[i] ^= provided_data[i];
            }
        }

        key.assign(temp.begin(), temp.begin() + key_len);
        V.assign(temp.begin() + key_len, temp.end());
    }

    void AESCTRRNG::incrementV()
    {
        for (int i = block_len - 1; i >= 0; i--)
        {
            if (++V[i] != 0)
                break;
        }
    }

    void
    AESCTRRNG::validate_entropy(const std::vector<unsigned char>& entropy_input)
    {
        if (entropy_input.size() < 16)
            throw std::runtime_error(
                "Insufficient entropy: minimum 16 bytes required");
    }

    const EVP_CIPHER* AESCTRRNG::getEVPCipherCBC() const
    {
        switch (security_level)
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
        switch (security_level)
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

    std::vector<unsigned char>
    AESCTRRNG::block_encrypt(const std::vector<unsigned char>& key,
                             const std::vector<unsigned char>& plaintext)
    {
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx)
        {
            throw std::runtime_error("EVP_CIPHER_CTX_new failed");
        }

        std::vector<unsigned char> ciphertext(plaintext.size() +
                                              EVP_MAX_BLOCK_LENGTH);
        int len = 0, ciphertext_len = 0;

        if (1 != EVP_EncryptInit_ex(ctx, getEVPCipherECB(), nullptr, key.data(),
                                    nullptr))
            throw std::runtime_error("EVP_EncryptInit_ex failed");

        EVP_CIPHER_CTX_set_padding(ctx, 0);

        if (1 != EVP_EncryptUpdate(ctx, ciphertext.data(), &len,
                                   plaintext.data(), plaintext.size()))
            throw std::runtime_error("EVP_EncryptUpdate failed");

        ciphertext_len = len;

        if (1 != EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len))
            throw std::runtime_error("EVP_EncryptFinal_ex failed");

        ciphertext_len += len;
        EVP_CIPHER_CTX_free(ctx);

        ciphertext.resize(ciphertext_len);
        return ciphertext;
    }

    std::vector<unsigned char>
    AESCTRRNG::BCC(const std::vector<unsigned char>& key,
                   const std::vector<unsigned char>& data)
    {
        if (data.size() % out_len != 0)
        {
            throw std::runtime_error(
                "BCC input data length is not a multiple of block size");
        }
        std::vector<unsigned char> X(out_len, 0x00);
        size_t num_blocks = data.size() / out_len;
        for (size_t i = 0; i < num_blocks; i++)
        {
            std::vector<unsigned char> block(data.begin() + i * out_len,
                                             data.begin() + (i + 1) * out_len);
            for (size_t j = 0; j < out_len; j++)
            {
                X[j] ^= block[j];
            }
            X = block_encrypt(key, X);
        }
        return X;
    }

} // namespace rngongpu
