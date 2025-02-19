// Copyright 2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "aes_rng.cuh"
#include "cuda_rng.cuh"
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
    cout << "TEST HERE!" << endl;
    rngongpu::test_aes();

    Modulus32 mert(15);
    cout << mert.value << endl;
    cout << mert.mu << endl;
    cout << mert.bit << endl;

    return EXIT_SUCCESS;
}