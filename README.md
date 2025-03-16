# 🎲 RNGonGPU - A GPU Based Random Number Generation Library Using DRBG

RNGonGPU is a GPU-based random number generation library engineered for secure applications using `CSPRNG`(Cryptographically Secure Pseudo-Random Number Generators). It is designed to comply with NIST’s [Recommendation for Random Number Generation Using Deterministic Random Bit Generators](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf), ensuring that the system meets stringent security and reproducibility requirements. Unlike [cuRAND](https://docs.nvidia.com/cuda/curand/index.html)(which is primarily tailored for simulations without cryptographic security), RNGonGPU guarantees both reproducible and secure outputs by employing `AES` to secure each generated value, thereby safeguarding against potential attacks.

Developed using `CUDA`, the library capitalizes on the parallel processing capabilities of GPUs to deliver high-performance random number generation. Its current implementation operates in two distinct modes:

- **CUDA Mode**:
    - This mode leverages CUDA to harness the inherent parallel processing power of GPUs, utilizing the [cuRAND](https://docs.nvidia.com/cuda/curand/index.html) library to accelerate random number generation. Although cuRAND is optimized for performance in simulation contexts, it does not fully address all rigorous security requirements. (NOT Cryptographically Secure)
    - The CUDA mode supports three [cuRAND](https://docs.nvidia.com/cuda/curand/index.html) state types: `curandStateXORWOW`, `curandStateMRG32k3a`, and `curandStatePhilox4_32_10`.

- **AES Mode**:
    - In this mode, an [AES-CTR](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936594) Deterministic Random Bit Generator (CTR_DRBG) architecture is employed to ensure secure random number generation in strict accordance with the [NIST SP800-90A](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf) guidelines. `AES` functions as a block cipher operating on fixed 128-bit blocks, thereby creating a robust security framework that classifies this mode as a secure DRBG.

    - During `instantiation`, the generator receives an entropy input, a nonce, and an optional personalization string. These inputs are processed through the DF, which derives a fixed-length seed that establishes the internal state of the DRBG. This method accommodates entropy inputs that may not be full-strength, with the nonce ensuring uniqueness.

    - For `byte generation`, the API provides multiple overloads of the generate_bytes function to produce the desired number of pseudorandom bytes. Whether using just the requested byte count or incorporating additional input (and even custom entropy for reseeding), the DF mode is employed to standardize and securely process the inputs before `byte generation`. This process uses parallelized techniques ([CUDA_AES](https://github.com/cihangirtezcan/CUDA_AES)) to achieve both performance and security.

    - To maintain long-term security, the internal state is periodically updated via an `update` function. This process uses provided data—again processed through the DF—to refresh the internal key and counter (V). Additionally, `reseeding` functions are available to reinitialize the DRBG when certain limits are reached or upon explicit request, ensuring continuous security. `Reseeding` can incorporate both new entropy and additional input, aligning with NIST guidelines.

    - Across all operations—instantiation, update, reseeding, and generation—the `DF` mode plays a central role in transforming variable-length inputs into fixed-length, standardized values. This consistent approach not only simplifies the overall design but also ensures that the security properties defined by [NIST SP800-90A](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf) are upheld throughout the DRBG's lifecycle.

The library is designed to do much more than simply generate bytes. It offers a range of functions for generating different types of random numbers, such as uniform, normal, and ternary distributions, through both CUDA-based and `AES` based approaches. This design provides the flexibility to work with standard output as well as with modular arithmetic, which enables users to generate numbers according to specific modulus criteria.

For example, the API includes overloads for functions like `uniform_random_number`, `normal_random_number`, and `ternary_random_number` that allow users to obtain a variety of random outputs. Additionally, modular versions of these functions let you specify a modulus or an array of moduli to suit different application needs, whether it be for simulation, statistical analysis, or other specialized computational tasks. This flexibility in generating both standard and modular random numbers enhances the presentation and adaptability of the output, making the library a versatile tool in diverse computational environments.
     
RNGonGPU is designed with extensibility in mind. Future enhancements include plans to integrate additional advanced algorithms beyond `AES` such as various `AES` variants or alternative block ciphers, to broaden the performance and security spectrum. Moreover, the roadmap envisions incorporating modular number generation capabilities supporting higher bit widths (e.g., 128, 256, and 512 bits), further enhancing the library’s versatility for applications ranging from scientific research to high-security cryptographic systems.

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=3.26.4
- [GCC](https://gcc.gnu.org/)
- [OpenSSL](https://www.openssl.org/) >= 1.1.0
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=11.4

### Third-Party Dependencies
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) (Just for arithmetic operations)


### Build & Install

To build and install RNGonGPU, follow the steps below. This includes configuring the project using CMake, compiling the source code, and installing the library on your system.

<div align="center">

| GPU Architecture | Compute Capability (CMAKE_CUDA_ARCHITECTURES Value) |
|:----------------:|:---------------------------------------------------:|
| Volta  | 70, 72 |
| Turing | 75 |
| Ampere | 80, 86 |
| Ada	 | 89, 90 |

</div>

```bash
$ cmake -S . -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/
$ sudo cmake --install build
```

## Examples

To run examples:

```bash
$ cmake -S . -D RNGonGPU_BUILD_EXAMPLES=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/example/<...>
$ Example: ./build/bin/example/aes_drng_example
```

## [Tests](/test/README.md)

To run tests:
```bash
$ cmake -S . -D RNGonGPU_BUILD_TESTS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/test/test
```

## Benchmarks

To run benchmarks:
```bash
$ cmake -S . -D RNGonGPU_BUILD_BENCHMARKS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/benchmark/benchmark --disable-blocking-kernel
```

## Using RNGonGPU in a downstream CMake project

Make sure RNGonGPU is installed before integrating it into your project. The installed RNGonGPU library provides a set of config files that make it easy to integrate RNGonGPU into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(RNGonGPU)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) RNGonGPU::rngongpu CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ...
```

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contributing
Contributions are welcome! Please check the [CONTRIBUTING](CONTRIBUTING.md) file for guidelines on how to contribute to the project.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)