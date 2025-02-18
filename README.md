# RNGonGPU 

RNGonGPU is a GPU-based random number generation library engineered for secure applications using
`CSPRNG`(Cryptographically Secure Pseudo-Random Number Generators). It is designed to be compliant with NIST's
[Recommendation for Random Number Generation Using Deterministic Random Bit Generators](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf)
standards, ensuring that the system meets stringent security and reproducibility requirements. 
Unlike cuRAND, which is geared toward simulations without cryptographic security, RNGonGPU 
ensures both reproducible and secure outputs. It employs `AES` to secure each generated value,
safeguarding against potential attacks. Designed with extensibility in mind, the framework will
integrate additional advanced algorithms over time, making it a versatile solution for applications
ranging from scientific research to high-security cryptographic systems.

Current RNG types:
- [cuRAND](https://docs.nvidia.com/cuda/curand/index.html) (NOT Cryptographically Secure)
- [AES-CTR](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=936594) 

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=3.26.4
- [GCC](https://gcc.gnu.org/)
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
```

## Examples

To run examples:

```bash
$ cmake -S . -D RNGonGPU_BUILD_EXAMPLES=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/examples/<...>
$ Example: ./build/bin/examples/example
```

## Using RNGonGPU in a downstream CMake project

Make sure RNGonGPU is installed before integrating it into your project. The installed RNGonGPU library provides a set of config files that make it easy to integrate RNGonGPU into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(RNGonGPU)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) RNGonGPU::RNGonGPU CUDA::cudart)
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