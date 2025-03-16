# Security Policy

## Overview
RNGonGPU is a GPU-based random number generation library for secure applications. It complies with NIST’s [Recommendation for Random Number Generation Using Deterministic Random Bit Generators](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf), ensuring stringent security and reproducibility. Unlike [cuRAND](https://docs.nvidia.com/cuda/curand/index.html), which is primarily tailored for simulations without cryptographic security, RNGonGPU guarantees both reproducible and secure outputs by employing AES to secure each generated value, thereby safeguarding against potential attacks. While every effort has been made to adhere to these rigorous standards, RNGonGPU is provided "as is" without any warranties, express or implied, and the developer(s) do not assume any liability for issues or damages arising from its use.

## Vulnerability Reporting
We value the contributions of the community in maintaining and improving the security of our library. If you discover any vulnerabilities, bugs, or any security-related issues, please report them responsibly by following the instructions below:

- **Email:** alisah@sabanciuniv.edu
- **Issue Tracker:** [GitHub Issues](https://github.com/Alisah-Ozcan/RNGonGPU/issues)

When submitting your report, please include:
- A detailed description of the issue.
- Steps to reproduce the problem.
- Any relevant logs, screenshots, or additional information that could help in diagnosing and resolving the issue.

## Liability Disclaimer
This library has been developed according to the NIST SP800-90A guidelines. However, its use is entirely at your own risk. The developer(s) and maintainers of this library shall not be held liable for any direct, indirect, incidental, or consequential damages arising from its use or misuse.

## Acknowledgements
We sincerely appreciate the efforts of all security researchers and users who responsibly disclose vulnerabilities or report issues. Your contributions are invaluable in helping us improve the security and reliability of this library.

## Updates
This security policy is subject to periodic updates. Please review this document regularly to stay informed about any changes to our security practices.
