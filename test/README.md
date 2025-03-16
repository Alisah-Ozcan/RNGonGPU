# RNGonGPU CTR_DRBG Test Cases – AES-based

This archive contains only the CTR_DRBG test vectors extracted from the NIST SP 800-90A DRBG example files. These vectors are used to validate RNGonGPU’s implementation against the NIST test cases for the AES-based CTR_DRBG mechanism. All values in these files pertain exclusively to CTR_DRBG; test vectors for other DRBG mechanisms (such as Hash_DRBG, HMAC_DRBG, etc.) are not included.

## Configuration Details

Depending on the original [NIST zip archive](https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/random-number-generators), the test vectors fall into one of the following configurations:

1. **No Reseed (Prediction Resistance NOT ENABLED, NO RESEED function):**  
   - The DRBG returned bits are four (4) output blocks long.
   - Tests consist of the following operations:
     - Instantiate
     - Generate Random Bits (first call)
     - Generate Random Bits (second call)
     - Uninstantiate

2. **Prediction Resistance NOT ENABLED (with Reseed):**  
   - The DRBG returned bits are four (4) output blocks long.
   - Tests consist of the following operations:
     - Instantiate
     - Reseed
     - Generate Random Bits (first call)
     - Generate Random Bits (second call)
     - Uninstantiate

3. **Prediction Resistance ENABLED:**  
   - The DRBG returned bits are four (4) output blocks long.
   - Tests consist of the following operations:
     - Instantiate
     - Generate Random Bits (first call)
     - Generate Random Bits (second call)
     - Uninstantiate



## Intermediate Value Files

The intermediate value files (with a `.txt` extension) display the working state of the CTR_DRBG after each DRBG operation. For CTR_DRBG, the working state consists of:

- **V:** The variable state.
- **Key:** The AES key used by the DRBG.

Each intermediate file shows the working state values indented by one tab and preceded by a line indicating the DRBG function just performed:
- **INSTANTIATE**
- **(RESEED)** – if applicable
- **GENERATE (FIRST CALL)**
- **GENERATE (SECOND CALL)**

---

## Test Environment

- **Testing Framework:** Tests are executed using [googletest](https://github.com/google/googletest).
- **Tested Application:** These test vectors are used to validate RNGonGPU’s AES-based CTR_DRBG implementation.