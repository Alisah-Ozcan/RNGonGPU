# example/CMakeLists.txt
# Only HIP executables – no CUDA

# Ensure the output directory exists before linking
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/example")

# AES DRNG example
add_executable(aes_drng_example
    example_aes_drng.hip
)
target_compile_features(aes_drng_example PRIVATE cxx_std_17)
target_link_libraries(aes_drng_example PRIVATE
    rngonAMDGPU
    hip::host
    roc::rocrand
)
set_target_properties(aes_drng_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
)

# HIP RNG example
add_executable(hip_rng_example
    example_hip_rng.hip
)
target_compile_features(hip_rng_example PRIVATE cxx_std_17)
target_link_libraries(hip_rng_example PRIVATE
    rngonAMDGPU
    hip::host
    roc::rocrand
)
set_target_properties(hip_rng_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
)
