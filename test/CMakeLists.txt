# test/CMakeLists.txt
# Only HIP test executable – no CUDA

# Ensure the output directory exists
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test")

add_executable(rngonamdgpu_tests
    test.hip
)
target_compile_features(rngonamdgpu_tests PRIVATE cxx_std_17)
target_link_libraries(rngonamdgpu_tests PRIVATE
    rngonAMDGPU
    hip::host
    roc::rocrand
    gtest_main          # ← provided by thirdparty/ (GoogleTest built in-tree)
)
set_target_properties(rngonamdgpu_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# Optionally register the tests with CTest
# include(GoogleTest)
# gtest_discover_tests(rngonamdgpu_tests)
