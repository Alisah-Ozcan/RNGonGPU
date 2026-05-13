#!/bin/bash
# Build script with automatic license acceptance

set -e

cd /shared/prerelease/home/cineca/nshukla/HPCTrainingExamples/HIPIFY/RNGonAMDGPU

# Run cmake to reconfigure if needed
cmake -B build \
  -DCMAKE_C_COMPILER=/shared/apps/ubuntu/opt/rocm-7.2.0/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/shared/apps/ubuntu/opt/rocm-7.2.0/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/shared/apps/ubuntu/opt/rocm-7.2.0/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON \
  -DRNGONAMDGPU_BUILD_BENCHMARKS=OFF \
  -DRNGONAMDGPU_BUILD_EXAMPLES=OFF \
  -DRNGONAMDGPU_BUILD_TESTS=OFF << 'EOF'
yes
EOF

# Build the project
cd /shared/prerelease/home/cineca/nshukla/HPCTrainingExamples/HIPIFY/RNGonAMDGPU/build
make -j$(nproc) << 'EOF'
yes
EOF

echo "Build completed successfully!"
