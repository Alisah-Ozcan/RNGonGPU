#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_amdgpu.sh  —  Build RNGonAMDGPU on AMD MI300A
# Target: ROCm 7.2.0 · amdclang 22.0.0-7.2.0
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Load modules (adjust to your cluster's module system) ────────────────────
module load rocm/7.2.0 amdclang/22.0.0-7.2.0

# ── ROCm paths ───────────────────────────────────────────────────────────────
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-$ROCM_PATH}"
export PATH="$ROCM_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ROCM_PATH:${CMAKE_PREFIX_PATH:-}"

# ── GPU target ───────────────────────────────────────────────────────────────
# MI300A = gfx942.  Override with:  GPU_ARCH=gfx940 ./build_amdgpu.sh
GPU_ARCH="${GPU_ARCH:-gfx942}"

# ── Derived paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
INSTALL_DIR="$SCRIPT_DIR/install"

echo "======================================================"
echo " RNGonAMDGPU  —  HIP/ROCm build"
echo " ROCM_PATH : $ROCM_PATH"
echo " GPU_ARCH  : $GPU_ARCH"
echo " Build dir : $BUILD_DIR"
echo "======================================================"

# CMake 3.29+ requires clang directly as HIP compiler — hipcc wrapper is NOT supported.
# amdclang++ is the correct compiler for ROCm 7.x + amdclang/22.0.0.
HIP_COMPILER="$ROCM_PATH/bin/amdclang++"

cmake -S "$SCRIPT_DIR" \
      -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="$HIP_COMPILER" \
      -DCMAKE_HIP_COMPILER="$HIP_COMPILER" \
      -DCMAKE_HIP_ARCHITECTURES="$GPU_ARCH" \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib -lamdhip64"

cmake --build "$BUILD_DIR" --parallel "$(nproc)"
echo ""
echo "✓ Build complete.  Artifacts in: $BUILD_DIR"
echo "  Run tests:  cd $BUILD_DIR && ctest --output-on-failure"
