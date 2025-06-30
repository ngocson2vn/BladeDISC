set -ve

#======================================================================
# Prerequisites
#======================================================================
# 1. Patch tf_community
# cd ../tf_community
# git apply ../patches/tf.patch

# 2. Create synlinks
# ln -sf ../pytorch_blade/bazel .
# ln -sf ../pytorch_blade/pytorch_blade .
# ln -sf ../pytorch_blade/tests .
#======================================================================

export BAZEL_CACHE_DIR=${HOME}/.cache/bazel

# SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# echo "SCRIPT_DIR: ${SCRIPT_DIR}"

# git submodule update --init --recursive ${SCRIPT_DIR}/tf_community
# git submodule update --init --recursive ${SCRIPT_DIR}/torch/mlir/disc/cutlass

# patch and create symbolic link to build compiler
# cd ${SCRIPT_DIR}/tf_community

# for rebuild.
# !!! Run the first time only!!!
# set +e
# patch -p1 < ${SCRIPT_DIR}/bazel/tf.patch
# cd tensorflow/compiler/mlir
# ln -sf ${SCRIPT_DIR}/torch/mlir/ral ral
# ln -sf ${SCRIPT_DIR}/torch/mlir/disc disc
# set -e


#bazel build //torch_zj/tools:mlir-opt
# cd ${SCRIPT_DIR}/torch && bazel build //mlir/disc:disc-opt

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4/}
export CUDA_TOOLKIT_PATH="${CUDA_HOME}"
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export TF_CUDA_PATHS="${CUDA_HOME},${HOME}/.cache/cudnn/"
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0,8.6"
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

bazel build //mlir/disc:disc_compiler_main --config=cuda

mkdir -p ./example/bin
cp -fv bazel-bin/mlir/disc/disc_compiler_main ./example/bin