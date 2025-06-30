# !/bin/bash
set -o pipefail

set -ex

#========================================================================================================================
# For debugging only
#========================================================================================================================
export CUSTOM_DEBUG=ON
#========================================================================================================================

REPO=`pwd`
mkdir -p example/bin

#========================================================================================================================
# Run the following snippet once
#========================================================================================================================
# cd $FRONT_END_DIR
# git submodule update --init --depth=1
# cd third_party/torch-mlir
# patch -p1 < ../../decomp_complex.patch
# cd -

# mkdir -p $HOME/.cache/

# {
#     cd $HOME/.cache/
#     mkdir .blade_tools && cd .blade_tools && ln -s /opt/tiger/typhoon-blade/bpt . && cd ..
#     DEVREGION='us' .blade_tools/bpt/bpt update
#     DEVREGION='us' .blade_tools/bpt/bpt install cudnn:cudnn-12.x-linux-x64-v8.9.2.26 --platform=x86_64-clang1101 --install-directory=cudnn
#     cd -
# }&

# bvc clone gnu/gcc/gcc9 $HOME/.cache/gcc9 -f
#========================================================================================================================

export PATH=$PATH:/usr/local/cuda/bin
export DEVREGION=${BUILD_REGION:-us} 
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda/}
export CUDA_TOOLKIT_PATH="${CUDA_HOME}"
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export TF_CUDA_PATHS="${CUDA_HOME},${HOME}/.cache/cudnn/"
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0,8.6"
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export TORCH_BLADE_BUILD_MLIR_SUPPORT=${TORCH_BLADE_BUILD_MLIR_SUPPORT:-ON}
export TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT=${TORCH_BLADE_BUILD_WITH_CUDA_SUPPORT:-ON}
export TORCH_BLADE_RUN_EXAMPLES=${TORCH_BLADE_RUN_EXAMPLES:-OFF}


export PYTHON=python3.11
export PIP=pip3.11


# pip_install_deps
export PATH=$HOME:$PATH

# bvc clone gnu/gcc/gcc9 $HOME/.cache/gcc9 -f
export PATH=$HOME/.cache/gcc9/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.cache/gcc9/lib:$HOME/.cache/gcc9/lib64:$LD_LIBRARY_PATH

# $(python3.11 -c "import os, torch; print(os.path.dirname(torch.__file__))")
export TORCH_BLADE_TORCH_INSTALL_PATH=/data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch
# $(python3.11 -c "import torch; print(torch.version.cuda)")
export CUDA_VERSION=12.4
# $(python3.11 -c "import torch; print(torch.version.__version__.split('+')[0])")
TORCH_VERSION=2.4.1
TORCH_MAJOR_VERSION=$(echo ${TORCH_VERSION} | cut -d'.' -f1)
TORCH_MINOR_VERSION=$(echo ${TORCH_VERSION} | cut -d'.' -f2)
bazel build //runner:runner --config=cuda --config=torch_cuda --config=torch_debug \
  --noremote_accept_cached \
  --copt=-DPYTORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION} \
  --copt=-DPYTORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}


rsync -arvP runner/run.sh example/

mkdir -p example/bin
rsync -arvP bazel-bin/runner/runner example/bin/

rsync -arvP bazel-bin/mlir/ral/libral_base_context.so        example/bin/
rsync -arvP bazel-bin/mlir/custom_ops/libdisc_custom_ops.so  example/bin/

rsync -arvP /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/lib/libc10.so               example/bin/
rsync -arvP /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/lib/libc10_cuda.so          example/bin/
rsync -arvP /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/lib/libtorch.so             example/bin/
rsync -arvP /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so         example/bin/
rsync -arvP /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so        example/bin/
rsync -arvP /data00/home/son.nguyen/.pyenv/versions/3.11.2/lib/python3.11/site-packages/torch/lib/libtorch_global_deps.so example/bin/

rsync -rvLP bazel-bin/external/org_tensorflow/tensorflow/libtensorflow_framework.so.2 example/bin/libtensorflow_framework.so.2