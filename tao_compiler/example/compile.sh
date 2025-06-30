#!/bin/bash

set -e

export TF_CPP_VMODULE=disc_compiler=1,zj_compiler=1

./bin/disc_compiler_main ./module.mlir ./module.so > ./compile.log.mlir 2>&1

echo "DONE"
find ./compile.log.mlir