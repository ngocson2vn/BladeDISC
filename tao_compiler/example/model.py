#!/usr/bin/env python3.11

#==================================================================
# Set necessary env vars
#==================================================================
import os

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RUN_STAGE"] = "runtime"
os.environ["COMPILATION"] = "serving"

os.environ["DYNAMO_CKPT_PATH"] = "./blade_debug/aot_compile_cache"
os.environ["TORCH_COMPILE_USE_LAZY_GRAPH_MODULE"] = "0"
os.environ["PYPILOT_DEPS"] = "/opt/tiger/pypilot"
os.environ["MODEL_NAME"] = "toy_model_v0"

# For debugging
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_BLADE_DUMP_FXGRAPH"] = "1"

# For torch_blade
os.environ["TORCH_BLADE_DEBUG_LOG"] = "1"
os.environ["TORCH_DISC_DUMP_PREFIX"] = "./blade_debug/dump_dir"
os.environ["TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL"] = "s0"
os.environ["TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED"] = "Eq(s0, s1)"
#==================================================================

import pdb
import random

import torch
from torch import nn
from torch.export import Dim
from torch_mlir.fx import export_and_import


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()

    def forward(self, x, y):
        res = torch.multiply(x, y)
        pred = torch.sum(res)
        return pred

toy = ToyModel()

# Convert to Torch-MLIR (requires Torch-MLIR installation)
batch_size = Dim("batch_size")
mlir_module = export_and_import(
    toy,
    x=torch.rand(3, 4), 
    y=torch.rand(3, 4),
    dynamic_shapes={
        "x": {0: batch_size},
        "y": {0: batch_size}
    },
)

output = "./module.mlir"
with open(output, "w") as fd:
    fd.write(str(mlir_module))

print()
mlir_module.dump()
print()

print(output)

# Torch IR -> MHLO
from torch_blade import mlir