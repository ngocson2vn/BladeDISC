#!/bin/bash

export TF_ENABLE_ONEDNN_OPTS=0

cd bin/
./runner --model-dir ..
