# Prerequisites
cuda-12.4
```Bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
```

cudnn-12.x-linux-x64-v8.9.2.26
```Bash
https://developer.nvidia.com/rdp/cudnn-archive
Download https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz/
tar -xf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo rsync -avP cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include ~/.cache/cudnn/
sudo rsync -avP cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib ~/.cache/cudnn/
```

# Build DISC compiler
```Bash
./sony_build_disc.sh
```

# Build runner
```Bash
./sony_build_runner.sh
```

# Example
```Bash
cd example/
./compile.sh
./run.sh
```