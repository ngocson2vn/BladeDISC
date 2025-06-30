#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>

#include "pytorch_blade/compiler/backends/engine_interface.h"
#include "pytorch_blade/compiler/mlir/runtime/disc_engine.h"

#include <torch/torch.h>

#include <gflags/gflags.h>

DEFINE_string(model_dir, "", "model directory");

using namespace torch::blade::backends;

std::string ReadFile(const std::string& file_path) {
  std::unique_ptr<char> buffer;
  std::ifstream reader(file_path, std::ios::binary);

  // Get the number of bytes
  reader.seekg(0, reader.end);
  int length = reader.tellg();
  reader.seekg(0, reader.beg);

  // Allocate buffer
  buffer.reset(new char[length]);

  // Read data to buffer
  reader.read(buffer.get(), length);
  reader.close();

  return std::string(buffer.get(), length);
}

at::List<at::Tensor> CreateSampleInputs(const std::vector<TensorInfo>& tensorInfoVec) {
  at::List<at::Tensor> inputs;
  for (const auto& tensorInfo : tensorInfoVec) {
    inputs.push_back(
      torch::rand(tensorInfo.sizes, torch::TensorOptions().dtype(tensorInfo.scalar_type).device(tensorInfo.device))
    );
  }

  return inputs;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_dir.empty()) {
    std::cerr << "ERROR: model_dir is empty!" << std::endl;
    return 1;
  }

  // Read engine data and metadata
  std::string module_path = FLAGS_model_dir + "/module.so";
  std::string pbtxt_path = module_path + ".pbtxt";
  std::cout << "module_path: " << module_path << std::endl;
  std::cout << "pbtxt_path: " << pbtxt_path << std::endl;

  int numInputs = 2;
  int numOutputs = 1;

  // Initialize backend
  torch::blade::disc::InitBladeDiscEngine();

  // Create engine
  EngineState state;
  state.engine_bytes = ReadFile(module_path);
  state.model_proto = ReadFile(pbtxt_path);
  state.backend_name = torch::blade::disc::GetBackendName();

  for (int i = 0; i < numInputs; i++) {
    auto arg = TensorInfo();
    arg.name = "arg" + std::to_string(i);
    arg.device = "cpu";
    arg.sizes = {100, 5};
    arg.scalar_type = torch::kFloat32;
    state.inputs.push_back(arg);
  }

  for (int i = 0; i < numOutputs; i++) {
    auto out = TensorInfo();
    out.name = "out" + std::to_string(i);
    out.device = "cpu";
    out.scalar_type = torch::kFloat32;
    out.sizes = {};
    state.outputs.push_back(out);
  }

  printf("Engine state:\n");
  printf("  - engine_bytes size: %lu\n", state.engine_bytes.size());
  printf("  - model_proto size: %lu\n", state.model_proto.size());
  printf("  - backend_name: %s\n", state.backend_name.c_str());

  auto engine = EngineInterface::CreateEngine(state);
  printf("Engine addr: %p\n", engine.get());

  // Create inputs
  at::List<at::Tensor> inputs = CreateSampleInputs(state.inputs);

  // Execute engine
  auto outputs = engine->Execute(inputs);
  std::cout << "\nOutput size: " << outputs.size() << std::endl;
  for (const auto& tensor : outputs) {
    std::cout << tensor << std::endl;
  }

  return 0;
}