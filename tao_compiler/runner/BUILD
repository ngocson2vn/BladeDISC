load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_default_copts",
    "if_cuda",
    "if_cuda_is_configured",
    "cuda_library",
)

cc_binary(
    name = "runner",
    srcs = ["runner.cc"],
    deps = [
	    "//pytorch_blade/compiler/backends:torch_blade_backends",
        "//pytorch_blade/compiler/mlir/runtime:torch_blade_disc_rt",
        "//pytorch_blade/compiler/mlir/runtime:torch_blade_ral_context",
        "@local_org_torch//:libtorch",
        "@com_google_protobuf//:protobuf",
        "@com_github_gflags_gflags//:gflags",
        "@org_disc_compiler//mlir/ral:libral_base_context.so",
        "@org_disc_compiler//mlir/custom_ops:libdisc_custom_ops.so",
        "@org_tensorflow//tensorflow:libtensorflow_framework_import_lib",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_driver",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    copts = ["-DTORCH_BLADE_BUILD_WITH_CUDA"],
    linkopts = ["-ldl"]
)
