load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

#: the interpreter the graph's actions run: the environment's base Python (a link it resolves to find its standard
#: library), standard library only
filegroup(name = "python", srcs = ["bin/python3"])


cc_library(
    name = "torch_headers",
    hdrs = glob(["site-packages/torch/include/**"]),
    includes = ["site-packages/torch/include", "site-packages/torch/include/torch/csrc/api/include"],
)

#: the torch libraries the extension links (its NEEDED)
filegroup(
    name = "torch_libraries",
    srcs = ["site-packages/torch/lib/libc10.so", "site-packages/torch/lib/libtorch_cpu.so"],
)

#: the libraries torch's own need beside them (libtorch_cpu's NEEDED: the OpenMP runtime its wheel carries, and CUPTI and
#: the CUDA runtime from the CUDA wheel), which an executable linked against torch's libraries resolves at its link
filegroup(
    name = "torch_runtime_libraries",
    srcs = [
        "site-packages/nvidia/cu13/lib/libcudart.so.13",
        "site-packages/nvidia/cu13/lib/libcupti.so.13",
        "site-packages/torch/lib/libgomp.so.1",
    ],
)

cc_library(
    name = "python_headers",
    hdrs = glob(["python_include/**"]),
    includes = ["python_include"],
)
