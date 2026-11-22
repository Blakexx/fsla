load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files(["bin/cuobjdump", "bin/nvdisasm", "bin/nvcc", "bin/ptxas", "bin/cu++filt"])

#: the pinned toolkit whole, the tree a test tier hands its tests (nvcc reads nvcc.profile, include/, lib64/ and nvvm/
#: beside it); its lib64 is a link to lib, which the glob would list twice
filegroup(name = "toolkit", srcs = glob(["**"], exclude = ["BUILD.bazel", "REPO.bazel", "redistrib.json", "curand/**", "lib64/**"]))

#: cuRAND's headers alone: clang's CUDA front end includes one (its runtime wrapper's `curand_mtgp32_kernel.h`)
filegroup(name = "curand_headers", srcs = glob(["curand/include/**"]))

#: the toolkit's compute-sanitizer, its libraries beside it (the sanitizer lane's)
filegroup(name = "compute_sanitizer", srcs = glob(["compute-sanitizer/**"]))

#: THE COMPILER AND WHAT IT READS (the graph's CUDA compile, bazel/cuda.bzl): nvcc, ptxas, fatbinary, cudafe++ and the crt stubs in
#: `bin/`, cicc and libdevice under `nvvm/`; the headers it includes are `cuda_headers`
filegroup(name = "nvcc_files", srcs = glob(["bin/**", "nvvm/**"]))

#: THE CUDA RUNTIME'S HEADERS, without its libraries: the units' runtime (the library links the runtime itself). The root an
#: installed toolkit's nvcc.profile adds (`include/`) and the root of CCCL's libraries (`include/cccl`)
cc_library(
    name = "cuda_headers",
    hdrs = glob(["include/**"]),
    includes = ["include", "include/cccl"],
)

#: what the library links by path: the runtime by its soname and the driver's stub (the real driver at load)
filegroup(name = "runtime_libs", srcs = ["lib/libcudart.so.13", "lib/stubs/libcuda.so"])

#: the runtime linked statically, for the lab's experiments and the probes
cc_library(
    name = "cuda_runtime_static",
    srcs = ["lib/libcudart_static.a", "lib/libcudadevrt.a"],
    linkopts = ["-ldl", "-lpthread", "-lrt"],
    deps = [":cuda_headers"],
)
