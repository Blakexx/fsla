// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE HOST TOOLCHAIN'S PROBE (`//tests:host_abi`, tools/host_abi.py): the compiler, the C library and the C++ library the
// graph's one C++ toolchain compiled this with, as the macros its own headers define, one `name value` a line.
#include <cstdio>
#include <string>

int main() {
    std::printf("compiler %s\n", __VERSION__);
    std::printf("glibc %d.%d\n", __GLIBC__, __GLIBC_MINOR__);
    std::printf("libstdc++ %d\n", _GLIBCXX_RELEASE);
    std::printf("cxx11_abi %d\n", _GLIBCXX_USE_CXX11_ABI);
    return std::string("probe").size() == 5 ? 0 : 1;
}
