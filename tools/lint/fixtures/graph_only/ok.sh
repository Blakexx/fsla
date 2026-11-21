#!/bin/bash
# nvcc, ptxas and make named in a comment
echo "nvcc ptxas make"
bazel run //tools:lab -- sass nvcc
grep nvcc docs.md
