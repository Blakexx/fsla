#!/bin/bash
# nvcc, ptxas and make named in a comment
echo "nvcc ptxas make"
bazel run //measure/lab:nvcc.run -- sass
grep nvcc docs.md
