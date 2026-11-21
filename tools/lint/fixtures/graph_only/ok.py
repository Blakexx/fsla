"""Fixture of tools/lint/graph_only.py: the same tools named as text, never run -- every line must pass.

nvcc, ptxas and `pip install` named in a docstring.
"""
import subprocess

# subprocess.run(["nvcc", "-c", "x.cu"]) in a comment
TOOLS = ["nvcc", "ptxas", "cuobjdump"]


def named():
    assert "nvcc" in "nvcc is a compiler"
    print("run nvcc by hand: no")
    subprocess.run(["git", "status"], check=True)
    subprocess.run(["bazel", "run", "//tools:lab", "--", "sass", "nvcc"], check=True)
    subprocess.run("echo nvcc ptxas | cat", shell=True, check=True)
