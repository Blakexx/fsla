"""Fixture of tools/lint/graph_only.py: each function runs a tool outside the graph, and each must be found."""
import os
import subprocess
import tempfile


def tool(name):
    return name


def compile(source):
    subprocess.run(["nvcc", "-O3", "-c", source], check=True)


def shell_form(source):
    os.system(f"ptxas -arch=sm_90a {source}")


def install():
    subprocess.check_call(["uv", "pip", "install", "numpy"])


def child(path):
    subprocess.run([os.sys.executable, "-m", "pytest", path])


def getter(binary):
    return subprocess.run([tool("cuobjdump"), "-sass", binary], capture_output=True).stdout


def scratch():
    return tempfile.mkdtemp(prefix="fsla_nvcc_")


def via_bash():
    subprocess.run(["bash", "-c", "cd build && ninja -C out"])
