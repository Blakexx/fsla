"""The action hook `.claude/hooks/graph_only.py` refuses a toolchain or install command typed in a checkout and names the
graph target that runs it; a `bazel` command, a quoted word, a pipe's argument and the override pass."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".claude" / "hooks" / "graph_only.py"


def run(command):
    call = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}
    ran = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(call), capture_output=True, text=True, check=True)
    return json.loads(ran.stdout)["hookSpecificOutput"] if ran.stdout.strip() else None


@pytest.mark.parametrize("command,target", [
    ("nvcc -arch=sm_90a x.cu", "//:library"),
    ("/opt/cuda/bin/cuobjdump -sass f", "//:static_manifest"),
    ("ptxas -arch=sm_90a x.ptx", "//csrc/fsla:<unit>_sass"),
    ("nvdisasm f.cubin", "//:static_manifest"),
    ("ncu --metric m ./x", "//measure/lab:<name>.run -- ncu"),
    ("nsys profile ./x", "//measure/lab:<name>.run"),
    ("compute-sanitizer ./x", "//tests:sanitizer"),
    ("cmake -B build", "//:library"),
    ("cd build && make -j8", "//:library"),
    ("ninja -C build", "//:library"),
    ("pip install x", "//:dev_env"),
    ("uv pip install x", "//:dev_env"),
    ("python3 -m pip install -U x", "//:dev_env"),
    ("CUDA_VISIBLE_DEVICES=0 sudo ncu ./x", "//measure/lab:<name>.run -- ncu"),
    ("echo ok; /usr/local/cuda/bin/nvcc x.cu", "//:library"),
    ("x=$(nvcc --version)", "//:library"),
    ("git worktree add ../w origin/master", "bazel --worktree add"),
    ("git -C /tmp/repo worktree add -b x ../w HEAD", "bazel --worktree add"),
    ("cd r && git worktree remove --force ../w", "bazel --worktree remove"),
])
def test_a_toolchain_or_install_command_is_refused_with_its_target(command, target):
    out = run(command)
    assert out["permissionDecision"] == "deny" and target in out["permissionDecisionReason"]
    assert "GRAPH_OVERRIDE" in out["permissionDecisionReason"]


@pytest.mark.parametrize("command", [
    "bazel run //measure/lab:x.run -- pack",
    "bazel run //measure/lab:x.run -- ncu --metric m",
    "bazelisk test //tests:sanitizer -- nvcc",
    'echo "nvcc is a compiler"',
    "git log | grep nvcc",
    "grep -rn 'pip install' docs",
    "cat > f.txt <<'EOF'\nnvcc x.cu\nEOF",
    'GRAPH_OVERRIDE="probe" nvcc -arch=sm_90a x.cu',
    "nvcc --version  # GRAPH_OVERRIDE=probe",
    "pip list",
    "python3 -m pytest tests/unit",
    "ls /opt/cuda/bin/nvcc-notes",
    "git commit -m 'make it so'",
    "git worktree list",
    "git worktree prune",
    "bazel --worktree add ../w HEAD",
])
def test_a_graph_command_text_or_override_passes(command):
    assert run(command) is None
