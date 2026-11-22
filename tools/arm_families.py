"""THE ONE READER OF `ARM_FAMILIES`: the registry of the arm families this tree ships (`bazel/arms.bzl`, a literal read by
AST). The lint and the measure provider both call it, so the registry has one reader and one refusal text."""
from __future__ import annotations

import ast
from pathlib import Path


def arm_families(root: Path) -> list[str]:
    """The families `ARM_FAMILIES` declares in `<root>/bazel/arms.bzl`, in declared order."""
    for node in ast.parse((root / "bazel" / "arms.bzl").read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ARM_FAMILIES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise SystemExit("bazel/arms.bzl declares no `ARM_FAMILIES`")
