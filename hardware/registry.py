# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT REGISTRY: the schema of every fact, the composable parts, the descriptors that compose them, and their expansion to
full run-time contexts (docs/internals/hardware/registry.md). A context is the contract's unit: every fact, each a point, a range or a set.
The matchers and the run time see contexts only; a part or a composition never leaves this module.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parent

#: how a value is known: the lab record that measured it, the test range that validated it, the document that states it, the
#: declaration that makes it the contract
PROVENANCE = ("measured_by", "validated_by", "documented_by", "declared_by")
READS = ("DEVICE", "CALL")
#: the folder of the device parts: a device part states an architecture whole (`arch`, `family` and the device's facts)
DEVICE = "device/"
#: the contract a build compiles against when it names none: every descriptor of `descriptors/`. A development build names a NARROWED contract
#: instead (`descriptors/dev/<name>.json`, marked `"narrowed": true`)
FULL = "full"
#: the folder of the OWED descriptors (`descriptors/owed/<name>.json`): a call the contract does not guarantee yet because the kernels it waits on are not built.
#: It composes parts and is validated as any descriptor is, and is never in the full contract, a narrowed one, the README's table or the arms a build compiles
OWED = "descriptors/owed"
#: the suffix of a part the device probe wrote (`hardware/parts/device/<id>.candidate.json`): a candidate is never a part
CANDIDATE = ".candidate"
TYPES = ("int", "set<enum>")


class RegistryError(ValueError):
    """A schema, part, descriptor or context the registry refuses."""


@dataclass(frozen=True)
class Fact:
    name: str
    type: str
    unit: str
    read: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class Context:
    """One supported run-time context: `descriptor` composed it, `facts` holds every schema fact (an int as (lo, hi), a set as
    its sorted members), `alternatives` names the facts whose choice made this context one of several, `arch` is the architecture of the descriptor's
    one device part."""
    descriptor: str
    facts: dict
    alternatives: tuple[str, ...]
    gpus: str
    arch: str

    @property
    def id(self) -> str:
        chosen = " ".join(f"{n}={_spell(self.facts[n])}" for n in self.alternatives)
        return f"{self.descriptor} {chosen}".rstrip()

    def key(self) -> tuple:
        return tuple((n, self.facts[n]) for n in sorted(self.facts))


def _spell(value) -> str:
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return str(value[0]) if value[0] == value[1] else f"{value[0]}..{value[1]}"
    return "{" + ",".join(value) + "}"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RegistryError(f"{path}: {e}") from None


def schema(root: Path = ROOT) -> dict[str, Fact]:
    """The schema's facts by name, each refused when malformed."""
    path = root / "schema.json"
    out: dict[str, Fact] = {}
    for f in _read(path).get("facts", []):
        name = f.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise RegistryError(f"{path}: a fact's name is an identifier, not {name!r}")
        if name in out:
            raise RegistryError(f"{path}: fact {name} is declared twice")
        if f.get("type") not in TYPES:
            raise RegistryError(f"{path}: fact {name} has type {f.get('type')!r}, which is not one of {TYPES}")
        if f.get("read") not in READS:
            raise RegistryError(f"{path}: fact {name} is read from {f.get('read')!r}, which is not one of {READS}")
        if not isinstance(f.get("unit"), str) or not f["unit"]:
            raise RegistryError(f"{path}: fact {name} names no unit")
        members = tuple(f.get("members", ()))
        if (f["type"] == "set<enum>") != bool(members) or len(set(members)) != len(members) or len(members) > 64:
            raise RegistryError(f"{path}: fact {name}: a set<enum> fact names its 1..64 distinct members, and no other fact has any")
        out[name] = Fact(name, f["type"], f["unit"], f["read"], members)
    if not out:
        raise RegistryError(f"{path}: the schema declares no fact")
    return out


def _normalize(fact: Fact, value, where: str):
    """One alternative of a fact's value: an int as (lo, hi), a set as its sorted members."""
    if fact.type == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return (value, value)
        if isinstance(value, dict) and set(value) == {"lo", "hi"} and all(isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
            if value["lo"] > value["hi"]:
                raise RegistryError(f"{where}: range {value} has lo above hi")
            return (value["lo"], value["hi"])
        raise RegistryError(f"{where}: {fact.name} is an int: a point, {{\"lo\", \"hi\"}} or a list of them (not {value!r})")
    if isinstance(value, dict) and set(value) == {"set"} and isinstance(value["set"], list):
        bad = [m for m in value["set"] if m not in fact.members]
        if bad:
            raise RegistryError(f"{where}: {fact.name} has no member {bad}; its members are {list(fact.members)}")
        if len(set(value["set"])) != len(value["set"]):
            raise RegistryError(f"{where}: {fact.name} names a member twice")
        return tuple(sorted(value["set"]))
    raise RegistryError(f"{where}: {fact.name} is a set<enum>: {{\"set\": [members]}} or a list of them (not {value!r})")


def part(name: str, root: Path = ROOT) -> dict:
    """A part by its name (`device/hopper_sm90a`), refused when malformed; the facts it states are checked against the schema by `expand`."""
    path = root / "parts" / f"{name}.json"
    if name.endswith(CANDIDATE):
        raise RegistryError(f"{path}: a CANDIDATE part, written by the device probe (`//measure/lab:device_probe`): a reviewer reads it and renames it "
                            f"to {name[:-len(CANDIDATE)]}.json, which admits it; until then the registry refuses it")
    if not path.is_file():
        raise RegistryError(f"no part {name!r}: {path} does not exist")
    data = _read(path)
    if data.get("name") != name:
        raise RegistryError(f"{path}: the part's name is {data.get('name')!r}, its path says {name!r}")
    facts = data.get("facts")
    if not isinstance(facts, dict) or not facts:
        raise RegistryError(f"{path}: a part states facts")
    for fname, record in facts.items():
        if not isinstance(record, dict) or "value" not in record:
            raise RegistryError(f"{path}: fact {fname} is {{\"value\", <provenance>}}")
        cited = [k for k in PROVENANCE if k in record]
        if len(cited) != 1 or set(record) != {"value", cited[0]} or not isinstance(record[cited[0]], str) or not record[cited[0]].strip():
            raise RegistryError(f"{path}: fact {fname} cites exactly one of {PROVENANCE} as a non-empty string, beside its value")
    return data


def parts(root: Path = ROOT) -> list[str]:
    base = root / "parts"
    return sorted(str(p.relative_to(base).with_suffix("")) for p in base.rglob("*.json"))


def descriptors(root: Path = ROOT, folder: str = "descriptors") -> dict[str, dict]:
    out = {}
    for path in sorted((root / folder).glob("*.json")):
        data = _read(path)
        if data.get("name") != path.stem:
            raise RegistryError(f"{path}: the descriptor's name is {data.get('name')!r}, its file says {path.stem!r}")
        if not data.get("compose") or not isinstance(data["compose"], list):
            raise RegistryError(f"{path}: a descriptor composes at least one part")
        if not isinstance(data.get("gpus"), str) or not data["gpus"]:
            raise RegistryError(f"{path}: a descriptor names the GPUs it covers (`gpus`), the label the README's table prints")
        out[path.stem] = data
    return out


def owed_descriptors(root: Path = ROOT) -> dict[str, dict]:
    """The OWED descriptors by name (`descriptors/owed/<name>.json`), each naming the kernels its call waits on (`"owed": ["<kernel>", ...]`, the names
    `fsla.ops._ext.not_built` and the tests' `requires_kernel` use), or a RegistryError."""
    found = descriptors(root, OWED)
    for name, data in found.items():
        kernels = data.get("owed")
        if (not isinstance(kernels, list) or not kernels or len(set(kernels)) != len(kernels)
                or not all(isinstance(k, str) and k.isidentifier() for k in kernels)):
            raise RegistryError(f"{root / OWED / (name + '.json')}: an owed descriptor names the kernels its call waits on (`\"owed\": [\"prefill\", ...]`, "
                                "distinct identifiers)")
    return found


def expand(descriptor: dict, facts: dict[str, Fact], root: Path = ROOT) -> list[Context]:
    """A descriptor's full contexts: its parts' facts merged (a fact stated twice, a schema fact left out, or a fact the schema does not name is
    refused), a list of alternatives crossed into one context each, in the order the facts are listed."""
    stated: dict[str, list] = {}
    for name in descriptor["compose"]:
        for fname, record in part(name, root)["facts"].items():
            where = f"descriptor {descriptor['name']}, part {name}"
            if fname not in facts:
                raise RegistryError(f"{where}: {fname} is no fact of the schema")
            if fname in stated:
                raise RegistryError(f"{where}: {fname} is stated by another part of the descriptor too")
            value = record["value"]
            stated[fname] = [_normalize(facts[fname], v, where) for v in (value if isinstance(value, list) else [value])]
            if not stated[fname]:
                raise RegistryError(f"{where}: {fname} lists no alternative")
    missing = [n for n in facts if n not in stated]
    if missing:
        raise RegistryError(f"descriptor {descriptor['name']} states no value for {missing}: a context holds every fact of the schema")
    devices = [n for n in descriptor["compose"] if n.startswith(DEVICE)]
    if len(devices) != 1:
        raise RegistryError(f"descriptor {descriptor['name']} composes {len(devices)} device parts {devices}: a descriptor is built for the one architecture its device part states")
    arch = part(devices[0], root).get("arch")
    alternatives = tuple(n for n in facts if len(stated[n]) > 1)
    names = list(facts)
    return [Context(descriptor["name"], dict(zip(names, combo)), alternatives, descriptor["gpus"], arch)
            for combo in itertools.product(*(stated[n] for n in names))]


@dataclass(frozen=True)
class Architecture:
    """One architecture as its device part states it: the build's name (`sm_90a`), the device's compute capability, regs and instruction capabilities and
    the family."""
    name: str
    part: str
    cc: int
    family: str
    regs_per_sm: int
    instructions: tuple[str, ...]


def cc_of(arch: str) -> int:
    """An architecture's compute capability x 10: `sm_90a` (built with its architecture-specific features) 900, as `__CUDA_ARCH__` reads there."""
    return int(arch.removeprefix("sm_").rstrip("a")) * 10


def _point(data: dict, fname: str, path: Path) -> int:
    value = data["facts"].get(fname, {}).get("value")
    if not isinstance(value, int) or isinstance(value, bool):
        raise RegistryError(f"{path}: a device part states {fname} as one point, not {value!r}")
    return value


def architecture(name: str, root: Path = ROOT) -> Architecture:
    """The architecture the device part `name` states, or a RegistryError: a part names its `arch` and `family`, states `cc` (the compute capability its `arch` names),
    `regs_per_sm` and `instructions`, each as one point or set."""
    data = part(name, root)
    path = root / "parts" / f"{name}.json"
    arch = data.get("arch")
    if not isinstance(arch, str) or not re.fullmatch(r"sm_\d+a?", arch):
        raise RegistryError(f"{path}: a device part names the architecture it is built as (`arch`: `sm_90a`)")
    family = data.get("family")
    if not isinstance(family, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", family):
        raise RegistryError(f"{path}: a device part names its family (`family`: `hopper`)")
    cc, regs = (_point(data, f, path) for f in ("cc", "regs_per_sm"))
    if cc != cc_of(arch):
        raise RegistryError(f"{path}: cc is {cc}, which is not the compute capability {arch} names ({cc_of(arch)})")
    held = data["facts"].get("instructions", {}).get("value")
    if not isinstance(held, dict) or set(held) != {"set"}:
        raise RegistryError(f"{path}: a device part states instructions as one {{\"set\": [members]}}")
    return Architecture(arch, name, cc, data["family"], regs, tuple(held["set"]))


def architectures(root: Path = ROOT) -> list[Architecture]:
    """The architectures the device parts state, by compute capability, each once."""
    found = [architecture(n, root) for n in parts(root) if n.startswith(DEVICE)]
    for a in found:
        twin = next((b for b in found if b is not a and b.name == a.name), None)
        if twin is not None:
            raise RegistryError(f"parts {a.part} and {twin.part} both state the architecture {a.name}: an architecture is stated once")
    return sorted(found, key=lambda a: a.cc)


def device_point(name: str = "device/hopper_sm90a", root: Path = ROOT) -> dict[str, int]:
    """The integer facts of a device part as one device states them, in the schema's order: a point as it is, a range at its upper end (the largest device the part
    admits). The synthetic devices of the tests are made from it."""
    facts = schema(root)
    data = part(name, root)["facts"]
    return {n: _normalize(f, data[n]["value"], name)[1] for n, f in facts.items() if f.read == "DEVICE" and f.type == "int" and n in data}


def built(root: Path = ROOT) -> list[str]:
    """The BUILT architectures, by compute capability: those a device part states (docs/internals/hardware/registry.md#architectures). The one
    statement of which architectures the graph builds by default (`bazel/flags.py`'s `BUILT_ARCHS`, `--//:archs`)."""
    return [a.name for a in architectures(root)]


def arch_of(cc: int, root: Path = ROOT) -> str:
    """The architecture a compute capability x 10 names (900 `sm_90a`): the one map back from `cc_of`."""
    for a in architectures(root):
        if a.cc == cc:
            return a.name
    raise SystemExit(f"compute capability {cc // 100}.{cc // 10 % 10} has no device part in the registry: an unbuilt architecture")


def device_arch(major: int, minor: int, root: Path = ROOT) -> str:
    """A device's architecture by its compute capability (9, 0 `sm_90a`)."""
    return arch_of((10 * major + minor) * 10, root)


def builds_header(archs: list[str], root: Path = ROOT) -> str:
    """`build/generated/builds.inc`: the builds a binary of these architectures (`["90a"]`) carries, one
    each -- the set `common/select.cuh`'s `build_switch` branches over (docs/internals/common/select.md#carried)."""
    found = architectures(root)
    want = {f"sm_{a}" for a in archs}
    unknown = sorted(want - {a.name for a in found})
    if unknown:
        raise SystemExit(f"builds: {', '.join(unknown)} has no device part in the registry")
    cases = " ".join(f"F({a.cc})" for a in found if a.name in want)
    return ("// GENERATED by hardware/registry.py (builds_header) -- do not edit, regenerate.\n"
            "// THE BUILDS THIS BINARY CARRIES. see docs/internals/common/select.md#carried\n"
            "#pragma once\n"
            f"#define FSLA_BUILDS(F) {cases}\n")


def narrowed(name: str, root: Path = ROOT) -> dict:
    """A DECLARED NARROWED CONTRACT by name (`descriptors/dev/<name>.json`): a descriptor of its own, marked `"narrowed": true`, whose every
    context lies inside a context of the full contract (a narrowed contract promises less, never more), or a RegistryError."""
    found = descriptors(root, "descriptors/dev")
    if name not in found:
        raise RegistryError(f"no narrowed contract {name!r}: the declared ones are {sorted(found) or 'none'} (`descriptors/dev/<name>.json`); "
                            f"`--//:contract={FULL}` is the full contract")
    descriptor = found[name]
    if descriptor.get("narrowed") is not True:
        raise RegistryError(f"descriptors/dev/{name}.json is not marked narrowed (`\"narrowed\": true`): only a declared narrowed contract may "
                            "narrow a build")
    facts = schema(root)
    held = contexts(root)
    for context in expand(descriptor, facts, root):
        if not any(_inside(context, whole, facts) for whole in held):
            raise RegistryError(f"narrowed contract {name}: its context '{context.id}' lies outside the full contract: a narrowed contract may "
                                "promise less than the full one, never more")
    return descriptor


def _inside(context: Context, whole: Context, facts: dict[str, Fact]) -> bool:
    for name, fact in facts.items():
        a, b = context.facts[name], whole.facts[name]
        if fact.type == "int":
            if a[0] < b[0] or a[1] > b[1]:
                return False
        elif not set(a) <= set(b):
            return False
    return True


def _reaches(a: Context, b: Context, facts: dict[str, Fact]) -> bool:
    """Whether some call lies in both contexts: every int fact's ranges meet, every set equal."""
    for name, fact in facts.items():
        x, y = a.facts[name], b.facts[name]
        if fact.type == "int":
            if x[1] < y[0] or y[1] < x[0]:
                return False
        elif x != y:
            return False
    return True


def owed(root: Path = ROOT) -> list[tuple[Context, tuple[str, ...]]]:
    """Every context of every OWED descriptor with the kernels it waits on, refused when a call of it is also in the full contract (a call is held by
    the contract or owed, never both: the run time refuses an owed call naming its kernels, which a guaranteed call must not meet)."""
    facts = schema(root)
    held = contexts(root)
    out: list[tuple[Context, tuple[str, ...]]] = []
    for name, descriptor in owed_descriptors(root).items():
        for context in expand(descriptor, facts, root):
            twin = next((whole for whole in held if _reaches(context, whole, facts)), None)
            if twin is not None:
                raise RegistryError(f"owed descriptor {name}: its context '{context.id}' is reachable from the full contract's '{twin.id}': a call is "
                                    "guaranteed or owed, never both")
            out.append((context, tuple(descriptor["owed"])))
    return out


def contexts(root: Path = ROOT, contract: str = FULL) -> list[Context]:
    """The contract a build compiles against: every context of every descriptor (`contract` the full one), or of the one narrowed contract it
    names; refused when two are identical in every fact."""
    if contract != FULL:
        facts = schema(root)
        return expand(narrowed(contract, root), facts, root)
    facts = schema(root)
    refused = [n for n in parts(root) if n.endswith(CANDIDATE)]
    if refused:
        part(refused[0], root)
    out: list[Context] = []
    seen: dict[tuple, Context] = {}
    for descriptor in descriptors(root).values():
        for context in expand(descriptor, facts, root):
            twin = seen.get(context.key())
            if twin is not None:
                raise RegistryError(f"contexts {twin.id!r} and {context.id!r} are identical in every fact: the contract holds a context once")
            seen[context.key()] = context
            out.append(context)
    if not out:
        raise RegistryError("the registry holds no descriptor: nothing is supported")
    return out


def check(root: Path = ROOT) -> None:
    """The schema, every part (composed or not), every architecture a device part states and every descriptor (the full contract's, the narrowed ones and the
    owed ones), the contract's duplicates and the owed contexts' overlap with the contract, or a RegistryError."""
    facts = schema(root)
    for name in parts(root):
        for fname in part(name, root)["facts"]:
            if fname not in facts:
                raise RegistryError(f"part {name}: {fname} is no fact of the schema")
    for cap, _, source, _ in CAPS:
        if source == "instruction" and cap not in facts["instructions"].members:
            raise RegistryError(f"the generated Caps holds {cap}, which is no member of the schema's `instructions`")
    contexts(root)
    if not architectures(root):
        raise RegistryError("the registry holds no device part: no architecture is built")
    for name in descriptors(root, "descriptors/dev"):
        narrowed(name, root)
    owed(root)


def contexts_text(contexts_: list[Context], contract: str = FULL) -> str:
    """The contract as `arms_gen` reads it: a first line naming the contract (`# contract: full`), then a line a context, tab-separated: its id,
    then `fact=lo:hi` or `fact={member,...}`."""
    lines = [f"# contract: {contract}"]
    for c in contexts_:
        cells = [c.id]
        for name, value in c.facts.items():
            cells.append(f"{name}={value[0]}:{value[1]}" if isinstance(value[0], int) else f"{name}={{{','.join(value)}}}")
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def facts_header(facts: dict[str, Fact]) -> str:
    """`plan/facts.h`: the `Facts` struct with one member per fact (an `Interval` or an `EnumSet`), the enum of each set fact, and the name lookups the
    contract's reader uses."""
    out = ["// GENERATED from hardware/schema.json by hardware/emit.py -- do not edit. NEVER CHECKED IN.\n",
           "#pragma once\n\n#include <string_view>\n\n#include \"plan/rules.h\"\n\nnamespace fsla::plan {\n\n"]
    for f in facts.values():
        if f.type == "set<enum>":
            out.append(f"enum class {_camel(f.name)} : int {{ {', '.join(f.members)} }};\n")
    out.append("\n//: the contract's facts, one member each (an Interval or an EnumSet). A context is WRITTEN through `assign`; it is READ only by the\n")
    out.append("//: functions of common/plan/, through `Reader`: no launcher or kernel reads a raw fact\n")
    out.append("struct Reader;\n\nclass Facts {\n public:\n")
    out.append("  constexpr bool assign(std::string_view fact, Interval v) {\n")
    for f in facts.values():
        if f.type == "int":
            out.append(f"    if (fact == \"{f.name}\") {{ {f.name}_ = v; return true; }}\n")
    out.append("    return false;\n  }\n\n")
    out.append("  constexpr bool assign(std::string_view fact, EnumSet v) {\n")
    for f in facts.values():
        if f.type == "set<enum>":
            out.append(f"    if (fact == \"{f.name}\") {{ {f.name}_ = v; return true; }}\n")
    out.append("    return false;\n  }\n\n private:\n  friend struct Reader;\n\n")
    for f in facts.values():
        out.append(f"  {'Interval' if f.type == 'int' else 'EnumSet'} {f.name}_{{}};  // {f.unit}, read from the {f.read}\n")
    out.append("};\n\n//: the one reader of a context's facts; only the functions of common/plan/ use it\n")
    out.append("struct Reader {\n")
    for f in facts.values():
        out.append(f"  static constexpr {'Interval' if f.type == 'int' else 'EnumSet'} {f.name}(const Facts& f) {{ return f.{f.name}_; }}\n")
    for kind in ("Interval", "EnumSet"):
        out.append(f"  static constexpr bool get(const Facts& f, std::string_view fact, {kind}& out) {{\n")
        for f in facts.values():
            if (f.type == "int") == (kind == "Interval"):
                out.append(f"    if (fact == \"{f.name}\") {{ out = f.{f.name}_; return true; }}\n")
        out.append("    return false;\n  }\n")
    out.append("};\n\n")
    out.append("inline constexpr std::string_view kFactNames[] = {" + ", ".join(f'"{n}"' for n in facts) + "};\n\n")
    out.append("inline constexpr std::string_view kCallFactNames[] = {" + ", ".join(f'"{f.name}"' for f in facts.values() if f.read == "CALL") + "};\n\n")
    out.append("//: whether the CALL supplies the fact at run time (the DEVICE's facts are read from the device)\n")
    out.append("constexpr bool is_call(std::string_view fact) {\n")
    for f in facts.values():
        if f.read == "CALL":
            out.append(f"  if (fact == \"{f.name}\") return true;\n")
    out.append("  return false;\n}\n\n")
    out.append("constexpr bool is_set(std::string_view fact) {\n")
    for f in facts.values():
        if f.type == "set<enum>":
            out.append(f"  if (fact == \"{f.name}\") return true;\n")
    out.append("  return false;\n}\n\n")
    out.append("//: the index of a set fact's member, -1 when the fact or the member is unknown\n")
    out.append("constexpr int member_of(std::string_view fact, std::string_view member) {\n")
    for f in facts.values():
        for i, m in enumerate(f.members):
            out.append(f"  if (fact == \"{f.name}\" && member == \"{m}\") return {i};\n")
    out.append("  return -1;\n}\n\n}  // namespace fsla::plan\n")
    return "".join(out)


#: the fields of the generated `Caps`, in order: (name, C++ type, where a row's value comes from, doc). A `fact` is the device part's fact of that name, an
#: `instruction` whether the part's `instructions` set holds a member of that name; only what a kernel source reads is a field
CAPS = (
    ("cc", "int", "fact", "compute capability x10, e.g. 900"),
    ("bf16_cvt_packed", "bool", "instruction", "two fp32 -> one b32 of two bf16, one instruction"),
    ("bf16_simd_arith", "bool", "instruction", "elementwise bf16 arithmetic on a packed pair (`mul.rn.bf16x2`). see docs/internals/common/arch_caps.md#bf16-simd-arith"),
    ("bf16_simd_fma", "bool", "instruction", "FUSED elementwise bf16 arithmetic on a packed pair (`fma.rn.bf16x2`). see docs/internals/common/arch_caps.md#bf16-simd-fma"),
    ("regs_per_sm", "int", "fact", ""),
)


def _camel_k(name: str) -> str:
    return "k" + "".join(p[:1].upper() + p[1:] for p in name.split("_"))


def arch_header(archs: list[Architecture]) -> str:
    """`common/hardware.cuh`'s text: the Caps row of every architecture a device part states, the families, the builds and each build's traits."""
    def cap(a: Architecture, name: str, kind: str) -> str:
        value = {"cc": a.cc, "regs_per_sm": a.regs_per_sm}[name] if kind == "fact" else ("true" if name in a.instructions else "false")
        return f".{name} = {value}"

    #: a family is one a device part states, numbered in the architectures' order
    families = list(dict.fromkeys(a.family for a in archs))
    out = ["// common/hardware.cuh -- GENERATED by the build graph (hardware/emit.py --arch-header) from the registry's device parts (hardware/parts/device)",
           "// (edit those, not this). see docs/internals/common/hardware.md",
           "#pragma once", "", "// clang-format off", "namespace denseref {", "namespace arch {", "", "struct Caps {"]
    for name, ctype, _, doc in CAPS:
        out.append(f"  {ctype} {name};" + (f"  // {doc}" if doc else ""))
    out += ["};", "", "//: THE TABULATED SET -- see docs/internals/common/arch_caps.md#tabulated",
            "constexpr bool tabulated(int cc) {",
            "  return " + (" || ".join(f"cc == {a.cc}" for a in archs) or "false") + ";", "}", "",
            "constexpr Caps caps_of(int cc) {"]
    for a in archs:
        lines, line = [], ""
        for init in (cap(a, name, kind) for name, _, kind, _ in CAPS):
            if line and len(line) + len(init) + 2 > 108:
                lines.append(line + ",")
                line = ""
            line = f"{line}, {init}" if line else init
        lines.append(line)
        out.append(f"  // {a.name}: {a.part}")
        out.append(f"  if (cc == {a.cc})")
        out.append("    return Caps{" + lines[0])
        out += [f"                {text}" for text in lines[1:]]
        out[-1] += "};"
    out += ["  return Caps{};", "}", "",
            "//: THE ARCHITECTURE FAMILIES -- see docs/internals/common/hardware.md#families",
            "enum class Family : int { " + ", ".join(_camel_k(f) for f in families) + " };",
            "//: the families' names, in `Family`'s order: the one source the host's readers (`device_card`) take a name from",
            "constexpr const char* kFamilyNames[] = {" + ", ".join(f'"{f}"' for f in families) + "};", "",
            "//: A BUILD: the key a kernel compiles at, its compute capability. see docs/internals/common/hardware.md#builds",
            "struct Build {", "  int cc;", "};", "",
            "//: A BUILD'S TRAITS: what a kernel's plan reads. see docs/internals/common/hardware.md#traits",
            "struct Traits {", "  Family family;", "  Caps caps;", "};", "",
            "constexpr Traits traits_of(Build b) {"]
    for a in archs:
        out.append(f"  if (b.cc == {a.cc})")
        out.append(f"    return Traits{{Family::{_camel_k(a.family)}, caps_of({a.cc})}};")
    out += ["  return Traits{};", "}", "",
            "//: A BUILD AS A TYPE, the form a kernel takes it in: docs/internals/common/hardware.md#build-tag",
            "template <Build B>", "struct BuildTag {", "  static constexpr Build build = B;",
            "  static constexpr Traits traits = traits_of(B);", "};", "",
            "}  // namespace arch", "}  // namespace denseref", "", "// clang-format on", ""]
    return "\n".join(out)


def contract_header(contexts_: list[Context], facts: dict[str, Fact], contract: str = FULL) -> str:
    """`plan/contract.h`: the contract as the run time holds it, `contract()` a context each (its id and its facts, each a point or a range), and
    its name: a narrowed contract is named wherever the run time refuses a call outside it."""
    out = ["// GENERATED from hardware/descriptors by hardware/emit.py -- do not edit. NEVER CHECKED IN.\n",
           "#pragma once\n\n#include <vector>\n\n#include \"plan/facts.h\"\n\nnamespace fsla::plan {\n\n",
           f"//: the contract this build compiled against: `full`, or the narrowed contract a development build named\ninline constexpr const char* kContractName = \"{contract}\";\ninline constexpr bool kContractNarrowed = {'true' if contract != FULL else 'false'};\n\n",
           "struct ContractContext {\n  const char* id;\n  Facts facts;\n};\n\n",
           "//: every context FSLA accepts\ninline const std::vector<ContractContext>& contract() {\n  static const std::vector<ContractContext> rows = [] {\n    std::vector<ContractContext> out;\n"]
    for c in contexts_:
        out.append("    {\n      Facts f;\n")
        for name, value in c.facts.items():
            if facts[name].type == "int":
                out.append(f"      f.assign(\"{name}\", Interval{{{value[0]}, {value[1]}}});\n")
            else:
                bits = sum(1 << facts[name].members.index(m) for m in value)
                out.append(f"      f.assign(\"{name}\", EnumSet{{{bits}}});\n")
        out.append(f"      out.push_back({{\"{c.id}\", f}});\n    }}\n")
    out.append("    return out;\n  }();\n  return rows;\n}\n\n}  // namespace fsla::plan\n")
    return "".join(out)


def owed_header(owed_: list[tuple[Context, tuple[str, ...]]], facts: dict[str, Fact]) -> str:
    """`plan/owed.h`: the OWED contexts as the run time holds them, `owed()` a context each (its id, its facts and the kernels it waits on). The planner
    refuses a call inside one, naming the kernels; no launch function reads it."""
    out = ["// GENERATED from hardware/descriptors/owed by hardware/emit.py -- do not edit. NEVER CHECKED IN.\n",
           "#pragma once\n\n#include <vector>\n\n#include \"plan/facts.h\"\n\nnamespace fsla::plan {\n\n",
           "struct OwedContext {\n  const char* id;\n  Facts facts;\n  std::vector<const char*> kernels;\n};\n\n",
           "//: every context the contract owes: the call's kernels are not built\ninline const std::vector<OwedContext>& owed() {\n  static const std::vector<OwedContext> rows = [] {\n    std::vector<OwedContext> out;\n"]
    for c, kernels in owed_:
        out.append("    {\n      Facts f;\n")
        for name, value in c.facts.items():
            if facts[name].type == "int":
                out.append(f"      f.assign(\"{name}\", Interval{{{value[0]}, {value[1]}}});\n")
            else:
                bits = sum(1 << facts[name].members.index(m) for m in value)
                out.append(f"      f.assign(\"{name}\", EnumSet{{{bits}}});\n")
        listed = ", ".join(f'"{k}"' for k in kernels)
        out.append(f"      out.push_back({{\"{c.id}\", f, {{{listed}}}}});\n    }}\n")
    out.append("    return out;\n  }();\n  return rows;\n}\n\n}  // namespace fsla::plan\n")
    return "".join(out)


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))
