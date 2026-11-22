# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CONTRACT REGISTRY: the schema of every fact, the composable parts, the descriptors that compose them, and their expansion to
full run-time contexts (docs/internals/hardware/registry.md). A context is the contract's unit: every fact, each a point, a range or a set.
The matchers and the run time see contexts only; a part or a composition never leaves this module.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parent

#: how a value is known: the lab record that measured it, the test range that validated it, the document that states it, the
#: declaration that makes it the contract
PROVENANCE = ("measured_by", "validated_by", "documented_by", "declared_by")
READS = ("DEVICE", "CALL")
#: the contract a build compiles against when it names none: every descriptor of `descriptors/`. A development build names a NARROWED contract
#: instead (`descriptors/dev/<name>.json`, marked `"narrowed": true`)
FULL = "full"
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
    its sorted members), `alternatives` names the facts whose choice made this context one of several."""
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
        if not isinstance(data.get("arch"), str) or not data["arch"].startswith("sm_"):
            raise RegistryError(f"{path}: a descriptor names the build architecture it is built for (`arch`: `sm_90a`)")
        out[path.stem] = data
    return out


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
    alternatives = tuple(n for n in facts if len(stated[n]) > 1)
    names = list(facts)
    return [Context(descriptor["name"], dict(zip(names, combo)), alternatives, descriptor["gpus"], descriptor["arch"])
            for combo in itertools.product(*(stated[n] for n in names))]


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
    """The schema, every part (composed or not) and every descriptor, and the contract's duplicates, or a RegistryError."""
    facts = schema(root)
    for name in parts(root):
        for fname in part(name, root)["facts"]:
            if fname not in facts:
                raise RegistryError(f"part {name}: {fname} is no fact of the schema")
    contexts(root)
    for name in descriptors(root, "descriptors/dev"):
        narrowed(name, root)


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


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))
