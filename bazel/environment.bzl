"""THE GRAPH'S ENVIRONMENT: the repositories made from the tree's pins -- the Python environment built from
requirements.lock into the shared store (`@rola_env`), the pinned linker (`@mold`), and the host C++ compiler's
identity (`@host_cc`), and the toolkit's disassemblers (`@cuda_tools`). Docs: docs/build-graph.md#environment."""

_ENV_BUILD = """\
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

#: the interpreter the graph's actions run: the environment's base Python (a link it resolves to find its standard
#: library), standard library only
filegroup(name = "python", srcs = ["bin/python3"])

cc_library(
    name = "torch_headers",
    hdrs = glob(["site-packages/torch/include/**"]),
    includes = ["site-packages/torch/include", "site-packages/torch/include/torch/csrc/api/include"],
)

#: the environment's own interpreter, its packages beside it (the store's; an action that imports them runs outside
#: the sandbox, the lock its input)
filegroup(name = "venv_python", srcs = ["venv/bin/python3", "venv/pyvenv.cfg", "requirements.lock"])

#: the torch libraries the extension links (its NEEDED, as setup.py's link leaves them)
filegroup(
    name = "torch_libraries",
    srcs = ["site-packages/torch/lib/libc10.so", "site-packages/torch/lib/libtorch_cpu.so"],
)

cc_library(
    name = "python_headers",
    hdrs = glob(["python_include/**"]),
    includes = ["python_include"],
)
"""

def _store(rctx):
    store = rctx.getenv("ROLA_GRAPH_STORE")
    if not store:
        fail("ROLA_GRAPH_STORE is unset: run Bazel through its front door, `bazel` (docs/build-graph.md#front-door)")
    return store

def _env_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.uv_pin))
    rctx.download_and_extract(url = pin["release_url"], sha256 = pin["asset_sha256"], type = "zip", output = "uv", stripPrefix = pin["strip_prefix"])
    record = json.decode(rctx.read(rctx.attr.toolchain))
    #: the environment's inputs: a change to any of them is a new environment (the lock and the devtools pin by content)
    for source in (rctx.attr.lock, rctx.attr.devtools, rctx.attr.make_env, rctx.attr.flags, rctx.attr.build_flags):
        rctx.watch(rctx.path(source))
    rctx.watch_tree(rctx.path(rctx.attr.manifests).dirname)
    built = rctx.execute([
        rctx.which("python3"),
        rctx.path(rctx.attr.make_env),
        "--store",
        _store(rctx),
        "--uv",
        str(rctx.path("uv/uv")),
        "--python",
        pin["python"],
        "--lock",
        str(rctx.path(rctx.attr.lock)),
        "--index",
        record["torch_index"],
        "--devtools",
        str(rctx.path(rctx.attr.devtools)),
    ], timeout = 7200, quiet = False)
    if built.return_code != 0:
        fail("the environment's store build failed (bazel/make_env.py):\n" + built.stderr)
    paths = json.decode(built.stdout.strip().splitlines()[-1])
    rctx.symlink(paths["site"], "site-packages")
    rctx.symlink(paths["include"], "python_include")
    rctx.symlink(paths["base"] + "/bin/python3", "bin/python3")
    rctx.symlink(paths["venv"], "venv")
    rctx.symlink(rctx.path(rctx.attr.lock), "requirements.lock")
    threads = rctx.getenv("ROLA_NVCC_THREADS")
    if not threads:
        fail("ROLA_NVCC_THREADS is unset: run Bazel through its front door, `bazel` (docs/build-graph.md#front-door)")
    flags = rctx.execute([
        paths["python"],
        rctx.path(rctx.attr.flags),
        "--toolchain",
        str(rctx.path(rctx.attr.toolchain)),
        "--nvcc-threads",
        threads,
    ], timeout = 600)
    if flags.return_code != 0:
        fail("the graph's flags (bazel/flags.py) failed:\n" + flags.stderr)
    #: the environment's own interpreter and tools by their store path: the path carries the environment's key, so a
    #: command naming it is keyed by the environment
    rctx.file("flags.bzl", flags.stdout + "VENV = %s\n" % json.encode(paths["venv"]))
    rctx.file("BUILD.bazel", _ENV_BUILD)

_env_repo = repository_rule(
    implementation = _env_impl,
    attrs = {
        "lock": attr.label(default = "//:requirements.lock"),
        "devtools": attr.label(default = "//tools:devtools.txt"),
        "uv_pin": attr.label(default = "//tools:uv_pin.json"),
        "toolchain": attr.label(default = "//tools:toolchains/cu13.json"),
        "manifests": attr.label(default = "//tools:manifests/cu13/sm_86.json"),
        "make_env": attr.label(default = "//bazel:make_env.py"),
        "flags": attr.label(default = "//bazel:flags.py"),
        "build_flags": attr.label(default = "//tools:build_flags.py"),
    },
    doc = "The Python environment: requirements.lock built once into the shared store, and the graph's flags from it.",
)

def _pinned_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.pin))
    #: the release URL, then its mirrors: the same asset through the host's API, which serves it as octet-stream
    rctx.download_and_extract(
        url = [pin["release_url"]] + pin.get("mirror_urls", []),
        sha256 = pin["asset_sha256"],
        type = "tar.gz",
        stripPrefix = pin["release_asset"].removesuffix(".tar.gz"),
        headers = {"Accept": ["application/octet-stream"]},
    )
    rctx.file("BUILD.bazel", 'filegroup(name = "linker", srcs = ["bin/ld.mold", "bin/mold"], visibility = ["//visibility:public"])\n')

_pinned_repo = repository_rule(
    implementation = _pinned_impl,
    attrs = {"pin": attr.label(mandatory = True)},
    doc = "A pinned release, downloaded by its pin's digest.",
)

def _host_cc_impl(rctx):
    cc = rctx.which("gcc")
    if not cc:
        fail("no gcc on PATH: the host C++ compiler nvcc drives is the machine's (docs/build-graph.md#environment)")
    tools = {
        "gcc": cc,
        "g++": rctx.which("g++"),
        "cc1plus": rctx.path(rctx.execute([cc, "-print-prog-name=cc1plus"]).stdout.strip()),
    }
    parts = []
    for tool, path in tools.items():
        rctx.watch(path)
        digest = rctx.execute(["sha256sum", str(path)]).stdout.split(" ")[0]
        parts.append("// %s %s %s" % (tool, path, digest))
    version = rctx.execute([cc, "--version"]).stdout.splitlines()[0]
    rctx.file("host_cc_identity.h", "// THE HOST C++ COMPILER'S IDENTITY, an input of every compile (docs/build-graph.md#rules)\n// %s\n%s\n" % (version, "\n".join(parts)))
    rctx.file("BUILD.bazel", 'load("@rules_cc//cc:defs.bzl", "cc_library")\n\ncc_library(name = "identity", hdrs = ["host_cc_identity.h"], visibility = ["//visibility:public"])\n')

_host_cc_repo = repository_rule(
    implementation = _host_cc_impl,
    doc = "The host C++ compiler's version and digests, as a header every compile takes as an input.",
)

_CUDA_TOOLS_BUILD = """\
package(default_visibility = ["//visibility:public"])

exports_files(["bin/cuobjdump", "bin/nvdisasm"])
"""

def _cuda_tools_impl(rctx):
    rctx.download(url = rctx.attr.manifest, sha256 = rctx.attr.sha256, output = "redistrib.json")
    redist = json.decode(rctx.read("redistrib.json"))
    root = rctx.attr.manifest.rsplit("/", 1)[0]
    #: one bin/ for both, as an installed toolkit has them: `cuobjdump -sass` runs the nvdisasm beside it
    for component in ("cuobjdump", "nvdisasm"):
        entry = redist["cuda_" + component]["linux-x86_64"]
        rctx.download_and_extract(
            url = root + "/" + entry["relative_path"],
            sha256 = entry["sha256"],
            rename_files = {entry["relative_path"].rsplit("/", 1)[1].removesuffix(".tar.xz") + "/LICENSE": component + "/LICENSE"},
            stripPrefix = entry["relative_path"].rsplit("/", 1)[1].removesuffix(".tar.xz"),
        )
    rctx.file("BUILD.bazel", _CUDA_TOOLS_BUILD)

_cuda_tools_repo = repository_rule(
    implementation = _cuda_tools_impl,
    attrs = {"manifest": attr.string(mandatory = True), "sha256": attr.string(mandatory = True)},
    doc = "The toolkit's disassemblers (cuobjdump, nvdisasm), from the redistributable manifest by its digests.",
)

def _tree_impl(rctx):
    root = str(rctx.workspace_root)
    git = rctx.which("git")
    index = rctx.execute([git, "-C", root, "rev-parse", "--path-format=absolute", "--git-path", "index"]).stdout.strip()
    #: re-read when the index moves: a file added, removed or staged
    rctx.watch(index)
    listed = rctx.execute([git, "-C", root, "ls-files", "-z"])
    if listed.return_code != 0:
        fail("git ls-files failed in %s: %s" % (root, listed.stderr))
    files = [f for f in listed.stdout.split("\0") if f]
    packages = sorted([f.rsplit("/", 1)[0] if "/" in f else "" for f in files if f.rsplit("/", 1)[-1] == "BUILD.bazel"])
    tracked = {pkg: [] for pkg in packages}
    for f in files:
        owner = ""
        for pkg in packages:
            if pkg and f.startswith(pkg + "/") and len(pkg) > len(owner):
                owner = pkg
        tracked[owner].append(f[len(owner) + 1:] if owner else f)
    rctx.file("tree.bzl", "\"\"\"GENERATED from `git ls-files`: the checkout's tracked files, by the package that holds them.\"\"\"\n\nTRACKED = %s\n" % json.encode_indent(tracked, indent = "    "))
    rctx.file("BUILD.bazel", 'exports_files(["tree.bzl"], visibility = ["//visibility:public"])\n')

_tree_repo = repository_rule(
    implementation = _tree_impl,
    doc = "The checkout's tracked files (`git ls-files`), by package: the tree the lints read, re-read when the index moves.",
)

def _environment_impl(mctx):
    _env_repo(name = "rola_env")
    _pinned_repo(name = "mold", pin = "//tools:mold_pin.json")
    _host_cc_repo(name = "host_cc")
    _tree_repo(name = "rola_tree")
    for module in mctx.modules:
        for tag in module.tags.cuda_tools:
            _cuda_tools_repo(name = "cuda_tools", manifest = tag.manifest, sha256 = tag.sha256)

environment = module_extension(
    implementation = _environment_impl,
    tag_classes = {
        "cuda_tools": tag_class(attrs = {
            "manifest": attr.string(mandatory = True, doc = "the toolkit's redistributable manifest"),
            "sha256": attr.string(mandatory = True, doc = "its digest: the one rules_cuda's toolkit is read by"),
        }),
    },
    doc = "The graph's environment: the Python environment, the pinned linker, the host compiler's identity and the " +
          "toolkit's disassemblers.",
)
