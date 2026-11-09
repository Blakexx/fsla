"""THE GRAPH'S ENVIRONMENT: the repositories made from the tree's pins and its index -- the Python environment built
from requirements.lock and the pinned rola-devtools into the shared store (`@rola_env`), the pinned linker (`@mold`),
the pinned uv and clang-tidy (`@uv`, `@clang_tidy`), the host C++ compiler's and devices' identities (`@host_cc`,
`@host_gpu`), the pinned CUDA toolkit and cuRAND's headers (`@cuda_tools`), and the checkout's tracked files
(`@rola_tree`). Docs: docs/build-graph.md#environment."""

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
        str(rctx.path(rctx.attr.uv)),
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
        "uv": attr.label(default = "@uv//:uv", allow_single_file = True, doc = "the pinned uv the store is built with"),
        "toolchain": attr.label(default = "//tools:toolchains/cu13.json"),
        "manifests": attr.label(default = "//tools:manifests/cu13/sm_86.json"),
        "make_env": attr.label(default = "//bazel:make_env.py"),
        "flags": attr.label(default = "//bazel:flags.py"),
        "build_flags": attr.label(default = "//tools:build_flags.py"),
    },
    doc = "The Python environment: requirements.lock built once into the shared store, and the graph's flags from it.",
)

_MOLD_BUILD = 'filegroup(name = "linker", srcs = ["bin/ld.mold", "{binary}"], visibility = ["//visibility:public"])\n'

_CLANG_TIDY_BUILD = """\
package(default_visibility = ["//visibility:public"])

filegroup(name = "clang_tidy", srcs = ["{binary}"])

#: its resource headers (clang's CUDA wrappers among them), beside the binary where clang looks for them
filegroup(name = "resources", srcs = glob(["clang_tidy/data/lib/clang/**"]))
"""

def _pinned_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.pin))
    asset = pin["release_asset"]
    #: an archive unpacks from its pin's folder (`archive_root`), as tools/dev.py's installer reads it -- a wheel (a
    #: zip) names its root; a bare binary lands at its `binary_path`
    wheel = asset.endswith(".whl")
    #: the release URL, then its mirrors: the same asset through the host's API, which serves it as octet-stream
    urls = [pin["release_url"]] + pin.get("mirror_urls", [])
    headers = {"Accept": ["application/octet-stream"]}
    if pin.get("asset_kind") == "binary":
        rctx.download(url = urls, sha256 = pin["asset_sha256"], output = pin["binary_path"], executable = True, headers = headers)
    else:
        rctx.download_and_extract(
            url = urls,
            sha256 = pin["asset_sha256"],
            type = "zip" if wheel else "tar.gz",
            stripPrefix = pin["archive_root"],
            headers = headers,
        )
    if "binary_sha256" in pin:
        binary = str(rctx.path(pin["binary_path"]))
        digest = rctx.execute(["sha256sum", binary]).stdout.split(" ")[0]
        if digest != pin["binary_sha256"]:
            fail("%s: %s's sha256 is %s, not the pin's %s" % (pin["name"], pin["binary_path"], digest, pin["binary_sha256"]))
        rctx.execute(["chmod", "0755", binary])
    #: the repository's BUILD file names the binary by its pin
    rctx.file("BUILD.bazel", rctx.attr.build.replace("{binary}", pin["binary_path"]))

_pinned_repo = repository_rule(
    implementation = _pinned_impl,
    attrs = {
        "pin": attr.label(mandatory = True),
        "build": attr.string(mandatory = True, doc = "the repository's BUILD file: the targets it exposes"),
    },
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
        parts.append("# %s %s %s" % (tool, path, digest))
    version = rctx.execute([cc, "--version"]).stdout.splitlines()[0]
    rctx.file("identity.txt", "# THE HOST C++ COMPILER'S IDENTITY, an input of every compile that no compile prunes (docs/build-graph.md#rules)\n# %s\n%s\n" % (version, "\n".join(parts)))
    rctx.file("BUILD.bazel", 'exports_files(["identity.txt"], visibility = ["//visibility:public"])\n')

_host_cc_repo = repository_rule(
    implementation = _host_cc_impl,
    doc = "The host C++ compiler's version and digests, as a file every compile takes as an input (never a header, " +
          "which a compile drops from its key when it does not read it).",
)

def _host_gpu_impl(rctx):
    #: only a device tier reads this repository, and a device tier on a machine that cannot name its devices refuses
    smi = rctx.which("nvidia-smi")
    if not smi:
        fail("no nvidia-smi on PATH: a device test tier is keyed on the machine's devices -- load the NVIDIA driver, " +
             "whose nvidia-smi names them, or run the tier that needs none: bazel test //tests:cpu")
    #: read again when the driver moves: its tools move with it
    rctx.watch(smi)
    found = rctx.execute([smi, "--query-gpu=index,name,uuid,driver_version,compute_cap,memory.total", "--format=csv,noheader"])
    if found.return_code != 0 or not found.stdout.strip():
        fail("nvidia-smi names no device (exit %d): %s -- a device test tier needs one; the tier that needs none: " % (
            found.return_code, found.stderr) + "bazel test //tests:cpu")
    rctx.file("identity.txt", "# THE MACHINE'S DEVICES, an input of every device test tier (docs/build-graph.md#rules)\n" + found.stdout)
    rctx.file("BUILD.bazel", 'exports_files(["identity.txt"], visibility = ["//visibility:public"])\n')

_host_gpu_repo = repository_rule(
    implementation = _host_gpu_impl,
    #: read at every fetch, so a device swapped under one driver is the key's too
    local = True,
    doc = "The machine's devices (each one's name, UUID, driver and compute capability), an input of every device " +
          "test tier: a pass is the device's, never replayed for another.",
)

_CUDA_TOOLS_BUILD = """\
package(default_visibility = ["//visibility:public"])

exports_files(["bin/cuobjdump", "bin/nvdisasm", "bin/nvcc"])

#: the pinned toolkit whole, the tree a test tier hands its tests (nvcc reads nvcc.profile, include/, lib64/ and nvvm/
#: beside it); its lib64 is a link to lib, which the glob would list twice
filegroup(name = "toolkit", srcs = glob(["**"], exclude = ["BUILD.bazel", "REPO.bazel", "redistrib.json", "curand/**", "lib64/**"]))

#: cuRAND's headers alone: clang's CUDA front end includes one (its runtime wrapper's `curand_mtgp32_kernel.h`)
filegroup(name = "curand_headers", srcs = glob(["curand/include/**"]))

#: the toolkit's compute-sanitizer, its libraries beside it (the sanitizer lane's)
filegroup(name = "compute_sanitizer", srcs = glob(["compute-sanitizer/**"]))
"""

def _component(rctx, root, entry, output, license):
    """One redistributable component by the manifest's digest, its archive's own folder stripped, its LICENSE kept at
    `license` (every component carries one at the same path)."""
    folder = entry["relative_path"].rsplit("/", 1)[1].removesuffix(".tar.xz")
    rctx.download_and_extract(
        url = root + "/" + entry["relative_path"],
        sha256 = entry["sha256"],
        stripPrefix = folder,
        #: a rename comes before the folder is stripped, so the new name is under the folder too
        rename_files = {folder + "/LICENSE": folder + "/" + license},
        output = output,
    )

def _cuda_tools_impl(rctx):
    pin = json.decode(rctx.read(rctx.attr.pin))
    rctx.download(url = pin["manifest_url"], sha256 = pin["manifest_sha256"], output = "redistrib.json")
    redist = json.decode(rctx.read("redistrib.json"))
    root = pin["manifest_url"].rsplit("/", 1)[0]
    #: the pinned toolkit, as tools/dev.py's installer lays it out: every component in one tree, then the pin's links
    for component in pin["components"]:
        _component(rctx, root, redist[component]["linux-x86_64"], ".", "licenses/%s/LICENSE" % component)
    for link, target in pin["links"].items():
        rctx.execute(["ln", "-sfn", target, link])
    #: cuRAND's headers, the graph's alone: its libraries are not the graph's
    _component(rctx, root, redist["libcurand"]["linux-x86_64"], "curand", "LICENSE")
    rctx.delete("curand/lib")
    rctx.file("BUILD.bazel", _CUDA_TOOLS_BUILD)

_cuda_tools_repo = repository_rule(
    implementation = _cuda_tools_impl,
    attrs = {"pin": attr.label(mandatory = True)},
    doc = "The pinned CUDA toolkit (tools/cuda_toolkit_pin.json: the redistributable manifest by its digest, each " +
          "component by the manifest's) and cuRAND's headers: the binary gates' disassemblers, the sanitizer lane's " +
          "compute-sanitizer and the test tiers' toolkit.",
)

def _tree_impl(rctx):
    root = str(rctx.workspace_root)
    git = rctx.which("git")
    #: the commit gate's tree is the commit's index (a partial commit's is not the checkout's): the hook hands it in
    index_file = rctx.getenv("GIT_INDEX_FILE")
    env = {"GIT_INDEX_FILE": index_file} if index_file else {}
    index = rctx.execute([git, "-C", root, "rev-parse", "--path-format=absolute", "--git-path", "index"], environment = env).stdout.strip()
    #: re-read when the index moves: a file added, removed or staged
    rctx.watch(index)
    listed = rctx.execute([git, "-C", root, "ls-files", "-z"], environment = env)
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
    #: the pinned uv the environment's store is built with, and the run targets that write venvs over it (`//:dev_env`)
    _pinned_repo(name = "uv", pin = "//tools:uv_pin.json", build = 'exports_files(["{binary}"], visibility = ["//visibility:public"])\n')
    _env_repo(name = "rola_env")
    _pinned_repo(name = "mold", pin = "//tools:mold_pin.json", build = _MOLD_BUILD)
    _pinned_repo(
        name = "clang_tidy",
        pin = "//tools:clang_tidy_pin.json",
        build = _CLANG_TIDY_BUILD,
    )
    _host_cc_repo(name = "host_cc")
    _host_gpu_repo(name = "host_gpu")
    _tree_repo(name = "rola_tree")
    _cuda_tools_repo(name = "cuda_tools", pin = "//tools:cuda_toolkit_pin.json")

environment = module_extension(
    implementation = _environment_impl,
    doc = "The graph's environment: the Python environment, the pinned linker and clang-tidy, the host compiler's " +
          "and devices' identities, the toolkit's disassemblers and the checkout's tracked files.",
)
