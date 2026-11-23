# `common/select.cuh` — the selection layer

Mirrors `csrc/fsla/src/common/select.{cu,cuh}`.

Every kernel adapts to its card (the constitution's R8; [`KERNEL_STANDARDS.md` section
8](../../KERNEL_STANDARDS.md#s8)): its sizes come from its plan at a
BUILD ([`hardware.md#builds`](hardware.md#builds)), and the host reaches the build of the device it holds through
this one layer -- the device's card, the builds the binary carries, one branch to the card's build at compile time --
rather than a switch over compute capabilities in each kernel.

<a id="card-of"></a>
## `card_of(device)`, `current_card()`

A device's `Card`: its compute capability, its SM
count (the planner's read: a part's SKUs differ in it), the build it runs and that build's traits. The first call for a device ADMITS it -- the closed-world checks
([`arch_runtime.md#check-arch-table`](arch_runtime.md#check-arch-table)): the planner's read of the device, refused when no context of the contract holds
the driver's numbers ([`plan/environment.md`](plan/environment.md)), and its build one this binary carries -- and keeps the card; every later call is a lookup.
A device whose build the binary does not carry is refused by name with the builds it does carry, so an architecture is
never run on another's image (an sm_100 card is not run on the sm_90a image).

<a id="carried"></a>
## The builds a binary carries

`build/generated/builds.inc` (`FSLA_BUILDS(F)`, one `F(cc)` a build) is written before every compile by
the graph's header action from the build's architectures (`hardware/registry.py`'s `builds_header`): one
build of every architecture the binary compiles.

<a id="in-image"></a>
## `in_image(b)`

A build's kernels compile in its own image only. `in_image(b)` is true for every build in the host pass and for the
image's own build in a device pass (`b.cc == __CUDA_ARCH__`), and a launch site instantiates a build's kernels under
`if constexpr (in_image(B))` -- so one architecture's image holds no other's kernel and the binary's entry points are one a
kernel a build, with no trapping copies in the other images. A launch of another image's kernel is refused by the
runtime (an invalid device function) and, before that, by `card_of`.

<a id="build-switch"></a>
## `build_switch(card, body)`

The host's one branch to the card's build: `body(BuildTag<B>{})`, `B` the card's build as a compile-time value and
`BuildTag<B>::traits` its traits, over `FSLA_BUILDS`: where a kernel's launch takes its build.
