// Fixture for tools/lint/rules/torch_include_in_arm_tu.yml -- must fire.
// (The rule itself is scoped to every registered family's sources, `csrc/rola/src/<family>/**`; this fixture
// file lives under tools/lint/fixtures/ only so the rule has something to
// exercise without landing a bad file in a family's sources.)
#include <torch/extension.h>
__global__ void k() {}
