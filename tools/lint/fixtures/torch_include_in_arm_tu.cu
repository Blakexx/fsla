// Fixture for tools/lint/rules/torch_include_in_arm_tu.yml -- must fire.
// (The rule itself is scoped to csrc/rola/src/carry/**; this fixture
// file lives under tools/lint/fixtures/ only so the rule has something to
// exercise without landing a bad file in the carry's sources.)
#include <torch/extension.h>
__global__ void k() {}
