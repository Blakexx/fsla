// The host arm switch, exercised against a synthetic selection header.
// See docs/internals/common/arm_switch.md and tests/unit/test_arm_switch.py

#include <cstdio>
#include <string>

#include <c10/util/Exception.h>

#include "fixture_selection.inc"
#include "common/arm_switch.cuh"

namespace {

ROLA_DEFINE_ARM_SET(FixtureArmSet, FIXTURE);

void probe(const char* label, FixtureArmSet::Key k) {
  int index = -1, d_v = -1, levels = -1, decay = -1;
  try {
    rola::arm_switch<FixtureArmSet>(k, [&](auto arm) {
      index = decltype(arm)::index;
      d_v = decltype(arm)::values[0];
      levels = decltype(arm)::values[1];
      decay = decltype(arm)::values[2];
    });
  } catch (const std::exception& e) {
    std::printf("%s\trefused\t%s\n", label, std::string(e.what()).c_str());
    return;
  }
  std::printf("%s\tdispatched\t%d\t%d\t%d\t%d\n", label, index, d_v, levels, decay);
}

}  // namespace

int main() {
  std::printf("count\t%d\t%d\n", FixtureArmSet::count, FixtureArmSet::declared);
  std::printf("members\t%s\n", FixtureArmSet::members().c_str());
  std::printf("rows\t%zu\n", FixtureArmSet::rows().size());
  probe("built_first", {{32, 1, 0}});
  probe("built_last", {{64, 2, 1}});
  probe("declared_unbuilt", {{32, 1, 1}});
  probe("undeclared_d_v", {{16, 1, 0}});
  probe("undeclared_levels", {{32, 3, 0}});
  probe("undeclared_decay", {{32, 1, 2}});
  return 0;
}
