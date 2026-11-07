// the closed-world arch rule's runtime half -- see docs/internals/common/arch_runtime.md
#pragma once

namespace rola {

void check_arch_table();

const long long* mma_quanta(int device);  // arch_runtime.md#mma-quanta

}  // namespace rola
