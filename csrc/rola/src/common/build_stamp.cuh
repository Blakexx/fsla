// the extension's freshness fact -- see docs/internals/common/build_stamp.md
#pragma once

#include <cstdint>

namespace rola {

int64_t csrc_build_stamp();  // the csrc content hash, read back off the device

}  // namespace rola
