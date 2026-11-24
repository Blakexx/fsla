// THE ENTMAX HOST ANSWERS the matcher states -- docs/internals/entmax/entmax.md

#include "entmax.cuh"
#include "plan/entmax_launch.h"

namespace fsla {
namespace entmax {

//: THE LEVEL BOUND and THE WIDTH CLASS, for the host: pure functions of the matcher, no device
int64_t max_levels() { return plan::kEntmaxMaxLevels; }

int64_t width_class(int64_t width) { return plan::entmax_padded_width(static_cast<int>(width)); }

}  // namespace entmax
}  // namespace fsla
