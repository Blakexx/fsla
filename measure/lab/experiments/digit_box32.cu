// measure/lab/experiments/digit_box32.cu -- QUESTION: the 32-digit box's reference for `digit_box.cu`: its readout and
// fold stand-in free-running on a static image, a token's cost to set the 16-digit box's against. METHOD:
// `digit_box.cu` built for 32 digits a box (4 a warp), on `read_image.inputs.py`'s 32-digit files. see
// docs/internals/tools/lab.md#digit-box
#define READ_IMAGE_DIGITS 32
#include "digit_box.cu"
