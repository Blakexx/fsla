// The hermetic toolkit's smoke kernel: compiled by the graph's nvcc, proving the toolchain resolves.
__global__ void toolchain_smoke(int* out) { *out = 1; }
