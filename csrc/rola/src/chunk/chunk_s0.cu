// GENERATED-BY-HAND at P69 stage B -- generator integration into
// tools/gen_shards.py is a stage-D item; until then this file is the
// matrix (edits here must update chunk.cu's CHUNK arms macro in step).
// SHARD chunk_s0 -- 8 of the chunk consumer's 24 instantiations.
#include "chunk_launch.cuh"

namespace rola::chunk {

template cudaError_t launch_chunk_arm<16, false, 64, 2, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 64, 2, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<64, false, 64, 2, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<16, true, 64, 2, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, true, 64, 2, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<64, true, 64, 2, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<16, false, 64, 2, 64>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 64, 2, 64>(const ChunkParams&, cudaStream_t);

template std::array<int64_t, 9> query_chunk_census_arm<16, false, 64, 2, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 64, 2, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<64, false, 64, 2, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<16, true, 64, 2, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<32, true, 64, 2, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<64, true, 64, 2, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<16, false, 64, 2, 64>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 64, 2, 64>();

//: the union-table arms (one per topology x BC) ride shard 0.
template cudaError_t launch_chunk_table_arm<2, 8, 64>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                      uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<2, 16, 64>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                       uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<2, 64, 32>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                       uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<2, 64, 64>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                       uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<2, 64, 128>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                        uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<2, 64, 256>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                        uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<3, 16, 128>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                        uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<4, 8, 32>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                      uint32_t*, int, int, cudaStream_t);
template cudaError_t launch_chunk_table_arm<4, 8, 128>(const __nv_bfloat16*, const __nv_bfloat16*,
                                                       uint32_t*, int, int, cudaStream_t);

}  // namespace rola::chunk
