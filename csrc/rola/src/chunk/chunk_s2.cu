// GENERATED-BY-HAND at P69 stage B -- generator integration into
// tools/gen_shards.py is a stage-D item; until then this file is the
// matrix (edits here must update chunk.cu's CHUNK arms macro in step).
// SHARD chunk_s2 -- 8 of the chunk consumer's 24 instantiations.
#include "chunk_launch.cuh"

namespace rola::chunk {

template cudaError_t launch_chunk_arm<64, false, 128, 2, 64>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 256, 2, 64>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<16, false, 64, 2, 16>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 64, 2, 16>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<64, false, 64, 2, 16>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 128, 3, 16>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 128, 4, 8>(const ChunkParams&, cudaStream_t);
template cudaError_t launch_chunk_arm<32, false, 32, 4, 8>(const ChunkParams&, cudaStream_t);

template std::array<int64_t, 9> query_chunk_census_arm<64, false, 128, 2, 64>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 256, 2, 64>();
template std::array<int64_t, 9> query_chunk_census_arm<16, false, 64, 2, 16>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 64, 2, 16>();
template std::array<int64_t, 9> query_chunk_census_arm<64, false, 64, 2, 16>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 128, 3, 16>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 128, 4, 8>();
template std::array<int64_t, 9> query_chunk_census_arm<32, false, 32, 4, 8>();

}  // namespace rola::chunk
