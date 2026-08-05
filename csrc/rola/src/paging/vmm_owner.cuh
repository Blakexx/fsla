#pragma once

#include <torch/extension.h>
#include <ATen/ops/from_blob.h>

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rola::paging {

struct VmmProbe {
  int device = -1;
  bool supported = false;
  std::uint64_t allocation_granularity = 0;
  std::string reason;
};

// Owns only the CUDA-driver backing. Logical page identity, allocation plans, and
// publication remain in SparseRoLAState so dense and VMM backing share one ABI.
class VmmOwner final : public std::enable_shared_from_this<VmmOwner> {
 public:
  static VmmProbe probe(int device);

  static std::shared_ptr<VmmOwner> create(int device, std::int64_t max_physical_pages,
                                          std::int64_t bp, std::int64_t dqk, std::int64_t dv,
                                          bool global_mass, std::int64_t target_chunk_bytes);

  ~VmmOwner() noexcept;

  VmmOwner(const VmmOwner&) = delete;
  VmmOwner& operator=(const VmmOwner&) = delete;

  // These tensors are contiguous logical views over the reserved address ranges.
  // Only the prefix reported by mapped_capacity_pages is mapped and may be touched.
  torch::Tensor value_base();
  torch::Tensor mass_base();

  void grow(std::int64_t required_pages);
  void rollback_to(std::int64_t target_pages);
  void reset();
  void close();

  bool closed() const { return closed_; }
  std::int64_t mapped_capacity_pages() const;
  std::int64_t max_physical_pages() const { return max_physical_pages_; }
  std::uint64_t committed_bytes() const { return committed_bytes_; }
  std::uint64_t virtual_bytes() const { return virtual_bytes_; }
  std::uint64_t allocation_granularity() const { return allocation_granularity_; }
  std::int64_t chunk_pages() const { return chunk_pages_; }
  std::string backing_kind() const { return "cuda_driver_vmm"; }

 private:
  struct Chunk {
    CUdeviceptr address = 0;
    std::size_t bytes = 0;
    CUmemGenericAllocationHandle handle = 0;
  };

  struct Plane {
    CUdeviceptr address = 0;
    std::size_t virtual_bytes = 0;
    std::size_t mapped_bytes = 0;
    std::size_t page_bytes = 0;
    std::vector<Chunk> chunks;
  };

  struct ViewTracker {
    std::atomic<std::int64_t> live_views{0};
  };

  VmmOwner(int device, std::int64_t max_physical_pages, std::int64_t bp, std::int64_t dqk,
           std::int64_t dv, bool global_mass, std::uint64_t allocation_granularity,
           std::int64_t target_chunk_bytes);

  static std::size_t round_up(std::size_t value, std::size_t alignment);
  static std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* what);
  static std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* what);
  static std::size_t checked_size(std::uint64_t value, const char* what);
  static int normalize_device(int device);
  static void check(CUresult result, const char* operation);
  static void initialize_driver();

  bool quiesce() noexcept;

  void reserve_planes();
  void map_initial_chunk();
  struct GrowthMark {
    Plane* plane = nullptr;
    std::size_t chunk_count = 0;
    std::size_t mapped_bytes = 0;
  };

  void grow_plane(Plane& plane, std::size_t target_bytes, std::vector<GrowthMark>& mapped_now);
  void rollback_growth(std::vector<GrowthMark>& mapped_now) noexcept;
  void remove_latest_chunk(Plane& plane);
  torch::Tensor make_view(const Plane& plane, bool mass);
  void release_planes() noexcept;

  int device_;
  std::int64_t max_physical_pages_;
  std::int64_t bp_;
  std::int64_t dqk_;
  std::int64_t dv_;
  bool global_mass_;
  std::uint64_t allocation_granularity_;
  std::int64_t chunk_pages_;
  std::uint64_t virtual_bytes_ = 0;
  std::uint64_t committed_bytes_ = 0;
  Plane value_;
  Plane mass_;
  std::shared_ptr<ViewTracker> view_tracker_;
  bool closed_ = false;
};

std::shared_ptr<VmmOwner> create_vmm_owner(int device, std::int64_t max_physical_pages,
                                           std::int64_t bp, std::int64_t dqk, std::int64_t dv,
                                           bool global_mass, std::int64_t target_chunk_bytes);

}  // namespace rola::paging
