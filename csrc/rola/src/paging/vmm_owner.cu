#include "p3_vmm_owner.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

namespace fla_rola::p3 {
namespace {

std::string driver_error(CUresult result) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::ostringstream stream;
  stream << (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name);
  if (description != nullptr) stream << ": " << description;
  return stream.str();
}

CUmemAllocationProp allocation_prop(int device) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}

}  // namespace

void VmmOwner::initialize_driver() {
  static std::once_flag once;
  static CUresult init_result = CUDA_SUCCESS;
  std::call_once(once, [] { init_result = cuInit(0); });
  check(init_result, "cuInit");
}

void VmmOwner::check(CUresult result, const char* operation) {
  TORCH_CHECK(result == CUDA_SUCCESS, "CUDA VMM ", operation, " failed: ",
              driver_error(result));
}

std::size_t VmmOwner::round_up(std::size_t value, std::size_t alignment) {
  TORCH_CHECK(alignment != 0 && value <= std::numeric_limits<std::size_t>::max() -
                                      (alignment - 1),
              "CUDA VMM size overflows size_t");
  return ((value + alignment - 1) / alignment) * alignment;
}

std::uint64_t VmmOwner::checked_mul(
    std::uint64_t lhs, std::uint64_t rhs, const char* what) {
  TORCH_CHECK(rhs == 0 || lhs <= std::numeric_limits<std::uint64_t>::max() / rhs,
              "CUDA VMM ", what, " overflows uint64");
  return lhs * rhs;
}

std::uint64_t VmmOwner::checked_add(
    std::uint64_t lhs, std::uint64_t rhs, const char* what) {
  TORCH_CHECK(lhs <= std::numeric_limits<std::uint64_t>::max() - rhs,
              "CUDA VMM ", what, " overflows uint64");
  return lhs + rhs;
}

std::size_t VmmOwner::checked_size(std::uint64_t value, const char* what) {
  TORCH_CHECK(value <= std::numeric_limits<std::size_t>::max(),
              "CUDA VMM ", what, " exceeds host size_t");
  return static_cast<std::size_t>(value);
}

int VmmOwner::normalize_device(int device) {
  if (device >= 0) return device;
  int current = -1;
  TORCH_CHECK(cudaGetDevice(&current) == cudaSuccess && current >= 0,
              "CUDA VMM could not resolve the current CUDA device");
  return current;
}

VmmProbe VmmOwner::probe(int device) {
  initialize_driver();
  VmmProbe result;
  if (device < 0) {
    const auto status = cudaGetDevice(&device);
    if (status != cudaSuccess || device < 0) {
      result.reason = status == cudaSuccess
          ? "CUDA returned an invalid current device"
          : cudaGetErrorString(status);
      return result;
    }
  }
  result.device = device;

  CUdevice cu_device;
  CUresult status = cuDeviceGet(&cu_device, device);
  if (status != CUDA_SUCCESS) {
    result.reason = driver_error(status);
    return result;
  }
  int supported = 0;
  status = cuDeviceGetAttribute(
      &supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_device);
  if (status != CUDA_SUCCESS || supported == 0) {
    result.reason = status == CUDA_SUCCESS
        ? "CUDA driver reports virtual memory management unsupported"
        : driver_error(status);
    return result;
  }

  c10::cuda::CUDAGuard guard(device);
  auto prop = allocation_prop(device);
  std::size_t granularity = 0;
  status = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (status != CUDA_SUCCESS || granularity == 0) {
    result.reason = status == CUDA_SUCCESS ? "driver returned zero allocation granularity"
                                           : driver_error(status);
    return result;
  }
  result.allocation_granularity = granularity;

  // A probe is not complete until the driver can reserve, map, grant access,
  // unmap, and release one granularity-sized allocation in this context.
  CUdeviceptr address = 0;
  CUmemGenericAllocationHandle handle = 0;
  bool mapped = false;
  status = cuMemAddressReserve(&address, granularity, 0, 0, 0);
  if (status == CUDA_SUCCESS) {
    status = cuMemCreate(&handle, granularity, &prop, 0);
  }
  if (status == CUDA_SUCCESS) {
    status = cuMemMap(address, granularity, 0, handle, 0);
    mapped = status == CUDA_SUCCESS;
  }
  if (status == CUDA_SUCCESS) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    status = cuMemSetAccess(address, granularity, &access, 1);
  }
  if (address != 0) {
    if (mapped) {
      // Mapping may have succeeded even when access setup failed. The driver
      // cleanup must follow the actual map state, not the final status.
      cuMemUnmap(address, granularity);
    }
    cuMemAddressFree(address, granularity);
  }
  if (handle != 0) cuMemRelease(handle);
  if (status != CUDA_SUCCESS) {
    result.reason = driver_error(status);
    return result;
  }
  result.supported = true;
  result.reason = "CUDA driver VMM reserve/map/access probe passed";
  return result;
}

std::shared_ptr<VmmOwner> VmmOwner::create(
    int device,
    std::int64_t max_physical_pages,
    std::int64_t bp,
    std::int64_t dqk,
    std::int64_t dv,
    bool global_mass,
    std::int64_t target_chunk_bytes) {
  TORCH_CHECK(max_physical_pages > 0 && bp > 0 && dqk > 0 && dv > 0,
              "CUDA VMM cache dimensions must be positive");
  const auto normalized_device = normalize_device(device);
  const auto capability = probe(normalized_device);
  TORCH_CHECK(capability.supported,
              "progressive_memory=True requires CUDA virtual memory management on device ",
              capability.device, "; ", capability.reason,
              ". Set progressive_memory=False explicitly for fully committed backing.");
  auto owner = std::shared_ptr<VmmOwner>(new VmmOwner(
      capability.device, max_physical_pages, bp, dqk, dv, global_mass,
      capability.allocation_granularity, target_chunk_bytes));
  owner->reserve_planes();
  owner->map_initial_chunk();
  return owner;
}

VmmOwner::VmmOwner(
    int device,
    std::int64_t max_physical_pages,
    std::int64_t bp,
    std::int64_t dqk,
    std::int64_t dv,
    bool global_mass,
    std::uint64_t allocation_granularity,
    std::int64_t target_chunk_bytes)
    : device_(device),
      max_physical_pages_(max_physical_pages),
      bp_(bp),
      dqk_(dqk),
      dv_(dv),
      global_mass_(global_mass),
      allocation_granularity_(allocation_granularity),
      view_tracker_(std::make_shared<ViewTracker>()) {
  TORCH_CHECK(target_chunk_bytes > 0, "CUDA VMM target_chunk_bytes must be positive");
  const auto bp_u = static_cast<std::uint64_t>(bp);
  const auto dqk_u = static_cast<std::uint64_t>(dqk);
  const auto dv_u = static_cast<std::uint64_t>(dv);
  const auto value_elements = checked_mul(
      checked_mul(bp_u, dqk_u, "value page element count"), dv_u,
      "value page element count");
  const auto mass_elements = checked_mul(bp_u, dqk_u, "mass page element count");
  const auto value_page_bytes = checked_mul(
      value_elements, sizeof(float), "value page byte count");
  const auto mass_page_bytes = checked_mul(
      mass_elements, sizeof(float), "mass page byte count");
  value_.page_bytes = checked_size(value_page_bytes, "value page byte count");
  mass_.page_bytes = checked_size(mass_page_bytes, "mass page byte count");
  const auto chunk_pages_u = std::max<std::uint64_t>(
      1, static_cast<std::uint64_t>(target_chunk_bytes) / value_page_bytes);
  chunk_pages_ = static_cast<std::int64_t>(std::min<std::uint64_t>(
      static_cast<std::uint64_t>(max_physical_pages), chunk_pages_u));
}

void VmmOwner::reserve_planes() {
  c10::cuda::CUDAGuard guard(device_);
  const auto reserve = [this](Plane& plane) {
    const auto payload = checked_mul(
        static_cast<std::uint64_t>(plane.page_bytes),
        static_cast<std::uint64_t>(max_physical_pages_),
        "virtual extent");
    plane.virtual_bytes = round_up(
        checked_size(payload, "virtual extent"), allocation_granularity_);
    check(cuMemAddressReserve(&plane.address, plane.virtual_bytes, 0, 0, 0),
          "cuMemAddressReserve");
    virtual_bytes_ = checked_add(
        virtual_bytes_, static_cast<std::uint64_t>(plane.virtual_bytes),
        "total virtual extent");
  };
  reserve(value_);
  if (global_mass_) reserve(mass_);
}

void VmmOwner::map_initial_chunk() {
  grow(std::min<std::int64_t>(max_physical_pages_, chunk_pages_));
}

void VmmOwner::grow_plane(
    Plane& plane,
    std::size_t target_bytes,
    std::vector<GrowthMark>& mapped_now) {
  if (target_bytes <= plane.mapped_bytes) return;
  target_bytes = round_up(target_bytes, allocation_granularity_);
  TORCH_CHECK(target_bytes <= plane.virtual_bytes,
              "CUDA VMM growth exceeds reserved virtual extent");
  const auto old_chunk_count = plane.chunks.size();
  mapped_now.push_back({&plane, old_chunk_count, plane.mapped_bytes});
  const auto bytes = target_bytes - plane.mapped_bytes;
  auto prop = allocation_prop(device_);
  CUmemGenericAllocationHandle handle = 0;
  bool mapped = false;
  check(cuMemCreate(&handle, bytes, &prop, 0), "cuMemCreate");
  try {
    TORCH_CHECK(plane.mapped_bytes <=
                    std::numeric_limits<CUdeviceptr>::max() - plane.address,
                "CUDA VMM mapped address overflows CUdeviceptr");
    const auto address = plane.address + plane.mapped_bytes;
    check(cuMemMap(address, bytes, 0, handle, 0), "cuMemMap");
    mapped = true;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(cuMemSetAccess(address, bytes, &access, 1), "cuMemSetAccess");
    plane.chunks.push_back({address, bytes, handle});
    plane.mapped_bytes = target_bytes;
  } catch (...) {
    if (mapped) cuMemUnmap(plane.address + plane.mapped_bytes, bytes);
    cuMemRelease(handle);
    throw;
  }
}

void VmmOwner::rollback_growth(
    std::vector<GrowthMark>& mapped_now) noexcept {
  for (auto it = mapped_now.rbegin(); it != mapped_now.rend(); ++it) {
    auto* plane = it->plane;
    while (plane->chunks.size() > it->chunk_count) {
      auto chunk = plane->chunks.back();
      plane->chunks.pop_back();
      if (chunk.address != 0 && chunk.bytes != 0) cuMemUnmap(chunk.address, chunk.bytes);
      if (chunk.handle != 0) cuMemRelease(chunk.handle);
    }
    plane->mapped_bytes = it->mapped_bytes;
  }
}

void VmmOwner::grow(std::int64_t required_pages) {
  TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  TORCH_CHECK(required_pages >= 0 && required_pages <= max_physical_pages_,
              "CUDA VMM required page count exceeds the reserved maximum");
  if (required_pages <= mapped_capacity_pages()) return;
  c10::cuda::CUDAGuard guard(device_);
  const auto current_pages = mapped_capacity_pages();
  const auto required_chunks =
      (required_pages + chunk_pages_ - 1) / chunk_pages_;
  const auto target_chunks = current_pages == 0 ? required_chunks : required_chunks >=
          (max_physical_pages_ + chunk_pages_ - 1) / chunk_pages_
      ? required_chunks
      : required_chunks + 1;
  const auto target_pages = std::min<std::int64_t>(
      max_physical_pages_, target_chunks * chunk_pages_);
  const auto target_value_bytes = round_up(
      checked_size(checked_mul(static_cast<std::uint64_t>(target_pages),
                               static_cast<std::uint64_t>(value_.page_bytes),
                               "value growth extent"),
                   "value growth extent"),
      allocation_granularity_);
  const auto target_mass_bytes = round_up(
      checked_size(checked_mul(static_cast<std::uint64_t>(target_pages),
                               static_cast<std::uint64_t>(mass_.page_bytes),
                               "mass growth extent"),
                   "mass growth extent"),
      allocation_granularity_);
  std::vector<GrowthMark> mapped_now;
  const auto old_committed = committed_bytes_;
  try {
    grow_plane(value_, target_value_bytes, mapped_now);
    if (global_mass_) grow_plane(mass_, target_mass_bytes, mapped_now);
    committed_bytes_ = checked_add(
        static_cast<std::uint64_t>(value_.mapped_bytes),
        global_mass_ ? static_cast<std::uint64_t>(mass_.mapped_bytes) : 0,
        "committed extent");
  } catch (...) {
    rollback_growth(mapped_now);
    committed_bytes_ = old_committed;
    throw;
  }
}

void VmmOwner::remove_latest_chunk(Plane& plane) {
  TORCH_CHECK(!plane.chunks.empty(),
              "CUDA VMM rollback has no committed growth chunk");
  const auto chunk = plane.chunks.back();
  TORCH_CHECK(chunk.address + chunk.bytes == plane.address + plane.mapped_bytes,
              "CUDA VMM rollback chunk does not end at the mapped extent");
  check(cuMemUnmap(chunk.address, chunk.bytes), "rollback cuMemUnmap");
  check(cuMemRelease(chunk.handle), "rollback cuMemRelease");
  plane.chunks.pop_back();
  plane.mapped_bytes -= chunk.bytes;
}

void VmmOwner::rollback_to(std::int64_t target_pages) {
  TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  TORCH_CHECK(target_pages >= 0 && target_pages <= mapped_capacity_pages(),
              "CUDA VMM rollback capacity is outside the mapped extent");
  c10::cuda::CUDAGuard guard(device_);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "CUDA VMM rollback could not quiesce the device");
  if (target_pages == mapped_capacity_pages()) return;

  auto target_bytes = [&](const Plane& plane, const char* what) {
    return round_up(
        checked_size(
            checked_mul(static_cast<std::uint64_t>(target_pages),
                        static_cast<std::uint64_t>(plane.page_bytes), what),
            what),
        allocation_granularity_);
  };
  const auto value_target = target_bytes(value_, "value rollback extent");
  const auto mass_target = global_mass_
      ? target_bytes(mass_, "mass rollback extent")
      : 0;

  auto validate_boundary = [](const Plane& plane, std::size_t target,
                              const char* what) {
    TORCH_CHECK(target <= plane.mapped_bytes, what, " exceeds mapped extent");
    std::size_t boundary = 0;
    bool found = target == 0;
    for (const auto& chunk : plane.chunks) {
      TORCH_CHECK(boundary <= std::numeric_limits<std::size_t>::max() - chunk.bytes,
                  what, " overflows size_t");
      boundary += chunk.bytes;
      found = found || boundary == target;
    }
    TORCH_CHECK(found, what, " is not a committed chunk boundary");
    TORCH_CHECK(!plane.chunks.empty() && target >= plane.chunks.front().bytes,
                "CUDA VMM rollback cannot remove the initial headroom chunk");
  };
  validate_boundary(value_, value_target, "value rollback target");
  if (global_mass_) {
    validate_boundary(mass_, mass_target, "mass rollback target");
  }

  auto restore_plane = [&](Plane& plane, std::size_t target) {
    while (plane.mapped_bytes > target) {
      remove_latest_chunk(plane);
    }
    TORCH_CHECK(plane.mapped_bytes == target,
                "CUDA VMM rollback did not restore a plane exactly");
  };
  restore_plane(value_, value_target);
  if (global_mass_) restore_plane(mass_, mass_target);
  TORCH_CHECK(mapped_capacity_pages() == target_pages,
              "CUDA VMM rollback did not restore the requested prior capacity");
  committed_bytes_ = checked_add(
      static_cast<std::uint64_t>(value_.mapped_bytes),
      global_mass_ ? static_cast<std::uint64_t>(mass_.mapped_bytes) : 0,
      "committed rollback extent");
}

std::int64_t VmmOwner::mapped_capacity_pages() const {
  const auto value_pages = value_.page_bytes == 0 ? 0 : value_.mapped_bytes / value_.page_bytes;
  if (!global_mass_) {
    return static_cast<std::int64_t>(std::min<std::size_t>(
        value_pages, static_cast<std::size_t>(max_physical_pages_)));
  }
  const auto mass_pages = mass_.page_bytes == 0 ? 0 : mass_.mapped_bytes / mass_.page_bytes;
  return static_cast<std::int64_t>(std::min({
      value_pages, mass_pages, static_cast<std::size_t>(max_physical_pages_)}));
}

torch::Tensor VmmOwner::make_view(const Plane& plane, bool mass) {
  TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  view_tracker_->live_views.fetch_add(1, std::memory_order_relaxed);
  auto tracker = view_tracker_;
  std::vector<std::int64_t> shape = mass
      ? std::vector<std::int64_t>{max_physical_pages_, bp_, dqk_}
      : std::vector<std::int64_t>{max_physical_pages_, bp_, dqk_, dv_};
  auto options = torch::TensorOptions().device(torch::kCUDA, device_).dtype(torch::kFloat32);
  // Keep the driver allocation alive for every tensor view. This prevents a
  // caller holding a slice from outliving the native owner and observing an
  // unmapped raw pointer.
  auto owner_lifetime = shared_from_this();
  try {
    return at::from_blob(
        reinterpret_cast<void*>(plane.address), shape,
        [tracker, owner_lifetime](void*) {
          tracker->live_views.fetch_sub(1, std::memory_order_relaxed);
          (void)owner_lifetime;
        }, options);
  } catch (...) {
    view_tracker_->live_views.fetch_sub(1, std::memory_order_relaxed);
    throw;
  }
}

torch::Tensor VmmOwner::value_base() { return make_view(value_, false); }

torch::Tensor VmmOwner::mass_base() {
  TORCH_CHECK(global_mass_, "raw CUDA VMM backing has no mass plane");
  return make_view(mass_, true);
}

void VmmOwner::reset() {
  TORCH_CHECK(!closed_, "CUDA VMM owner is closed");
  // Reset is intentionally logical only. Mappings and their stable addresses survive.
}

void VmmOwner::release_planes() noexcept {
  for (Plane* plane : {&value_, &mass_}) {
    if (plane->address == 0) continue;
    for (auto it = plane->chunks.rbegin(); it != plane->chunks.rend(); ++it) {
      if (it->address != 0 && it->bytes != 0) cuMemUnmap(it->address, it->bytes);
      if (it->handle != 0) cuMemRelease(it->handle);
    }
    plane->chunks.clear();
    plane->mapped_bytes = 0;
    cuMemAddressFree(plane->address, plane->virtual_bytes);
    plane->address = 0;
  }
  committed_bytes_ = 0;
  virtual_bytes_ = 0;
}

void VmmOwner::close() {
  if (closed_) return;
  TORCH_CHECK(view_tracker_->live_views.load(std::memory_order_acquire) == 0,
              "cannot close CUDA VMM owner while exported tensor views are live");
  c10::cuda::CUDAGuard guard(device_);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess,
              "CUDA VMM close could not quiesce the device");
  release_planes();
  closed_ = true;
}

bool VmmOwner::quiesce() noexcept {
  try {
    c10::cuda::CUDAGuard guard(device_);
    return cudaDeviceSynchronize() == cudaSuccess;
  } catch (...) {
    return false;
  }
}

VmmOwner::~VmmOwner() noexcept {
  if (!closed_) {
    // Never unmap while work may still reference the reserved ranges. If the
    // context is already unavailable, leaking the mapping is safer than
    // tearing it down under in-flight CUDA work; process teardown reclaims it.
    if (quiesce()) {
      try {
        c10::cuda::CUDAGuard guard(device_);
        release_planes();
      } catch (...) {
        // Destructors cannot report driver failures.
      }
    }
  }
}

std::shared_ptr<VmmOwner> create_vmm_owner(
    int device,
    std::int64_t max_physical_pages,
    std::int64_t bp,
    std::int64_t dqk,
    std::int64_t dv,
    bool global_mass,
    std::int64_t target_chunk_bytes) {
  return VmmOwner::create(
      device, max_physical_pages, bp, dqk, dv, global_mass, target_chunk_bytes);
}

}  // namespace fla_rola::p3
