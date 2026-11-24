// THE SHARED-MEMORY LAYOUT (docs/internals/common/smem_layout.md): a kernel's shared memory declared once, as named tiles
// sized from the card's budget at compile time, so that a kernel states no size of its own.
#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>

#include "common/hardware.cuh"

namespace fsla {

//: A TILE OF EXACTLY `COUNT` elements of `T`, named by the type `Tag`
template <class Tag, class T, int COUNT>
struct Fixed {
  using tag = Tag;
  using type = T;
  static constexpr std::int64_t least = COUNT;
  static constexpr std::int64_t weight = 0;
};

//: A TILE OF AT LEAST `LEAST` elements of `T`, and `WEIGHT` parts of what the budget leaves -- see docs/internals/common/smem_layout.md#share
template <class Tag, class T, int LEAST, int WEIGHT>
struct Share {
  using tag = Tag;
  using type = T;
  static constexpr std::int64_t least = LEAST;
  static constexpr std::int64_t weight = WEIGHT;
};

//: A CTA'S SHARED MEMORY on the build `Bt`, `CTAS` CTAs sharing an SM -- see docs/internals/common/smem_layout.md#layout
template <class Bt, int CTAS, class... Tiles>
struct SmemLayout {
  static constexpr int kTiles = static_cast<int>(sizeof...(Tiles));
  static constexpr std::int64_t kAlign = 16;

  //: one CTA's budget: the SM's shared memory over the CTAs that share it, less what the card keeps of each CTA's
  static constexpr std::int64_t kept =
      Bt::traits.caps.smem_per_sm - Bt::traits.caps.smem_per_cta_optin;
  static constexpr std::int64_t budget = Bt::traits.caps.smem_per_sm / CTAS - kept;

  template <class Tile>
  static constexpr std::int64_t element = static_cast<std::int64_t>(sizeof(typename Tile::type));

  template <class Tile>
  static constexpr std::int64_t least_bytes =
      (Tile::least * element<Tile> + kAlign - 1) & ~(kAlign - 1);

  static constexpr std::int64_t least = (std::int64_t{0} + ... + least_bytes<Tiles>);
  static constexpr std::int64_t weights = (std::int64_t{0} + ... + Tiles::weight);
  static_assert(least <= budget, "SmemLayout: the least sizes do not fit this card");

  //: what the budget leaves, dealt to the shares by weight in whole aligned bytes
  static constexpr std::int64_t spare = budget - least;

  template <class Tile>
  static constexpr std::int64_t extra_bytes =
      Tile::weight == 0 ? 0 : (spare * Tile::weight / (weights == 0 ? 1 : weights)) & ~(kAlign - 1);

  template <class Tile>
  static constexpr std::int64_t tile_bytes = least_bytes<Tile> + extra_bytes<Tile>;

  template <class Tile>
  static constexpr std::int64_t tile_count = Tile::least + extra_bytes<Tile> / element<Tile>;

  //: the bytes the launch reserves
  static constexpr std::int64_t bytes = (std::int64_t{0} + ... + tile_bytes<Tiles>);

  //: the place of the tile named `Tag` among the tiles, -1 where no tile or more than one has the name
  template <class Tag>
  static constexpr int index() {
    constexpr bool named[kTiles] = {std::is_same_v<Tag, typename Tiles::tag>...};
    int at = -1, found = 0;
    for (int i = 0; i < kTiles; ++i) {
      if (named[i]) {
        at = i;
        ++found;
      }
    }
    return found == 1 ? at : -1;
  }

  //: `index`, where exactly one tile has the name: any other name does not compile
  template <class Tag>
  static constexpr int place() {
    static_assert(index<Tag>() >= 0, "SmemLayout: exactly one tile has this name");
    return index<Tag>();
  }

  template <class Tag>
  using type = std::tuple_element_t<place<Tag>(), std::tuple<typename Tiles::type...>>;

  //: the elements the tile named `Tag` holds on this card
  template <class Tag>
  static constexpr std::int64_t counted() {
    constexpr std::int64_t each[kTiles] = {tile_count<Tiles>...};
    return each[place<Tag>()];
  }

  template <class Tag>
  static constexpr std::int64_t count = counted<Tag>();

  template <class Tag>
  static constexpr std::int64_t offset() {
    constexpr std::int64_t each[kTiles] = {tile_bytes<Tiles>...};
    std::int64_t at = 0;
    for (int i = 0; i < place<Tag>(); ++i) at += each[i];
    return at;
  }

  //: THE CTA'S SHARED MEMORY, declared for every kernel that lays it out here
  __device__ __forceinline__ static unsigned char* base() {
    extern __shared__ __align__(16) unsigned char fsla_shared_memory[];
    return fsla_shared_memory;
  }

  //: the tile named `Tag`, as its elements
  template <class Tag>
  __device__ __forceinline__ static type<Tag>* get() {
    return reinterpret_cast<type<Tag>*>(base() + offset<Tag>());
  }
};

namespace smem_proof {

//: two cards that differ in shared memory alone: the layout below must fit both, and give the larger card more
struct Large {
  static constexpr ::denseref::arch::Traits traits = {
      {}, ::denseref::arch::Caps{.smem_per_sm = 66560, .smem_per_cta_optin = 65536}};
};

struct Small {
  static constexpr ::denseref::arch::Traits traits = {
      {}, ::denseref::arch::Caps{.smem_per_sm = 33280, .smem_per_cta_optin = 32256}};
};

struct Table;
struct Ring;
struct Panel;

template <class Bt, int CTAS>
using Proof = SmemLayout<Bt, CTAS, Fixed<Table, std::int32_t, 100>, Share<Ring, std::int64_t, 2, 3>,
                         Share<Panel, std::int16_t, 8, 1>>;

using L = Proof<Large, 1>;
using S = Proof<Small, 1>;
static_assert(L::bytes <= L::budget && S::bytes <= S::budget, "a layout fits its card");
static_assert(L::count<Table> == 100 && S::count<Table> == 100, "a fixed tile is the same");
static_assert(L::count<Ring> > S::count<Ring>, "a share is larger on the larger card");
static_assert(S::count<Ring> > 2 && S::count<Panel> > 8, "a share is above its least");
static_assert(L::budget == 65536 && S::budget == 32256, "a lone CTA reserves the card's limit");
static_assert(2 * (Proof<Large, 2>::budget + 1024) == 66560, "two CTAs' budgets fill the SM");
static_assert(S::offset<Ring>() == 400, "a tile's offset is the bytes before it");

}  // namespace smem_proof

}  // namespace fsla
