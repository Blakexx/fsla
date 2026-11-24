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

//: A TILE OF ONE `T`
template <class Tag, class T>
struct One {
  using tag = Tag;
  using type = T;
  //: FIXED by the tile's kind: one element
  static constexpr std::int64_t least = 1;
  //: FIXED by the tile's kind: no part of what the budget leaves
  static constexpr std::int64_t weight = 0;
};

//: A TILE OF WHAT THE BUDGET LEAVES, one part of it: its elements are addressed at run time -- see docs/internals/common/smem_layout.md#rest
template <class Tag, class T>
struct Rest {
  using tag = Tag;
  using type = T;
  //: FIXED by the tile's kind: no least
  static constexpr std::int64_t least = 0;
  //: FIXED by the tile's kind: one part of what the budget leaves
  static constexpr std::int64_t weight = 1;
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
  //: FIXED by the widest load a tile is read with: a tile starts on a 16-byte boundary
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

  template <class Tile>
  static constexpr std::int64_t reserved_least = Tile::weight == 0 ? 0 : least_bytes<Tile>;

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

  //: A TILE WITH NO SHARE IS ITS OWN SHARED OBJECT, so the compiler knows no two overlap; a tile with a share is in what the launch reserves -- see docs/internals/common/smem_layout.md#own
  template <class Tile>
  static constexpr std::int64_t reserved_bytes = Tile::weight == 0 ? 0 : tile_bytes<Tile>;

  static constexpr std::int64_t own_bytes = least - (std::int64_t{0} + ... + reserved_least<Tiles>);

  //: the bytes the launch reserves: the shares'
  static constexpr std::int64_t bytes = (std::int64_t{0} + ... + reserved_bytes<Tiles>);

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

  //: where a tile with a share starts in what the launch reserves
  template <class Tag>
  static constexpr std::int64_t offset() {
    constexpr std::int64_t each[kTiles] = {reserved_bytes<Tiles>...};
    std::int64_t at = 0;
    for (int i = 0; i < place<Tag>(); ++i) at += each[i];
    return at;
  }

  //: the bytes a launch reserves for a call that uses `used` elements of the tile named `Tag`, the last with a share: what a call does not reserve stays the SM's cache -- see docs/internals/common/smem_layout.md#reserve
  template <class Tag>
  static constexpr std::int64_t reserved_to(std::int64_t used) {
    constexpr std::int64_t each[kTiles] = {reserved_bytes<Tiles>...};
    static_assert(offset<Tag>() + each[place<Tag>()] == bytes,
                  "SmemLayout: only the last tile with a share is reserved in part");
    const std::int64_t held = used * static_cast<std::int64_t>(sizeof(type<Tag>));
    return offset<Tag>() + ((held + kAlign - 1) & ~(kAlign - 1));
  }

  //: what the launch reserves of the CTA's shared memory
  __device__ __forceinline__ static unsigned char* base() {
    extern __shared__ __align__(16) unsigned char fsla_shared_memory[];
    return fsla_shared_memory;
  }

  //: a tile with no share: its own shared object
  template <class Tile>
  __device__ __forceinline__ static typename Tile::type* own() {
    __shared__ __align__(16) typename Tile::type cells[Tile::least];
    return cells;
  }

  template <class Tag, class Tile>
  __device__ __forceinline__ static void find(type<Tag>*& at) {
    if constexpr (std::is_same_v<Tag, typename Tile::tag>) {
      if constexpr (Tile::weight == 0) {
        at = own<Tile>();
      } else {
        at = reinterpret_cast<type<Tag>*>(base() + offset<Tag>());
      }
    }
  }

  //: the tile named `Tag`, as its elements
  template <class Tag>
  __device__ __forceinline__ static type<Tag>* get() {
    type<Tag>* at = nullptr;
    (find<Tag, Tiles>(at), ...);
    return at;
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

using RingTile = Share<Ring, std::int64_t, 2, 3>;

template <class Bt, int CTAS>
using Proof = SmemLayout<Bt, CTAS, Fixed<Table, std::int32_t, 100>, RingTile,
                         Share<Panel, std::int16_t, 8, 1>>;

using L = Proof<Large, 1>;
using S = Proof<Small, 1>;
static_assert(L::bytes + L::own_bytes <= L::budget, "a layout fits the larger card");
static_assert(S::bytes + S::own_bytes <= S::budget, "a layout fits the smaller card");
static_assert(L::count<Table> == 100 && S::count<Table> == 100, "a fixed tile is the same");
static_assert(L::count<Ring> > S::count<Ring>, "a share is larger on the larger card");
static_assert(S::count<Ring> > 2 && S::count<Panel> > 8, "a share is above its least");
static_assert(L::budget == 65536 && S::budget == 32256, "a lone CTA reserves the card's limit");
static_assert(2 * (Proof<Large, 2>::budget + 1024) == 66560, "two CTAs' budgets fill the SM");
static_assert(S::own_bytes == 400 && S::offset<Ring>() == 0, "a fixed tile is its own object");
static_assert(S::offset<Panel>() == S::tile_bytes<RingTile>, "shares follow shares");
static_assert(S::reserved_to<Panel>(9) == S::offset<Panel>() + 32, "a call reserves what it uses");

}  // namespace smem_proof

}  // namespace fsla
