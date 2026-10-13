# MortonPlus

Exact disk reporting over `u32` points, built on the Morton-order layout of
[MortonSIMD](../mortonSIMD.cc) and adding query techniques from the SPRK-tree.
Points stay sorted by Morton code, and queries start from the same covering cells as
MortonSIMD. The additions change only how candidate ranges are found and how results
are written.

| SPRK-tree technique | MortonPlus equivalent |
|---|---|
| Radius reduction | Below the (at most four) covering cells, descend the quadtree and drop sub-cells whose exact minimum distance exceeds the radius. |
| Lookup table instead of binary search | A table over the top `2L` Morton bits gives each prefix's first rank (about 8 points per bucket). Large buckets in skewed data fall back to MortonSIMD's static B-tree, so a lookup never costs more than in MortonSIMD. |
| Leaf bucket size | A cell is split only while it holds more than `min_points` points (default 128), using the table's exact counts. Empty cells are skipped without any lookup. |
| Two-phase query, bulk compress | Collect merged code intervals first, then scan them. Compressed vectors are stored whole instead of pushed point by point. |

A cell lying entirely inside the disk is copied without distance tests.

Membership is exact for every `u32` coordinate and every `u128` squared radius, including
squared distances up to 2^65. The result is the same multiset of points as MortonSIMD
returns. `Options` switches each technique on or off for ablation; every combination
reports the same points.

## Usage

```rust
use mortonplus::{MortonPlus, Options};

let index = MortonPlus::<8>::new(&points, Options::default()); // points: &[[u32; 2]]
let mut out = Vec::new();
index.query([cx, cy], r2, &mut out); // appends the exact results; r2: u128
```

`query` takes `&self`, so concurrent queries need no index per thread. Each thread keeps
a reusable interval buffer, as SPRK does with its `SCRATCH`.

## Kernels and block size

`W` (the const generic, default 8) is the number of points processed per step, a
multiple of `CHUNK`. The arithmetic is explicit 256-bit SIMD, 4 packed points per
vector. The tier is chosen at compile time (`KERNEL` reports it):

- `avx512`: AVX-512VL compares into mask registers, register-form `vpcompressq` and
  masked tail loads. Memory-destination compress is avoided because it is microcoded
  on Zen 4.
- `avx2`: emulated unsigned compares and permutation-table compression, as in MortonSIMD.
- `portable`: scalar.

Build with `RUSTFLAGS="-Ctarget-cpu=native"` (or an explicit target such as
`x86-64-v3`). Without it the portable tier is used. A global `~/.cargo/config.toml`
with target-specific `rustflags` overrides a crate's `[build] rustflags`, so set
`RUSTFLAGS` explicitly when benchmarking.

Letting the compiler choose the width (plain loops over `[_; W]`) was tried and
rejected: LLVM left W = 4 and 8 scalar inside the query loop and rebuilt comparison
masks bit by bit, and was slower than the explicit kernel everywhere.

## Verification

`cargo test --release` compares every option combination and W ∈ {4, 8, 16} against
brute force on:

- sizes 0 to 40, uniform points, all-identical points, a duplicate-heavy grid,
  clusters, points on a line and coordinates at the domain extremes;
- radii of 0, exactly at real point distances and ±1 around them, at the 2^64 and 2^65
  boundaries, and covering the full domain.

The tests pass on the AVX-512, AVX2, portable and AVX-512F-without-VL builds, and in a
debug build with overflow checks.

## Regression sweep against MortonSIMD

```sh
cargo build --release                     # with RUSTFLAGS as above
target/release/sweep gen DIR              # workload files, about 600 MB
target/release/sweep run DIR > plus.csv   # MortonPlus, W = 4, 8, 16
g++ -std=c++20 -O3 -DNDEBUG -march=native -mavx2 -mbmi2 bench/morton_simd_sweep.cc -o simd_sweep
./simd_sweep DIR > simd.csv               # MortonSIMD on the same files
```

The workloads cover four distributions: uniform, 64 tight clusters, a thin line, and a
256 × 256 duplicate grid. Sizes run from n = 1k to 10M. Query centers are uniform, and
for non-uniform data also placed near existing points. Radii target k = 1, 16, 256 and
4096 answers, plus n/4 for n ≤ 1M, each taken from the median k-th-neighbour distance
over 16 pilot centers. That gives 161 workloads in total. Both programs report answer
counts, which must match.

### Results

One pinned core of an AMD Ryzen 7 7840U (Zen 4). Times are ratios to MortonSIMD on
identical workloads; lower is faster.

| Build | Query time (geometric mean) | Best | Workloads slower than MortonSIMD | Build time |
|---|---|---|---|---|
| `avx512`, W = 4 / 8 / 16 | 0.28 / 0.29 / 0.29 | 0.06 | 9 / 8 / 8 of 161 | 0.4–0.6 |
| `avx2`, W = 4 / 8 / 16 | 0.31 / 0.31 / 0.30 | 0.06 | 12 / 14 / 12 of 161 | 0.5–0.6 |

All variants match MortonSIMD's answer counts on every workload.

The remaining slower cases are all small queries (k ≤ 16) centered inside dense data:
clusters, lines and duplicate grids. They are typically 1.02–1.6× MortonSIMD's time,
5–50 ns per query. Both indexes then cost about one B-tree descent, and MortonPlus adds
a small fixed overhead. These timings are noisy: one workload ranges from 1.02× to 2.44×
across W variants that run the same code path. Larger queries and uniform data are
2–16× faster than MortonSIMD.
