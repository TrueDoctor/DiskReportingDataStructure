//! MortonPlus: a Morton-order disk reporting index with SPRK-style query techniques.
//!
//! Points are sorted by Morton code, as in MortonSIMD. A query covers the disk's
//! bounding square with at most four aligned quadtree cells, then optionally:
//! - `refine`: descends below those cells, pruning sub-cells by exact minimum
//!   distance (the quadtree analogue of SPRK radius reduction);
//! - `lut`: finds rank ranges with a prefix table instead of a search;
//! - `min_points`: splits a cell only while it holds more points than this,
//!   using the table's exact counts (the analogue of SPRK's leaf bucket size);
//! - `inside`: copies cells lying entirely inside the disk without distance tests;
//! - `bulk`: writes whole compressed vectors instead of pushing points one by one.
//!
//! Membership is exact for u32 coordinates and u128 squared radii.

pub type Point = [u32; 2];

/// Query techniques; every combination reports the same points. The defaults enable all.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Descend below the covering cells, pruning sub-cells that miss the disk.
    pub refine: bool,
    /// Resolve rank ranges through the prefix table (otherwise only the B-tree).
    pub lut: bool,
    /// Copy cells lying entirely inside the disk without distance tests.
    pub inside: bool,
    /// Write compressed vectors in bulk (otherwise push accepted points one by one).
    pub bulk: bool,
    /// Maximum refinement depth below the covering cells.
    pub extra_levels: u32,
    /// With `lut`, split only cells holding more points than this (0: fixed depth).
    pub min_points: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { refine: true, lut: true, inside: true, bulk: true, extra_levels: 6, min_points: 128 }
    }
}

thread_local! {
    static SCRATCH: std::cell::Cell<Vec<Interval>> = const { std::cell::Cell::new(Vec::new()) };
}

#[derive(Clone, Copy)]
struct Interval {
    first: u64,
    last: u64,
    inside: bool,
}

/// Static B-tree over the sorted codes (MortonSIMD's layered rank index): 8 keys per
/// node, levels stored bottom-up. Level 0 is the sorted code array itself, padded.
struct RankTree {
    n: usize,
    offsets: Vec<usize>,
    keys: Vec<u64>,
}

impl RankTree {
    const B: usize = 8;

    fn blocks(n: usize) -> usize {
        n.div_ceil(Self::B)
    }
    fn prev_keys(n: usize) -> usize {
        (Self::blocks(n) + Self::B) / (Self::B + 1) * Self::B
    }
    fn height(n: usize) -> usize {
        if n <= Self::B {
            1
        } else {
            Self::height(Self::prev_keys(n)) + 1
        }
    }

    fn new(codes: Vec<u64>) -> Self {
        let (b, n) = (Self::B, codes.len());
        let h = Self::height(n);
        let mut offsets = vec![0; h + 1];
        let (mut k, mut m) = (0, n);
        for level in offsets.iter_mut().take(h).skip(1) {
            k += Self::blocks(m) * b;
            m = Self::prev_keys(m);
            *level = k;
        }
        offsets[h] = k + b;
        let mut keys = codes;
        keys.resize(offsets[h], u64::MAX);
        for level in 1..h {
            for i in 0..offsets[level + 1] - offsets[level] {
                let mut k = (i / b) * (b + 1) + i % b + 1;
                for _ in 0..level - 1 {
                    k *= b + 1;
                }
                keys[offsets[level] + i] = if k * b < n { keys[k * b] } else { u64::MAX };
            }
        }
        Self { n, offsets, keys }
    }

    /// The sorted codes.
    fn codes(&self) -> &[u64] {
        &self.keys[..self.n]
    }

    /// Number of keys below `x` in the node starting at `offset`.
    #[inline(always)]
    fn count(&self, offset: usize, x: u64) -> usize {
        debug_assert!(offset + Self::B <= self.keys.len());
        // SAFETY: every node offset produced by lower_bound lies within keys.
        unsafe { kernel::count_below8(self.keys.as_ptr().add(offset), x) }
    }

    /// First rank whose code is at least `x`.
    fn lower_bound(&self, x: u64) -> usize {
        let mut k = 0;
        for level in (1..self.offsets.len() - 1).rev() {
            k = k * (Self::B + 1) + self.count(self.offsets[level] + k, x) * Self::B;
        }
        (k + self.count(k, x)).min(self.n)
    }
}

/// `W` is the number of points processed per step, a multiple of [`CHUNK`].
pub struct MortonPlus<const W: usize = 8> {
    opt: Options,
    ranks: RankTree,
    points: Vec<Point>,
    /// `lut[b]`: first rank whose code prefix (top `2 * lut_level` bits) is at least `b`.
    lut: Vec<u32>,
    lut_level: u32,
}

const EVEN: u64 = 0x5555_5555_5555_5555;
const ODD: u64 = 0xaaaa_aaaa_aaaa_aaaa;

#[inline(always)]
fn deposit(v: u64, mask: u64) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        std::arch::x86_64::_pdep_u64(v, mask)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        // Spread the low 32 bits of v to every other bit, starting at mask's lowest bit.
        let mut x = v & 0xffff_ffff;
        x = (x | x << 16) & 0x0000_ffff_0000_ffff;
        x = (x | x << 8) & 0x00ff_00ff_00ff_00ff;
        x = (x | x << 4) & 0x0f0f_0f0f_0f0f_0f0f;
        x = (x | x << 2) & 0x3333_3333_3333_3333;
        x = (x | x << 1) & EVEN;
        x << mask.trailing_zeros()
    }
}

#[inline(always)]
fn extract(v: u64, mask: u64) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    unsafe {
        std::arch::x86_64::_pext_u64(v, mask) as u32
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let mut x = (v >> mask.trailing_zeros()) & EVEN;
        x = (x | x >> 1) & 0x3333_3333_3333_3333;
        x = (x | x >> 2) & 0x0f0f_0f0f_0f0f_0f0f;
        x = (x | x >> 4) & 0x00ff_00ff_00ff_00ff;
        x = (x | x >> 8) & 0x0000_ffff_0000_ffff;
        x = (x | x >> 16) & 0x0000_0000_ffff_ffff;
        x as u32
    }
}

#[inline(always)]
pub fn morton(p: Point) -> u64 {
    deposit(p[0] as u64, EVEN) | deposit(p[1] as u64, ODD)
}

#[inline(always)]
fn cell_code(x: u32, y: u32, depth: u32) -> u64 {
    if depth == 0 {
        return 0;
    }
    let shift = 2 * (32 - depth);
    if shift == 64 {
        0
    } else {
        morton([x, y]) >> shift << shift
    }
}

/// Appends the cell's code interval, merging it with an adjacent predecessor.
#[inline(always)]
fn push_interval(intervals: &mut Vec<Interval>, code: u64, depth: u32, inside: bool) {
    let last = code | cell_span(depth);
    match intervals.last_mut() {
        Some(prev) if prev.inside == inside && prev.last.wrapping_add(1) == code => prev.last = last,
        _ => intervals.push(Interval { first: code, last, inside }),
    }
}

#[inline(always)]
fn cell_span(depth: u32) -> u64 {
    if depth == 0 {
        u64::MAX
    } else {
        (1u64 << (2 * (32 - depth))) - 1
    }
}

#[inline(always)]
fn axis_dist(lo: u32, hi: u64, c: u32) -> u64 {
    (lo as u64).saturating_sub(c as u64).max((c as u64).saturating_sub(hi))
}

#[inline(always)]
fn axis_far(lo: u32, hi: u64, c: u32) -> u64 {
    (c as u64).abs_diff(lo as u64).max((c as u64).abs_diff(hi))
}

/// Exact disk membership and output on chunks of `C` packed points.
///
/// Tiers, chosen at compile time:
/// - `avx512`: 256-bit arithmetic (4 points per ymm) with AVX-512VL compares into mask
///   registers, register-form `vpcompressq` and masked tail loads;
/// - `avx2`: 4 points per ymm, emulated unsigned compares, permutation-table compress;
/// - `portable`: scalar.
mod kernel {
    use super::Point;

    /// Query disk. Callers guarantee r2 < 2^65, so the high word is 0 or 1.
    #[derive(Clone, Copy)]
    pub struct Disk {
        pub c: Point,
        pub r2: u128,
    }

    #[inline(always)]
    pub fn contains(p: Point, d: &Disk) -> bool {
        let dx = p[0].abs_diff(d.c[0]) as u128;
        let dy = p[1].abs_diff(d.c[1]) as u128;
        dx * dx + dy * dy <= d.r2
    }

    // ── AVX-512 (VL), 256-bit arithmetic ──────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512vl"))]
    mod imp {
        use super::{Disk, Point};
        use std::arch::x86_64::*;
        pub const NAME: &str = "avx512";
        pub const C: usize = 4;
        pub const MASKED_TAIL: bool = true;
        pub type V = __m256i;
        pub struct Prepared {
            cx: __m256i,
            cy: __m256i,
            low: __m256i,
            r2: __m256i,
            hi: bool,
        }

        #[inline(always)]
        pub fn prepare(d: &Disk) -> Prepared {
            unsafe {
                Prepared {
                    cx: _mm256_set1_epi64x(d.c[0] as i64),
                    cy: _mm256_set1_epi64x(d.c[1] as i64),
                    low: _mm256_set1_epi64x(0xFFFF_FFFF),
                    r2: _mm256_set1_epi64x(d.r2 as u64 as i64),
                    hi: d.r2 >> 64 != 0,
                }
            }
        }
        #[inline(always)]
        pub unsafe fn load(p: *const Point) -> V {
            unsafe { _mm256_loadu_si256(p as *const V) }
        }
        /// Loads the first `count` (< C) points; masked-off lanes are not read.
        #[inline(always)]
        pub unsafe fn load_partial(p: *const Point, count: usize) -> (V, u32) {
            let k = (1u8 << count) - 1;
            unsafe { (_mm256_maskz_loadu_epi64(k, p as *const i64), k as u32) }
        }
        #[inline(always)]
        pub fn mask(v: V, pr: &Prepared) -> u32 {
            unsafe {
                let x = _mm256_and_si256(v, pr.low);
                let y = _mm256_srli_epi64::<32>(v);
                let dx = _mm256_sub_epi32(_mm256_max_epu32(x, pr.cx), _mm256_min_epu32(x, pr.cx));
                let dy = _mm256_sub_epi32(_mm256_max_epu32(y, pr.cy), _mm256_min_epu32(y, pr.cy));
                let xx = _mm256_mul_epu32(dx, dx);
                let sum = _mm256_add_epi64(xx, _mm256_mul_epu32(dy, dy));
                // The true squared distance is sum + 2^64 * carry, with carry <= 1.
                let carry = _mm256_cmplt_epu64_mask(sum, xx);
                let le = _mm256_cmple_epu64_mask(sum, pr.r2);
                (if pr.hi { le | !carry } else { le & !carry }) as u32 & 0xF
            }
        }
        /// Writes the selected points to dst (C slots of room); returns their count.
        #[inline(always)]
        pub unsafe fn emit(v: V, mask: u32, dst: *mut Point) -> usize {
            // Register compress, then a plain store: memory-destination compress is microcoded on Zen 4.
            unsafe { _mm256_storeu_si256(dst as *mut V, _mm256_maskz_compress_epi64(mask as u8, v)) };
            mask.count_ones() as usize
        }
    }

    // ── AVX2 ──────────────────────────────────────────────────────────────────
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(target_feature = "avx512f", target_feature = "avx512vl"))
    ))]
    mod imp {
        use super::{Disk, Point};
        use std::arch::x86_64::*;
        pub const NAME: &str = "avx2";
        pub const C: usize = 4;
        pub const MASKED_TAIL: bool = false;
        pub type V = __m256i;
        pub struct Prepared {
            cx: __m256i,
            cy: __m256i,
            low: __m256i,
            sign: __m256i,
            r2: __m256i,
            hi: bool,
        }

        /// For each 4-bit mask, the u32 lane order that packs the selected points first.
        static PERMUTATIONS: [[i32; 8]; 16] = {
            let mut table = [[0i32; 8]; 16];
            let mut mask = 0;
            while mask < 16 {
                let (mut next, mut lane) = (0, 0);
                while lane < 4 {
                    if mask & (1 << lane) != 0 {
                        table[mask][next] = 2 * lane;
                        table[mask][next + 1] = 2 * lane + 1;
                        next += 2;
                    }
                    lane += 1;
                }
                mask += 1;
            }
            table
        };

        #[inline(always)]
        pub fn prepare(d: &Disk) -> Prepared {
            unsafe {
                let sign = _mm256_set1_epi64x(i64::MIN);
                Prepared {
                    cx: _mm256_set1_epi64x(d.c[0] as i64),
                    cy: _mm256_set1_epi64x(d.c[1] as i64),
                    low: _mm256_set1_epi64x(0xFFFF_FFFF),
                    sign,
                    // Pre-flipped sign bit for the unsigned comparison.
                    r2: _mm256_xor_si256(_mm256_set1_epi64x(d.r2 as u64 as i64), sign),
                    hi: d.r2 >> 64 != 0,
                }
            }
        }
        #[inline(always)]
        pub unsafe fn load(p: *const Point) -> V {
            unsafe { _mm256_loadu_si256(p as *const V) }
        }
        pub unsafe fn load_partial(_p: *const Point, _count: usize) -> (V, u32) {
            unreachable!()
        }
        #[inline(always)]
        pub fn mask(v: V, pr: &Prepared) -> u32 {
            unsafe {
                let x = _mm256_and_si256(v, pr.low);
                let y = _mm256_srli_epi64::<32>(v);
                let dx = _mm256_sub_epi32(_mm256_max_epu32(x, pr.cx), _mm256_min_epu32(x, pr.cx));
                let dy = _mm256_sub_epi32(_mm256_max_epu32(y, pr.cy), _mm256_min_epu32(y, pr.cy));
                let xx = _mm256_mul_epu32(dx, dx);
                let sum = _mm256_add_epi64(xx, _mm256_mul_epu32(dy, dy));
                let (sum_s, xx_s) = (_mm256_xor_si256(sum, pr.sign), _mm256_xor_si256(xx, pr.sign));
                let carry = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(xx_s, sum_s))) as u32;
                let gt = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(sum_s, pr.r2))) as u32;
                let le = !gt & 0xF;
                (if pr.hi { le | !carry } else { le & !carry }) & 0xF
            }
        }
        #[inline(always)]
        pub unsafe fn emit(v: V, mask: u32, dst: *mut Point) -> usize {
            unsafe {
                let perm = _mm256_loadu_si256(PERMUTATIONS[mask as usize].as_ptr() as *const V);
                _mm256_storeu_si256(dst as *mut V, _mm256_permutevar8x32_epi32(v, perm));
            }
            mask.count_ones() as usize
        }
    }

    // ── Portable ──────────────────────────────────────────────────────────────
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    mod imp {
        use super::{contains, Disk, Point};
        pub const NAME: &str = "portable";
        pub const C: usize = 4;
        pub const MASKED_TAIL: bool = false;
        pub type V = [Point; 4];
        pub struct Prepared(Disk);

        #[inline(always)]
        pub fn prepare(d: &Disk) -> Prepared {
            Prepared(*d)
        }
        #[inline(always)]
        pub unsafe fn load(p: *const Point) -> V {
            unsafe { (p as *const V).read_unaligned() }
        }
        pub unsafe fn load_partial(_p: *const Point, _count: usize) -> (V, u32) {
            unreachable!()
        }
        #[inline(always)]
        pub fn mask(v: V, pr: &Prepared) -> u32 {
            let mut m = 0;
            for (i, &p) in v.iter().enumerate() {
                m |= (contains(p, &pr.0) as u32) << i;
            }
            m
        }
        #[inline(always)]
        pub unsafe fn emit(v: V, mask: u32, dst: *mut Point) -> usize {
            let mut n = 0;
            for (i, &p) in v.iter().enumerate() {
                unsafe { dst.add(n).write(p) };
                n += (mask >> i & 1) as usize;
            }
            n
        }
    }

    /// Number of the 8 keys at `p` that are below `x` (one B-tree node).
    #[inline(always)]
    pub unsafe fn count_below8(p: *const u64, x: u64) -> usize {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            // One compare into a mask register, popcounted (autovectorized code extracts the
            // mask bit by bit instead). Measured faster here than two 256-bit compares.
            use std::arch::x86_64::*;
            let keys = _mm512_loadu_si512(p as *const _);
            _mm512_cmplt_epu64_mask(keys, _mm512_set1_epi64(x as i64)).count_ones() as usize
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f")))]
        unsafe {
            use std::arch::x86_64::*;
            let sign = _mm256_set1_epi64x(i64::MIN);
            let xs = _mm256_xor_si256(_mm256_set1_epi64x(x as i64), sign);
            let below = |v: __m256i| {
                let lt = _mm256_cmpgt_epi64(xs, _mm256_xor_si256(v, sign));
                _mm256_movemask_pd(_mm256_castsi256_pd(lt)).count_ones() as usize
            };
            below(_mm256_loadu_si256(p as *const __m256i)) + below(_mm256_loadu_si256(p.add(4) as *const __m256i))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            (0..8).filter(|&i| unsafe { *p.add(i) } < x).count()
        }
    }

    pub use imp::{emit, load, load_partial, mask, prepare, C, MASKED_TAIL, NAME};
}

/// Name of the compiled membership kernel: "avx512", "avx2" or "portable".
pub const KERNEL: &str = kernel::NAME;
/// Points per vector chunk of the compiled kernel; `W` must be a multiple of it.
pub const CHUNK: usize = kernel::C;
use kernel::Disk;

impl<const W: usize> MortonPlus<W> {
    /// Builds the index. Panics if there are 2^32 or more points (ranks are stored as u32).
    pub fn new(input: &[Point], opt: Options) -> Self {
        const { assert!(W > 0 && W.is_multiple_of(kernel::C), "W must be a positive multiple of CHUNK") };
        assert!(input.len() < u32::MAX as usize, "MortonPlus supports fewer than 2^32 points");
        let mut sorted: Vec<(u64, Point)> = input.iter().map(|&p| (morton(p), p)).collect();
        sorted.sort_unstable_by_key(|e| e.0);
        let codes: Vec<u64> = sorted.iter().map(|e| e.0).collect();
        let points: Vec<Point> = sorted.iter().map(|e| e.1).collect();
        // About 8 points per table bucket.
        let mut lut_level = 1;
        while lut_level < 13 && (1usize << (2 * lut_level)) * 8 < codes.len() {
            lut_level += 1;
        }
        let buckets = 1usize << (2 * lut_level);
        let mut lut = vec![codes.len() as u32; buckets + 1];
        for (i, &code) in codes.iter().enumerate().rev() {
            lut[(code >> (64 - 2 * lut_level)) as usize] = i as u32;
        }
        for b in (0..buckets).rev() {
            lut[b] = lut[b].min(lut[b + 1]);
        }
        Self { opt, ranks: RankTree::new(codes), points, lut, lut_level }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Appends every point with squared distance at most `r2` from `c` to `out`.
    pub fn query(&self, c: Point, r2: u128, out: &mut Vec<Point>) {
        if self.points.is_empty() {
            return;
        }
        let max = u32::MAX as u128;
        if r2 >= 2 * max * max {
            self.scan_range(0, self.points.len(), true, c, r2, out);
            return;
        }
        // r2 < 2^65 here: convert the two words natively (u128 -> f64 is a slow libcall).
        let r2f = (r2 >> 64) as u64 as f64 * 18_446_744_073_709_551_616.0 + r2 as u64 as f64;
        let mut r = r2f.sqrt().ceil() as u64;
        while (r as u128) * (r as u128) < r2 {
            r += 1;
        }
        while r > 0 && ((r - 1) as u128) * ((r - 1) as u128) >= r2 {
            r -= 1;
        }
        let r = r.min(u32::MAX as u64);
        let lo = [c[0].saturating_sub(r as u32), c[1].saturating_sub(r as u32)];
        let hi = [(c[0] as u64 + r).min(u32::MAX as u64) as u32, (c[1] as u64 + r).min(u32::MAX as u64) as u32];
        let diameter = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let depth = if diameter == 0 { 32 } else { diameter.leading_zeros() };
        // The same covering cells as Morton: the bounding square meets at most four.
        let mut cells = [
            cell_code(lo[0], lo[1], depth),
            cell_code(hi[0], lo[1], depth),
            cell_code(lo[0], hi[1], depth),
            cell_code(hi[0], hi[1], depth),
        ];
        cells.sort_unstable();
        let target = if self.opt.refine { (depth + self.opt.extra_levels).min(32) } else { depth };
        let splittable = depth < target && !(self.opt.min_points > 0 && depth >= self.lut_level);
        if !splittable {
            // Fast path, as in MortonSIMD: no covering cell can be split, so at most four
            // intervals; keep them on the stack. A covering cell is wider than the disk's
            // diameter, so it is never inside.
            let mut buf = [Interval { first: 0, last: 0, inside: false }; 4];
            let mut len = 0;
            for i in 0..4 {
                if (i == 0 || cells[i] != cells[i - 1]) && Self::cell_meets_disk(cells[i], depth, c, r2) {
                    let last = cells[i] | cell_span(depth);
                    if len > 0 && buf[len - 1].last.wrapping_add(1) == cells[i] {
                        buf[len - 1].last = last;
                    } else {
                        buf[len] = Interval { first: cells[i], last, inside: false };
                        len += 1;
                    }
                }
            }
            self.scan_intervals(&buf[..len], c, r2, out);
            return;
        }
        // Per-thread interval buffer, reused across queries (as SPRK's SCRATCH).
        let mut intervals = SCRATCH.take();
        intervals.clear();
        for i in 0..4 {
            if i == 0 || cells[i] != cells[i - 1] {
                self.descend(cells[i], depth, target, c, r2, &mut intervals);
            }
        }
        self.scan_intervals(&intervals, c, r2, out);
        SCRATCH.set(intervals);
    }

    /// Scans ascending code intervals. The previous end bounds the next start (as in
    /// MortonSIMD): search only when codes lie in between, and skip the end search for
    /// empty intervals.
    #[inline(always)]
    fn scan_intervals(&self, intervals: &[Interval], c: Point, r2: u128, out: &mut Vec<Point>) {
        let (codes, n) = (self.ranks.codes(), self.points.len());
        let mut cursor = 0;
        for iv in intervals {
            let s = if cursor < n && codes[cursor] < iv.first { self.rank_lower(iv.first) } else { cursor };
            if s == n || codes[s] > iv.last {
                cursor = s;
                continue;
            }
            let e = if iv.last == u64::MAX { n } else { self.rank_lower(iv.last + 1) };
            self.scan_range(s, e, iv.inside, c, r2, out);
            cursor = e;
        }
    }

    /// Exact test whether the cell `code` at `depth` intersects the disk.
    #[inline(always)]
    fn cell_meets_disk(code: u64, depth: u32, c: Point, r2: u128) -> bool {
        let (x0, y0) = (extract(code, EVEN), extract(code, ODD));
        let side = 1u64 << (32 - depth);
        let (x1, y1) = (x0 as u64 + side - 1, y0 as u64 + side - 1);
        let (dx, dy) = (axis_dist(x0, x1, c[0]) as u128, axis_dist(y0, y1, c[1]) as u128);
        dx * dx + dy * dy <= r2
    }

    /// Depth-first over the quadtree in Morton order; leaves become merged code intervals.
    fn descend(&self, code: u64, depth: u32, target: u32, c: Point, r2: u128, intervals: &mut Vec<Interval>) {
        if !Self::cell_meets_disk(code, depth, c, r2) {
            return; // pruned: the cell misses the disk
        }
        let (x0, y0) = (extract(code, EVEN), extract(code, ODD));
        let side = 1u64 << (32 - depth);
        let (x1, y1) = (x0 as u64 + side - 1, y0 as u64 + side - 1);
        // Exact point count from the prefix table, for cells no deeper than the table.
        let mut count = usize::MAX;
        if self.opt.lut && self.opt.min_points > 0 && depth <= self.lut_level {
            let shift = 2 * (self.lut_level - depth);
            let bucket = if depth == 0 { 0 } else { (code >> (64 - 2 * depth)) << shift } as usize;
            count = (self.lut[bucket + (1usize << shift)] - self.lut[bucket]) as usize;
            if count == 0 {
                return; // empty cell: no range lookup needed
            }
        }
        let mut inside = false;
        // The farthest corner is at least half a diagonal away: skip cells that cannot fit.
        let half_diag2 = (side as u128 - 1) * (side as u128 - 1) / 2;
        if self.opt.inside && half_diag2 <= r2 {
            let (fx, fy) = (axis_far(x0, x1, c[0]) as u128, axis_far(y0, y1, c[1]) as u128);
            inside = fx * fx + fy * fy <= r2;
        }
        let small = self.opt.min_points > 0 && (count <= self.opt.min_points || depth >= self.lut_level);
        if !inside && depth < target && !small {
            let shift = 2 * (32 - depth - 1);
            for child in 0..4u64 {
                self.descend(code | child << shift, depth + 1, target, c, r2, intervals);
            }
            return;
        }
        push_interval(intervals, code, depth, inside);
    }

    fn rank_lower(&self, key: u64) -> usize {
        if !self.opt.lut {
            return self.ranks.lower_bound(key);
        }
        let bucket = (key >> (64 - 2 * self.lut_level)) as usize;
        let (mut lo, hi) = (self.lut[bucket] as usize, self.lut[bucket + 1] as usize);
        if key & ((1u64 << (64 - 2 * self.lut_level)) - 1) == 0 {
            return lo;
        }
        if hi - lo > 16 {
            // Skewed data: a large bucket. The B-tree bounds the cost to MortonSIMD's.
            return self.ranks.lower_bound(key);
        }
        let codes = self.ranks.codes();
        while lo < hi && codes[lo] < key {
            lo += 1; // small bucket
        }
        lo
    }

    fn scan_range(&self, start: usize, end: usize, inside: bool, c: Point, r2: u128, out: &mut Vec<Point>) {
        if inside {
            out.extend_from_slice(&self.points[start..end]);
            return;
        }
        let d = Disk { c, r2 };
        if self.opt.bulk {
            self.report_bulk(start, end, &d, out);
        } else {
            self.report_push(start, end, &d, out);
        }
    }

    /// Morton's original output: one push per accepted point.
    fn report_push(&self, start: usize, end: usize, d: &Disk, out: &mut Vec<Point>) {
        let pr = kernel::prepare(d);
        let mut i = start;
        while end - i >= kernel::C {
            // SAFETY: C points from i are in bounds.
            let mask = kernel::mask(unsafe { kernel::load(self.points.as_ptr().add(i)) }, &pr);
            for j in 0..kernel::C {
                if mask >> j & 1 == 1 {
                    out.push(self.points[i + j]);
                }
            }
            i += kernel::C;
        }
        out.extend(self.points[i..end].iter().filter(|&&p| kernel::contains(p, d)));
    }

    /// Bulk output: reserve once per range, then per step of W points compute all
    /// chunk masks first and store whole compressed vectors, advancing by popcount.
    fn report_bulk(&self, start: usize, end: usize, d: &Disk, out: &mut Vec<Point>) {
        let pr = kernel::prepare(d);
        out.reserve(end - start + W);
        let mut n = out.len();
        let dst = out.as_mut_ptr();
        let src = self.points.as_ptr();
        let mut i = start;
        // SAFETY (whole block): reads stay within [start, end); writes stay within the
        // reserved capacity, which covers the range plus W slack slots.
        unsafe {
            while end - i >= W {
                // Masks for all chunks of the step first, then the stores (only every
                // C-th slot of these arrays is used).
                let mut v = [kernel::load(src.add(i)); W];
                let mut m = [0u32; W];
                for j in (0..W).step_by(kernel::C) {
                    v[j] = kernel::load(src.add(i + j));
                    m[j] = kernel::mask(v[j], &pr);
                }
                for j in (0..W).step_by(kernel::C) {
                    n += kernel::emit(v[j], m[j], dst.add(n));
                }
                i += W;
            }
            while end - i >= kernel::C {
                let v = kernel::load(src.add(i));
                n += kernel::emit(v, kernel::mask(v, &pr), dst.add(n));
                i += kernel::C;
            }
            if i < end {
                if kernel::MASKED_TAIL {
                    let (v, valid) = kernel::load_partial(src.add(i), end - i);
                    n += kernel::emit(v, kernel::mask(v, &pr) & valid, dst.add(n));
                } else {
                    for &p in &self.points[i..end] {
                        dst.add(n).write(p);
                        n += kernel::contains(p, d) as usize;
                    }
                }
            }
            out.set_len(n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn u32(&mut self) -> u32 {
            (self.next() >> 32) as u32
        }
    }

    fn d2(p: Point, c: Point) -> u128 {
        let (dx, dy) = (p[0].abs_diff(c[0]) as u128, p[1].abs_diff(c[1]) as u128);
        dx * dx + dy * dy
    }

    fn variants() -> Vec<Options> {
        let mut v = Vec::new();
        for refine in [false, true] {
            for lut in [false, true] {
                for inside in [false, true] {
                    for bulk in [false, true] {
                        for (extra_levels, min_points) in [(0, 0), (2, 0), (6, 128), (32, 1), (32, 8)] {
                            v.push(Options { refine, lut, inside, bulk, extra_levels, min_points });
                        }
                    }
                }
            }
        }
        v
    }

    /// Radii that stress the exact kernel: real point distances and their neighbours,
    /// the 2^64 boundary of the low word, and the full-domain limit.
    fn radii(points: &[Point], c: Point, rng: &mut Rng) -> Vec<u128> {
        let max = u32::MAX as u128;
        let mut r = vec![0, 1, 1 << 64, (1 << 64) - 1, (1 << 64) + 1, 2 * max * max - 1, 2 * max * max, u128::MAX];
        for _ in 0..6 {
            if let Some(&p) = points.get(rng.next() as usize % points.len().max(1)) {
                let d = d2(p, c);
                r.extend([d, d.saturating_sub(1), d + 1, d / (1 + rng.next() as u128 % 4000)]);
            }
        }
        r
    }

    fn check(points: &[Point], queries: usize, rng: &mut Rng) {
        check_w::<{ CHUNK }>(points, queries, rng);
        check_w::<{ 2 * CHUNK }>(points, queries, rng);
        check_w::<{ 4 * CHUNK }>(points, queries, rng);
    }

    fn check_w<const W: usize>(points: &[Point], queries: usize, rng: &mut Rng) {
        let variants = variants();
        let indexes: Vec<_> = variants.iter().map(|&o| MortonPlus::<W>::new(points, o)).collect();
        let mut out = Vec::new();
        let corners = [[0, 0], [u32::MAX, u32::MAX], [0, u32::MAX], [u32::MAX, 0], [1 << 31, 1 << 31]];
        for q in 0..queries {
            let c = match q % 3 {
                0 => corners[q / 3 % corners.len()],
                1 if !points.is_empty() => points[rng.next() as usize % points.len()],
                _ => [rng.u32(), rng.u32()],
            };
            for r2 in radii(points, c, rng) {
                let mut truth: Vec<Point> = points.iter().copied().filter(|&p| d2(p, c) <= r2).collect();
                truth.sort_unstable();
                for index in &indexes {
                    out.clear();
                    out.push([7, 7]); // results are appended, not overwritten
                    index.query(c, r2, &mut out);
                    assert_eq!(out[0], [7, 7]);
                    out.remove(0);
                    out.sort_unstable();
                    assert_eq!(out, truth, "W {W}, options {:?}, center {c:?}, r2 {r2}", index.opt);
                }
            }
        }
    }

    #[test]
    fn exact_on_tiny_inputs() {
        let mut rng = Rng(3);
        for n in 0..=40 {
            let points: Vec<Point> = (0..n).map(|_| [rng.u32(), rng.u32()]).collect();
            check(&points, 6, &mut rng);
        }
    }

    #[test]
    fn exact_on_uniform_points() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let points: Vec<Point> = (0..5_000).map(|_| [rng.u32(), rng.u32()]).collect();
        check(&points, 30, &mut rng);
    }

    #[test]
    fn exact_on_duplicates() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let grid: Vec<Point> = (0..5_000).map(|_| [rng.u32() & 0xFF00_0000, rng.u32() & 0xFF00_0000]).collect();
        check(&grid, 30, &mut rng);
        let same = vec![[123_456, 654_321]; 3_000];
        check(&same, 15, &mut rng);
    }

    #[test]
    fn exact_on_clusters_and_lines() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let centers: Vec<Point> = (0..8).map(|_| [rng.u32(), rng.u32()]).collect();
        let clusters: Vec<Point> = (0..5_000)
            .map(|_| {
                let c = centers[rng.next() as usize % centers.len()];
                [c[0].wrapping_add(rng.u32() >> 16), c[1].wrapping_add(rng.u32() >> 16)]
            })
            .collect();
        check(&clusters, 30, &mut rng);
        let line: Vec<Point> = (0..5_000)
            .map(|_| {
                let x = rng.u32();
                [x, x ^ (rng.u32() >> 24)]
            })
            .collect();
        check(&line, 30, &mut rng);
    }

    #[test]
    fn exact_near_coordinate_extremes() {
        let mut rng = Rng(11);
        let points: Vec<Point> =
            (0..5_000).map(|_| [(rng.u32() % 64).wrapping_sub(32), (rng.u32() % 64).wrapping_sub(32)]).collect();
        check(&points, 30, &mut rng);
    }

    #[test]
    fn rank_tree_matches_partition_point() {
        let mut rng = Rng(5);
        for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 72, 73, 80, 81, 1000, 4097, 50_000] {
            for spread in [u64::MAX, 1000, 3] {
                let mut codes: Vec<u64> = (0..n).map(|_| rng.next() % spread.max(1)).collect();
                codes.sort_unstable();
                let tree = RankTree::new(codes.clone());
                assert_eq!(tree.codes(), &codes[..]);
                let mut probes: Vec<u64> =
                    codes.iter().flat_map(|&c| [c, c.wrapping_add(1), c.wrapping_sub(1)]).collect();
                probes.extend([0, 1, u64::MAX, u64::MAX - 1]);
                probes.extend((0..200).map(|_| rng.next() % spread.max(1)));
                for x in probes {
                    assert_eq!(tree.lower_bound(x), codes.partition_point(|&k| k < x), "n {n}, x {x}");
                }
            }
        }
    }

    #[test]
    fn reports_kernel() {
        println!("kernel={KERNEL} chunk={CHUNK}");
    }

    #[test]
    fn morton_matches_reference_interleave() {
        let mut rng = Rng(7);
        for _ in 0..1000 {
            let p = [rng.u32(), rng.u32()];
            let mut expect = 0u64;
            for bit in 0..32 {
                expect |= ((p[0] as u64 >> bit) & 1) << (2 * bit) | ((p[1] as u64 >> bit) & 1) << (2 * bit + 1);
            }
            assert_eq!(morton(p), expect);
            assert_eq!([extract(expect, EVEN), extract(expect, ODD)], p);
        }
    }
}
