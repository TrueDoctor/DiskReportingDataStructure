//! Local exact-integer reporting adapter around the upstream SPRK tree.
//! The f32 tree selects leaf ranges with a conservative radius. Each leaf block
//! also stores its original uint32 points, so membership is decided exactly in
//! integer arithmetic and accepted points are written straight from the leaf.
use crate::output::IdDist;
use crate::scalar::IdStorage;
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount};
use crate::tree::{LeafRange, Sprk};

/// Conservative envelope for uint32 coordinates normalized by 2^32.
/// The absolute term covers f32 coordinate rounding; the relative term covers
/// distance arithmetic. Higher-dimensional half-distance kernels also need a
/// cancellation allowance because they subtract dot products of absolute positions.
pub fn candidate_radius<const D: usize>(radius: f64) -> f32 {
    let epsilon = f32::EPSILON as f64;
    let rounded = radius + 2.0 * (D as f64).sqrt() * epsilon;
    let squared = rounded * rounded * (1.0 + 64.0 * D as f64 * epsilon)
        + if D >= 6 { 32.0 * (D * D) as f64 * epsilon } else { 0.0 };
    (squared.sqrt() as f32).next_up()
}

const W: usize = 8;

/// The points of one tree block, packed as `x | y << 32` (the `[u32; 2]` layout).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block {
    points: [u64; W],
    ids: [u32; W],
    valid: u8,
}

pub struct IntegerPointSprk {
    tree: Option<Sprk<2, W, f32, u32>>,
    blocks: Vec<Block>,
    ranges: Vec<LeafRange>,
}

#[inline(always)]
fn pack(p: [u32; 2]) -> u64 { p[0] as u64 | (p[1] as u64) << 32 }

/// Exact membership of a block's points; bit i is set when point i is inside.
/// Squared distances are below 2^65, so a carry out of the sum means outside.
#[inline(always)]
fn inside_mask(block: &Block, cx: u64, cy: u64, radius2: u64) -> u8 {
    // Autovectorized code rebuilds the mask bit by bit; compare into a mask register instead.
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::*;
        let p = _mm512_load_epi64(block.points.as_ptr().cast());
        let (cx, cy) = (_mm512_set1_epi64(cx as i64), _mm512_set1_epi64(cy as i64));
        let x = _mm512_and_si512(p, _mm512_set1_epi64(0xFFFF_FFFF));
        let y = _mm512_srli_epi64::<32>(p);
        let dx = _mm512_sub_epi64(_mm512_max_epu64(x, cx), _mm512_min_epu64(x, cx));
        let dy = _mm512_sub_epi64(_mm512_max_epu64(y, cy), _mm512_min_epu64(y, cy));
        let xx = _mm512_mul_epu32(dx, dx);
        let sum = _mm512_add_epi64(xx, _mm512_mul_epu32(dy, dy));
        let no_carry = _mm512_mask_cmpge_epu64_mask(block.valid, sum, xx);
        return _mm512_mask_cmple_epu64_mask(no_carry, sum, _mm512_set1_epi64(radius2 as i64));
    }
    #[allow(unreachable_code)]
    let mut mask = 0u8;
    for i in 0..W {
        let p = block.points[i];
        let dx = (p & 0xFFFF_FFFF).abs_diff(cx);
        let dy = (p >> 32).abs_diff(cy);
        let xx = dx * dx;
        let sum = xx.wrapping_add(dy * dy);
        mask |= ((sum >= xx) & (sum <= radius2)) as u8 * (1 << i);
    }
    mask & block.valid
}

/// Appends the selected lanes of `src` to `out`, which must have room for W more.
#[inline(always)]
unsafe fn compress_store(src: &[u64; W], mask: u8, out: *mut u64) -> usize {
    #[cfg(target_feature = "avx512f")]
    unsafe {
        use std::arch::x86_64::*;
        // Compress in a register: memory-destination compress is microcoded on Zen 4.
        let v = _mm512_maskz_compress_epi64(mask, _mm512_load_epi64(src.as_ptr().cast()));
        _mm512_storeu_si512(out.cast(), v);
    }
    #[cfg(not(target_feature = "avx512f"))]
    {
        let mut n = 0;
        for i in 0..W {
            unsafe { *out.add(n) = src[i] };
            n += (mask >> i & 1) as usize;
        }
    }
    mask.count_ones() as usize
}

impl IntegerPointSprk {
    pub fn new(points: &[[u32; 2]]) -> Self {
        if points.is_empty() {
            return Self { tree: None, blocks: Vec::new(), ranges: Vec::new() };
        }
        let coords: Vec<_> = points.iter().map(|p| p.map(|x| (x as f64 / 4294967296.0) as f32)).collect();
        let tree = Sprk::new(&coords);
        let blocks = tree.positions_sorted.iter().map(|pdvec| {
            let mut block = Block { points: [0; W], ids: pdvec.ids, valid: 0 };
            for (i, &id) in pdvec.ids.iter().enumerate() {
                if id == u32::SENTINEL { continue; }
                block.points[i] = pack(points[id as usize]);
                block.valid |= 1 << i;
            }
            block
        }).collect();
        Self { tree: Some(tree), blocks, ranges: Vec::new() }
    }

    /// Calls `emit` with each block range the disk may touch, plus the exact predicate.
    fn scan(&mut self, center: [u32; 2], radius2: u128, mut emit: impl FnMut(&Block, u8)) {
        let Some(tree) = &self.tree else { return };
        let pos = center.map(|x| (x as f64 / 4294967296.0) as f32);
        let radius = candidate_radius::<2>((radius2 as f64).sqrt() / 4294967296.0);
        self.ranges.clear();
        tree.collect_ranges(&pos, 0, 0, radius * radius, &mut [0.0; 2], &mut self.ranges);
        let (cx, cy) = (center[0] as u64, center[1] as u64);
        if radius2 <= u64::MAX as u128 {
            for range in &self.ranges {
                for block in &self.blocks[range.min_i..range.max_i] {
                    emit(block, inside_mask(block, cx, cy, radius2 as u64));
                }
            }
        } else {
            // Radius beyond 2^32: rare, use 128-bit arithmetic.
            for range in &self.ranges {
                for block in &self.blocks[range.min_i..range.max_i] {
                    let mut mask = 0u8;
                    for i in 0..W {
                        let p = block.points[i];
                        let dx = (p & 0xFFFF_FFFF).abs_diff(cx) as u128;
                        let dy = (p >> 32).abs_diff(cy) as u128;
                        mask |= ((dx * dx + dy * dy <= radius2) as u8) << i;
                    }
                    emit(block, mask & block.valid);
                }
            }
        }
    }

    pub fn query_points(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<[u32; 2]>) {
        self.scan(center, radius2, |block, mask| {
            out.reserve(W);
            let len = out.len();
            // SAFETY: room for W points was reserved; [u32; 2] and u64 share layout.
            unsafe {
                let n = compress_store(&block.points, mask, out.as_mut_ptr().add(len).cast());
                out.set_len(len + n);
            }
        });
    }

    pub fn query_ids(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<usize>) {
        self.scan(center, radius2, |block, mask| {
            for i in 0..W {
                if mask >> i & 1 == 1 { out.push(block.ids[i] as usize); }
            }
        });
    }
}

fn contains(point: [u32; 2], center: [u32; 2], radius2: u128) -> bool {
    let dx = (point[0] as i64 - center[0] as i64).unsigned_abs() as u128;
    let dy = (point[1] as i64 - center[1] as i64).unsigned_abs() as u128;
    dx * dx + dy * dy <= radius2
}

/// Minimal variant: the upstream f32 SIMD query and exact integer filter, unchanged,
/// except that points are stored in leaf order. Tree IDs are relabelled to their leaf
/// slot after construction, so gathering candidates reads memory sequentially.
/// `V` is the upstream lane count; `V = 4` takes upstream's scalar compress path.
pub struct LeafOrderSprk<const V: usize = W>
where
    LaneCount<V>: SupportedLaneCount,
{
    tree: Option<Sprk<2, V, f32, u32>>,
    /// Points in leaf order; relabelled tree IDs index this array.
    points: Vec<[u32; 2]>,
    /// Input index of each leaf-ordered point.
    input_ids: Vec<usize>,
    candidates: Vec<usize>,
}

impl<const V: usize> LeafOrderSprk<V>
where
    LaneCount<V>: SupportedLaneCount,
    PDVec<2, V, f32, u32>: CompressDispatch<V, f32, u32>,
{
    pub fn new(points: &[[u32; 2]]) -> Self {
        if points.is_empty() {
            return Self { tree: None, points: Vec::new(), input_ids: Vec::new(), candidates: Vec::new() };
        }
        let coords: Vec<_> = points.iter().map(|p| p.map(|x| (x as f64 / 4294967296.0) as f32)).collect();
        let mut tree = Sprk::new(&coords);
        let mut sorted = Vec::with_capacity(points.len());
        let mut input_ids = Vec::with_capacity(points.len());
        for pdvec in &mut tree.positions_sorted {
            for id in &mut pdvec.ids {
                if *id == u32::SENTINEL { continue; }
                sorted.push(points[*id as usize]);
                input_ids.push(*id as usize);
                *id = (sorted.len() - 1) as u32;
            }
        }
        Self { tree: Some(tree), points: sorted, input_ids, candidates: Vec::new() }
    }

    fn candidates(&mut self, center: [u32; 2], radius2: u128) {
        self.candidates.clear();
        if let Some(tree) = &self.tree {
            let pos = center.map(|x| (x as f64 / 4294967296.0) as f32);
            let radius = candidate_radius::<2>((radius2 as f64).sqrt() / 4294967296.0);
            tree.query_radius(&pos, radius, &mut self.candidates);
        }
    }

    pub fn query_points(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<[u32; 2]>) {
        self.candidates(center, radius2);
        for &id in &self.candidates {
            let point = self.points[id];
            if contains(point, center, radius2) { out.push(point); }
        }
    }

    pub fn query_ids(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<usize>) {
        self.candidates(center, radius2);
        for &id in &self.candidates {
            if contains(self.points[id], center, radius2) { out.push(self.input_ids[id]); }
        }
    }
}

/// f64 variant: uint32 coordinates are exact in f64, so the upstream query with a
/// slightly widened radius is a superset, and the returned squared distance decides
/// most points. Only candidates within 8 eps of the boundary need the exact test.
/// Points are stored in leaf order, as in [`LeafOrderSprk`].
pub struct ExactF64Sprk {
    tree: Option<Sprk<2, W, f64, u32>>,
    points: Vec<[u32; 2]>,
    input_ids: Vec<usize>,
    candidates: Vec<IdDist<u32, f64>>,
}

impl ExactF64Sprk {
    pub fn new(points: &[[u32; 2]]) -> Self {
        if points.is_empty() {
            return Self { tree: None, points: Vec::new(), input_ids: Vec::new(), candidates: Vec::new() };
        }
        let coords: Vec<_> = points.iter().map(|p| p.map(f64::from)).collect();
        let mut tree = Sprk::new(&coords);
        let mut sorted = Vec::with_capacity(points.len());
        let mut input_ids = Vec::with_capacity(points.len());
        for pdvec in &mut tree.positions_sorted {
            for id in &mut pdvec.ids {
                if *id == u32::SENTINEL { continue; }
                sorted.push(points[*id as usize]);
                input_ids.push(*id as usize);
                *id = (sorted.len() - 1) as u32;
            }
        }
        Self { tree: Some(tree), points: sorted, input_ids, candidates: Vec::new() }
    }

    /// Coordinates and their differences are exact; the computed squared distance has
    /// relative error below 3 eps. Search 8 eps outward; accept below 8 eps inward.
    fn query(&mut self, center: [u32; 2], radius2: u128, mut accept: impl FnMut(usize, [u32; 2])) {
        let Some(tree) = &self.tree else { return };
        let r2 = radius2 as f64;
        let radius = (r2 * (1.0 + 8.0 * f64::EPSILON)).sqrt() * (1.0 + 4.0 * f64::EPSILON);
        let inside = r2 * (1.0 - 8.0 * f64::EPSILON);
        self.candidates.clear();
        tree.query_radius(&center.map(f64::from), radius, &mut self.candidates);
        for c in &self.candidates {
            let point = self.points[c.id as usize];
            if c.dist <= inside || contains(point, center, radius2) {
                accept(c.id as usize, point);
            }
        }
    }

    pub fn query_points(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<[u32; 2]>) {
        self.query(center, radius2, |_, point| out.push(point));
    }

    pub fn query_ids(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<usize>) {
        let input_ids = std::mem::take(&mut self.input_ids);
        self.query(center, radius2, |id, _| out.push(input_ids[id]));
        self.input_ids = input_ids;
    }
}

#[cfg(test)]
mod tests {
    use super::{contains, ExactF64Sprk, IntegerPointSprk, LeafOrderSprk};

    trait Reporter {
        fn build(points: &[[u32; 2]]) -> Self;
        fn points(&mut self, c: [u32; 2], r2: u128, out: &mut Vec<[u32; 2]>);
        fn ids(&mut self, c: [u32; 2], r2: u128, out: &mut Vec<usize>);
    }
    macro_rules! reporter {
        ($t:ty) => {
            impl Reporter for $t {
                fn build(points: &[[u32; 2]]) -> Self { <$t>::new(points) }
                fn points(&mut self, c: [u32; 2], r2: u128, out: &mut Vec<[u32; 2]>) { self.query_points(c, r2, out) }
                fn ids(&mut self, c: [u32; 2], r2: u128, out: &mut Vec<usize>) { self.query_ids(c, r2, out) }
            }
        };
    }
    reporter!(IntegerPointSprk);
    reporter!(LeafOrderSprk);
    reporter!(LeafOrderSprk<4>);
    reporter!(ExactF64Sprk);

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn check(points: Vec<[u32; 2]>, rng: &mut Rng) {
        check_with::<IntegerPointSprk>(&points, rng);
        check_with::<LeafOrderSprk>(&points, rng);
        check_with::<LeafOrderSprk<4>>(&points, rng);
        check_with::<ExactF64Sprk>(&points, rng);
    }

    fn check_with<R: Reporter>(points: &[[u32; 2]], rng: &mut Rng) {
        let mut index = R::build(points);
        let (mut out, mut ids) = (Vec::new(), Vec::new());
        for q in 0..400 {
            let c = [rng.next() as u32, rng.next() as u32];
            // Radii from real point distances put many points exactly on the boundary.
            let p = points[rng.next() as usize % points.len()];
            let dx = (p[0] as i64 - c[0] as i64).unsigned_abs() as u128;
            let dy = (p[1] as i64 - c[1] as i64).unsigned_abs() as u128;
            let r2 = match q % 20 {
                0 => u128::MAX >> 60, // radius beyond 2^32
                1 => 0,
                _ => (dx * dx + dy * dy) / (1 + rng.next() as u128 % 4000),
            };
            let mut truth: Vec<_> = (0..points.len()).filter(|&i| contains(points[i], c, r2)).collect();
            out.clear();
            index.points(c, r2, &mut out);
            out.sort();
            let mut expected: Vec<_> = truth.iter().map(|&i| points[i]).collect();
            expected.sort();
            assert_eq!(out, expected);
            ids.clear();
            index.ids(c, r2, &mut ids);
            ids.sort();
            truth.sort();
            assert_eq!(ids, truth);
        }
    }

    #[test]
    fn exact_on_uniform_points() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let points = (0..20_000).map(|_| [rng.next() as u32, rng.next() as u32]).collect();
        check(points, &mut rng);
    }

    #[test]
    fn exact_on_duplicate_grid_points() {
        let mut rng = Rng(0xD1B54A32D192ED03);
        let points = (0..20_000)
            .map(|_| [rng.next() as u32 & 0xFFF0_0000, rng.next() as u32 & 0xFFF0_0000]).collect();
        check(points, &mut rng);
    }

    #[test]
    fn exact_near_coordinate_extremes() {
        let mut rng = Rng(0x2545F4914F6CDD1D);
        let points = (0..20_000)
            .map(|_| [(rng.next() as u32 % 64).wrapping_sub(32), (rng.next() as u32 % 64).wrapping_sub(32)])
            .collect();
        check(points, &mut rng);
    }
}
