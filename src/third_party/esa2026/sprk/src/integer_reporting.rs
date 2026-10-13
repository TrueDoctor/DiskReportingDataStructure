//! Local exact-integer reporting adapter around the upstream f32 SIMD query.
//! Candidate IDs use the native writer. Gathering, exact membership and writing
//! the requested output are all performed by the timed caller.
use crate::output::IdDist;
use crate::scalar::IdStorage;
use crate::simd::{CompressDispatch, LaneCount, PDVec, SupportedLaneCount};
use crate::tree::Sprk;

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

pub struct IntegerPointSprk {
    tree: Option<Sprk<2, 8, f32, u32>>,
    points: Vec<[u32; 2]>,
    candidates: Vec<usize>,
}
impl IntegerPointSprk {
    pub fn new(points: &[[u32; 2]]) -> Self {
        let coords: Vec<_> = points.iter().map(|p| p.map(|x| (x as f64 / 4294967296.0) as f32)).collect();
        Self { tree: if points.is_empty() { None } else { Some(Sprk::new(&coords)) },
               points: points.to_vec(), candidates: Vec::new() }
    }
    fn candidates(&mut self, center: [u32; 2], radius2: u128) {
        self.candidates.clear();
        if let Some(tree) = &self.tree {
            let pos = center.map(|x| (x as f64 / 4294967296.0) as f32);
            let radius = candidate_radius::<2>((radius2 as f64).sqrt() / 4294967296.0);
            tree.query_radius(&pos, radius, &mut self.candidates);
        }
    }
    fn contains(point: [u32; 2], center: [u32; 2], radius2: u128) -> bool {
        let dx = (point[0] as i64 - center[0] as i64).unsigned_abs() as u128;
        let dy = (point[1] as i64 - center[1] as i64).unsigned_abs() as u128;
        dx * dx + dy * dy <= radius2
    }
    pub fn query_points(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<[u32; 2]>) {
        self.candidates(center, radius2);
        for &id in &self.candidates {
            let point = self.points[id];
            if Self::contains(point, center, radius2) { out.push(point); }
        }
    }
    pub fn query_ids(&mut self, center: [u32; 2], radius2: u128, out: &mut Vec<usize>) {
        self.candidates(center, radius2);
        for &id in &self.candidates {
            if Self::contains(self.points[id], center, radius2) { out.push(id); }
        }
    }
}

const W: usize = 8;

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
