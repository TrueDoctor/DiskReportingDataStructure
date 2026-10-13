//! Regression sweep against MortonSIMD on identical workloads.
//!
//! `sweep gen DIR`  writes workload files (shared with bench/morton_simd_sweep.cc);
//! `sweep run DIR`  times MortonPlus variants on them and prints CSV rows
//! dist,n,centers,k,algo,build_s,ns_per_query,answers
use mortonplus::{MortonPlus, Options, Point, CHUNK, KERNEL};
use std::io::{Read, Write};
use std::time::Instant;

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

struct Workload {
    name: String,
    points: Vec<Point>,
    /// (k, r2, query centers)
    queries: Vec<(u64, u128, Vec<Point>)>,
}

fn points_for(dist: &str, n: usize, rng: &mut Rng) -> Vec<Point> {
    match dist {
        "uniform" => (0..n).map(|_| [rng.u32(), rng.u32()]).collect(),
        "clustered" => {
            let centers: Vec<Point> = (0..64).map(|_| [rng.u32(), rng.u32()]).collect();
            (0..n)
                .map(|_| {
                    let c = centers[rng.next() as usize % 64];
                    [c[0].wrapping_add(rng.u32() >> 12), c[1].wrapping_add(rng.u32() >> 12)]
                })
                .collect()
        }
        "line" => (0..n)
            .map(|_| {
                let x = rng.u32();
                [x, x / 2 + (rng.u32() >> 22)]
            })
            .collect(),
        "dupgrid" => (0..n).map(|_| [rng.u32() & 0xFF00_0000, rng.u32() & 0xFF00_0000]).collect(),
        _ => unreachable!(),
    }
}

fn center(points: &[Point], matched: bool, rng: &mut Rng) -> Point {
    if matched {
        let p = points[rng.next() as usize % points.len()];
        [p[0].wrapping_add(rng.u32() >> 24), p[1].wrapping_add(rng.u32() >> 24)]
    } else {
        [rng.u32(), rng.u32()]
    }
}

fn generate(dir: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for dist in ["uniform", "clustered", "line", "dupgrid"] {
        for n in [1_000usize, 10_000, 100_000, 1_000_000, 10_000_000] {
            let points = points_for(dist, n, &mut rng);
            let center_kinds: &[bool] = if dist == "uniform" { &[false] } else { &[false, true] };
            for &matched in center_kinds {
                let mut ks: Vec<u64> = vec![1, 16, 256, 4096];
                if n <= 1_000_000 {
                    ks.push(n as u64 / 4);
                }
                ks.retain(|&k| k as usize <= n);
                // Radius: median over 16 pilot centers of the squared distance to the k-th neighbour.
                let mut kth: Vec<Vec<u128>> = vec![Vec::new(); ks.len()];
                let mut dists = vec![0u128; n];
                for _ in 0..16 {
                    let c = center(&points, matched, &mut rng);
                    for (d, &p) in dists.iter_mut().zip(&points) {
                        *d = d2(p, c);
                    }
                    for (i, &k) in ks.iter().enumerate() {
                        kth[i].push(*dists.select_nth_unstable(k as usize - 1).1);
                    }
                }
                let queries = ks
                    .iter()
                    .zip(kth.iter_mut())
                    .map(|(&k, v)| {
                        v.sort_unstable();
                        let count = (4_000_000 / k).clamp(50, 2000) as usize;
                        (k, v[8], (0..count).map(|_| center(&points, matched, &mut rng)).collect())
                    })
                    .collect();
                let name = format!("{dist},{n},{}", if matched { "matched" } else { "uniform" });
                write(dir, &Workload { name, points: points.clone(), queries });
            }
        }
    }
}

fn write(dir: &str, w: &Workload) {
    let path = format!("{dir}/{}.bin", w.name.replace(',', "_"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let name = w.name.as_bytes();
    f.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    f.write_all(name).unwrap();
    f.write_all(&(w.points.len() as u64).to_le_bytes()).unwrap();
    for p in &w.points {
        f.write_all(&p[0].to_le_bytes()).unwrap();
        f.write_all(&p[1].to_le_bytes()).unwrap();
    }
    f.write_all(&(w.queries.len() as u64).to_le_bytes()).unwrap();
    for (k, r2, cs) in &w.queries {
        f.write_all(&k.to_le_bytes()).unwrap();
        f.write_all(&r2.to_le_bytes()).unwrap();
        f.write_all(&(cs.len() as u64).to_le_bytes()).unwrap();
        for c in cs {
            f.write_all(&c[0].to_le_bytes()).unwrap();
            f.write_all(&c[1].to_le_bytes()).unwrap();
        }
    }
}

struct Reader {
    bytes: Vec<u8>,
    at: usize,
}
impl Reader {
    fn take(&mut self, n: usize) -> &[u8] {
        self.at += n;
        &self.bytes[self.at - n..self.at]
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    fn u128(&mut self) -> u128 {
        u128::from_le_bytes(self.take(16).try_into().unwrap())
    }
    fn points(&mut self, n: usize) -> Vec<Point> {
        self.take(8 * n)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| [u32::from_le_bytes(c[0..4].try_into().unwrap()), u32::from_le_bytes(c[4..8].try_into().unwrap())])
            .collect()
    }
}

fn read(path: &std::path::Path) -> Workload {
    let mut r = Reader { bytes: Vec::new(), at: 0 };
    std::fs::File::open(path).unwrap().read_to_end(&mut r.bytes).unwrap();
    let len = r.u64() as usize;
    let name = String::from_utf8(r.take(len).to_vec()).unwrap();
    let n = r.u64() as usize;
    let points = r.points(n);
    let queries = (0..r.u64())
        .map(|_| {
            let (k, r2) = (r.u64(), r.u128());
            let count = r.u64() as usize;
            (k, r2, r.points(count))
        })
        .collect();
    Workload { name, points, queries }
}

fn run_variant<const W: usize>(w: &Workload, label: &str, opt: Options) {
    let t = Instant::now();
    let index = MortonPlus::<W>::new(&w.points, opt);
    let build = t.elapsed().as_secs_f64();
    let mut out: Vec<Point> = Vec::with_capacity(1 << 16);
    // Optional K=<k> environment filter, for profiling one query size.
    let only: Option<u64> = std::env::var("K").ok().and_then(|v| v.parse().ok());
    for (k, r2, cs) in &w.queries {
        if only.is_some_and(|o| o != *k) {
            continue;
        }
        let (mut best, mut total) = (f64::MAX, 0);
        let reps = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        for _ in 0..reps {
            total = 0;
            let t = Instant::now();
            for &c in cs {
                out.clear();
                index.query(c, *r2, &mut out);
                total += out.len();
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{},{k},{label},{build:.6},{:.1},{total}", w.name, best * 1e9 / cs.len() as f64);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => generate(&args[2]),
        Some("run") => {
            let mut files: Vec<_> = std::fs::read_dir(&args[2]).unwrap().map(|e| e.unwrap().path()).collect();
            files.sort();
            let default = Options::default();
            for f in files {
                let w = read(&f);
                run_variant::<{ CHUNK }>(&w, &format!("{KERNEL} W={CHUNK}"), default);
                run_variant::<{ 2 * CHUNK }>(&w, &format!("{KERNEL} W={}", 2 * CHUNK), default);
                run_variant::<{ 4 * CHUNK }>(&w, &format!("{KERNEL} W={}", 4 * CHUNK), default);
            }
        }
        _ => eprintln!("usage: sweep gen DIR | run DIR"),
    }
}
