// Times MortonSIMD on the workload files written by `sweep gen DIR`.
// Prints CSV rows: dist,n,centers,k,algo,build_s,ns_per_query,answers
#include "../../mortonSIMD.cc"
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
namespace ms = diskreport::morton_MortonSIMD;
using Wide = unsigned __int128;

struct Reader {
  std::vector<char> bytes; size_t at = 0;
  template <class T> T get() { T v; std::memcpy(&v, bytes.data() + at, sizeof v); at += sizeof v; return v; }
  std::vector<ms::Point> points(size_t n) {
    std::vector<ms::Point> p(n);
    std::memcpy(p.data(), bytes.data() + at, 8 * n); at += 8 * n;
    return p;
  }
};

int main(int, char** argv) {
  const char* only_k = std::getenv("K");
  const int reps = std::getenv("REPS") ? std::atoi(std::getenv("REPS")) : 3;
  std::vector<std::filesystem::path> files;
  for (auto& e : std::filesystem::directory_iterator(argv[1])) files.push_back(e.path());
  std::sort(files.begin(), files.end());
  for (auto& f : files) {
    std::ifstream in(f, std::ios::binary);
    Reader r{{std::istreambuf_iterator<char>(in), {}}};
    const auto len = r.get<uint64_t>();
    std::string name(r.bytes.data() + r.at, len); r.at += len;
    const auto points = r.points(r.get<uint64_t>());
    auto t0 = std::chrono::steady_clock::now();
    ms::MortonSIMD index(points);
    const double build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<ms::Point> out; out.reserve(1 << 16);
    for (uint64_t q = r.get<uint64_t>(); q-- > 0;) {
      const auto k = r.get<uint64_t>();
      const auto r2 = r.get<Wide>();
      const auto centers = r.points(r.get<uint64_t>());
      if (only_k && std::strtoull(only_k, nullptr, 10) != k) continue;
      double best = 1e30; size_t total = 0;
      for (int rep = 0; rep < reps; ++rep) {
        total = 0;
        auto t = std::chrono::steady_clock::now();
        for (auto c : centers) { out.clear(); index.query(ms::Circle{c, r2}, std::back_inserter(out)); total += out.size(); }
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
      }
      std::printf("%s,%llu,MortonSIMD,%.6f,%.1f,%zu\n", name.c_str(), (unsigned long long)k, build, best * 1e9 / centers.size(), total);
      std::fflush(stdout);
    }
  }
}
