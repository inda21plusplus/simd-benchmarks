#include <benchmark/benchmark.h>

#include <xmmintrin.h>

struct Vec4 {
    Vec4() : Vec4(0.0f, 0.0f, 0.0f, 0.0f) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    float x, y, z, w;
};

bool cmp_sisd(Vec4 a, Vec4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

bool cmp_simd(Vec4 a, Vec4 b) {
    const __m128 xmm_a = _mm_load_ps(&a.x);
    const __m128 xmm_b = _mm_load_ps(&b.x);
    const __m128 eq = _mm_cmpeq_ps(xmm_a, xmm_b);
    return ((_mm_movemask_ps(eq) & 0xf) == 0xf);
}

static void cmp_sisd_eq_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 1.0f, 1.0f, 1.0f};
    Vec4 b = {1.0f, 1.0f, 1.0f, 1.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    bool res = cmp_sisd(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(cmp_sisd_eq_bench);

static void cmp_simd_eq_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 1.0f, 1.0f, 1.0f};
    Vec4 b = {1.0f, 1.0f, 1.0f, 1.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    bool res = cmp_simd(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(cmp_simd_eq_bench);

static void cmp_sisd_neq_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 5.0f, 1.0f, 1.0f};
    Vec4 b = {1.0f, 1.0f, 1.0f, 1.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    bool res = cmp_sisd(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(cmp_sisd_eq_bench);

static void cmp_simd_neq_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 5.0f, 1.0f, 1.0f};
    Vec4 b = {1.0f, 1.0f, 1.0f, 1.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    bool res = cmp_simd(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(cmp_simd_eq_bench);
