#include <benchmark/benchmark.h>

#include <xmmintrin.h>

struct Vec4 {
    Vec4() : Vec4(0.0f, 0.0f, 0.0f, 0.0f) {}
    Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    float x, y, z, w;
};

union Vec4Union {
    Vec4Union() : Vec4Union(0.0f, 0.0f, 0.0f, 0.0f) {}
    Vec4Union(__m128 xmm) : xmm(xmm) {}
    Vec4Union(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    __m128 xmm;
    struct {
        float x, y, z, w;
    };
};

static_assert(sizeof(Vec4) == sizeof(Vec4Union), "");

Vec4 add_sisd(Vec4 a, Vec4 b) {
    return Vec4{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w,
    };
}

Vec4Union add_sisd_union(Vec4Union a, Vec4Union b) {
    return Vec4Union{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w,
    };
}

Vec4 add_simd(Vec4 a, Vec4 b) {
    const __m128 xmm_a = _mm_load_ps(&a.x);
    const __m128 xmm_b = _mm_load_ps(&b.x);
    const __m128 xmm_res = _mm_add_ps(xmm_a, xmm_b);
    Vec4 res;
    _mm_store_ps(&res.x, xmm_res);
    return res;
}

Vec4Union add_simd_union(Vec4Union a, Vec4Union b) {
  return Vec4Union{_mm_add_ps(a.xmm, b.xmm)};
}

static void union_add_sisd_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    Vec4 b = {1.5f, 1.9f, 0.0f, 12.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    Vec4 res = add_sisd(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(union_add_sisd_bench);

static void union_add_sisd_union_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4Union a = {1.0f, 0.5f, 0.72f, 5.0f};
    Vec4Union b = {1.5f, 1.9f, 0.0f, 12.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    Vec4Union res = add_sisd_union(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(union_add_sisd_union_bench);

static void union_add_simd_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    Vec4 b = {1.5f, 1.9f, 0.0f, 12.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    Vec4 res = add_simd(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(union_add_simd_bench);

static void union_add_simd_union_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4Union a = {1.0f, 0.5f, 0.72f, 5.0f};
    Vec4Union b = {1.5f, 1.9f, 0.0f, 12.0f};
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);

    Vec4Union res = add_simd_union(a, b);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(union_add_simd_bench);
