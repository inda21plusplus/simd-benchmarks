#include <benchmark/benchmark.h>

#include <xmmintrin.h>
#include <immintrin.h>
#include <cmath>

struct Vec4 {
    float x, y, z, w;
};

Vec4 normalize_sisd(Vec4 a) {
    const float square_magnitude = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
    const float magnitude = std::sqrt(square_magnitude);
    return Vec4{
        a.x / magnitude,
        a.y / magnitude,
        a.z / magnitude,
        a.w / magnitude,
    };
}

Vec4 normalize_sisd_fast(Vec4 a) {
    const float square_magnitude = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;

    const float inv_magnitude = 1.0f/std::sqrt(square_magnitude);
    return Vec4{
        a.x * inv_magnitude,
        a.y * inv_magnitude,
        a.z * inv_magnitude,
        a.w * inv_magnitude,
    };
}

Vec4 normalize_sisd_fast_cheat(Vec4 a) {
    const float square_magnitude = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;

    const __m128 sqr_mag = _mm_set1_ps(square_magnitude);
    const __m128 inv_mag = _mm_rsqrt_ps(sqr_mag);
    const float inv_magnitude = _mm_cvtss_f32(inv_mag);

    return Vec4{
        a.x * inv_magnitude,
        a.y * inv_magnitude,
        a.z * inv_magnitude,
        a.w * inv_magnitude,
    };
}


Vec4 normalize_simd(Vec4 a) {
    const __m128 xmm_a = _mm_load_ps(&a.x);
    const __m128 square_magnitude = _mm_dp_ps(xmm_a, xmm_a, 0xff);
    const __m128 magnitude = _mm_sqrt_ps(square_magnitude);
    const __m128 normalized_a = _mm_div_ps(xmm_a, magnitude);
    Vec4 res;
    _mm_store_ps(&res.x, normalized_a);
    return res;
}

Vec4 normalize_simd_fast(Vec4 a) {
    const __m128 xmm_a = _mm_load_ps(&a.x);
    const __m128 square_magnitude = _mm_dp_ps(xmm_a, xmm_a, 0xff);
    const __m128 inv_magnitude = _mm_rsqrt_ps(square_magnitude);
    const __m128 normalized_a = _mm_mul_ps(xmm_a, inv_magnitude);
    Vec4 res;
    _mm_store_ps(&res.x, normalized_a);
    return res;
}

static void normalize_sisd_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    benchmark::DoNotOptimize(a);

    Vec4 res = normalize_sisd(a);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(normalize_sisd_bench);

static void normalize_sisd_fast_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    benchmark::DoNotOptimize(a);

    Vec4 res = normalize_sisd_fast(a);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(normalize_sisd_fast_bench);

static void normalize_sisd_fast_cheat_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    benchmark::DoNotOptimize(a);

    Vec4 res = normalize_sisd_fast_cheat(a);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(normalize_sisd_fast_cheat_bench);

static void normalize_simd_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    benchmark::DoNotOptimize(a);

    Vec4 res = normalize_simd(a);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(normalize_simd_bench);

static void normalize_simd_fast_bench(benchmark::State& state) {
  for (auto _ : state) {
    Vec4 a = {1.0f, 0.5f, 0.72f, 5.0f};
    benchmark::DoNotOptimize(a);

    Vec4 res = normalize_simd_fast(a);
    benchmark::DoNotOptimize(res);
  }
}
BENCHMARK(normalize_simd_fast_bench);
