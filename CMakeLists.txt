cmake_minimum_required(VERSION 3.2)

project(
	simd_benchmarks
	VERSION 1.0.0
	LANGUAGES CXX
)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "")
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")

include(FetchContent)
FetchContent_Declare(benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.6.1
)

FetchContent_MakeAvailable(benchmark)

add_executable(simd_benchmarks
    main.cpp
    normalize.cpp
    union.cpp
    cmp.cpp
)

target_compile_options(simd_benchmarks
  PRIVATE
     -msse4.1
)

target_link_libraries(simd_benchmarks PRIVATE benchmark)
