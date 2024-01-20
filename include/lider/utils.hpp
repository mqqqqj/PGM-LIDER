#pragma once
#include <vector>
#include <type_traits>
#include <cassert>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include <immintrin.h> // for Intel Intrinsics

#define DATA_TYPE float
// #define AVX512
#define AVX2
struct LabelHashkey
{
    size_t label;
    size_t hashkey;
};

// DATA_TYPE euclidean_distance(const std::vector<DATA_TYPE> &v, const std::vector<DATA_TYPE> &u)
// {
// #ifdef AVX512
//     static_assert(std::is_same<DATA_TYPE, float>::value, "DATA_TYPE must be float");
//     assert(v.size() == u.size());
//     assert(v.size() % 16 == 0); // 假设v.size()是16的倍数

//     __m512 sum = _mm512_setzero_ps();
//     for (size_t i = 0; i < v.size(); i += 16)
//     {
//         __m512 a = _mm512_loadu_ps(v.data() + i);
//         __m512 b = _mm512_loadu_ps(u.data() + i);
//         __m512 diff = _mm512_sub_ps(a, b);
//         __m512 square = _mm512_mul_ps(diff, diff);
//         sum = _mm512_add_ps(sum, square);
//     }
//     __m256 sum_low = _mm512_castps512_ps256(sum);
//     __m256 sum_high = _mm512_extractf32x8_ps(sum, 1);
//     __m256 hsum = _mm256_hadd_ps(_mm256_hadd_ps(sum_low, sum_high), _mm256_setzero_ps());
//     float dist = _mm_cvtss_f32(_mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));
//     return std::sqrt(dist);
// #else
//     static_assert(std::is_same<DATA_TYPE, float>::value, "DATA_TYPE must be float");
//     // 使用 std::transform 和 std::inner_product 可以利用并行化提高性能
//     std::vector<DATA_TYPE> diff(v.size());
//     std::transform(v.begin(), v.end(), u.begin(), diff.begin(), std::minus<DATA_TYPE>());

//     DATA_TYPE sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
//     return std::sqrt(sum);
// #endif
// }

// for partial_sort : simd speed up
DATA_TYPE euclidean_distance(DATA_TYPE *v, DATA_TYPE *u, int dim)
{
#ifdef AVX512
    __m512 sum = _mm512_setzero_ps();
    assert(dim % 16 == 0); // 假设v.size()是16的倍数
    for (int i = 0; i < dim; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&v[i]);
        __m512 b = _mm512_loadu_ps(&u[i]);
        __m512 diff = _mm512_sub_ps(a, b);
        __m512 square = _mm512_mul_ps(diff, diff);
        sum = _mm512_add_ps(sum, square);
    }
    __m256 sum_low = _mm512_castps512_ps256(sum);
    __m256 sum_high = _mm512_extractf32x8_ps(sum, 1);
    __m256 hsum = _mm256_hadd_ps(_mm256_hadd_ps(sum_low, sum_high), _mm256_setzero_ps());
    float dist = _mm_cvtss_f32(_mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));
    return std::sqrt(dist);
#elif defined AVX2
    __m256 sum = _mm256_setzero_ps();
    assert(dim % 8 == 0); // 假设dim是8的倍数
    for (int i = 0; i < dim; i += 8)
    {
        __m256 a = _mm256_loadu_ps(&v[i]);
        __m256 b = _mm256_loadu_ps(&u[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        __m256 square = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, square);
    }
    __m256 hsum = _mm256_hadd_ps(sum, _mm256_setzero_ps());
    hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 1));
    float dist = _mm_cvtss_f32(_mm_hadd_ps(_mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum)));
    return std::sqrt(dist);
#else
    float dist = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        float diff = v[i] - u[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
#endif
}

template <typename T>
void print_vector(const std::vector<T> &v)
{
    std::cout << "(";
    for (auto &d : v)
    {
        std::cout << d << ",";
    }
    std::cout << ")" << std::endl;
}

std::vector<std::vector<std::vector<DATA_TYPE>>> gen_uniform_planes(int H, int M, int D)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.0, 1.0);
    std::vector<std::vector<std::vector<DATA_TYPE>>> planes;
    static_assert(std::is_same<DATA_TYPE, float>::value, "DATA_TYPE must be float");
    planes.reserve(H);

    for (int i = 0; i < H; ++i)
    {
        std::vector<std::vector<DATA_TYPE>> plane;
        plane.reserve(M); // 使用 reserve 方法预分配内存,避免不必要的动态内存分配

        for (int j = 0; j < M; ++j)
        {
            std::vector<DATA_TYPE> row;
            row.reserve(D);

            for (int k = 0; k < D; ++k)
            {
                row.emplace_back(distribution(gen));
            }

            plane.emplace_back(std::move(row));
        }

        planes.emplace_back(std::move(plane)); // 使用 emplace_back 方法添加元素：这可以避免创建临时对象和额外的复制或移动操作。
    }
    return planes;
}

size_t hash_in_bucket(const std::vector<std::vector<DATA_TYPE>> &plane, DATA_TYPE *v)
{
    std::bitset<32> hash_key;
    int M = plane.size();
    int D = plane[0].size();
#ifdef AVX512
    for (int m = 0; m < M; m++)
    {
        __m512 sum = _mm512_setzero_ps();
        for (int n = 0; n < D; n += 16)
        {
            __m512 a = _mm512_loadu_ps(&plane[m][n]);
            __m512 b = _mm512_loadu_ps(&v[n]);
            __m512 prod = _mm512_mul_ps(a, b);
            sum = _mm512_add_ps(sum, prod);
        }
        __m256 sum_low = _mm512_castps512_ps256(sum);
        __m256 sum_high = _mm512_extractf32x8_ps(sum, 1);
        __m256 h_vec = _mm256_hadd_ps(_mm256_hadd_ps(sum_low, sum_high), _mm256_setzero_ps());
        float h = _mm_cvtss_f32(_mm256_castps256_ps128(h_vec));
        if (h >= 0)
            hash_key.set(m);
    }
#elif defined AVX2
    for (int m = 0; m < M; m++)
    {
        __m256 sum = _mm256_setzero_ps();
        for (int n = 0; n < D; n += 8)
        {
            __m256 a = _mm256_loadu_ps(&plane[m][n]);
            __m256 b = _mm256_loadu_ps(&v[n]);
            __m256 prod = _mm256_mul_ps(a, b);
            sum = _mm256_add_ps(sum, prod);
        }
        __m256 h_vec = _mm256_hadd_ps(sum, _mm256_setzero_ps());
        h_vec = _mm256_add_ps(h_vec, _mm256_permute2f128_ps(h_vec, h_vec, 1));
        float h = _mm_cvtss_f32(_mm_hadd_ps(_mm256_castps256_ps128(h_vec), _mm256_castps256_ps128(h_vec)));
        if (h >= 0)
            hash_key.set(m);
    }
#else
    for (int m = 0; m < M; m++)
    {
        float h = 0.0;
        for (int n = 0; n < D; ++n)
        {
            h += plane[m][n] * v[n];
        }
        if (h >= 0)
            hash_key.set(m);
    }
#endif
    return hash_key.to_ulong();
}

std::vector<size_t> hash(const std::vector<std::vector<std::vector<DATA_TYPE>>> &planes, DATA_TYPE *v)
{
    static_assert(std::is_same<DATA_TYPE, float>::value, "DATA_TYPE must be float");
    int H = planes.size();
    int M = planes[0].size();
    std::vector<size_t> hashed_v(H);
    for (int table_id = 0; table_id < H; table_id++)
    {
        hashed_v[table_id] = hash_in_bucket(planes[table_id], v);
    }
    return hashed_v;
}

struct Groundtruth
{
    int label;
    float dist;
};

void calc_gt4cluster(std::string gtpath, DATA_TYPE **data, int data_size, DATA_TYPE **querys, int size, int dim)
{
    int topk = 100;
    std::vector<Groundtruth> gt(data_size); // 11273
    std::ofstream gtFile(gtpath, std::ios::binary);
    if (!gtFile.is_open())
    {
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    for (int i = 0; i < size; i++) // size:100
    {
        for (int j = 0; j < data_size; j++) // data_size
        {
            gt[j].label = j;
            gt[j].dist = euclidean_distance(data[j], querys[i], dim);
        }
        std::partial_sort(gt.begin(), gt.begin() + topk, gt.end(), [](const Groundtruth &x, const Groundtruth &y)
                          { return x.dist < y.dist; });
        for (int j = 0; j < topk; j++)
        {
            gtFile.write(reinterpret_cast<const char *>(&gt[j].label), sizeof(int));
        }
    }

    gtFile.close();
}

float Recall(std::vector<std::vector<size_t>> results, int **gt_set, int q_size, int k)
{
    int hit = 0;
    for (int i = 0; i < q_size; i++)
    {
        for (int top_k = 0; top_k < k; top_k++)
        {
            for (int j = 0; j < k; j++)
            {
                if (results[i][top_k] == gt_set[i][j])
                {
                    hit++;
                    break;
                }
            }
        }
    }
    // std::cout << hit << " / " << q_size * k << std ::endl;
    return static_cast<float>(hit) / (q_size * k);
}

template <typename T>
void debugInfo(const char *mes, T value)
{
    std::cout << "debug-" << mes << ": " << value << std::endl;
}

template <typename T>
void writeBinaryPOD(std::ostream &out, const T &podRef)
{
    out.write((char *)&podRef, sizeof(T));
}