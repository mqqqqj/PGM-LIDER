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
#define DATA_TYPE float
// #define CPP17 true

DATA_TYPE euclidean_distance(const std::vector<DATA_TYPE> &v, const std::vector<DATA_TYPE> &u)
{
    static_assert(std::is_same<DATA_TYPE, float>::value, "DATA_TYPE must be float");
    if (v.size() != u.size())
    {
        throw std::invalid_argument("In distance caculation function, vectors must be of the same size.");
    }
    // 使用 std::transform 和 std::inner_product 可以利用并行化提高性能
    std::vector<DATA_TYPE> diff(v.size());
    std::transform(v.begin(), v.end(), u.begin(), diff.begin(), std::minus<DATA_TYPE>());

    DATA_TYPE sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return std::sqrt(sum);
}

DATA_TYPE euclidean_distance(DATA_TYPE *v, DATA_TYPE *u, int dim)
{
#ifdef CPP17
    float dist = std::transform_reduce(v, v + dim, u, 0.0f, std::plus<>(), [](float a, float b)
                                       {
        float diff = a - b;
        return diff * diff; });
#else
    float dist = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        float diff = v[i] - u[i];
        dist += diff * diff;
    }
#endif
    return std::sqrt(dist);
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
    for (int m = 0; m < M; m++)
    {
        double h = std::inner_product(plane[m].begin(), plane[m].end(), v, 0.0);
        if (h >= 0)
            hash_key.set(m);
    }
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

void calc_gt4cluster(const char *gtpath, DATA_TYPE **data, int data_size, DATA_TYPE **querys, int size, int dim)
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
    std::cout << hit << " / " << q_size * k << std ::endl;
    return static_cast<float>(hit) / (q_size * k);
}

template <typename T>
void debugInfo(const char *mes, T value)
{
    std::cout << "debug-" << mes << ": " << value << std::endl;
}