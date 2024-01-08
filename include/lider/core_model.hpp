#pragma once
#include <set>
#include <mutex>
#include <thread>
#include "./pgm/pgm_index.hpp"
#include "utils.hpp"

// #define TEST_CORE

struct LabelHashkey
{
    size_t label;
    size_t hashkey;
};

template <typename data_type = int, size_t epsilon = 64>
class CoreModel
{
private:
    int km; // Number of output points by a core model
    int H;  // Number of hashkey arrays, number of planes
    int M;  // hashkey length in ESK-LSH
    int r0; // extend user-specific factor
    int R;  // R = r0 * km
    int N;  // cluster size / centroids number
    int D;  // dimension of data
    int id; // id of model 0~c
    DATA_TYPE **data;
    size_t *indices;
    std::vector<std::vector<LabelHashkey>> SKHashArray;
    std::vector<std::vector<DATA_TYPE>> RescaledArray;
    std::vector<pgm::PGMIndex<DATA_TYPE, epsilon>> pgm_indexes;

    // Extension on hashkey distance
    float ExtendLSHDistance(const size_t hash1, const size_t hash2)
    {
        // 原论文中的B不使用了
        int KL = M;
        int mask = 1 << (M - 1);
        while (mask > 0)
        {
            if ((hash1 & mask) == (hash2 & mask))
            {
                KL--;
                mask >>= 1;
            }
            else
            {
                break;
            }
        }
        if (KL == 0)
            return 0;
        float KD = std::abs(static_cast<int>(hash1 - hash2)) / std::ldexp(1.0, KL);
        assert(KD < 1);
        return KL + KD;
    }

public:
    CoreModel()
    {
    }

    CoreModel(int km, int H, int M, int r0, int N, int D, int id) : km(km), H(H), M(M), r0(r0), N(N), D(D), id(id)
    {

        if (id == 0)
            R = km;
        else
            R = km * r0;
        SKHashArray.resize(H, std::vector<LabelHashkey>(N));
    }

    void index(DATA_TYPE **data, size_t *indices, std::vector<std::vector<std::vector<DATA_TYPE>>> &planes)
    {
        // 将数据点hash
        this->data = data;
        this->indices = indices;
        for (int table_id = 0; table_id < H; table_id++)
        {
            for (int i = 0; i < N; i++)
            {
                SKHashArray[table_id][i].label = i;
                SKHashArray[table_id][i].hashkey = hash_in_bucket(planes[table_id], data[i]);
            }
            // 对SKHashArray按照hashkey大小重新由低到高排序
            // std::sort(SKHashArray[table_id].begin(), SKHashArray[table_id].end());
            std::sort(std::begin(SKHashArray[table_id]), std::end(SKHashArray[table_id]), [](const LabelHashkey &lhs, const LabelHashkey &rhs)
                      { return lhs.hashkey < rhs.hashkey; });
            // rescale
            size_t MinKey = SKHashArray[table_id][0].hashkey;
            size_t MaxKey = SKHashArray[table_id][N - 1].hashkey;
            // std::cout << "MinKey: " << MinKey << " MaxKey: " << MaxKey << std::endl;
            std::vector<DATA_TYPE> rescaled_array(N); // 对当前桶内的hashkey缩放到:[0,N-1]
            for (int i = 0; i < N; i++)
            {
                rescaled_array[i] = static_cast<float>(((static_cast<double>(SKHashArray[table_id][i].hashkey) - (static_cast<double>(MinKey))) / (MaxKey - MinKey)) * (N - 1));
            }

            RescaledArray.push_back(rescaled_array);                 // 将当前桶内的缩放结果保存
            pgm::PGMIndex<DATA_TYPE, epsilon> index(rescaled_array); // 使用pgmindex进行索引
            pgm_indexes.push_back(index);                            // 保存整个哈希桶的索引结果
        }
    }

    std::vector<size_t> query(DATA_TYPE *query, std::vector<size_t> &hashed_query)
    {
        // 1. 对查询点进行hash
        // 2. 对每个hashkey进行rescale
        // 3. 对每个hashkey进行查询
        // 4. 对每个查询结果进行合并
        // 5. 对合并结果进行排序
        // 6. 取前k个结果
        std::set<size_t> candidates;
        for (int table_id = 0; table_id < H; table_id++)
        {
            float rescaled_hashkey = (hashed_query[table_id] - SKHashArray[table_id][0].hashkey) * (N - 1) / (SKHashArray[table_id][N - 1].hashkey - SKHashArray[table_id][0].hashkey);
            auto range = pgm_indexes[table_id].search(rescaled_hashkey);
            auto lo = RescaledArray[table_id].begin() + range.lo;
            auto hi = RescaledArray[table_id].begin() + range.hi;
            int location = std::distance(RescaledArray[table_id].begin(), std::lower_bound(lo, hi, rescaled_hashkey));

            // ESK-extension bi-directional search
            int right = location;
            if (right == N)
            {
                right--;
            }
            int left = right > 0 ? right - 1 : right;
            int count = 0;
            while (count < R)
            {
                if (ExtendLSHDistance(SKHashArray[table_id][left].hashkey, hashed_query[table_id]) < ExtendLSHDistance(SKHashArray[table_id][right].hashkey, hashed_query[table_id]))
                {
                    candidates.insert(SKHashArray[table_id][left].label);
                    count++;
                    if (left == 0)
                    {
                        while (count < R && right < N)
                        {
                            candidates.insert(SKHashArray[table_id][right].label);
                            count++;
                            right++;
                        }
                        break;
                    }
                    else
                    {
                        left--;
                    }
                }
                else
                {
                    candidates.insert(SKHashArray[table_id][right].label);
                    count++;
                    if (right == N - 1)
                    {
                        while (count < R && left >= 0)
                        {
                            candidates.insert(SKHashArray[table_id][left].label);
                            count++;
                            left--;
                        }
                        break;
                    }
                    else
                    {
                        right++;
                    }
                }
            }
        }
        // debugInfo("search range:", (float)candidates.size() / N);
        std::vector<size_t> candidate_labels(candidates.begin(), candidates.end());
        // 使用 std::partial_sort 对 candidate_labels 进行部分排序，只保证前 km 个元素是有序的，
        std::partial_sort(std::begin(candidate_labels), candidate_labels.begin() + km, std::end(candidate_labels),
                          [this, &query](const size_t &l1, const size_t &l2)
                          {
                              return euclidean_distance(data[l1], query, D) < euclidean_distance(data[l2], query, D);
                          });
        std::vector<size_t> result(km);
        for (int i = 0; i < km; i++)
        {
#ifdef TEST_CORE
            result[i] = candidate_labels[i]; // result是类内的id
#else
            result[i] = indices[candidate_labels[i]]; // result是全体data的id
#endif
        }
        assert(result.size() == km);
        return result;
    }
};