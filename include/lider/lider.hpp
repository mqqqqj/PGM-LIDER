#include "../include/lider/core_model.hpp"
#include <unordered_map>

#define FLATSEARCH

template <typename data_type = int, size_t epsilon = 64>
class LIDER
{
private:
    int c0; // 第一层输出的结果数
    int km; // 第二层每个model输出的结果数
    int c;  // 第二层model的个数
    int k;  // 最终输出的结果数
    int D;
    DATA_TYPE **data; // 数据集的全部数据
    std::unordered_map<int, int> hotCluster;
    CoreModel<DATA_TYPE, 64> CentroidsRetriver;
    std::vector<CoreModel<DATA_TYPE, 64>> InClusterRetrivers;
    // std::set<size_t> candidates;
    std::vector<size_t> merge(std::vector<size_t> &candidates, DATA_TYPE *query)
    {

        // 遍历candidates，逐个与query比较欧式距离，选取最近的k个,按照距离query从小到大的排序存在result中
        std::partial_sort(std::begin(candidates), candidates.begin() + k, std::end(candidates),
                          [this, &query](const size_t &l1, const size_t &l2)
                          {
                              return euclidean_distance(data[l1], query, D) < euclidean_distance(data[l2], query, D);
                          });

        std::vector<size_t> result(candidates.begin(), candidates.begin() + k);
        return result;
    }

public:
    LIDER(CoreModel<DATA_TYPE, 64> CentroidsRetriver, std::vector<CoreModel<DATA_TYPE, 64>> InClusterRetrivers, DATA_TYPE **data, int c0, int km, int c, int k, int D) : CentroidsRetriver(CentroidsRetriver), InClusterRetrivers(InClusterRetrivers), data(data), c0(c0), km(km), c(c), k(k), D(D)
    {
    }

    std::vector<size_t> query(DATA_TYPE *query, std::vector<size_t> &hashed_query, bool fixed_extension)
    {
        // candidates.clear();
#ifdef FLATSEARCH
        size_t *s = CentroidsRetriver.getIndices();
        // printf("%d\n", s[12]);
        std::vector<size_t> retrivedCentroids = CentroidsRetriver.flatquery(query);
#else
        std::vector<size_t> retrivedCentroids = CentroidsRetriver.query(query, hashed_query, fixed_extension);
#endif
        assert(retrivedCentroids.size() == c0);
        // record hot cluster
        for (int i = 0; i < c0; i++)
        {
            if (hotCluster.count(retrivedCentroids[i]) == 0)
            {
                hotCluster[retrivedCentroids[i]] = 1;
            }
            else
            {
                hotCluster[retrivedCentroids[i]]++;
            }
        }
        std::vector<size_t> candidates(c0 * km);

#pragma omp parallel for // num_threads(72) // 拉满
        for (int i = 0; i < c0; i++)
        {
#ifdef FLATSEARCH
            std::vector<size_t> InClusterTOPKM = InClusterRetrivers[retrivedCentroids[i]].flatquery(query);
#else
            std::vector<size_t> InClusterTOPKM = InClusterRetrivers[retrivedCentroids[i]].query(query, hashed_query, fixed_extension);
#endif
            assert(InClusterTOPKM.size() == km);
            // #pragma omp critical
            //             candidates.insert(InClusterTOPKM.begin(), InClusterTOPKM.end());
            for (int j = 0; j < km; j++)
            {
                candidates[j + i * km] = InClusterTOPKM[j];
            }
        }
        assert(candidates.size() == c0 * km);
        // print_vector(candidates);
        // merge c0 * km results to k results
        // int candidate_size = candidates.size();
        // std::cout << candidate_size << std::endl;
        // std::vector<size_t> v_candidates(candidates.begin(), candidates.end());
        // std::vector<size_t> result = merge(v_candidates, query);
        std::vector<size_t> result = merge(candidates, query);
        return result;
    }
    std::unordered_map<int, int> getHotCluster()
    {
        return hotCluster;
    }
    void saveIndex(const std::string &location)
    {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;
        // 将lider模型保存到output文件中
        writeBinaryPOD(output, c0);
        writeBinaryPOD(output, km);
        writeBinaryPOD(output, c);
        writeBinaryPOD(output, k);
        writeBinaryPOD(output, D);
        // data不用保存
        output.close();
        // 将CentroidsRetriver保存到output文件中
        CentroidsRetriver.saveIndex(location);
        // 将InClusterRetrivers保存到output文件中
        for (int i = 0; i < c; i++)
        {
            InClusterRetrivers[i].saveIndex(location);
        }
    }
};