#include "../include/lider/core_model.hpp"

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

    CoreModel<DATA_TYPE, 64> CentroidsRetriver;
    std::vector<CoreModel<DATA_TYPE, 64>> InClusterRetrivers;

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

    std::vector<size_t> query(DATA_TYPE *query, std::vector<size_t> &hashed_query)
    {
        std::vector<size_t> retrivedCentroids = CentroidsRetriver.query(query, hashed_query);
        assert(retrivedCentroids.size() == c0);
        std::vector<size_t> candidates(c0 * km);
        // std::mutex mtx;
        std::vector<std::thread> threads; // 存储所有线程的向量
        for (int i = 0; i < c0; i++)
        {
            threads.push_back(std::thread([&, i]() { // 捕获i和所有其他变量
                std::vector<size_t> InClusterTOPKM = InClusterRetrivers[retrivedCentroids[i]].query(query, hashed_query);
                assert(InClusterTOPKM.size() == km);
                // std::lock_guard<std::mutex> lock(mtx); // 在修改candidates之前锁定互斥锁
                for (int j = 0; j < km; j++)
                {
                    candidates[j + i * km] = InClusterTOPKM[j];
                }
            }));
        }
        // 等待所有线程完成
        for (auto &thread : threads)
        {
            thread.join();
        }
        assert(candidates.size() == c0 * km);
        // print_vector(candidates);
        // merge c0 * km results to k results
        std::vector<size_t> result = merge(candidates, query);
        return result;
    }
};