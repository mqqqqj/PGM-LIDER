#include "../include/lider/hili.hpp"

int main()
{
    int dim = 128;   // Dimension of the elements
    int N = 1000000; // Maximum number of elements, should be known beforehand
    int layer_num = 8;
    double mL = 1.0;
    int H = 16;
    int km = 10;
    int k = 10; // Number of final output points
    int r0 = 30;
    int M = 20;
    HiLi<DATA_TYPE, 64> hili(dim, N, layer_num, mL, H, km, k, r0, M);
    // read data
    std::string dataPath = "/home/mqj/data/sift/sift_base.fvecs";
    std::ifstream dataFile(dataPath, std::ios::binary);
    DATA_TYPE **data = new DATA_TYPE *[N];
    for (int i = 0; i < N; i++)
    {
        data[i] = new DATA_TYPE[dim];
    }
    for (int i = 0; i < N; i++)
    {
        dataFile.read(reinterpret_cast<char *>(data[i]), sizeof(DATA_TYPE) * dim);
    }
    dataFile.close();
    std::vector<std::vector<std::vector<float>>> uniform_planes = gen_uniform_planes(H, M, dim);
    auto start_time = std::chrono::high_resolution_clock::now();
    hili.index(data, uniform_planes);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Construction time cost of HILI: " << duration.count() / 1000.0 << " seconds." << std::endl;

    // read query info
    const char *qpath = "/home/mqj/data/sift/sift_query.fvecs";
    std::ifstream qFile(qpath, std::ios::binary);
    int q_size = 1000;
    DATA_TYPE **query_set = new DATA_TYPE *[q_size];
    for (int i = 0; i < q_size; i++)
    {
        query_set[i] = new DATA_TYPE[dim];
    }
    for (int i = 0; i < q_size; i++)
    {
        qFile.seekg(4, std::ios::cur);
        qFile.read(reinterpret_cast<char *>(query_set[i]), sizeof(DATA_TYPE) * dim);
    }
    qFile.close();

    std::vector<std::vector<size_t>> results;
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < q_size; i++)
    {
        auto q_hashes = hash(uniform_planes, query_set[i]); // num_hashtable * hashkey_size
        auto result = hili.query(query_set[i], q_hashes);
        results.push_back(result);
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "QPS: " << q_size / (duration.count() / 1000.0) << std::endl;
    // read groundtruth
    // ivecs里的每一“行”里，第一个数据是查询答案的数量n，后面n个数是答案向量的id。(注：ivecs中的i指int32)
    const char *gtpath = "/home/mqj/data/sift/sift_groundtruth.ivecs";
    std::ifstream gtFile(gtpath, std::ios::binary);
    int **groundtruth = new int *[q_size];
    for (int i = 0; i < q_size; i++)
    {
        groundtruth[i] = new int[k];
        int n;
        gtFile.read(reinterpret_cast<char *>(&n), sizeof(int));
        gtFile.read(reinterpret_cast<char *>(groundtruth[i]), sizeof(int) * k);
        gtFile.seekg(4 * (n - k), std::ios::cur);
    }
    // std::cout << "groundtruth read" << std::endl;
    // calculate raceall
    float recall = Recall(results, groundtruth, q_size, k);
    std::cout << "RECALL: " << recall << std::endl;
    return 0;
}