#include <chrono>
#include "../include/lider/core_model.hpp"

void test_core()
{
    // read cluster0 info
    int c = 300;
    std::string filePath = "/home/mqj/data/sift/" + std::to_string(c) + "-kmeans/clustersInfo.num.vec";
    std::ifstream inputFile(filePath, std::ios::binary);
    int N;
    inputFile.read(reinterpret_cast<char *>(&N), sizeof(int));
    inputFile.close();

    // read cluster dataset
    const int dim = 128;
    DATA_TYPE **data = new DATA_TYPE *[N];
    std::cout << "num of cluster0 " << N << std::endl;
    for (int i = 0; i < N; i++)
    {
        data[i] = new DATA_TYPE[dim];
    }
    std::string filePath1 = "/home/mqj/data/sift/" + std::to_string(c) + "-kmeans/cluster0_data.bin";
    std::ifstream dataFile(filePath1, std::ios::binary);
    for (int i = 0; i < N; i++)
    {
        dataFile.read(reinterpret_cast<char *>(data[i]), sizeof(DATA_TYPE) * dim);
    }
    dataFile.close();

    size_t *indices = new size_t[N];
    std::string indicePath = "/home/mqj/data/sift/" + std::to_string(c) + "-kmeans/cluster0_indices.bin";
    std::ifstream indiceFile(indicePath, std::ios::binary);
    for (int i = 0; i < N; i++)
    {
        indiceFile.read(reinterpret_cast<char *>(&indices[i]), sizeof(size_t));
    }

    std::cout << "dim of data " << dim << std::endl;
    // build LIDER
    int H = 10;
    int km = 10;
    int r0 = 20;
    int M = 20;
    auto uniform_planes = gen_uniform_planes(H, M, dim);
    CoreModel<DATA_TYPE, 64> in_cluster_0(km, H, M, r0, N, dim, 1);
    in_cluster_0.index(data, indices, uniform_planes);

    // read query info
    const char *qpath = "/home/mqj/data/sift/sift_query.fvecs";
    std::ifstream qFile(qpath, std::ios::binary);
    int q_size = 100;
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

    // calculate groundtruth
    std::string gtpath = "./gt-cluster0-" + std::to_string(c) + "query-top100.bin";
    std::cout << "start calculate groundtruth" << std::endl;
    calc_gt4cluster(gtpath, data, N, query_set, 100, 128);

    // read groundtruth
    std::ifstream gtFile(gtpath, std::ios::binary);
    int **gt_set = new int *[q_size];
    for (int i = 0; i < q_size; i++)
    {
        gt_set[i] = new int[100];
    }
    for (int i = 0; i < q_size; i++)
    {
        gtFile.read(reinterpret_cast<char *>(gt_set[i]), sizeof(int) * 100);
    }
    gtFile.close();
    std::cout << "groundtruth read" << std::endl;

    // query
    std::vector<std::vector<size_t>> results;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < q_size; i++)
    {
        auto q_hashes = hash(uniform_planes, query_set[i]); // num_hashtable * hashkey_size
        auto result = in_cluster_0.query(query_set[i], q_hashes);
        results.push_back(result);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float recall = Recall(results, gt_set, q_size, km);
    std::cout << "RECALL: " << recall << std::endl;
    std::cout << "QPS: " << q_size / (duration.count() / 1000.0) << std::endl;
}

int main()
{
    test_core();
    return 0;
}