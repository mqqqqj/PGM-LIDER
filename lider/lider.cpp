#include "../include/lider/lider.hpp"
#include <sstream>
#include <thread>
#include <future>
const int D = 128;
int H = 10;
int km = 10;
int k = 10; // Number of output points by LIDER
int r0 = 20;
int M = 20;
int c0 = 20; // Number of output centroids by top level core model
int c = 300; // 在cluster.py中聚类
std::vector<std::vector<std::vector<float>>> uniform_planes = gen_uniform_planes(H, M, D);

void process_cluster(int in_cluster_id, int N, std::vector<CoreModel<DATA_TYPE, 64>> &InClusterRetrivers)
{
    DATA_TYPE **data = new DATA_TYPE *[N];
    for (int i = 0; i < N; i++)
    {
        data[i] = new DATA_TYPE[D];
    }
    std::string dataPath = "/home/mqj/data/sift/" + std::to_string(c) + "-kmeans/cluster" + std::to_string(in_cluster_id) + "_data.bin";
    std::ifstream dataFile(dataPath, std::ios::binary);
    for (int i = 0; i < N; i++)
    {
        dataFile.read(reinterpret_cast<char *>(data[i]), sizeof(DATA_TYPE) * D);
    }
    dataFile.close();
    size_t *indices = new size_t[N];
    std::string indicePath = "/home/mqj/data/sift/" + std::to_string(c) + "-kmeans/cluster" + std::to_string(in_cluster_id) + "_indices.bin";
    std::ifstream indiceFile(indicePath, std::ios::binary);
    for (int i = 0; i < N; i++)
    {
        indiceFile.read(reinterpret_cast<char *>(&indices[i]), sizeof(size_t));
    }
    // std::cout << "finish read cluster " << in_cluster_id << std::endl;

    CoreModel<DATA_TYPE, 64> InClusterRetriver(km, H, M, r0, N, D, in_cluster_id + 1); // in cluster id range:1~c
    InClusterRetriver.index(data, indices, uniform_planes);
    InClusterRetrivers[in_cluster_id] = InClusterRetriver;
    // std::cout << "finish build InClusterRetriver " << in_cluster_id << std::endl;
}

int main()
{
    CoreModel<DATA_TYPE, 64> CentroidsRetriver(c0, H, M, r0, c, D, 0); // centroid retriver id is 0
    DATA_TYPE **centroidsData = new DATA_TYPE *[c];
    // read data for CentroidsRetriver
    std::vector<CoreModel<DATA_TYPE, 64>> InClusterRetrivers(c);
    // read cluster info
    std::string filePath = "/home/mqj/data/sift/" + std::to_string(c) + "-kmeans/clustersInfo.num.vec";
    std::ifstream inputFile(filePath, std::ios::binary);
    int ClusetersSize[c];
    for (int in_cluster_id = 0; in_cluster_id < c; in_cluster_id++)
    {
        centroidsData[in_cluster_id] = new DATA_TYPE[D];
        inputFile.read(reinterpret_cast<char *>(&ClusetersSize[in_cluster_id]), sizeof(int));
        inputFile.read(reinterpret_cast<char *>(centroidsData[in_cluster_id]), sizeof(DATA_TYPE) * D);
        // std::cout << "cluster " << in_cluster_id << "'s size: " << ClusetersSize[in_cluster_id] << std::endl;
    }
    inputFile.close();

    // build LIDER
    // create indices for centroid data
    size_t *centroidIndices = new size_t[c];
    for (int i = 0; i < c; i++)
        centroidIndices[i] = i;
    auto start_time = std::chrono::high_resolution_clock::now();
    CentroidsRetriver.index(centroidsData, centroidIndices, uniform_planes);
    // std::cout << "finish build CentroidsRetriver" << std::endl;

    std::vector<std::thread> threads;
    for (int in_cluster_id = 0; in_cluster_id < c; in_cluster_id++)
    {
        // void process_cluster(int in_cluster_id, std::vector<CoreModel<DATA_TYPE, 64>> &InClusterRetrivers, int *ClusetersSize)
        threads.push_back(std::thread(process_cluster, in_cluster_id, ClusetersSize[in_cluster_id], std::ref(InClusterRetrivers)));
    }

    for (auto &th : threads)
    {
        th.join();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Construction time cost of LIDER: " << duration.count() / 1000.0 << " seconds." << std::endl;
    // read base data
    std::string base = "/home/mqj/data/sift/sift_base.fvecs";
    std::ifstream baseFile(base, std::ios::binary);
    int N = 1000000;
    DATA_TYPE **data = new DATA_TYPE *[N];
    for (int i = 0; i < N; i++)
    {
        data[i] = new DATA_TYPE[D];
        baseFile.seekg(4, std::ios::cur);
        baseFile.read(reinterpret_cast<char *>(data[i]), sizeof(DATA_TYPE) * D);
    }
    LIDER<DATA_TYPE, 64> lider(CentroidsRetriver, InClusterRetrivers, data, c0, km, c, k, D);

    // read query info
    const char *qpath = "/home/mqj/data/sift/sift_query.fvecs";
    std::ifstream qFile(qpath, std::ios::binary);
    int q_size = 100;
    DATA_TYPE **query_set = new DATA_TYPE *[q_size];
    for (int i = 0; i < q_size; i++)
    {
        query_set[i] = new DATA_TYPE[D];
    }
    for (int i = 0; i < q_size; i++)
    {
        qFile.seekg(4, std::ios::cur);
        qFile.read(reinterpret_cast<char *>(query_set[i]), sizeof(DATA_TYPE) * D);
    }
    qFile.close();

    // start query
    // std::cout << "start query." << std::endl;
    std::vector<std::vector<size_t>> results;
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < q_size; i++)
    {
        auto q_hashes = hash(uniform_planes, query_set[i]); // num_hashtable * hashkey_size
        auto result = lider.query(query_set[i], q_hashes);
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