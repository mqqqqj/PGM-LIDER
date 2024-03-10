#include <chrono>
#include "../include/lider/core_model.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <queue>
#include <condition_variable>
#include <functional>
#include <atomic>

class ThreadPool
{
public:
    ThreadPool(size_t threads) : stop(false)
    {
        for (size_t i = 0; i < threads; ++i)
            workers.emplace_back([this]
                                 {
                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                } });
    }

    template <class F>
    void enqueue(F &&f)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

void test_core()
{
    // read cluster0 info
    int c = 400;
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
    int H = 16;
    int km = 10;
    int r0 = 30;
    int M = 20;
    auto uniform_planes = gen_uniform_planes(H, M, dim);
    CoreModel<DATA_TYPE, 64> in_cluster_0(km, H, M, r0, N, dim, 1);
    in_cluster_0.index(data, indices, uniform_planes);

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

void naive_read()
{
    int D = 128;
    // read base data
    auto start_time = std::chrono::high_resolution_clock::now();
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
    // printf("%f \n", data[1236][45]);
    // read query info
    const char *qpath = "/home/mqj/data/sift/sift_query.fvecs";
    std::ifstream qFile(qpath, std::ios::binary);
    int q_size = 1000;
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
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Naive read base data and query data: " << duration.count() / 1000.0 << " seconds." << std::endl;
}

void fast_read()
{
    int D = 128;
    // read base data
    auto start_time = std::chrono::high_resolution_clock::now();
    int fd = open("/home/mqj/data/sift/sift_base.fvecs", O_RDONLY);
    int N = 1000000;
    char *addr = (char *)mmap(NULL, N * (D + 1) * sizeof(DATA_TYPE), PROT_READ, MAP_PRIVATE, fd, 0);
    DATA_TYPE **data = (DATA_TYPE **)_mm_malloc(N * sizeof(DATA_TYPE *), 32);
    for (int i = 0; i < N; i++)
    {
        data[i] = (DATA_TYPE *)_mm_malloc(D * sizeof(DATA_TYPE), 32);
        memcpy(data[i], addr + (i * (D + 1) + 1) * sizeof(DATA_TYPE), D * sizeof(DATA_TYPE));
    }
    munmap(addr, N * (D + 1) * sizeof(DATA_TYPE));
    close(fd); // unistd.h
    // printf("%f \n", data[1236][45]);
    // read query info
    fd = open("/home/mqj/data/sift/sift_query.fvecs", O_RDONLY);
    int q_size = 1000;
    addr = (char *)mmap(NULL, q_size * (D + 1) * sizeof(DATA_TYPE), PROT_READ, MAP_PRIVATE, fd, 0);
    DATA_TYPE **query_set = (DATA_TYPE **)_mm_malloc(q_size * sizeof(DATA_TYPE *), 32);
    for (int i = 0; i < q_size; i++)
    {
        query_set[i] = (DATA_TYPE *)_mm_malloc(D * sizeof(DATA_TYPE), 32);
        memcpy(query_set[i], addr + (i * (D + 1) + 1) * sizeof(DATA_TYPE), D * sizeof(DATA_TYPE));
    }
    munmap(addr, q_size * (D + 1) * sizeof(DATA_TYPE));
    close(fd);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Fast read base data and query data: " << duration.count() / 1000.0 << " seconds." << std::endl;
}

int main()
{
    // test_core();
    fast_read();
    naive_read();
    return 0;
}