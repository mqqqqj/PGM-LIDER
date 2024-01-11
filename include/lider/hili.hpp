#pragma once
#include <set>
#include <mutex>
#include <thread>
#include <array>
#include "./pgm/pgm_index.hpp"
#include "../include/lider/core_model.hpp"

#define DEBUG

template <typename data_type = int, size_t epsilon = 64>
class HiLi
{
private:
    int dim;
    int N;
    int LayerNum;
    double mL;
    int H;
    int km;
    int k; // Number of final output points
    int r0;
    int M;
    CoreModel<DATA_TYPE, 64> *models;
#ifdef DEBUG
    int layer[20];
#endif
public:
    HiLi(int dim, int N, int LayerNum, double mL, int H, int km, int k, int r0, int M) : dim(dim), N(N), LayerNum(LayerNum), mL(mL), H(H), km(km), k(k), r0(r0), M(M)
    {
        models = new CoreModel<DATA_TYPE, 64>[LayerNum];
#ifdef DEBUG
        for (int i = 0; i < 20; i++)
            layer[i] = 0;
#endif
    }
    void index(DATA_TYPE **alldata, std::vector<std::vector<std::vector<DATA_TYPE>>> &uniform_planes)
    {
        // indices
        // std::cout << alldata[999999][127] << std::endl;
        std::vector<std::vector<size_t>> indices(LayerNum);
        // data
        std::vector<std::vector<std::array<DATA_TYPE, 128>>> data(LayerNum);
        for (int i = 0; i < N; i++)
        {
            int l = setLayer();
            l = l >= (LayerNum - 1) ? (LayerNum - 1) : l;
            assert(l < LayerNum);
            assert(l >= 0);
            std::array<DATA_TYPE, 128> arr;
            std::copy(&alldata[i][0], &alldata[i][dim - 1], arr.begin());
            for (int j = 0; j <= l; j++)
            {
#ifdef DEBUG
                layer[j]++;
#endif
                indices[j].push_back(i);
                data[j].push_back(arr);
            }
        }
        showLayerDistribution();
        // build layermodels
        for (int i = 0; i < LayerNum; i++)
        {
            int layer_data_size = data[i].size();
            std::cout << "layer " << i << " data size: " << layer_data_size << std::endl;
            CoreModel<DATA_TYPE, 64> layermodel(km, H, M, r0, layer_data_size, dim, i);
            DATA_TYPE **layerdata;
            size_t *indice = new size_t[layer_data_size];
            if (i == 0)
            {
                layerdata = alldata;
                for (int j = 0; j < layer_data_size; j++)
                {
                    indice[j] = j;
                }
            }
            else
            {
                layerdata = new DATA_TYPE *[layer_data_size];
                for (int j = 0; j < layer_data_size; j++)
                {
                    layerdata[j] = new DATA_TYPE[dim];
                    assert(data[i][j].size() == dim);
                    std::copy(data[i][j].begin(), data[i][j].end(), layerdata[j]);
                    indice[j] = indices[i][j];
                }
            }
            layermodel.index(layerdata, indice, uniform_planes);
            models[i] = layermodel;
            std::cout << "layer " << i << " build finished." << std::endl;
        }
    }

    std::vector<size_t> query(DATA_TYPE *query, std::vector<size_t> &hashed_query)
    {
        std::vector<size_t> candidates = models[LayerNum - 1].query(query, hashed_query); // 最顶层是r0*km个
        int s = candidates.front(), e = candidates.back();
        for (int i = LayerNum - 2; i >= 0; i--)
        {
            // 找到这些点在i-1层的起始点和终止点
            int *startAndend = models[i].search4Hili(s, e);
            s = startAndend[0];
            e = startAndend[1];
        }
        return candidates;
    }

    int setLayer()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        double u = dis(gen);
        int l = std::floor(-std::log(u) * mL);
        return l;
    }

    void showLayerDistribution()
    {
#ifdef DEBUG
        for (int i = 0; i < 20; i++)
        {
            std ::cout << layer[i] << ", ";
        }
        std::cout << std::endl;
#else
        std::cout << "marco debug is off" << std::endl;
#endif
    }
};