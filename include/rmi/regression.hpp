#pragma once

#include <cstddef> // Include the <cstddef> header for size_t

struct ApproxPos
{
    size_t pos; ///< The approximate position of the key.
    size_t lo;  ///< The lower bound of the range.
    size_t hi;  ///< The upper bound of the range.
};

template <typename data_type = int, size_t epsilon = 64>
class LinearRegression
{
private:
public:
    LinearRegression(){};
    LinearRegression(std::vector<DATA_TYPE> &data)
    {
    }
    ApproxPos search(DATA_TYPE query) const // query is after hash
    {
        ApproxPos result{0, 0, 0};
        return result;
    }
};