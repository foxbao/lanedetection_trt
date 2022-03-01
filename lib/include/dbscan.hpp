#pragma once
#include <vector>
class DBSCAN
{
public:
        auto dbscan(const std::vector<std::vector<float>> &data, float eps, int min_pts)
            -> std::vector<std::vector<size_t>>;
        // auto sort_clusters(std::vector<std::vector<size_t>>& clusters);
        // auto get_query_point(const std::vector<std::vector<float>>& data, size_t index);
};
// auto dbscan(const std::vector<std::vector<float>>& data, float eps, int min_pts)
//         -> std::vector<std::vector<size_t>>;