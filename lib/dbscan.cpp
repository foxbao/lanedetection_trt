

#include "nanoflann.hpp"
#include <vector>
#include "include/dbscan.hpp"
// And this is the "dataset to kd-tree" adaptor class:
struct adaptor
{
    const std::vector<std::vector<float>>&  points;
    adaptor(const std::vector<std::vector<float>>&  points) : points(points) { }

    /// CRTP helper method
    //inline const Derived& derived() const { return obj; }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return points.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return points[idx][dim];
        // if (dim==0)
        // {
        //     return points[idx]
        // }
        // return (dim == 0)? points[idx].first: points[idx].second;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

};


auto get_query_point(const std::vector<std::vector<float>>& data, size_t index)
{
    return std::array<float, 3>({data[index][0], data[index][1],data[index][2]});
}



auto sort_clusters(std::vector<std::vector<size_t>>& clusters)
{
    for(auto& cluster: clusters)
    {
        std::sort(cluster.begin(), cluster.end());
    }
}


auto DBSCAN::dbscan(const std::vector<std::vector<float>>& data, float eps, int min_pts)-> std::vector<std::vector<size_t>>
{
    eps *= eps;
    const auto adapt = adaptor(data);
    using namespace nanoflann;
    using  my_kd_tree_t = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, decltype(adapt)>, decltype(adapt), 2>;

    auto index = my_kd_tree_t(2, adapt, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    auto visited  = std::vector<bool>(data.size());
    auto clusters = std::vector<std::vector<size_t>>();
    auto matches  = std::vector<std::pair<size_t, float>>();
    auto sub_matches = std::vector<std::pair<size_t, float>>();

    for(size_t i = 0; i < data.size(); i++)
    {
        if (visited[i]) continue;

        index.radiusSearch(get_query_point(data, i).data(), eps, matches, SearchParams(32, 0.f, false));
        if (matches.size() < static_cast<size_t>(min_pts)) continue;
        visited[i] = true;

        auto cluster = std::vector<size_t>({i});

        while (matches.empty() == false)
        {
            auto nb_idx = matches.back().first;
            matches.pop_back();
            if (visited[nb_idx]) continue;
            visited[nb_idx] = true;
            index.radiusSearch(get_query_point(data, nb_idx).data(), eps, sub_matches, SearchParams(32, 0.f, false));

            if (sub_matches.size() >= static_cast<size_t>(min_pts))
            {
                std::copy(sub_matches.begin(), sub_matches.end(), std::back_inserter(matches));
            }
            cluster.push_back(nb_idx);
        }
        clusters.emplace_back(std::move(cluster));
    }
    sort_clusters(clusters);
    return clusters;
}