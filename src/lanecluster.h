#include "util.h"
#include "dbscan.h"
#include "inner_types.h"
class LaneCluster
{
    public:
        LaneCluster();
        void apply_lane_feats_cluster(const cv::Mat &binary_seg_result,const cv::Mat &instance_seg_result,std::vector<inner_type::Lane> &lane_coords);
    private:
        void _embedding_feats_dbscan_cluster(const std::vector<std::vector<float>> &lane_embedding_feats,std::vector<int> &db_labels,std::vector<int> &unique_labels);
        void _get_lane_embedding_feats(const cv::Mat &binary_seg_result,const cv::Mat &instance_seg_result,std::vector<std::vector<float>> &lane_embedding_feats,inner_type::Lane &lane_coordinates);
        std::shared_ptr<DBSCAN> sp_dbscan;
};