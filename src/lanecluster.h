#include "util.h"
#include "dbscan.h"
class LaneCluster
{
    public:
        LaneCluster();
        void apply_lane_feats_cluster(cv::Mat &binary_seg_result,cv::Mat &instance_seg_result);
    private:
        void _embedding_feats_dbscan_cluster(const std::vector<std::vector<float>> &lane_embedding_feats);
        void _get_lane_embedding_feats(cv::Mat &binary_seg_result,cv::Mat &instance_seg_result,std::vector<std::vector<float>> &lane_embedding_feats,std::vector<std::vector<int>> &lane_coordinate);
        std::shared_ptr<DBSCAN> sp_dbscan;
};