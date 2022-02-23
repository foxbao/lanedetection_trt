#include "util.h"
class LaneCluster
{
    public:
        LaneCluster();

        void apply_lane_feats_cluster(cv::Mat &binary_seg_result,cv::Mat &instance_seg_result);
    private:
        void _embedding_feats_dbscan_cluster();
        void _get_lane_embedding_feats(cv::Mat &binary_seg_result,cv::Mat &instance_seg_result);

};