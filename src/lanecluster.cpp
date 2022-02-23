#include "lanecluster.h"

LaneCluster::LaneCluster()
{
}

void LaneCluster::apply_lane_feats_cluster(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result)
{
    this->_get_lane_embedding_feats(binary_seg_result, instance_seg_result);
}

void LaneCluster::_embedding_feats_dbscan_cluster()
{
}

void LaneCluster::_get_lane_embedding_feats(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result)
{
    int aaa=1;
    cv::MatIterator_<uchar> it, end;
    for (it = binary_seg_result.begin<uchar>(), end = binary_seg_result.end<uchar>(); it != end; ++it)
    {

    }
        // idx = np.where(binary_seg_ret == 255)
        // lane_embedding_feats = instance_seg_ret[:, idx[0],idx[1]].transpose()
        // lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        // assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        // ret = {
        //     'lane_embedding_feats': lane_embedding_feats,
        //     'lane_coordinates': lane_coordinate
        // }

        // return ret
}