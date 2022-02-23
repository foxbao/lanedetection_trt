#include "lanecluster.h"

LaneCluster::LaneCluster()
{
    sp_dbscan=std::make_shared<DBSCAN>();
}

void LaneCluster::apply_lane_feats_cluster(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result)
{
    std::vector<std::vector<float>> lane_embedding_feats;
    std::vector<std::vector<int>> lane_coordinate;
    this->_get_lane_embedding_feats(binary_seg_result, instance_seg_result,lane_embedding_feats,lane_coordinate);
    _embedding_feats_dbscan_cluster(lane_embedding_feats);
    
    int bbb=2;
}

void LaneCluster::_embedding_feats_dbscan_cluster(const std::vector<std::vector<float>> &lane_embedding_feats)
{
    sp_dbscan->cluster(lane_embedding_feats);
}


void LaneCluster::_get_lane_embedding_feats(cv::Mat &binary_seg_result, cv::Mat &instance_seg_result,std::vector<std::vector<float>> &lane_embedding_feats,std::vector<std::vector<int>> &lane_coordinate)
{
    for (size_t nrow = 0; nrow < binary_seg_result.rows; nrow++)
    {
        uchar *binary_ptr = binary_seg_result.ptr<uchar>(nrow);
        cv::Vec3b *instance_ptr = instance_seg_result.ptr<cv::Vec3b>(nrow);
        // uchar *morphological_ret_ptr = morphological_ret.ptr<uchar>(nrow);
        for (size_t ncol = 0; ncol < binary_seg_result.cols * binary_seg_result.channels(); ncol++)
        {
           if(int( binary_ptr[ncol])==255)
           {
                float r=instance_ptr[ncol][0];
                float g=instance_ptr[ncol][1];
                float b=instance_ptr[ncol][2];
                std::vector<float> feat;
                feat.push_back(r);
                feat.push_back(g);
                feat.push_back(b);
                lane_embedding_feats.push_back(feat);
                std::vector<int> coordinate;
                coordinate.push_back(nrow);
                coordinate.push_back(ncol);
                lane_coordinate.push_back(coordinate);
           }
        }
    }

}