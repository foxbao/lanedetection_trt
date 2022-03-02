#include "../includes/lanecluster.h"

using namespace std;
LaneCluster::LaneCluster()
{
    sp_dbscan = std::make_shared<DBSCAN>();
    eps_ = 0.35;
    min_pts_ = 100;
}

void LaneCluster::_get_mask_lanecoord(const vector<int> &unique_labels, const vector<int> &db_labels, vector<inner_type::LanePoint> &coord, cv::Mat &mask, vector<inner_type::Lane> &lane_coords)
{
    for (size_t index_unique_label = 0; index_unique_label < unique_labels.size(); index_unique_label++)
    {
        inner_type::Lane lane;
        int label = unique_labels[index_unique_label];
        for (size_t index_label = 0; index_label < db_labels.size(); index_label++)
        {
            if (unique_labels[index_unique_label] == db_labels[index_label])
            {
                lane.pts.push_back(inner_type::LanePoint(coord[index_label].row, coord[index_label].col));
                mask.at<cv::Vec3b>(coord[index_label].row, coord[index_label].col)[0] = color_map_[index_unique_label][0];
                mask.at<cv::Vec3b>(coord[index_label].row, coord[index_label].col)[1] = color_map_[index_unique_label][1];
                mask.at<cv::Vec3b>(coord[index_label].row, coord[index_label].col)[2] = color_map_[index_unique_label][2];
            }
        }
        lane_coords.push_back(lane);
    }
}

void LaneCluster::apply_lane_feats_cluster(const cv::Mat &binary_seg_result, const cv::Mat &instance_seg_result, std::vector<inner_type::Lane> &lane_coords, cv::Mat &mask)
{
    std::vector<std::vector<float>> lane_embedding_feats;
    std::vector<inner_type::LanePoint> coord;
    std::vector<int> db_labels;
    std::vector<int> unique_labels;
    mask = cv::Mat(binary_seg_result.rows, binary_seg_result.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    this->_get_lane_embedding_feats(binary_seg_result, instance_seg_result, lane_embedding_feats, coord);
    this->_embedding_feats_dbscan_cluster(lane_embedding_feats, db_labels, unique_labels);
    lane_coords.clear();
    this->_get_mask_lanecoord(unique_labels, db_labels, coord, mask, lane_coords);
}

void LaneCluster::_embedding_feats_dbscan_cluster(const std::vector<std::vector<float>> &lane_embedding_feats, std::vector<int> &db_labels, std::vector<int> &unique_labels)
{
    auto clusters = sp_dbscan->dbscan(lane_embedding_feats, eps_, min_pts_);

    for (size_t i = 0; i < clusters.size(); i++)
    {
        unique_labels.push_back(i);
    }
    db_labels.resize(lane_embedding_feats.size());
    for (size_t idx_cluster = 0; idx_cluster < clusters.size(); idx_cluster++)
    {
        for (auto db_idx : clusters[idx_cluster])
        {
            db_labels[db_idx] = idx_cluster;
        }
    }
}

void LaneCluster::_get_lane_embedding_feats(const cv::Mat &binary_seg_result, const cv::Mat &instance_seg_result, std::vector<std::vector<float>> &lane_embedding_feats, std::vector<inner_type::LanePoint> &lane_coordinates)
{
    for (size_t nrow = 0; nrow < binary_seg_result.rows; nrow++)
    {
        const uchar *binary_ptr = binary_seg_result.ptr<uchar>(nrow);
        const cv::Vec3b *instance_ptr = instance_seg_result.ptr<cv::Vec3b>(nrow);
        // uchar *morphological_ret_ptr = morphological_ret.ptr<uchar>(nrow);
        for (size_t ncol = 0; ncol < binary_seg_result.cols * binary_seg_result.channels(); ncol++)
        {
            if (int(255 == binary_ptr[ncol]))
            {
                float r = instance_ptr[ncol][0];
                float g = instance_ptr[ncol][1];
                float b = instance_ptr[ncol][2];
                std::vector<float> feat{r,g,b};
                lane_embedding_feats.push_back(feat);
                lane_coordinates.push_back(inner_type::LanePoint(nrow, ncol));
            }
        }
    }
}