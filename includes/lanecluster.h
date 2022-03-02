#pragma once
#include "util.h"
#include "dbscan.hpp"
#include "inner_types.h"
#include <vector>
using namespace std;

class LaneCluster
{
    public:
        LaneCluster();
        void apply_lane_feats_cluster(const cv::Mat &binary_seg_result,const cv::Mat &instance_seg_result,std::vector<inner_type::Lane> &lane_coords,cv::Mat &mask);
    private:
        int color_map_[8][3]={{255, 0, 0},{0, 255, 0},{0, 0, 255},{125, 125, 0},{0, 125, 125},{125, 0, 125},{50, 100, 50},{100, 50, 100}};
        void _embedding_feats_dbscan_cluster(const std::vector<std::vector<float>> &lane_embedding_feats,std::vector<int> &db_labels,std::vector<int> &unique_labels);
        void _get_lane_embedding_feats(const cv::Mat &binary_seg_result,const cv::Mat &instance_seg_result,std::vector<std::vector<float>> &lane_embedding_feats,std::vector<inner_type::LanePoint> &lane_coordinates);
        void _get_mask_lanecoord(const vector<int>& unique_labels,const vector<int>& db_labels,vector<inner_type::LanePoint>& coord,cv::Mat& mask,vector<inner_type::Lane> &lane_coords);
        std::shared_ptr<DBSCAN> sp_dbscan;
        double eps_ ;
        int min_pts_;
};