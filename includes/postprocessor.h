#include "util.h"
#include "lanecluster.h"
// #include "inner_types.h"
#include "polyfit.h"
// #include "dbscan.h"
class ImgPostProcessor
{
public:
    ImgPostProcessor();
    void CalInstance(const float *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm);
    void GenerateBinarySegmentThree(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm);
    void GenerateBinarySegment(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm);
    void ProcessLane(const int *buffer_binary,
                     const nvinfer1::Dims &dim_binary,
                     const float *buffer_instance,
                     const nvinfer1::Dims &dim_instance,
                     cv::Mat &mask,
                     std::vector<inner_type::Lane> &lanes_coords,
                     std::vector<std::vector<double>> &fit_params);
    void WriteImg(const std::string &filename, const util::PPM ppm);
    void LineFit(const std::vector<inner_type::Lane> &lanes_coords, std::vector<std::vector<double>> &fit_params);
    int volume(util::PPM &ppm);

private:
    cv::Mat MorphologicalProcess(const util::PPM &binary_ppm, int kernel_size = 5);
    cv::Mat MorphologicalProcess(const cv::Mat &image, int kernel_size = 5);
    void GenerateMatInstance(cv::Mat &mat_instance_seg_result, const float *buffer_instance, const nvinfer1::Dims &dim_instance);
    void ConnectComponentsAnalysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids);
    void RemoveSmallConnectComponents(const cv::Mat &labels, const cv::Mat &stats, cv::Mat &morphological_ret);
    bool IsInVector(const int &value, std::vector<int> &vec);

    std::shared_ptr<LaneCluster> sp_laneCluster_;
    std::shared_ptr<PolyFit> sp_polyfit_;
    int min_area_threshold_;
};