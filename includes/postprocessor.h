#include "util.h"
#include "lanecluster.h"
#include "inner_types.h"

// #include "dbscan.h"
class ImgPostProcessor
{
    public:
        ImgPostProcessor();
        void generateBinarySegmentThree(const int *buffer,const nvinfer1::Dims& dim,util::PPM &dst_ppm);
        void generateBinarySegment(const int *buffer,const nvinfer1::Dims& dim,util::PPM &dst_ppm);
        void calInstance(const float *buffer,const nvinfer1::Dims& dim,util::PPM &dst_ppm);
        void processLane(const int *buffer_binary,const nvinfer1::Dims& dim_binary,const float *buffer_instance,const nvinfer1::Dims& dim_instance,cv::Mat &mask);
        void write(const std::string& filename,const util::PPM ppm);
        int volume(util::PPM &ppm);
    private:
        std::shared_ptr<LaneCluster> sp_laneCluster_;
        
        int min_area_threshold;
        cv::Mat _morphological_process(const util::PPM &binary_ppm, int kernel_size = 5);
        cv::Mat _morphological_process(const cv::Mat &image, int kernel_size = 5);
        void _generateMatInstance(cv::Mat &mat_instance_seg_result,const float *buffer_instance, const nvinfer1::Dims &dim_instance);
        void _connect_components_analysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids);
        void _remove_small_connect_components(const cv::Mat &labels, const cv::Mat &stats,cv::Mat &morphological_ret);
        bool _isInVector(const int &value, std::vector<int> &vec);
};