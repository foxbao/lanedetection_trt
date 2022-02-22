#include "util.h"
class ImgPostProcessor
{
    public:
        void calBinary(const int *buffer,const nvinfer1::Dims& dim,util::PPM &dst_ppm);
        void calInstance(const float *buffer,const nvinfer1::Dims& dim,util::PPM &dst_ppm);
        void processLane(const int *buffer_binary,const nvinfer1::Dims& dim_binary,const float *buffer_instance,const nvinfer1::Dims& dim_instance);

        void write(const std::string& filename,const util::PPM ppm);
        int volume(util::PPM &ppm);
    private:
        int min_area_threshold;
        cv::Mat _morphological_process(const cv::Mat &image, int kernel_size = 5);
        void _connect_components_analysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids);
};