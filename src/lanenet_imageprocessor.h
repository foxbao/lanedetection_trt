#include "util.h"

class PostProcessor : public util::ImageBase
{
public:
    PostProcessor(const std::string &filename, const nvinfer1::Dims &dims, const std::vector<int> &palette, const int num_classes);
    void calBinary(const int *buffer);
    void calInstance(const float *buffer);
    void processLane(const int *buffer);


private:
    int mNumClasses;
    std::vector<int> mPalette;
    int min_area_threshold;
    cv::Mat _morphological_process(const cv::Mat &image, int kernel_size = 5);
    void _connect_components_analysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids);
};

class ImageProcessor
{
    public:
        void png2ppm(const char *input, util::PPM &ppm);
        void png2ppm(const char *input, const char *output);
    private:
};