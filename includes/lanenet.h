#pragma once
#include <opencv2/opencv.hpp>

#include "logger.h"
#include "util.h"
#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "imageprocessor.h"
#include "../includes/postprocessor.h"
constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;
class Lanenet
{
public:
    Lanenet(const std::string &config_file);
    ~Lanenet();
    void LoadEngine();
    bool InferenceFolder(const std::string &folder_name);

private:
    bool EngineInference(const std::vector<std::string> &image_list,void **buffers_gpu,
            const std::vector<int64_t> &bufferSize,const std::vector<nvinfer1::Dims> &data_dims, cudaStream_t stream);

    std::vector<float> PrepareImage(std::vector<cv::Mat> & image);
    std::vector<cv::Mat> PostProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void Overlap(const int *buffer, int H,int W,cv::Mat &resized_img);
    void PlotImgs(const std::string &file_name_no_extension,std::shared_ptr<int>output_buffer_cpu_1,std::shared_ptr<float>output_buffer_cpu_2,const std::vector<nvinfer1::Dims> &data_dims,const cv::Mat &mask,cv::Mat &resized_img);
    std::string onnx_file_;
    std::string engine_file_;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int OUT_WIDTH;
    int OUT_HEIGHT;
    int IMAGE_RESIZE_WIDTH;
    int IMAGE_RESIZE_HEIGHT;
    int CATEGORY;
    std::vector<float> img_mean_;
    std::vector<float> img_std_;
    util::UniquePtr<nvinfer1::IExecutionContext> context_ =nullptr;;
    std::string foldername_;

    nvinfer1::Dims inputDims_;                      //!< The dimensions of the input to the network.
    nvinfer1::Dims outputDims_;                     //!< The dimensions of the output to the network.

    util::UniquePtr<nvinfer1::ICudaEngine> mEngine_; //!< The TensorRT engine used to run the network
    std::shared_ptr<ImageProcessor> sp_ImageProcessor_;
    std::shared_ptr<ImgPostProcessor> sp_ImgPostProcessor_;
};

