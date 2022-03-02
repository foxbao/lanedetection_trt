#ifndef LANENET_TRT_LANENET_H
#define LANENET_TRT_LANENET_H

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

    std::vector<float> prepareImage(std::vector<cv::Mat> & image);
    std::vector<cv::Mat> postProcess(const std::vector<cv::Mat> &vec_Mat, float *output, const int &outSize);
    void plotImgs(const std::string &file_name_no_extension,std::shared_ptr<int>output_buffer_cpu_1,std::shared_ptr<float>output_buffer_cpu_2,const std::vector<nvinfer1::Dims> &data_dims,const cv::Mat &mask);
    std::string onnx_file;
    std::string engine_file;
    int BATCH_SIZE;
    int INPUT_CHANNEL;
    int IMAGE_WIDTH;
    int IMAGE_HEIGHT;
    int OUT_WIDTH;
    int OUT_HEIGHT;
    int IMAGE_RESIZE_WIDTH;
    int IMAGE_RESIZE_HEIGHT;
    int CATEGORY;
    std::vector<float> img_mean;
    std::vector<float> img_std;
    // nvinfer1::ICudaEngine *engine = nullptr;
    util::UniquePtr<nvinfer1::IExecutionContext> context =nullptr;;
    std::string m_foldername;
    // nvinfer1::IExecutionContext *context = nullptr;
    // std::vector<cv::Scalar> class_colors;

    // std::string mEngineFilename;                    //!< Filename of the serialized engine.

    nvinfer1::Dims mInputDims;                      //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims;                     //!< The dimensions of the output to the network.

    util::UniquePtr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::shared_ptr<ImageProcessor> pImageProcessor;
    std::shared_ptr<ImgPostProcessor> pImgPostProcessor= std::make_shared<ImgPostProcessor>();
    void test_func(std::unique_ptr<int>output_buffer_cpu_1);
};

#endif //LENET_TRT_LENET_H
