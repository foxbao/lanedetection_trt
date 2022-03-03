#include "../includes/lanenet.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"
#include <cassert>
#include <sys/time.h>
#include <vector>
#include <opencv2/opencv.hpp>

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

Lanenet::Lanenet(const std::string &config_file)
{
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["Lanenet"];
    onnx_file_ = config["onnx_file"].as<std::string>();
    engine_file_ = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    OUT_WIDTH = config["OUT_WIDTH"].as<int>();
    OUT_HEIGHT = config["OUT_HEIGHT"].as<int>();
    IMAGE_RESIZE_WIDTH = config["IMAGE_RESIZE_WIDTH"].as<int>();
    IMAGE_RESIZE_HEIGHT = config["IMAGE_RESIZE_HEIGHT"].as<int>();
    CATEGORY = config["CATEGORY"].as<int>();
    img_mean_ = config["img_mean"].as<std::vector<float>>();
    img_std_ = config["img_std"].as<std::vector<float>>();
    // class_colors.resize(CATEGORY);
    srand((int)time(nullptr));
    // for (cv::Scalar &class_color : class_colors)
    //     class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

    sp_ImageProcessor_ = std::make_unique<ImageProcessor>();
    sp_ImgPostProcessor_ = std::make_shared<ImgPostProcessor>();
}

Lanenet::~Lanenet() = default;

void Lanenet::LoadEngine()
{
    std::ifstream engineFile(engine_file_, std::ios::binary);
    if (engineFile.fail())
    {
        return;
    }
    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    // mEngine_.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    mEngine_.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
    assert(mEngine_.get() != nullptr);
}

bool Lanenet::InferenceFolder(const std::string &folder_name)
{
    foldername_ = folder_name;
    context_ = util::UniquePtr<nvinfer1::IExecutionContext>(mEngine_->createExecutionContext());
    if (!context_)
    {
        return false;
    }
    auto input_idx = mEngine_->getBindingIndex("input");
    if (input_idx == -1)
    {
        return false;
    }
    assert(mEngine_->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, 3 /* channels */, IMAGE_HEIGHT, IMAGE_WIDTH};
    context_->setBindingDimensions(input_idx, input_dims); // set the height and width to the context
    // auto input_size = util::getMemorySize(input_dims, sizeof(mEngine_->getBindingDataType(input_idx)));

    int nbBindings = mEngine_->getNbBindings();
    void *buffers_gpu[nbBindings];

    std::vector<int64_t> bufferSize;
    bufferSize.resize(nbBindings);

    std::string output_label = "output";
    std::string input_label = "input";

    // allocate gpu memories
    std::vector<nvinfer1::Dims> data_dims;
    for (size_t i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = context_->getBindingDimensions(i);
        data_dims.push_back(dims);
        nvinfer1::DataType dtype = mEngine_->getBindingDataType(i);
        std::cout << "binding:" << i << " name"
                  << ":" << mEngine_->getBindingName(i) << " dims:" << dims.d[0] << "," << dims.d[1] << "," << dims.d[2] << "," << dims.d[3];
        std::cout << " sizeof(dtype):" << sizeof(dtype);
        int64_t totalSize = util::getMemorySize(dims, sizeof(mEngine_->getBindingDataType(i)));
        bufferSize[i] = totalSize;
        cudaMalloc(&buffers_gpu[i], totalSize);
        std::cout << " total size:" << totalSize << std::endl;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    std::vector<std::string> image_list = readFolder(folder_name);
    this->EngineInference(image_list, buffers_gpu, bufferSize, data_dims, stream);

    // Free CUDA resources
    cudaStreamDestroy(stream);
    this->context_ = nullptr;
    cudaFree(buffers_gpu[0]);
    cudaFree(buffers_gpu[1]);
    cudaFree(buffers_gpu[2]);
    cudaFree(buffers_gpu[3]);

    return true;
}

void Lanenet::PlotImgs(const std::string &file_name_no_extension,
                       std::shared_ptr<int> output_buffer_cpu_1,
                       std::shared_ptr<float> output_buffer_cpu_2,
                       const std::vector<nvinfer1::Dims> &data_dims,
                       const cv::Mat &mask,
                       cv::Mat &resized_img,
                       std::vector<inner_type::Lane> &lanes_coords,
                       std::vector<std::vector<double>> &fit_params)
{
    // cv2.FONT_HERSHEY_SIMPLEX
    auto font = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 2;
    // binary
    std::string binary_file_path = foldername_ + "_binary/" + file_name_no_extension + ".ppm";
    util::PPM ppm_binary;
    sp_ImgPostProcessor_->GenerateBinarySegmentThree(output_buffer_cpu_1.get(), data_dims[2], ppm_binary); // binary output
    sp_ImgPostProcessor_->WriteImg(binary_file_path, ppm_binary);
    cv::Mat mat_binary;
    sp_ImageProcessor_->PPM2Mat(ppm_binary, mat_binary);
    cv::putText(mat_binary, "binary", cv::Point(200, 20), font, 1.0, cv::Scalar(255, 255, 255), thickness);

    // instance
    std::string instance_file_path = foldername_ + "_instance/" + file_name_no_extension + ".jpg";
    util::PPM ppm_instance;
    sp_ImgPostProcessor_->CalInstance(output_buffer_cpu_2.get(), data_dims[3], ppm_instance); // binary output
    // sp_ImgPostProcessor_->WriteImg(instance_file_path, ppm_instance);

    cv::Mat mat_instance;
    sp_ImageProcessor_->PPM2Mat(ppm_instance, mat_instance);
    cv::putText(mat_instance, "instance", cv::Point(200, 20), font, 1.0, cv::Scalar(255, 255, 255), thickness);
    cv::imwrite(instance_file_path, mat_instance);

    // mask
    cv::putText(mask, "cluster", cv::Point(200, 20), font, 1.0, cv::Scalar(255, 255, 255), thickness);
    std::string mask_file_path = foldername_ + "_mask/" + file_name_no_extension + ".jpg";
    // cv::imwrite(mask_file_path, mask);

    // combined fitted
    std::string combined_file_path = foldername_ + "_combined/" + file_name_no_extension + ".jpg";

    // for (auto fit_param : fit_params)
    // {
    //     fit_param[0];
    //     fit_param[1];
    //     fit_param[2];
    //     for (int row = 100; row < resized_img.rows; row=row+10)
    //     {
    //         int col=fit_param[0]+fit_param[1]*row+fit_param[2]*row*row;
    //         if (col<resized_img.cols)
    //         {
    //             cv::circle(resized_img,cv::Point(col,row),1,cv::Scalar(0,255,0),2);
    //         }
            
    //     }
    // }
    // sp_ImageProcessor_->Overlap(output_buffer_cpu_1.get(), data_dims[2].d[2], data_dims[2].d[3], resized_img);
    cv::putText(resized_img, "fitted", cv::Point(200, 20), font, 1.0, cv::Scalar(255, 255, 255), thickness);

    int idx_lane = 0;
    for (auto&& lane : lanes_coords)
    {
        int row_start=lane.pts[0].row_;
        for (int row = row_start; row < resized_img.rows; row=row+10)
        {
            int col=fit_params[idx_lane][0]+fit_params[idx_lane][1]*row+fit_params[idx_lane][2]*row*row;
            if (col<resized_img.cols)
            {
                cv::circle(resized_img,cv::Point(col,row),1,cv::Scalar(0,255,0),2);
            }
        }

        auto position = lane.pts[int(lane.pts.size() / 2)];
        cv::putText(resized_img, "line" + std::to_string(idx_lane), cv::Point(position.col_, position.row_), font, 0.5, cv::Scalar(255, 255, 255), 1);
        idx_lane++;
    }



    std::vector<cv::Mat> vImgs;
    vImgs.push_back(resized_img);
    vImgs.push_back(mat_binary);
    vImgs.push_back(mat_instance);
    vImgs.push_back(mask);
    cv::Mat FeatureMat;
    cv::hconcat(vImgs, FeatureMat);
    cv::imwrite(combined_file_path, FeatureMat);
}

bool Lanenet::EngineInference(const std::vector<std::string> &image_list,
                              void **buffers_gpu,
                              const std::vector<int64_t> &bufferSize,
                              const std::vector<nvinfer1::Dims> &data_dims,
                              cudaStream_t stream)
{

    auto output_buffer_cpu_0 = std::shared_ptr<float>{new float[bufferSize[1]]}; //
    auto output_buffer_cpu_1 = std::shared_ptr<int>{new int[bufferSize[2]]};     //
    auto output_buffer_cpu_2 = std::shared_ptr<float>{new float[bufferSize[3]]}; //

    int index = 0;
    std::string file_name_no_extension;
    std::string input_file_png_name;
    std::string input_file_ppm_name;
    util::PPM ppm;

    auto sp_ImgPostProcessor_ = std::make_unique<ImgPostProcessor>();
    struct timeval t1, t2;
    gettimeofday(&t1, NULL);
    for (const std::string &image_name : image_list)
    {
        index++;
        std::cout << "Processing: " << image_name << std::endl;
        file_name_no_extension = util::get_file_name_no_extension(image_name);
        input_file_png_name = foldername_ + "/" + file_name_no_extension + ".png";
        input_file_ppm_name = foldername_ + "_ppm/" + file_name_no_extension + ".ppm";
        cv::Mat resized_img;
        sp_ImageProcessor_->png2ppm(input_file_png_name.c_str(), ppm, IMAGE_RESIZE_HEIGHT, IMAGE_RESIZE_WIDTH, resized_img);

        auto input_image{util::RGBImageReader(input_file_ppm_name, data_dims[0], this->img_mean_, this->img_std_)};

        input_image.read(ppm);
        // normalize with mean and std
        auto input_buffer = input_image.process();

        // Copy image data to input binding memory
        // if (cudaMemcpyAsync(buffers_gpu[0], input_buffer.get(), bufferSize[0], cudaMemcpyHostToDevice, stream) != cudaSuccess)
        // {
        //     gLogError << "ERROR: CUDA memory copy of input failed, size = " << bufferSize[0] << " bytes" << std::endl;
        //     return false;
        // }
        HANDLE_ERROR(cudaMemcpyAsync(buffers_gpu[0], input_buffer.get(), bufferSize[0], cudaMemcpyHostToDevice, stream));

        // Run TensorRT inference
        void *bindings[] = {buffers_gpu[0], buffers_gpu[1], buffers_gpu[2], buffers_gpu[3]};
        bool status = context_->enqueueV2(bindings, stream, nullptr);
        if (!status)
        {
            gLogError << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        }

        HANDLE_ERROR(cudaMemcpyAsync(output_buffer_cpu_0.get(), buffers_gpu[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
        HANDLE_ERROR(cudaMemcpyAsync(output_buffer_cpu_1.get(), buffers_gpu[2], bufferSize[2], cudaMemcpyDeviceToHost, stream));
        HANDLE_ERROR(cudaMemcpyAsync(output_buffer_cpu_2.get(), buffers_gpu[3], bufferSize[3], cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);
        std::vector<inner_type::Lane> lanes_coords;
        cv::Mat mask;
        std::vector<std::vector<double>> fit_params;
        sp_ImgPostProcessor_->ProcessLane(output_buffer_cpu_1.get(), data_dims[2], output_buffer_cpu_2.get(), data_dims[3], mask, lanes_coords, fit_params);

        bool plot = true;
        // bool plot = false;
        if (plot)
        {
            this->PlotImgs(file_name_no_extension,
                           output_buffer_cpu_1,
                           output_buffer_cpu_2,
                           data_dims, mask,
                           resized_img,
                           lanes_coords,
                           fit_params);
        }

        //
    }

    gettimeofday(&t2, NULL);
    double deltaT = (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
    double deltaTsec = deltaT / 1000000;
    std::cout << "time_comsumed:" << deltaT / 1000000 << std::endl;
    std::cout << "FPS:" << index / deltaTsec;

    return true;
}
