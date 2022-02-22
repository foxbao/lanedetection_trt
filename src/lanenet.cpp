#include "lanenet.h"
#include "yaml-cpp/yaml.h"
#include "common.hpp"
#include "postprocessor.h"

// #include "common.h"
#include <vector>
Lanenet::Lanenet(const std::string &config_file)
{
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node config = root["Lanenet"];
    onnx_file = config["onnx_file"].as<std::string>();
    engine_file = config["engine_file"].as<std::string>();
    BATCH_SIZE = config["BATCH_SIZE"].as<int>();
    INPUT_CHANNEL = config["INPUT_CHANNEL"].as<int>();
    IMAGE_WIDTH = config["IMAGE_WIDTH"].as<int>();
    IMAGE_HEIGHT = config["IMAGE_HEIGHT"].as<int>();
    OUT_WIDTH = config["OUT_WIDTH"].as<int>();
    OUT_HEIGHT = config["OUT_HEIGHT"].as<int>();
    IMAGE_RESIZE_WIDTH = config["IMAGE_RESIZE_WIDTH"].as<int>();
    IMAGE_RESIZE_HEIGHT = config["IMAGE_RESIZE_HEIGHT"].as<int>();
    CATEGORY = config["CATEGORY"].as<int>();
    img_mean = config["img_mean"].as<std::vector<float>>();
    img_std = config["img_std"].as<std::vector<float>>();
    // class_colors.resize(CATEGORY);
    srand((int)time(nullptr));
    // for (cv::Scalar &class_color : class_colors)
    //     class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);

    pImageProcessor=std::make_unique<ImageProcessor>();
}

Lanenet::~Lanenet() = default;

void Lanenet::LoadEngine()
{
    std::ifstream engineFile(engine_file, std::ios::binary);
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
    mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    assert(mEngine.get() != nullptr);
}


bool Lanenet::InferenceFolder(const std::string &folder_name)
{
    m_foldername=folder_name;
    context = util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    auto input_idx = mEngine->getBindingIndex("input");
    if (input_idx == -1)
    {
        return false;
    }
    assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
    auto input_dims = nvinfer1::Dims4{1, 3 /* channels */, IMAGE_HEIGHT, IMAGE_WIDTH};
    context->setBindingDimensions(input_idx, input_dims); // set the height and width to the context
    // auto input_size = util::getMemorySize(input_dims, sizeof(mEngine->getBindingDataType(input_idx)));

    int nbBindings = mEngine->getNbBindings();
    void *buffers_gpu[nbBindings];
    
    std::vector<int64_t> bufferSize;
    bufferSize.resize(nbBindings);

    std::string output_label = "output";
    std::string input_label = "input";

    std::vector<nvinfer1::Dims> data_dims;
    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        data_dims.push_back(dims);
        nvinfer1::DataType dtype = mEngine->getBindingDataType(i);
        std::cout << "binding:" << i << " name"
                  << ":" << mEngine->getBindingName(i) << " dims:" << dims.d[0] << "," << dims.d[1] << "," << dims.d[2] << "," << dims.d[3];
        std::cout << " sizeof(dtype):" << sizeof(dtype);
        int64_t totalSize = util::getMemorySize(dims, sizeof(mEngine->getBindingDataType(i)));
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

    this->EngineInference(image_list, buffers_gpu,bufferSize, data_dims,stream);

    // Free CUDA resources
    cudaStreamDestroy(stream);
    this->context=nullptr;
    cudaFree(buffers_gpu[0]);
    cudaFree(buffers_gpu[1]);
    cudaFree(buffers_gpu[2]);
    cudaFree(buffers_gpu[3]);

    return true;
}


bool Lanenet::EngineInference(const std::vector<std::string> &image_list, void **buffers_gpu,
                     const std::vector<int64_t> &bufferSize, const std::vector<nvinfer1::Dims> &data_dims,cudaStream_t stream)
{

    auto output_buffer_cpu_0 = std::unique_ptr<float>{new float[bufferSize[1]]};
    auto output_buffer_cpu_1 = std::unique_ptr<int>{new int[bufferSize[2]]};
    auto output_buffer_cpu_2 = std::unique_ptr<float>{new float[bufferSize[3]]};

    int index = 0;
    std::string file_name_no_extension;
    std::string input_file_png_name;
    std::string input_file_ppm_name;
    util::PPM ppm;

    auto pImgPostProcessor=std::make_unique<ImgPostProcessor>();
    for (const std::string &image_name : image_list)
    {
        index++;
        std::cout << "Processing: " << index << std::endl;
        file_name_no_extension = util::get_file_name_no_extension(image_name);
        input_file_png_name = m_foldername+"/" + file_name_no_extension + ".png";
        input_file_ppm_name = m_foldername+"_ppm/"+ file_name_no_extension + ".ppm";
        pImageProcessor->png2ppm(input_file_png_name.c_str(), ppm,IMAGE_RESIZE_HEIGHT,IMAGE_RESIZE_WIDTH);

        auto input_image{util::RGBImageReader(input_file_ppm_name, data_dims[0], this->img_mean, this->img_std)};

        // input_image.read();
        input_image.read(ppm);
        // normalize with mean and std
        auto input_buffer = input_image.process();

        // Copy image data to input binding memory
        if (cudaMemcpyAsync(buffers_gpu[0], input_buffer.get(), bufferSize[0], cudaMemcpyHostToDevice, stream) != cudaSuccess)
        {
            gLogError << "ERROR: CUDA memory copy of input failed, size = " << bufferSize[0] << " bytes" << std::endl;
            return false;
        }

        // Run TensorRT inference
        void *bindings[] = {buffers_gpu[0], buffers_gpu[1], buffers_gpu[2], buffers_gpu[3]};
        bool status = context->enqueueV2(bindings, stream, nullptr);
        if (!status)
        {
            gLogError << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        }


        if (cudaMemcpyAsync(output_buffer_cpu_0.get(), buffers_gpu[1], bufferSize[1], cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            gLogError << "ERROR: CUDA memory copy of output failed, size = " << bufferSize[1] << " bytes" << std::endl;
            return false;
        }

        if (cudaMemcpyAsync(output_buffer_cpu_1.get(), buffers_gpu[2], bufferSize[2], cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            gLogError << "ERROR: CUDA memory copy of output failed, size = " << bufferSize[2] << " bytes" << std::endl;
            return false;
        }

        if (cudaMemcpyAsync(output_buffer_cpu_2.get(), buffers_gpu[3], bufferSize[3], cudaMemcpyDeviceToHost, stream) != cudaSuccess)
        {
            gLogError << "ERROR: CUDA memory copy of output failed, size = " << bufferSize[3] << " bytes" << std::endl;
            return false;
        }
        cudaStreamSynchronize(stream);

        // instance_pred=output_buffers[2].reshape((3,image_height, image_width))* 255
        // Plot the semantic segmentation predictions of 21 classes in a colormap image and write to file
        const int num_classes{21};
        const std::vector<int> palette{(0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 << 21) - 1};
        // std::string output_filename = "../lane_samples/output.ppm";
        std::string binary_file_path = m_foldername+"_result/output_" + file_name_no_extension + ".ppm";


        pImgPostProcessor->processLane(output_buffer_cpu_1.get(),data_dims[2],output_buffer_cpu_2.get(),data_dims[3]);
        
        
        
        util::PPM ppm_binary;
        pImgPostProcessor->calBinary(output_buffer_cpu_1.get(),data_dims[2],ppm_binary);// binary output
        pImgPostProcessor->write(binary_file_path,ppm_binary);

        std::string instance_file_path = m_foldername+"_result/output_instance_" + file_name_no_extension + ".ppm";
        util::PPM ppm_instance;
        pImgPostProcessor->calInstance(output_buffer_cpu_2.get(),data_dims[3],ppm_instance);// binary output
        pImgPostProcessor->write(instance_file_path,ppm_instance);



        // auto output_image{PostProcessor(output_file_path, data_dims[2], palette, num_classes)};
        // output_image.processLane(output_buffer_cpu_1.get());
        // output_image.write();

        // std::string instance_file_path = "../../lane_samples_result/output_instance_" + file_name_no_extension + ".ppm";
        // auto instance_image{PostProcessor(instance_file_path, data_dims[3], palette, num_classes)};
        // instance_image.calInstance(output_buffer_cpu_2.get());
        // instance_image.write();
    }

}

