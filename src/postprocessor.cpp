#include "postprocessor.h"

void ImgPostProcessor::calBinary(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    // util::PPM ppm;
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (int j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        // auto clsid{static_cast<uint8_t>(buffer[j])};
        if (0 != buffer[j])
        {
            // std::cout<<buffer[j];
            dst_ppm.buffer.data()[j * 3] = 255;
            dst_ppm.buffer.data()[j * 3 + 1] = 255;
            dst_ppm.buffer.data()[j * 3 + 2] = 255;
        }
    }
}

void ImgPostProcessor::calInstance(const float *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (int j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        // if (0 != buffer[j])
        // {
        // std::cout<<buffer[j];
        dst_ppm.buffer.data()[j * 3] = buffer[j] * 255;
        dst_ppm.buffer.data()[j * 3 + 1] = buffer[j + HW] * 255;
        dst_ppm.buffer.data()[j * 3 + 2] = buffer[j + 2 * HW] * 255;
    }
}


cv::Mat ImgPostProcessor::_morphological_process(const cv::Mat &image, int kernel_size)
{

    if (image.channels() == 3)
    {
        throw "Binary segmentation result image should be a single channel image";
    }

    // if image.dtype is not np.uint8:
    //     image = np.array(image, np.uint8)

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat closing;
    cv::morphologyEx(image, closing, cv::MORPH_CLOSE, kernel);
    return closing;
}

void ImgPostProcessor::_connect_components_analysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids)
{
    //         if len(image.shape) == 3:
    //     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    // else:
    //     gray_image = image
    cv::Mat gray_image;
    if (3 == image.channels())
    {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray_image = image;
    }
    // cv::Mat labels;
    // cv::Mat stats;
    // cv::Mat centroids;
    cv::connectedComponentsWithStats(gray_image, labels, stats, centroids, 8);
    // return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)
}

void ImgPostProcessor::processLane(const int *buffer_binary,const nvinfer1::Dims& dim_binary,const float *buffer_instance,const nvinfer1::Dims& dim_instance)
{
    int a=1;
}

int ImgPostProcessor::volume(util::PPM &ppm)
{
    return ppm.w * ppm.h * 3;
}

void ImgPostProcessor::write(const std::string &filename, util::PPM ppm)
{
    std::ofstream outfile(filename, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << filename << std::endl;
    }
    outfile << ppm.magic << " " << ppm.w << " " << ppm.h << " " << ppm.max << std::endl;
    outfile.write(reinterpret_cast<char *>(ppm.buffer.data()), volume(ppm));
    outfile.close();
}