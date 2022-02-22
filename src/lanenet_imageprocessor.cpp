#include "lanenet_imageprocessor.h"

PostProcessor::PostProcessor(const std::string &filename, const nvinfer1::Dims &dims, const std::vector<int> &palette, const int num_classes)
    : ImageBase(filename, dims), mNumClasses(num_classes), mPalette(palette)
{
    min_area_threshold = 100;
}


void PostProcessor::calBinary(const int *buffer)
{
    mPPM.magic = "P6";
    mPPM.w = mDims.d[3];
    mPPM.h = mDims.d[2];
    mPPM.max = 255;
    mPPM.buffer.resize(volume());

    for (int j = 0, HW = mPPM.h * mPPM.w; j < HW; ++j)
    {
        // auto clsid{static_cast<uint8_t>(buffer[j])};
        if (0 != buffer[j])
        {
            // std::cout<<buffer[j];
            mPPM.buffer.data()[j * 3] = 255;
            mPPM.buffer.data()[j * 3 + 1] = 255;
            mPPM.buffer.data()[j * 3 + 2] = 255;
        }
    }
}

void PostProcessor::calInstance(const float *buffer)
{
    mPPM.magic = "P6";
    mPPM.w = mDims.d[3];
    mPPM.h = mDims.d[2];
    mPPM.max = 255;
    mPPM.buffer.resize(volume());

    for (int j = 0, HW = mPPM.h * mPPM.w; j < HW; ++j)
    {
        // if (0 != buffer[j])
        // {
            // std::cout<<buffer[j];
            mPPM.buffer.data()[j * 3] = buffer[j]*255;
            mPPM.buffer.data()[j * 3 + 1] = buffer[j+HW]*255;
            mPPM.buffer.data()[j * 3 + 2] = buffer[j+2*HW]*255;

    }
}

void PostProcessor::processLane(const int *buffer)
{
    this->calBinary(buffer);
    // cv::Mat img;
    // cv::Mat morphological_ret = this->_morphological_process(img);
    // cv::Mat labels;
    // cv::Mat stats;
    // cv::Mat centroids;
}

cv::Mat PostProcessor::_morphological_process(const cv::Mat &image, int kernel_size)
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

void PostProcessor::_connect_components_analysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids)
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


void ImageProcessor::png2ppm(const char *input, const char *output)
{
    using namespace std;
    cv::Size sz;
    sz.height = 256;
    sz.width = 512;

    ofstream ppm(output, ios::app | ios::binary);
    ppm << "P6" << endl
        << sz.width << " " << sz.height << endl
        << 255 << endl;

    cv::Mat ori_img = cv::imread(input);
    cv::Mat resized_img;
    cv::resize(ori_img, resized_img, sz);

    uchar ptrBlue, ptrGreen, ptrRed;
    cv::MatIterator_<cv::Vec3b> it, end;
    for (it = resized_img.begin<cv::Vec3b>(), end = resized_img.end<cv::Vec3b>(); it != end; ++it)
    {
        ptrBlue = (*it)[0];
        ptrGreen = (*it)[1];
        ptrRed = (*it)[2];
        ppm << ptrRed << ptrGreen << ptrBlue;
    }
    ppm.close();
}


void ImageProcessor::png2ppm(const char *input, util::PPM &ppm,int height,int width)
{
    using namespace std;
    cv::Size sz;
    sz.height = height;
    sz.width = width;
    ppm.magic = "P6";
    ppm.h = sz.height;
    ppm.w = sz.width;
    ppm.c = 3;
    ppm.max = 255;
    cv::Mat ori_img = cv::imread(input);
    cv::Mat resized_img;
    cv::resize(ori_img, resized_img, sz);
    uchar ptrBlue, ptrGreen, ptrRed;
    cv::MatIterator_<cv::Vec3b> it, end;
    ppm.buffer.resize(ppm.c * ppm.h * ppm.w);

    int idx;
    for (it = resized_img.begin<cv::Vec3b>(), end = resized_img.end<cv::Vec3b>(), idx = 0; it != end; ++it, ++idx)
    {
        ptrBlue = (*it)[0];
        ptrGreen = (*it)[1];
        ptrRed = (*it)[2];
        ppm.buffer[3 * idx] = ptrRed;
        ppm.buffer[3 * idx + 1] = ptrGreen;
        ppm.buffer[3 * idx + 2] = ptrBlue;
    }
}