#include "../includes/imageprocessor.h"

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

void ImageProcessor::png2ppm(const char *input,
                             util::PPM &ppm,
                             int height,
                             int width,
                             cv::Mat &resized_img)
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
    // cv::Mat resized_img;
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

void ImageProcessor::PPM2Mat(const util::PPM &ppm, cv::Mat &img)
{
    int C = ppm.c;
    int H = ppm.h;
    int W = ppm.w;
    int bbb = 2;
    img = cv::Mat(H, W, CV_8UC3);
    cv::MatIterator_<cv::Vec3b> it = img.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> end = img.end<cv::Vec3b>();

    uchar ptrBlue, ptrGreen, ptrRed;
    for (int idx = 0, HW = H * W; idx < HW, it != end; ++idx, ++it)
    {
        ptrRed = ppm.buffer[3 * idx];       //red
        ptrGreen = ppm.buffer[3 * idx + 1]; //green
        ptrBlue = ppm.buffer[3 * idx + 2];  //blue
        (*it)[0] = ptrBlue;
        (*it)[1] = ptrGreen;
        (*it)[2] = ptrRed;
    }
}

void ImageProcessor::Overlap(const int *buffer, int H, int W, cv::Mat &resized_img)
{
    if (resized_img.cols != W)
    {
        return;
    }

    if (resized_img.rows != H)
    {
        return;
    }

    cv::MatIterator_<cv::Vec3b> it, end;
    it = resized_img.begin<cv::Vec3b>();
    end = resized_img.end<cv::Vec3b>();
    for (int j = 0; j < H * W, it != end; ++j, ++it)
    {
        // std::cout<<buffer[j]<<std::endl;
        if (0 != buffer[j])
        {
            (*it)[0] = 0;   //Blue
            (*it)[1] = 255; //Green
            (*it)[2] = 0;   //Red
        }
    }
}