#pragma once
#include "util.h"
#include <opencv2/opencv.hpp>
class ImageProcessor
{
    public:
        void png2ppm(const char *input, util::PPM &ppm,int height,int width,cv::Mat& resized_img);
        void png2ppm(const char *input, const char *output);
        void PPM2Mat(const util::PPM &ppm,cv::Mat& img);
        void Overlap(const int *buffer, int H,int W,cv::Mat &resized_img);
    private:
};