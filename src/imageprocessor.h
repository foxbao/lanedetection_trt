#pragma once
#include "util.h"

class ImageProcessor
{
    public:
        void png2ppm(const char *input, util::PPM &ppm,int height,int width);
        void png2ppm(const char *input, const char *output);
    private:
};