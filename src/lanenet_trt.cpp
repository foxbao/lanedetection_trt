#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"
#include "lanenet.h"



int main(int argc, char **argv)
{

    // jpg2ppm("./lane_samples/0073.png", "./lane_samples/0073.ppm");
    // jpg2ppm("./lane_samples/0553.png", "./lane_samples/0553.ppm");
    // std::vector<uint8_t> buffer;
    if (argc < 3)
    {
        std::cout << "Please design config file and image folder!" << std::endl;
        return -1;
    }
    std::string config_file = argv[1];
    std::string folder_name = argv[2];
    Lanenet lanenet(config_file);
    lanenet.LoadEngine();
    lanenet.InferenceFolder(folder_name);
    std::cout << "finished!" << std::endl;
}