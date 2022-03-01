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



#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <stack>
using namespace std;


#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <stack>

// #include "kdtree.h"
// #include "point.h"
#include "dbscan.hpp"
using namespace std;

#define EXAMPLAR_NUM 100
#define EXAMPLAR_DIM 3
// void test_dbscan()
// {

// 	std::vector<std::vector<double>> features;
//     double eps=100;
//     int minPts=2;
// 	std::shared_ptr<DBSCAN> sp_dbscan =std::make_shared<DBSCAN>(eps,minPts);
// 	vector<point> dataset = sp_dbscan->openFile("dataset3.txt");

//     for (const auto& point:dataset)
//     {
//         std::vector<double> feature;
//         feature.push_back(point.x);
//         feature.push_back(point.y);
//         feature.push_back(point.z);
//         features.push_back(feature);
//     }
//     sp_dbscan->cluster(features);
// 	// sp_dbscan->test_cluster(dataset);
// }

int main(int argc, char **argv)
{
	// test_dbscan();
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