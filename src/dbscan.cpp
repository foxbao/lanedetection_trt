#include "dbscan.h"
#include <numeric>
DBSCAN::DBSCAN()
{
    eps = 0.35;
    min_samples = 100;
}

void DBSCAN::cluter(const std::vector<std::vector<float>> &lane_embedding_feats)
{
    // normalize the data

    // calculate mean and stdev

    // use mean and stdev to transform the data to [0,1]
    int channel_num=3;
    std::vector<double> channels[channel_num];
    std::vector<double> features[channel_num];
    // vector<int>* p;
    for (auto feat : lane_embedding_feats)
    {
        for (int i = 0; i < feat.size(); i++)
        {
            channels[i].push_back(feat[i]);
        }
    }
    int szChannel=channels->size();
    for (int i=0;i<channel_num;i++)
    {
        double sum = std::accumulate(std::begin(channels[i]), std::end(channels[i]), 0.0);
        double mean = sum / channels[i].size(); //均值

        double accum = 0.0;
        std::for_each(std::begin(channels[i]), std::end(channels[i]), [&](const double d)
                      { accum += (d - mean) * (d - mean); });
        double stdev = sqrt(accum / (channels[i].size() - 1)); //方差
        for (auto channel_value:channels[i])
        {
            features[i].push_back((channel_value-mean)/stdev);
        }
        int aaa=1;
    }

    std::vector<double> feature;

    int aaa = 1;
    int bbb = 1;
    // calculate the
}

float DBSCAN::squareDistance(const point &a, const point &b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}