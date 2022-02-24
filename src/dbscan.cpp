#include "dbscan.h"
#include <numeric>
#include <stack>

point::point(float in_x, float in_y, float in_z, int in_cluster)
{
	x = in_x;
	y = in_y;
	z = in_z;
	cluster = in_cluster;
}

DBSCAN::DBSCAN()
{
	eps_ = 0.35;
	min_samples_ = 100;
}

void DBSCAN::clear_variables()
{
	labels_.clear();
	core_sample_indices_.clear();
	unique_labels_.clear();
	num_clusters_=0;
}

void DBSCAN::fit_transform(const std::vector<std::vector<float>> &lane_embedding_feats, std::vector<std::vector<double>> &features)
{
	int channel_num = 3;
	std::vector<double> channels[channel_num];
	std::vector<double> features_channels[channel_num]; // vector<int>* p;
	for (auto feat : lane_embedding_feats)
	{
		for (int i = 0; i < feat.size(); i++)
		{
			channels[i].push_back(feat[i]);
		}
	}
	int szChannel = channels->size();
	for (int i = 0; i < channel_num; i++)
	{
		// calculate mean and stdev
		double sum = std::accumulate(std::begin(channels[i]), std::end(channels[i]), 0.0);
		double mean = sum / channels[i].size(); //mean

		double accum = 0.0;
		std::for_each(std::begin(channels[i]), std::end(channels[i]), [&](const double d)
					  { accum += (d - mean) * (d - mean); });
		double stdev = sqrt(accum / (channels[i].size() - 1)); //dev
		// use mean and stdev to transform the data to [0,1]
		for (auto channel_value : channels[i])
		{
			features_channels[i].push_back((channel_value - mean) / stdev);
		}
	}
	int szemb = lane_embedding_feats.size();
	for (int i = 0; i < lane_embedding_feats.size(); i++)
	{
		std::vector<double> feature;
		feature.push_back(features_channels[0][i]);
		feature.push_back(features_channels[1][i]);
		feature.push_back(features_channels[2][i]);
		features.push_back(feature);
	}
}

void DBSCAN::cluster(const std::vector<std::vector<float>> &lane_embedding_feats)
{
	// normalize the data
	clear_variables();
	std::vector<std::vector<double>> features;
	fit_transform(lane_embedding_feats, features);
	// cluster the data
	cluster(features);
}

void DBSCAN::calculatePts(std::vector<point> &dataset)
{
	int len = dataset.size();
	//calculate the neighbours of pts which are inside eps
	std::cout << "calculate pts" << std::endl;
	for (int i = 0; i < len; i++)
	{
		for (int j = i + 1; j < len; j++)
		{
			if (squareDistance(dataset[i], dataset[j]) < eps_)
			{
				dataset[i].pts++;
				dataset[j].pts++;
			}
		}
	}
}

void DBSCAN::calculateCorePts(std::vector<point> &dataset, std::vector<point> &corePoints)
{
	std::cout << "core point " << std::endl;
	// calculate the core points who has more than the threshold neighbours
	for (int i = 0; i < dataset.size(); i++)
	{
		if (dataset[i].pts >= min_samples_)
		{
			dataset[i].pointType = 3;
			corePoints.push_back(dataset[i]);
		}
	}
}

void DBSCAN::jointCorePts(std::vector<point> &corePoints)
{
	std::cout << "joint core point" << std::endl;
	//joint core point
	for (int i = 0; i < corePoints.size(); i++)
	{
		for (int j = i + 1; j < corePoints.size(); j++)
		{
			if (squareDistance(corePoints[i], corePoints[j]) < eps_)
			{
				corePoints[i].corepts.push_back(j);
				corePoints[j].corepts.push_back(i);
			}
		}
	}

	for (int i = 0; i < corePoints.size(); i++)
	{
		std::stack<point *> ps;
		if (corePoints[i].visited == 1)
			continue;
		ps.push(&corePoints[i]);
		point *v;
		while (!ps.empty())
		{
			v = ps.top();
			v->visited = 1;
			ps.pop();
			for (int j = 0; j < v->corepts.size(); j++)
			{
				if (corePoints[v->corepts[j]].visited == 1)
					continue;
				corePoints[v->corepts[j]].cluster = corePoints[i].cluster;
				corePoints[v->corepts[j]].visited = 1;
				ps.push(&corePoints[v->corepts[j]]);
			}
		}
	}
}

void DBSCAN::border(std::vector<point> &dataset, std::vector<point> &corePoints)
{
	std::cout << "border point,joint border point to core point" << std::endl;
	//border point,joint border point to core point
	int len = dataset.size();
	for (int i = 0; i < len; i++)
	{
		if (dataset[i].pointType == 3)
			continue;
		for (int j = 0; j < corePoints.size(); j++)
		{
			if (squareDistance(dataset[i], corePoints[j]) < eps_)
			{
				dataset[i].pointType = 2;
				dataset[i].cluster = corePoints[j].cluster;
				break;
			}
		}
	}
}

void DBSCAN::output(std::vector<point> &dataset, std::vector<point> &corePoints)
{
	std::cout << "output" << std::endl;
	//output
	std::fstream clustering;
	int len = dataset.size();
	clustering.open("clustering.txt", std::ios::out);
	for (int i = 0; i < len; i++)
	{
		if (dataset[i].pointType == 2)
			clustering << dataset[i].x << "," << dataset[i].y << "," << corePoints[i].z << "," << dataset[i].cluster << "\n";
	}

	// the corePointss
	for (int i = 0; i < corePoints.size(); i++)
	{
		clustering << corePoints[i].x << "," << corePoints[i].y << "," << corePoints[i].z << "," << corePoints[i].cluster << "\n";
	}
	clustering.close();
}

void DBSCAN::calLabels(const std::vector<point> &corePoints)
{
	for (auto cp : corePoints)
	{
		labels_.push_back(cp.cluster);
	}
	

	unique_labels_ = labels_;
	std::sort(unique_labels_.begin(), unique_labels_.end());
	auto it = std::unique(unique_labels_.begin(), unique_labels_.end());
	unique_labels_.erase(it, unique_labels_.end());
	num_clusters_ = unique_labels_.size();
}

void DBSCAN::cluster(const std::vector<std::vector<double>> &features)
{
	std::vector<point> dataset;
	for (int i = 0; i < features.size(); i++)
	{
		point p(features[i][0], features[i][1], features[i][2], i);
		dataset.push_back(p);
	}

	int len = dataset.size();
	std::vector<point> corePoints;
	calculatePts(dataset);
	calculateCorePts(dataset, corePoints);
	jointCorePts(corePoints);
	border(dataset, corePoints);
	output(dataset, corePoints);
	calLabels(corePoints);
	// for (auto cp : corePoints)
	// {
	// 	labels_.push_back(cp.cluster);
	// }
	// std::sort(labels_.begin(), labels_.end());
	// unique_labels_ = labels_;
	// auto it = std::unique(unique_labels_.begin(), unique_labels_.end());
	// unique_labels_.erase(it, unique_labels_.end());
	// int num_clusters = unique_labels_.size();
}

float DBSCAN::squareDistance(const point &a, const point &b)
{
	return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}