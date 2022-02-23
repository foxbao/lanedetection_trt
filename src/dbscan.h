#include "util.h"
class point
{
public:
	float x;
	float y;
	float z;
	int cluster = 0;
	int pointType = 1; //1 noise 2 border 3 core
	int pts = 0;	   //points in MinPts
	std::vector<int> corepts;
	int visited = 0;
	point() {}
	point(float in_x, float in_y, float in_z, int in_cluster);
	// {
	// 	x = in_x;
	// 	y = in_y;
	// 	z = in_z;
	// 	cluster = in_cluster;
	// }
};

class DBSCAN
{
public:
	DBSCAN();
	void cluster(const std::vector<std::vector<float>> &lane_embedding_feats);
	

private:
	void calculatePts(std::vector<point> &dataset);
	void calculateCorePts(std::vector<point> &dataset, std::vector<point> &corePoints);
	void jointCorePts(std::vector<point> &corePoints);
	void border(std::vector<point> &dataset, std::vector<point> &corePoints);
	void output(std::vector<point> &dataset, std::vector<point> &corePoints);
	void fit_transform(const std::vector<std::vector<float>> &lane_embedding_feats, std::vector<std::vector<double>> &features);
	float squareDistance(const point &a, const point &b);
	void cluster(const std::vector<std::vector<double>> &features);
	double eps_;
	double min_samples_;
};