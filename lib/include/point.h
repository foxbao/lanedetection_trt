#pragma once
#include <vector>


class point
{
public:
	float x;
	float y;
	float z;
	int cluster = 0;
	int pointType = 1; //1 noise 2 border 3 core
	int pts = 0;	   //points in MinPts
	std::vector<int> neighborCoreIdx; // the core points in MinPts
	std::vector<int> neibougPts;
	int visited = 0;
	point() {}
	point(float in_x, float in_y, float in_z, int in_cluster);
	int idx;
	// {
	// 	x = in_x;
	// 	y = in_y;
	// 	z = in_z;
	// 	cluster = in_cluster;
	// }
};