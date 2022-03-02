#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/QR>

class PolyFit
{
public:
	void polyfit(const std::vector<double> &t,
				 const std::vector<double> &v,
				 std::vector<double> &coeff,
				 int order
	);
};
// void baobao()
// {
// 	int a=1;
// }
