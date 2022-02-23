#include "util.h"
class point{
public:
	float x;
	float y;
	float z;
	int cluster=0;
	int pointType=1;//1 noise 2 border 3 core
	int pts=0;//points in MinPts 
	std::vector<int> corepts;
	int visited = 0;
	point (){}
	point (float a,float b,int c){
		x = a;
		y = b;
		cluster = c;
	}
};

class DBSCAN
{
    public:
        DBSCAN();
    private:
        float squareDistance(point a,point b);
};