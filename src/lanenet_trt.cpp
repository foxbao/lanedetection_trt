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
using namespace std;
class point{
public:
	float x;
	float y;
	float z;
	int cluster=0;
	int pointType=1;//1 noise 2 border 3 core
	int pts=0;//points in MinPts 
	vector<int> corepts;
	int visited = 0;
	point (){}
	point (float a,float b,int c){
		x = a;
		y = b;
		cluster = c;
	}
};
float stringToFloat(string i){
	stringstream sf;
	float score=0;
	sf<<i;
	sf>>score;
	return score;
}
vector<point> openFile(const char* dataset){
	fstream file;
	file.open(dataset,ios::in);
	if(!file) 
    {
        cout <<"Open File Failed!" <<endl;
        vector<point> a;
        return a;
    } 
	vector<point> data;
	int i=1;
	while(!file.eof()){
		string temp;
		file>>temp;
		int split = temp.find(',',0);
		point p(stringToFloat(temp.substr(0,split)),stringToFloat(temp.substr(split+1,temp.length()-1)),i++);
		data.push_back(p);
	}
	file.close();
	cout<<"successful!"<<endl;
	return data;
}
float squareDistance(point a,point b){
	return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}
void DBSCAN(vector<point> dataset,float Eps,int MinPts){
	int len = dataset.size();
	//calculate pts
	cout<<"calculate pts"<<endl;
	for(int i=0;i<len;i++){
		for(int j=i+1;j<len;j++){
			int aaa=squareDistance(dataset[i],dataset[j]);
			if(squareDistance(dataset[i],dataset[j])<Eps)
			{
				dataset[i].pts++;
				dataset[j].pts++;
			}

		}
	}
	//core point 
	cout<<"core point "<<endl;
	vector<point> corePoints;
	for(int i=0;i<len;i++){
		if(dataset[i].pts>=MinPts) {
			dataset[i].pointType = 3;
			corePoints.push_back(dataset[i]);
		}
	}
	cout<<"joint core point"<<endl;
	//joint core point
	for(int i=0;i<corePoints.size();i++){
		for(int j=i+1;j<corePoints.size();j++){
			if(squareDistance(corePoints[i],corePoints[j])<Eps){
				corePoints[i].corepts.push_back(j);
				corePoints[j].corepts.push_back(i);
			}
		}
	}
	for(int i=0;i<corePoints.size();i++){
		stack<point*> ps;
		if(corePoints[i].visited == 1) continue;
		ps.push(&corePoints[i]);
		point *v;
		while(!ps.empty()){
			v = ps.top();
			v->visited = 1;
			ps.pop();
			for(int j=0;j<v->corepts.size();j++){
				if(corePoints[v->corepts[j]].visited==1) continue;
				corePoints[v->corepts[j]].cluster = corePoints[i].cluster;
				corePoints[v->corepts[j]].visited = 1;
				ps.push(&corePoints[v->corepts[j]]);				
			}
		}		
	}
	cout<<"border point,joint border point to core point"<<endl;
	//border point,joint border point to core point
	for(int i=0;i<len;i++){
		if(dataset[i].pointType==3) continue;
		for(int j=0;j<corePoints.size();j++){
			if(squareDistance(dataset[i],corePoints[j])<Eps) {
				dataset[i].pointType = 2;
				dataset[i].cluster = corePoints[j].cluster;
				break;
			}
		}
	}
	cout<<"output"<<endl;
	//output
	fstream clustering;
	clustering.open("clustering.txt",ios::out);
	for(int i=0;i<len;i++){
		if(dataset[i].pointType == 2)
			clustering<<dataset[i].x<<","<<dataset[i].y<<","<<dataset[i].cluster<<"\n";
	}
	for(int i=0;i<corePoints.size();i++){
			clustering<<corePoints[i].x<<","<<corePoints[i].y<<","<<corePoints[i].cluster<<"\n";
	}
	clustering.close();
}


void plotPoints(vector<point> dataset)
{
	cv::Mat img(500,500,CV_8UC3, cv::Scalar(255,255,255));

	//画空心点
	for (auto pt:dataset)
	{
		cv::Point p(pt.x, pt.y);//初始化点坐标为(20,20)
		circle(img, p, 5, cv::Scalar(0, 255, 0),-1); //第三个参数表示点的半径，第四个参数选择颜色。这样子我们就画出了
	}
	cv::imwrite("imgtest.jpg",img);
}


int main(int argc, char **argv)
{
	// vector<point> dataset = openFile("dataset3.txt");
	// plotPoints(dataset);
	// DBSCAN(dataset,100,2);
	
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