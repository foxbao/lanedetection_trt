#pragma once
#include <iostream>
#include <vector>
#include<memory>

namespace inner_type
{

struct LanePoint
{
    LanePoint();
    LanePoint(int in_row,int in_col){
        row=in_row;
        col=in_col;
    };
    int row;
    int col;
};

class Lane
{
    public:
        Lane();
        std::vector<inner_type::LanePoint> pts;
        std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> GetPairXYVectorPtr() const;
    private:
        
};

}