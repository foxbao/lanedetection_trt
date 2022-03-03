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
        row_=in_row;
        col_=in_col;
    };
    int row_;
    int col_;
};

class Lane
{
    public:
        Lane();
        std::vector<inner_type::LanePoint> pts;
        std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> GetPairRowColVectorPtr() const;
    private:
        
};

}