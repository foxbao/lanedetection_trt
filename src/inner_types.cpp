#include "../includes/inner_types.h"


namespace inner_type
{

    Lane::Lane()
    {
        
    }
    std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> Lane::GetPairXYVectorPtr() const
    {
        std::vector<int> vector_X(pts.size());
        std::vector<int> vector_Y(pts.size());
        for (auto pt:pts)
        {
            vector_X.push_back(pt.row);
            vector_Y.push_back(pt.col);
        }
        std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> sp_pair;
        sp_pair=std::make_shared<std::pair<std::vector<int>,std::vector<int>>>(vector_X,vector_Y);
        return sp_pair;
    }
}