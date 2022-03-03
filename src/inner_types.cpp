#include "../includes/inner_types.h"


namespace inner_type
{

    Lane::Lane()
    {
        
    }
    std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> Lane::GetPairRowColVectorPtr() const
    {
        std::vector<int> vector_row(pts.size());
        std::vector<int> vector_col(pts.size());
        int idx_pt=0;
        for (int idx_pt=0;idx_pt<pts.size();idx_pt++)
        {
            vector_row[idx_pt]=pts[idx_pt].row_;
            vector_col[idx_pt]=pts[idx_pt].col_;
            
        }
        std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> sp_pair;
        sp_pair=std::make_shared<std::pair<std::vector<int>,std::vector<int>>>(vector_row,vector_col);
        return sp_pair;
    }
}