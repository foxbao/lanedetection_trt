#include "../includes/postprocessor.h"
#include "../includes/inner_types.h"
ImgPostProcessor::ImgPostProcessor()
{
    min_area_threshold_ = 100;
    sp_laneCluster_ = std::make_shared<LaneCluster>();
    sp_polyfit_=std::make_shared<PolyFit>();
}
void ImgPostProcessor::GenerateBinarySegmentThree(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    // util::PPM ppm;
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.c = 3;
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (int j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        // auto clsid{static_cast<uint8_t>(buffer[j])};
        if (0 != buffer[j])
        {
            // std::cout<<buffer[j];
            dst_ppm.buffer.data()[j * 3] = 255;
            dst_ppm.buffer.data()[j * 3 + 1] = 255;
            dst_ppm.buffer.data()[j * 3 + 2] = 255;
        }
    }
}

void ImgPostProcessor::GenerateBinarySegment(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    // util::PPM ppm;
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.c = 1;
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (size_t j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        if (0 != buffer[j])
        {
            dst_ppm.buffer.data()[j] = 255;
        }
    }
}

void ImgPostProcessor::CalInstance(const float *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.c = 3;
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (int j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        dst_ppm.buffer.data()[j * 3] = buffer[j] * 255;
        dst_ppm.buffer.data()[j * 3 + 1] = buffer[j + HW] * 255;
        dst_ppm.buffer.data()[j * 3 + 2] = buffer[j + 2 * HW] * 255;
    }
}

cv::Mat ImgPostProcessor::MorphologicalProcess(const util::PPM &binary_ppm, int kernel_size)
{
    cv::Mat binary_img(binary_ppm.h, binary_ppm.w, CV_8UC1, cv::Scalar(0));

    cv::MatIterator_<uchar> it, end;

    int idx;

    for (it = binary_img.begin<uchar>(), end = binary_img.end<uchar>(), idx = 0; it != end; ++it, ++idx)
    {
        (*it) = binary_ppm.buffer.data()[idx];
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat closing;
    cv::morphologyEx(binary_img, closing, cv::MORPH_CLOSE, kernel);
    return closing;
}

cv::Mat ImgPostProcessor::MorphologicalProcess(const cv::Mat &image, int kernel_size)
{

    if (image.channels() == 3)
    {
        throw "Binary segmentation result image should be a single channel image";
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat closing;
    cv::morphologyEx(image, closing, cv::MORPH_CLOSE, kernel);
    return closing;
}

void ImgPostProcessor::GenerateMatInstance(cv::Mat &mat_instance_seg_result,const float *buffer_instance, const nvinfer1::Dims &dim_instance)
{
    int H=dim_instance.d[2];//height
    int W=dim_instance.d[3];//width
    mat_instance_seg_result=cv::Mat(H,W,CV_8UC3);
    int idx;
    cv::MatIterator_<cv::Vec3b> it, end;

    // buffer_instance format is CHW, whicle cv::Mat is HWC
    for (it = mat_instance_seg_result.begin<cv::Vec3b>(), end = mat_instance_seg_result.end<cv::Vec3b>(), idx = 0; it != end; ++it, ++idx)
    {

        (*it)[0]=buffer_instance[idx]*255;
        (*it)[1]=buffer_instance[idx+H*W]*255;
        (*it)[2]=buffer_instance[idx+2*H*W]*255;
    }
}

void ImgPostProcessor::ConnectComponentsAnalysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids)
{
    cv::Mat gray_image;
    if (3 == image.channels())
    {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray_image = image;
    }
    cv::connectedComponentsWithStats(gray_image, labels, stats, centroids, 8);
}



bool ImgPostProcessor::IsInVector(const int &value, std::vector<int> &vec)
{
    std::vector<int>::iterator ret;
    ret = std::find(vec.begin(), vec.end(), value);
    if (ret != vec.end())
    {
        return true;
    }
    else
    {
        return false;
    }
}


void ImgPostProcessor::RemoveSmallConnectComponents(const cv::Mat &labels, const cv::Mat &stats,cv::Mat &morphological_ret)
{
    std::vector<int> idx_to_remove;
    for (size_t nrow = 0; nrow < stats.rows; nrow++)
    {
        const int *stats_ptr = stats.ptr<int>(nrow);
        if (int(stats_ptr[4]) < min_area_threshold_)
        {
            idx_to_remove.push_back(nrow);
        }
    }

    for (size_t nrow = 0; nrow < labels.rows; nrow++)
    {
        const int *labels_ptr = labels.ptr<int>(nrow);
        uchar *morphological_ret_ptr = morphological_ret.ptr<uchar>(nrow);
        for (size_t ncol = 0; ncol < labels.cols * labels.channels(); ncol++)
        {
            if (int(labels_ptr[ncol]) != 0)
            {
                if (IsInVector(int(labels_ptr[ncol]), idx_to_remove))
                {
                    morphological_ret_ptr[ncol]=0;
                }
            }
        }
    }
}

void ImgPostProcessor::ProcessLane(const int *buffer_binary, const nvinfer1::Dims &dim_binary, const float *buffer_instance, const nvinfer1::Dims &dim_instance,cv::Mat &mask,std::vector<inner_type::Lane> &lanes_coords)
{
    util::PPM ppm_binary;
    this->GenerateBinarySegment(buffer_binary, dim_binary, ppm_binary); // binary output
    // apply image morphology operation to fill in the hold and reduce the small area
    cv::Mat morphological_ret = this->MorphologicalProcess(ppm_binary);
    cv::Mat mat_instance_seg_result;
    this->GenerateMatInstance(mat_instance_seg_result,buffer_instance,dim_instance);
    // apply connect component to connect the areas
    cv::Mat labels, stats, centroids;
    this->ConnectComponentsAnalysis(morphological_ret, labels, stats, centroids);
    // remove the very small connected components
    this->RemoveSmallConnectComponents(labels,stats,morphological_ret);
    this->sp_laneCluster_->apply_lane_feats_cluster(morphological_ret,mat_instance_seg_result,lanes_coords,mask);


    std::vector<std::vector<double>> fit_params;
    this->LineFit(lanes_coords,fit_params);
}

int ImgPostProcessor::volume(util::PPM &ppm)
{
    return ppm.w * ppm.h * ppm.c;
}

void ImgPostProcessor::WriteImg(const std::string &filename, util::PPM ppm)
{
    std::ofstream outfile(filename, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << filename << std::endl;
    }
    outfile << ppm.magic << " " << ppm.w << " " << ppm.h << " " << ppm.max << std::endl;
    outfile.write(reinterpret_cast<char *>(ppm.buffer.data()), volume(ppm));
    outfile.close();


    
}

void ImgPostProcessor::LineFit(const std::vector<inner_type::Lane>& lanes_coords,std::vector<std::vector<double>>& fit_params)
{
    for(const auto& lane:lanes_coords)
    {
        std::vector<double> fit_param;
        std::shared_ptr<std::pair<std::vector<int>,std::vector<int>>> sp_pair=lane.GetPairXYVectorPtr();
        std::vector<double> doubleVecX(sp_pair->first.begin(), sp_pair->first.end());
        std::vector<double> doubleVecY(sp_pair->second.begin(), sp_pair->second.end());
        sp_polyfit_->polyfit(doubleVecX,doubleVecY,fit_param,2);
        fit_params.push_back(fit_param);
    }
}