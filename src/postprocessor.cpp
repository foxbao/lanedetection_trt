#include "postprocessor.h"

ImgPostProcessor::ImgPostProcessor()
{
    min_area_threshold = 100;
    sp_laneCluster = std::make_shared<LaneCluster>();
    // sp_dbscan= std::make_shared<DBSCAN>();
    sp_dbscan=std::make_shared<DBSCAN>();
}
void ImgPostProcessor::generateBinarySegmentThree(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
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

void ImgPostProcessor::generateBinarySegment(const int *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    // util::PPM ppm;
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.c = 1;
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (int j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        if (0 != buffer[j])
        {
            dst_ppm.buffer.data()[j] = 255;
        }
    }
}

void ImgPostProcessor::calInstance(const float *buffer, const nvinfer1::Dims &dim, util::PPM &dst_ppm)
{
    dst_ppm.magic = "P6";
    dst_ppm.w = dim.d[3];
    dst_ppm.h = dim.d[2];
    dst_ppm.c = 3;
    dst_ppm.max = 255;
    dst_ppm.buffer.resize(volume(dst_ppm));

    for (int j = 0, HW = dst_ppm.h * dst_ppm.w; j < HW; ++j)
    {
        // if (0 != buffer[j])
        // {
        // std::cout<<buffer[j];
        dst_ppm.buffer.data()[j * 3] = buffer[j] * 255;
        dst_ppm.buffer.data()[j * 3 + 1] = buffer[j + HW] * 255;
        dst_ppm.buffer.data()[j * 3 + 2] = buffer[j + 2 * HW] * 255;
    }
}

cv::Mat ImgPostProcessor::_morphological_process(const util::PPM &binary_ppm, int kernel_size)
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

cv::Mat ImgPostProcessor::_morphological_process(const cv::Mat &image, int kernel_size)
{

    if (image.channels() == 3)
    {
        throw "Binary segmentation result image should be a single channel image";
    }

    // if image.dtype is not np.uint8:
    //     image = np.array(image, np.uint8)

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat closing;
    cv::morphologyEx(image, closing, cv::MORPH_CLOSE, kernel);
    return closing;
}

void ImgPostProcessor::_generateMatInstance(cv::Mat &mat_instance_seg_result,const float *buffer_instance, const nvinfer1::Dims &dim_instance)
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

void ImgPostProcessor::_connect_components_analysis(const cv::Mat &image, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids)
{
    //         if len(image.shape) == 3:
    //     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    // else:
    //     gray_image = image
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
    // return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)
}

// template <class T>
// bool findValue(const cv::Mat &mat, T value) {
//     for(int i = 0;i < mat.rows;i++) {
//         const T* row = mat.ptr<T>(i);
//         if(std::find(row, row + mat.cols, value) != row + mat.cols)
//             return true;
//     }
//     return false;
// }

bool isInVector(const int &value, std::vector<int> &vec)
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

void ImgPostProcessor::processLane(const int *buffer_binary, const nvinfer1::Dims &dim_binary, const float *buffer_instance, const nvinfer1::Dims &dim_instance)
{
    util::PPM ppm_binary;
    this->generateBinarySegment(buffer_binary, dim_binary, ppm_binary); // binary output
    // apply image morphology operation to fill in the hold and reduce the small area
    cv::Mat morphological_ret = this->_morphological_process(ppm_binary);
    cv::Mat mat_instance_seg_result;
    this->_generateMatInstance(mat_instance_seg_result,buffer_instance,dim_instance);


    // cv::imwrite("generatedMatInstance.jpg",mat_instance_seg_result);
    // apply connect component to connect the areas
    cv::Mat labels, stats, centroids;
    this->_connect_components_analysis(morphological_ret, labels, stats, centroids);

    // remove the very small connected components
    std::vector<int> idx_to_remove;
    for (size_t nrow = 0; nrow < stats.rows; nrow++)
    {
        int *stats_ptr = stats.ptr<int>(nrow);
        if (int(stats_ptr[4]) < min_area_threshold)
        {
            idx_to_remove.push_back(nrow);
        }
    }

    for (size_t nrow = 0; nrow < labels.rows; nrow++)
    {
        int *labels_ptr = labels.ptr<int>(nrow);
        uchar *morphological_ret_ptr = morphological_ret.ptr<uchar>(nrow);
        for (size_t ncol = 0; ncol < labels.cols * labels.channels(); ncol++)
        {
            if (int(labels_ptr[ncol]) != 0)
            {
                if (isInVector(int(labels_ptr[ncol]), idx_to_remove))
                {
                    morphological_ret_ptr[ncol]=0;
                }
            }
        }
    }

    this->sp_laneCluster->apply_lane_feats_cluster(morphological_ret,mat_instance_seg_result);



    // cv::imwrite("morphological_ret_processed.jpg", morphological_ret);

    // auto maxPosition=std::max_element(label_values.begin(),label_values.end());
    // std::cout <<"max position"<<*maxPosition<<std::endl;
    // int aaa = 1;

    // std::cout<<img_pseudo<<std::endl;
    // std::cout<<int(img_pseudo.at<uchar>(1 , 1))<<std::endl;
    // int label_channels = labels.channels();
    // std::cout<<labels<<std::endl;
    // int img_pseudo_channels = img_pseudo.channels();
    // std::cout<<int(labels.at<uchar>(161 , 181))<<std::endl;
    // std::cout<<int(labels.at<uchar>(181 , 161))<<std::endl;
    // for (int h = 0; h < labels.rows; ++h)
    // {
    //     for (int w = 0; w < labels.cols/2; ++w)
    //     {
    //         uchar *ptr = img_pseudo.ptr<uchar>(h, w);
    //         *ptr = 254;
    //         // if(int(labels.at<uchar>(h , w))!=0)
    //         // {
    //         //     uchar* ptr=binary_img.ptr<uchar>(h,w);
    //         //     *ptr=254;
    //         // }
    //     }
    // }
    // cv::imwrite("img_pseudo.jpg", img_pseudo);
    // int dadsf = 2;

    // for (size_t nrow = 0; nrow < labels.rows; nrow++)
    // {
    //     uchar *data = labels.ptr<uchar>(nrow);
    //     for(size_t ncol = 0; ncol < labels.cols * labels.channels(); ncol++)
    //     {
    //         if (0!=data[ncol])
    //         {   std::cout<< nrow <<" "<<ncol <<" haha ";
    //             std::cout << data[ncol] <<std::endl;
    //         }

    //     }

    // }

    // for (size_t nrow = 0; nrow < stats.rows; nrow++)
    // {
    //     uchar *data = stats.ptr<uchar>(nrow);
    //     for(size_t ncol = 0; ncol < stats.cols * stats.channels(); ncol++)
    //     {
    //         std::cout << int( data[ncol] ) <<" ";

    //     }
    //     std::cout <<std::endl;

    //     if(int( data[4] )<min_area_threshold)
    //     {

    //         for(int h = 0 ; h < labels.rows ; ++ h)
    //         {
    //             for(int w = 0 ; w < labels.cols ; ++ w)
    //             {
    //                 if (labels.at<uchar>(h , w) == nrow)
    //                 {
    //                     int aaa=1;
    //                 }
    //             }
    //         }
    //     }
    // }

    // for index, stat in enumerate(stats):
    //     if stat[4] <= min_area_threshold:
    //         idx = np.where(labels == index)
    //         morphological_ret[idx] = 0

    
}

int ImgPostProcessor::volume(util::PPM &ppm)
{
    return ppm.w * ppm.h * ppm.c;
}

void ImgPostProcessor::write(const std::string &filename, util::PPM ppm)
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