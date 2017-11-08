#ifndef MTCNN_DETECT_ALIGN_H_
#define MTCNN_DETECT_ALIGN_H_

#include "mtcnn.h"
#include "lbf/lbf.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <string>

using namespace cv;
using namespace std;
using namespace lbf;

typedef struct JDFaceInfo {
    float score;
    cv::Rect face_bbox;
    vector<cv::Point> face_points;
};

class MTCNNDetectAlign {
public:
    MTCNNDetectAlign(){}
    MTCNNDetectAlign(const string& model_dir);
    ~MTCNNDetectAlign(){}
    bool Init(const string& model_dir);
    int Detect(const cv::Mat& image, vector<JDFaceInfo>& face_infos);
 private:
    
    boost::shared_ptr<MTCNN> mtcnn;
    boost::shared_ptr<LbfCascador> lbf_cascador;;

};

#endif //MTCNN_DETECT_ALIGN_H_