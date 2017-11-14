#include <math.h>
#include <stdio.h>  // for snprintf
#include <caffe/caffe.hpp>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
using namespace caffe;
using namespace std;

struct FrcnnFaceInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score; /**< Larger score should mean higher confidence. */
  int type;
};
/*
* ===  Class
* ======================================================================
*         Name:  Detector
*  Description:  FasterRCNN CXX Detector
* =====================================================================================
*/
class FRCNNDetector {
 public:
  FRCNNDetector() {}
  FRCNNDetector(const string& model_dir);
  FRCNNDetector(const string& model_file, const string& weights_file);
  void Detect(const cv::Mat& image, vector<FrcnnFaceInfo>& faceinfos);
  void Detect(string im_name);
  void Detect_video(string im_name);
  void bbox_transform_inv(const int num, const float* box_deltas,
                          const float* pred_cls, float* boxes, float* pred,
                          int img_height, int img_width);
  void vis_detections(cv::Mat image, vector<vector<float> > pred_boxes,
                      vector<float> confidence, float CONF_THRESH);
  void boxes_sort(int num, const float* pred, float* sorted_pred);
  void apply_nms(vector<vector<float> >& pred_boxes, vector<float>& confidence);

 private:
  boost::shared_ptr<Net<float> > net_;
};

struct Info {
  float score;
  const float* head = NULL;
};
