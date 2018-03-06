#include <caffe/caffe.hpp>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

using namespace caffe;
using namespace std;
using namespace cv;

namespace frcnn {

struct FrcnnBox {
  int x;
  int y;
  int width;
  int height;
  int type;
  float score;
};

class FasterRCNN {
 public:
  FasterRCNN(int class_num = 5, float nms_thresh = 0.3,
             float confidence_thresh = 0.8f)
      : class_num_(class_num),
        nms_thresh_(nms_thresh),
        confidence_thresh_(confidence_thresh) {}
  bool Init(const std::string& model_file, const std::string& weights_file);
  void Detect(const cv::Mat& image, vector<FrcnnBox>& detect_boxes,
              bool vis_result = false);
  void Detect(const string& image_name);

  void VisResult(cv::Mat& show_image, const vector<vector<float>>& pred_boxes,
                 const vector<float>& confidence);

 private:
  shared_ptr<Net<float>> net_;
  int class_num_;
  float nms_thresh_;
  float confidence_thresh_;
};
}