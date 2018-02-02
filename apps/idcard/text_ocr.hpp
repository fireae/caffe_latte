#pragma once
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace caffe;

typedef std::pair<std::string, float> Prediction;
typedef std::pair<int, float> PredictionIdx;

namespace jdcn {
class TextOCR {
 public:
  TextOCR() {}

  bool Init(const string& model_path, bool gpu_mode = true);
  bool Init(const string& model_file, const string& weight_file,
            const string& label_file, bool gpu_mode);

  void PrecessImage(const cv::Mat& image);
  vector<float> GetPrediction(const cv::Mat& image, std::vector<int>& outshape);

  vector<string> GetLabels(const string& label_file);

 private:
  std::shared_ptr<Net<float> > net_;
  int image_width_;
  int image_height_;
  int image_channels_;
  std::vector<std::string> labels_;
  float ratio_;
};

}  // namespace jdcn