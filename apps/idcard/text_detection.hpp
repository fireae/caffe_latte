#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"

namespace jdcn {
class TextDetector {
 public:
  TextDetector() {}
  ~TextDetector() {}
  bool Init(const std::string& model_file, const std::string& weight_file);
  void PrecessImage(const cv::Mat& image);
  void DetectTextRegions(std::vector<std::vector<float>>& text_proposals,
                         std::vector<float>& text_scores, float text_thresh);

  void DetectTextLines(vector<vector<float>>& text_lines);
  void ClipBoxes(std::vector<float>& box, int height, int width);
  void FiterTextBoxes(std::vector<std::vector<float>>& text_proposals,
                      std::vector<float>& text_scores, float text_thresh);

 private:
  std::shared_ptr<caffe::Net<float>> net_;
  int image_height_;
  int image_width_;
  float ratio_;
};

}  // namespace xiangyin