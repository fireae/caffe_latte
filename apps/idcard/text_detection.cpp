#include "text_detection.hpp"
#include <caffe/caffe.hpp>
#include <limits>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "polyfit.h"
#include "text_builder.hpp"

using namespace std;
using namespace caffe;
using namespace cv;

namespace jdcn {
bool TextDetector::Init(const string& model_file, const string& weight_file) {
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weight_file);
  return true;
}

void TextDetector::PrecessImage(const cv::Mat& im_org) {
  std::shared_ptr<Blob<float>> blob_data = net_->blob_by_name("data");
  cv::Mat im;
  ratio_ = 800.0 / std::max(im_org.rows, im_org.cols);
  cv::resize(im_org, im, cv::Size(), ratio_, ratio_);
  cv::Mat image;
  im.convertTo(image, CV_32FC3);
  // mean value (102.9801, 115.9465, 122.7717)
  image -= cv::Scalar(102.9801, 115.9465, 122.7717);
  image_width_ = image.cols;
  image_height_ = image.rows;
  blob_data->Reshape(1, 3, image_height_, image_width_);
  float* blob_data_ptr = blob_data->mutable_cpu_data();
  for (int h = 0; h < image_height_; ++h) {
    for (int w = 0; w < image_width_; ++w) {
      blob_data_ptr[(0 * image_height_ + h) * image_width_ + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[0]);
      blob_data_ptr[(1 * image_height_ + h) * image_width_ + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[1]);
      blob_data_ptr[(2 * image_height_ + h) * image_width_ + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[2]);
    }
  }
}

void TextDetector::ClipBoxes(vector<float>& box, int height, int width) {
  if (box[0] < 0) {
    box[0] = 0;
  }
  if (box[0] > width - 1) {
    box[0] = width - 1;
  }
  if (box[2] < 0) {
    box[2] = 0;
  }
  if (box[2] > width - 1) {
    box[2] = width - 1;
  }
  if (box[1] < 0) {
    box[1] = 0;
  }
  if (box[1] > height - 1) {
    box[1] = height - 1;
  }
  if (box[3] < 0) {
    box[3] = 0;
  }
  if (box[3] > height - 1) {
    box[3] = height - 1;
  }
}

void TextDetector::FiterTextBoxes(vector<vector<float>>& text_proposals,
                                  vector<float>& text_scores,
                                  float text_thresh) {
  std::shared_ptr<Blob<float>> rois = net_->blob_by_name("rois");
  std::shared_ptr<Blob<float>> scores = net_->blob_by_name("scores");
  float* scores_data = scores->mutable_cpu_data();
  float* rois_data = rois->mutable_cpu_data();
  for (int i = 0; i < scores->shape()[0]; i++) {
    if (scores_data[i] > text_thresh) {
      vector<float> vec_roi;
      vec_roi.push_back(rois_data[i * 4 + 0]);
      vec_roi.push_back(rois_data[i * 4 + 1]);
      vec_roi.push_back(rois_data[i * 4 + 2]);
      vec_roi.push_back(rois_data[i * 4 + 3]);
      ClipBoxes(vec_roi, image_height_, image_width_);
      text_proposals.push_back(vec_roi);
      text_scores.push_back(scores_data[i]);
    }
  }
}
void TextDetector::DetectTextRegions(vector<vector<float>>& text_proposals,
                                     vector<float>& text_scores,
                                     float text_thresh) {
  std::shared_ptr<Blob<float>> im_info = net_->blob_by_name("im_info");
  float* im_data = im_info->mutable_cpu_data();
  im_data[0] = image_height_;
  im_data[1] = image_width_;
  net_->Forward();
  FiterTextBoxes(text_proposals, text_scores, text_thresh);
}

void TextDetector::DetectTextLines(vector<vector<float>>& text_lines) {
  vector<vector<float>> text_proposals;
  vector<float> text_scores;
  float text_thresh = 0.7;
  DetectTextRegions(text_proposals, text_scores, text_thresh);
  vector<int> im_size(2);
  im_size[0] = image_height_;
  im_size[1] = image_width_;
  TextProposalGraphBuilder builder;
  Graph g = builder.BuildGraph(text_proposals, text_scores, im_size);
  vector<vector<int>> sub_graphs = g.SubGraphsConnected();

  for (int i = 0; i < sub_graphs.size(); i++) {
    vector<int> sub_graph = sub_graphs[i];
    vector<vector<float>> text_line_boxes;
    float x0 = numeric_limits<float>::max();
    float x1 = 0.0;
    vector<float> lt_x0;
    vector<float> rt_y0;
    vector<float> lb_x0;
    vector<float> rb_y0;
    for (int j = 0; j < sub_graph.size(); j++) {
      int index = sub_graph[j];
      text_line_boxes.push_back(text_proposals[index]);
      if (text_proposals[index][0] < x0) {
        x0 = text_proposals[index][0];
      }
      if (text_proposals[index][2] > x1) {
        x1 = text_proposals[index][2];
      }
      lt_x0.push_back(text_proposals[index][0]);
      rt_y0.push_back(text_proposals[index][1]);

      lb_x0.push_back(text_proposals[index][0]);
      rb_y0.push_back(text_proposals[index][3]);
    }

    float offset = (text_line_boxes[0][2] - text_line_boxes[0][0]) * 0.5;
    vector<float> factor(2);
    polyfit(lt_x0.data(), rt_y0.data(), lt_x0.size(), 1, factor.data());
    float lt_y = factor[1] * (x0 + offset) + factor[0];
    float rt_y = factor[1] * (x1 - offset) + factor[0];

    factor[0] = 0.0;
    factor[1] = 0.0;
    polyfit(lb_x0.data(), rb_y0.data(), lb_x0.size(), 1, factor.data());
    float lb_y = factor[1] * (x0 + offset) + factor[0];
    float rb_y = factor[1] * (x1 - offset) + factor[0];

    vector<float> text_line(5);
    text_line[0] = (x0);
    text_line[1] = std::min((lt_y), (rt_y));
    text_line[2] = (x1);
    text_line[3] = std::max((lb_y), (rb_y));

    double sum = 0.0;
    for (int k = 0; k < sub_graph.size(); k++) {
      sum += text_scores[sub_graph[k]];
    }
    text_line[4] = (sum / sub_graph.size());

    text_lines.push_back(text_line);
  }

  for (int i = 0; i < text_lines.size(); ++i) {
    vector<float>& text_line = text_lines[i];
    for (int j = 0; j < 4; ++j) {
      text_line[j] /= ratio_;
    }
  }
}

}  // namespace jdcn