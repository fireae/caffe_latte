#include "frcnn.h"
#include "caffe/util/frcnn_util.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace frcnn {

bool FasterRCNN::Init(const string& model_file, const string& weights_file) {
  net_ = boost::shared_ptr<Net<float>>(new Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);
  return true;
}
void FasterRCNN::Detect(const cv::Mat& image, vector<FrcnnBox>& detect_boxes,
                        bool vis_result) {
  const int max_input_side = 1000;
  const int min_input_side = 600;

  int max_side = std::max(image.rows, image.cols);
  int min_side = std::min(image.rows, image.cols);
  float max_side_scale = float(max_side) / float(max_input_side);
  float min_side_scale = float(min_side) / float(min_input_side);
  float max_scale = max(max_side_scale, min_side_scale);

  float img_scale = 1.0;
  if (max_scale < 1.0) {
    img_scale = float(1.0) / max_scale;
  }
  int height = int(image.rows * img_scale);
  int width = int(image.cols * img_scale);

  cv::Mat float_image;
  image.convertTo(float_image, CV_32FC3);
  for (int h = 0; h < float_image.rows; h++) {
    for (int w = 0; w < float_image.cols; w++) {
      float_image.at<cv::Vec3f>(cv::Point(w, h))[0] -= float(102.9801);
      float_image.at<cv::Vec3f>(cv::Point(w, h))[1] -= float(115.9465);
      float_image.at<cv::Vec3f>(cv::Point(w, h))[2] -= float(122.7717);
    }
  }

  cv::Mat rsz_float_image;
  cv::resize(float_image, rsz_float_image, cv::Size(width, height));

  boost::shared_ptr<Blob<float>> blob_data = net_->blob_by_name("data");
  blob_data->Reshape(1, 3, height, width);
  float* blob_data_ptr = blob_data->mutable_cpu_data();

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      blob_data_ptr[(0 * height + h) * width + w] =
          float(rsz_float_image.at<cv::Vec3f>(cv::Point(w, h))[0]);
      blob_data_ptr[(1 * height + h) * width + w] =
          float(rsz_float_image.at<cv::Vec3f>(cv::Point(w, h))[1]);
      blob_data_ptr[(2 * height + h) * width + w] =
          float(rsz_float_image.at<cv::Vec3f>(cv::Point(w, h))[2]);
    }
  }

  float* blob_im_info_ptr = net_->blob_by_name("im_info")->mutable_cpu_data();
  blob_im_info_ptr[0] = height;
  blob_im_info_ptr[1] = width;
  blob_im_info_ptr[2] = img_scale;

  net_->Forward();

  const float* bbox_pred = net_->blob_by_name("bbox_pred")->cpu_data();
  int num = net_->blob_by_name("rois")->num();
  const float* rois = net_->blob_by_name("rois")->cpu_data();
  const float* cls_prob = net_->blob_by_name("cls_prob")->cpu_data();

  float* rois_box = new float[4 * num];
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < 4; ++c) {
      rois_box[n * 4 + c] = rois[n * 5 + c + 1] / img_scale;
    }
  }
  float* pred = new float[num * 5 * class_num_];
  BBoxTransformInv(num, bbox_pred, cls_prob, rois_box, pred, image.rows,
                   image.cols, class_num_);

  float* pred_per_cls = new float[num * 5];
  for (int cls = 1; cls < class_num_; cls++) {
    for (int n = 0; n < num; n++) {
      for (int k = 0; k < 5; k++) {
        pred_per_cls[n * 5 + k] = pred[(cls * num + n) * 5 + k];
      }
    }

    vector<vector<float>> pred_boxes;
    vector<float> confidence;
    for (int m = 0; m < num; m++) {
      vector<float> tmp_box;
      tmp_box.push_back(pred_per_cls[m * 5 + 0]);
      tmp_box.push_back(pred_per_cls[m * 5 + 1]);
      tmp_box.push_back(pred_per_cls[m * 5 + 2]);
      tmp_box.push_back(pred_per_cls[m * 5 + 3]);
      pred_boxes.push_back(tmp_box);
      confidence.push_back(pred_per_cls[m * 5 + 4]);
      //std::cout << m << " : " << pred_per_cls[m * 5 + 4] << std::endl;
    }

    ApplyNMS(pred_boxes, confidence, nms_thresh_);

    for (int i = 0; i < pred_boxes.size(); i++) {
      if (confidence[i] < nms_thresh_) continue;

      FrcnnBox box;
      box.x = pred_boxes[i][0];
      box.y = pred_boxes[i][1];
      box.width = pred_boxes[i][2] - pred_boxes[i][0];
      box.height = pred_boxes[i][3] - pred_boxes[i][1];
      box.type = cls;
      box.score = confidence[i];
      detect_boxes.push_back(box);
    }

    if (vis_result) {
      cv::Mat show_image = image.clone();
      VisResult(show_image, pred_boxes, confidence);
      char name[256];
      sprintf(name, "class_%d.jpg", cls);
      cv::imwrite(name, show_image);
    }
  }

  delete[] rois_box;
  delete[] pred;
  delete[] pred_per_cls;
}

void FasterRCNN::VisResult(cv::Mat& show_image,
                           const vector<vector<float>>& pred_boxes,
                           const vector<float>& confidence) {
  for (int i = 0; i < pred_boxes.size(); i++) {
    if (confidence[i] < nms_thresh_) continue;
    cv::rectangle(show_image, cv::Point(pred_boxes[i][0], pred_boxes[i][1]),
                  cv::Point(pred_boxes[i][2], pred_boxes[i][3]),
                  cv::Scalar(255, 0, 0));
  }
}
}