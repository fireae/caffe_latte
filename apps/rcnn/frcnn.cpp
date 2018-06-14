#include "frcnn.h"
#include "caffe/util/frcnn_utils.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <memory>
using namespace std;

namespace frcnn {
	using std::shared_ptr;

bool FasterRCNN::Init(const string& model_file, const string& weights_file) {
  net_ = shared_ptr<Net<float>>(new Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(weights_file);
  return true;
}

void FasterRCNN::Detect(const cv::Mat& im, vector<FrcnnBox>& detect_boxes,
                        bool vis_result) {
  const int kTestMaxSize = 640;
  const int kTestTagetSize = 600;

  cv::Mat img_org;
  im.convertTo(img_org, CV_32FC3);

  img_org -= Scalar(102.9801, 115.9465, 122.7717);
  int im_size_min = std::min(img_org.cols, img_org.rows);
  int im_size_max = std::max(img_org.cols, img_org.rows);

  float im_scale = 1.0;
  im_scale = float(kTestTagetSize) / float(im_size_min);
  if (im_scale * im_size_max > kTestMaxSize) {
    im_scale = float(kTestMaxSize) / float(im_size_max);
  }

  cv::Mat image;
  cv::resize(img_org, image, Size(), im_scale, im_scale, cv::INTER_LINEAR);

  int height = int(image.rows);
  int width = int(image.cols);

  shared_ptr<Blob<float>> blob_data = net_->blob_by_name("data");
  blob_data->Reshape(1, 3, height, width);
  float* blob_data_ptr = blob_data->mutable_cpu_data();
  printf("%d %d %f\n", width, height, im_scale);

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      blob_data_ptr[(0 * height + h) * width + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[0]);
      blob_data_ptr[(1 * height + h) * width + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[1]);
      blob_data_ptr[(2 * height + h) * width + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[2]);
    }
  }

  float* blob_im_info_ptr = net_->blob_by_name("im_info")->mutable_cpu_data();
  blob_im_info_ptr[0] = height;
  blob_im_info_ptr[1] = width;
  blob_im_info_ptr[2] = im_scale;

  net_->Forward();

  const float* bbox_pred = net_->blob_by_name("bbox_pred")->cpu_data();

  int num = net_->blob_by_name("rois")->num();
  const float* rois = net_->blob_by_name("rois")->cpu_data();
  float* rois_box = new float[4 * num];
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < 4; ++c) {
      rois_box[n * 4 + c] = float(rois[n * 5 + c + 1]) / float(im_scale);
    }
  }

  const float* cls_prob = net_->blob_by_name("cls_prob")->cpu_data();
  float* pred = new float[num * 5 * class_num_];
  BBoxTransformInv(num, bbox_pred, cls_prob, rois_box, pred, im.rows, im.cols,
                   class_num_);

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
      printf("%d: %f->(%f %f %f %f)\n", m, confidence[m], pred_boxes[m][0],
             pred_boxes[m][1], pred_boxes[m][2], pred_boxes[m][3]);
    }

    ApplyNMS(pred_boxes, confidence, nms_thresh_);

    for (int i = 0; i < pred_boxes.size(); i++) {
      if (confidence[i] < confidence_thresh_) continue;

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
      cv::Mat show_image = im.clone();
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
    if (confidence[i] < confidence_thresh_) continue;
    printf("%f : (%f %f %f %f) \n", confidence[i], pred_boxes[i][0],
           pred_boxes[i][1], pred_boxes[i][2], pred_boxes[i][3]);
    cv::rectangle(show_image, cv::Point(pred_boxes[i][0], pred_boxes[i][1]),
                  cv::Point(pred_boxes[i][2], pred_boxes[i][3]),
                  cv::Scalar(255, 0, 0));
  }
}
}
