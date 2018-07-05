#pragma once
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/transforms/transforms.hpp"
#include "caffe/util/io.hpp"
namespace caffe {
class Transformer {
 public:
  Transformer(const TransformationParameter& transform_param)
      : param_(transform_param) {
    Init();
  }
  ~Transformer() {
    for (auto transformer : transformations_) {
      delete transformer;
      transformer = nullptr;
    }
  }
  void Init() {
    if (param_.force_color()) {
      Transformation* a = new RGB2BGR();
      transformations_.push_back(a);
    }
    if (param_.force_gray()) {
      Transformation* a = new Grayscale();
      transformations_.push_back(a);
    }

    // if (param_.mean_file() != "") {
    //   BlobProto blob_proto;
    //   ReadProtoFromBinaryFileOrDie(param_.mean_file().c_str(), &blob_proto);
    //   data_mean_.FromProto(blob_proto);
    // }
    if (param_.mean_value_size() > 0) {
      CHECK(param_.has_mean_file() == false)
          << "Cannot specify mean_file and mean_value at the same time";
      vector<float> mean_values;
      for (int c = 0; c < param_.mean_value_size(); ++c) {
        mean_values.push_back(param_.mean_value(c));
      }
      Transformation* a = new Mean(mean_values);
      transformations_.push_back(a);
    }

    if (param_.scale() != 1.0) {
      vector<float> scales;
      scales.push_back(param_.scale());
      Transformation* a = new Scale(scales);
      transformations_.push_back(a);
    }

    if (param_.mirror()) {
    }

    if (param_.crop_width() != 0 && param_.crop_height() != 0) {
    }
  }

  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    CImg<unsigned char> new_image = image;
    for (auto transformer : transformations_) {
      CImg<unsigned char> trans_image = transformer->Transform(new_image);
      new_image = trans_image;
    }
    return new_image;
  }

  template <typename Dtype>
  void ToBlob(CImg<unsigned char>& image, Blob<Dtype>* blob) {
    int channels = image.spectrum();
    int height = image.height();
    int width = image.width();

    Dtype* blob_data = blob->mutable_cpu_data();
    cimg_forXYC(image, x, y, c) {
      int blob_idx = c * height * width + height * y + x;
      blob_data[blob_idx] = image(x, y, 0, c);
    }
  }

  vector<Transformation*> transformations_;
  TransformationParameter param_;
  // Blob<Dtype> data_mean_;
};

}  // namespace caffe