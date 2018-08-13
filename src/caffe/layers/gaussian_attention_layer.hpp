#ifndef CAFFE_GAUSSIAN_ATTENTION_LAYER_HPP_
#define CAFFE_GAUSSIAN_ATTENTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Gaussian attention 
 */
template <typename Dtype>
class GaussianAttentionLayer : public Layer<Dtype> {
 public:
  explicit GaussianAttentionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GaussianAttention"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // Blob<Dtype> mean_, variance_, temp_, x_norm_;
  Blob<Dtype> mask_;
  Blob<Dtype> tmp_;
  Blob<Dtype> ones_;
  Dtype sigma_;
  int num_;
  int channels_, num_locs_;
  int height_, width_;
  // The number of channels for each attention location
  int channel_stride_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
