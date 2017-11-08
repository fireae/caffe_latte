#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief During training only, sets a random portion of @f$x@f$ to 0, adjusting
 *        the rest of the vector magnitude accordingly.
 *
 * For a given convolution feature tensor of size (C, H, W), we perform only C dropout trails,
 * abd extebd the dropout value across the entire feature map. Therefore, adjacent pixels in the dropped-out
 * feature map are either all 0 (dropped-out) or all active.
 * We have found that this modified dropout (relative to standard dropout) implementation improves performace,
 * especially on the FLIC dataset, where the training set size is small.
 * Note that spatial-dropout happens before 1*1 convolution layer.
 *
 * Please refer to `Efficient Object Localization Using Convolutional Networks, CVPR 2015.`
 */
template <typename Dtype>
class SpatialDropoutLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides SpatialDropoutParameter SpatialDropout_param,
   *     with SpatialDropoutLayer options:
   *   - SpatialDropout_ratio (\b optional, default 0.5).
   *     Sets the probability @f$ p @f$ that any given unit is dropped.
   */
  explicit SpatialDropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialDropout"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  /// when divided by UINT_MAX, the randomly generated values @f$u\sim U(0,1)@f$
  Blob<unsigned int> rand_vec_;
  /// the probability @f$ p @f$ of dropping any input
  Dtype threshold_;
  /// the scale for undropped inputs at train time @f$ 1 / (1 - p) @f$
  Dtype scale_;
  unsigned int uint_thres_;
};

}
