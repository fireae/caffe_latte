#include <vector>

#include "caffe/layers/signed_power_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/filler.hpp"

namespace caffe {

template <typename Dtype>
void SignedPowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(top[0] != bottom[0]) << "In-place computation is not supported in SignedPowerLayer!";
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  eps_ = this->layer_param_.power_param().eps();
  diff_scale_ = power_  * scale_;
  
  // Learnable parameter: exponent gamma
  this->blobs_.resize(1);
  vector<int> sz;
  sz.push_back(1);
  this->blobs_[0].reset(new Blob<Dtype>(sz));
  shared_ptr<Filler<Dtype> > filler;
  FillerParameter filler_param;
  filler_param.set_type("constant");
  filler_param.set_value(power_);
  filler.reset(GetFiller<Dtype>(filler_param));
  filler->Fill(this->blobs_[0].get());
}

// Compute y = (shift + scale * x)^power
template <typename Dtype>
void SignedPowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

		CHECK(false) << "No CPU implementation so far for signed power layer. Please use GPU mode." << std::endl;
}

template <typename Dtype>
void SignedPowerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
		CHECK(false) << "No CPU implementation so far for signed power layer. Please use GPU mode." << std::endl;
}

#ifndef USE_CUDA
STUB_GPU(SignedPowerLayer);
#endif

INSTANTIATE_CLASS(SignedPowerLayer);
REGISTER_LAYER_CLASS(SignedPower);

}  // namespace caffe
