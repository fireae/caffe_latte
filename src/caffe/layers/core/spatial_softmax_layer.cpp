#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  */
  temp_ = Dtype(1.0) / Dtype(this->layer_param_.spatial_softmax_param().temperature());
  top[0]->ReshapeLike(*bottom[0]);
  // spatial uses the scale data differently -- 1 dim on last two axis
  vector<int> scale_dims = bottom[0]->shape();
  int num_dims = scale_dims.size();
  scale_dims[num_dims - 1] = 1;
  scale_dims[num_dims - 2] = 1;
  scale_.Reshape(scale_dims);
  temp_data_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Unimplemented";
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Unimplemented";
}


#ifndef USE_CUDA
STUB_GPU(SpatialSoftmaxLayer);
#endif

INSTANTIATE_CLASS(SpatialSoftmaxLayer);
REGISTER_LAYER_CLASS(SpatialSoftmax);

}  // namespace caffe
