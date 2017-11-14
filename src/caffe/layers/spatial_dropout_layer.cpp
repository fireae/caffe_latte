
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_dropout_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK(this->layer_param_.has_spatial_dropout_param());
  const SpatialDropoutParameter spatial_dropout_param = 
      this->layer_param_.spatial_dropout_param();
  threshold_ = spatial_dropout_param.dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  rand_vec_.Reshape(num, channels, 1, 1);
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  if(this->phase_ == TRAIN) {
    // Create random numbers
    const int mask_count = rand_vec_.count();
    unsigned int* mask = rand_vec_.mutable_cpu_data();
    caffe_rng_bernoulli(mask_count, 1. - threshold_, mask);
    
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int sub_count = height * width;
    const int count = bottom[0]->count();
    CHECK_EQ(count, mask_count * sub_count);

    int d_offset = 0;
    int m_offset = 0;
    for(int n = 0; n < num; n++) {
      for(int c = 0; c < channels; c++) {
        const Dtype alpha = Dtype(mask[m_offset++] * this->scale_);
        for(int c = 0; c < sub_count; c++) {
          top_data[d_offset] = bottom_data[d_offset] * alpha;
          d_offset++;
        }
      }
    }
    CHECK_EQ(d_offset, count);
    CHECK_EQ(m_offset, mask_count);
  } else if(this->phase_ == TEST) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  } else {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
void SpatialDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if(propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if(this->phase_ == TRAIN) {
      const int num = top[0]->num();
      const int channels = top[0]->channels();
      const int height = top[0]->height();
      const int width = top[0]->width();
      const int count = top[0]->count();
      const int sub_count = height * width;
      const int mask_count = rand_vec_.count();
      CHECK_EQ(count, mask_count * sub_count);
      
      int d_offset = 0;
      int m_offset = 0;
      const unsigned int* mask = rand_vec_.cpu_data();
      for(int n = 0; n < num; n++) {
        for(int c = 0; c < channels; c++) {
          const Dtype alpha = Dtype(mask[m_offset++] * this->scale_);
          for(int c = 0; c < sub_count; c++) {
            bottom_diff[d_offset] = top_diff[d_offset] * alpha;
            d_offset++;
          }
        }
      }
      CHECK_EQ(d_offset, count);
      CHECK_EQ(m_offset, mask_count);
    } else if(this->phase_ == TEST) {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    } else {
      NOT_IMPLEMENTED;
    }
  }
}


// #ifndef USE_CUDA
// STUB_GPU(SpatialDropoutLayer);
// #endif

INSTANTIATE_CLASS(SpatialDropoutLayer);
REGISTER_LAYER_CLASS(SpatialDropout);

}  // namespace caffe
