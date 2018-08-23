#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/internal_thread.hpp"

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter &param)
    : Layer<Dtype>(param), transform_param_(param.transform_param()) {}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  if (top.size() == 3) {
    output_roi_ = true;
  } else {
    output_roi_ = false;
  }
  if (top.size() == 4) {
    output_pts_ = true;
  } else {
    output_pts_ = false;
  }
  if (top.size() == 5) {
    output_weights_ = true;
  } else {
    output_weights_ = false;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter &param)
    : BaseDataLayer<Dtype>(param), prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
    if (this->output_roi_) {
      prefetch_[i]->roi_.mutable_cpu_data();
    }
    if (this->output_pts_) {
      prefetch_[i]->pts_.mutable_cpu_data();
    }
    if (this->output_weights_) {
      prefetch_[i]->weight_.mutable_cpu_data();
    }
  }
#ifdef USE_CUDA
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
      if (this->output_roi_) {
        prefetch_[i]->roi_.mutable_gpu_data();
      }
      if (this->output_pts_) {
        prefetch_[i]->pts_.mutable_gpu_data();
      }
      if (this->output_weights_) {
        prefetch_[i]->weight_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifdef USE_CUDA
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype> *batch = prefetch_free_.pop();
      load_batch(batch);
#ifdef USE_CUDA
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        if (this->output_roi_) {
          batch->roi_.data().get()->async_gpu_push(stream);
        }
        if (this->output_pts_) {
          batch->pts_.data().get()->async_gpu_push(stream);
        }
        if (this->output_weights_) {
          batch->weight_.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (std::exception &) {
    // Interrupted exception is expected on shutdown
  }
#ifdef USE_CUDA
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
  if (this->output_roi_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(prefetch_current_->roi_);
    top[2]->set_cpu_data(prefetch_current_->roi_.mutable_cpu_data());
  }
  if (this->output_pts_) {
    // Reshape to loaded labels.
    top[3]->ReshapeLike(prefetch_current_->pts_);
    top[3]->set_cpu_data(prefetch_current_->pts_.mutable_cpu_data());
  }
  if (this->output_weights_) {
    // Reshape to loaded weights_.
    top[4]->ReshapeLike(prefetch_current_->weight_);
    top[4]->set_cpu_data(prefetch_current_->weight_.mutable_cpu_data());
  }
}

#ifndef USE_CUDA
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

} // namespace caffe
