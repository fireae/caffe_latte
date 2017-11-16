

#ifndef CAFFE_PROPOSAL_TARGET_LAYER_HPP_
#define CAFFE_PROPOSAL_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/frcnn_util.hpp"

namespace caffe {

/**
 * bottom: 'rpn_rois'
  bottom: 'gt_boxes'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
 * **/

template <typename Dtype>
class ProposalTargetLayer : public Layer<Dtype> {
 public:
  explicit ProposalTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top) {
    // LOG(FATAL) << "Reshaping happens during the call to forward.";
  }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 5; }
  virtual inline int MinTopBlobs() const { return 5; }
  virtual inline int MaxTopBlobs() const { return 5; }
  virtual inline const char* type() const { return "ProposalTarget"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    Forward_cpu(bottom, top);
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {}
  void _sample_rois(const vector<Point4f<Dtype> >& all_rois,
                    const vector<Point4f<Dtype> >& gt_boxes,
                    const vector<int>& gt_label, const int fg_rois_per_image,
                    const int rois_per_image, vector<int>& labels,
                    vector<Point4f<Dtype> >& rois,
                    vector<vector<Point4f<Dtype> > >& bbox_targets,
                    vector<vector<Point4f<Dtype> > >& bbox_inside_weights);
  int n_classes_;
  shared_ptr<Caffe::RNG> rng_;
  int count_;
  int fg_num_;
  int bg_num_;
  float fg_thresh_;
  float bg_thresh_hi_;
  float bg_thresh_lo_;
  vector<Dtype> bbox_normalize_means_;
  vector<Dtype> bbox_normalize_stds_;
  vector<Dtype> bbox_inside_weights_;
  bool bbox_normalize_targets_;
};

}  // namespace caffe

#endif  // CAFFE_ProposalTarget_LAYER_HPP_