#ifndef CAFFE_FRCNN_PROPOSAL_LAYER_HPP_
#define CAFFE_FRCNN_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
  /*************************************************
  faster-rcnn ProposalLayer
  Outputs object detection proposals by applying estimated bounding-box
  transformations to a set of regular boxes (called "anchors").
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rpn_rois'
  **************************************************/
  template <typename Dtype>
  class ProposalLayer : public Layer<Dtype> {
  public:
    explicit ProposalLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      // LOG(FATAL) << "Reshaping happens during the call to forward.";
    }

    virtual inline const char* type() const { return "ProposalLayer"; }
    virtual inline int MinBottomBlobs() const { return 3; }
    virtual inline int MaxBottomBlobs() const { return 3; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {}

    int base_h_;
    int base_w_;
    int feat_stride_h_;
    int feat_stride_w_;
    Blob<int> nms_mask_;
    int pre_nms_topn_;
    int post_nms_topn_;
    Dtype nms_thresh_;
    int min_size_;
    Blob<Dtype> anchors_;
    Blob<Dtype> proposals_;
    Blob<int> roi_indices_;

    vector<vector<int> > vec_anchors_;
  };

}  // namespace caffe

#endif  // CAFFE_PROPOSAL_LAYER_HPP_
