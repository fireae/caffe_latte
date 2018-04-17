#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/frcnn_utils.hpp"
#include "caffe/util/nms_util.hpp"

#define ROUND(x) ((int)((x) + (Dtype)0.5))

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  ProposalParameter param = this->layer_param_.proposal_param();

  base_h_ = param.base_h();
  base_w_ = param.base_w();
  feat_stride_h_ = param.feat_stride_h();
  feat_stride_w_ = param.feat_stride_w();
  pre_nms_topn_ = param.pre_nms_topn();
  post_nms_topn_ = param.post_nms_topn();
  nms_thresh_ = param.nms_thresh();
  min_size_ = param.min_size();

  vector<int> anchors_shape(2);
  anchors_shape[0] = param.anchor_size();
  anchors_shape[1] = 4;
  anchors_.Reshape(anchors_shape);
  Dtype* anchors_ptr_ = anchors_.mutable_cpu_data();
  for (int i = 0; i < param.anchor_size(); ++i) {
    //vector<int> anchor_pt;
    //anchor_pt.push_back(param.anchor(i).tl_x());
    //anchor_pt.push_back(param.anchor(i).tl_y());
    //anchor_pt.push_back(param.anchor(i).br_x());
    //anchor_pt.push_back(param.anchor(i).br_y());
    //vec_anchors_.push_back(anchor_pt);
    anchors_ptr_[i * 4 + 0] = static_cast<Dtype>(param.anchor(i).tl_x());
    anchors_ptr_[i * 4 + 1] = static_cast<Dtype>(param.anchor(i).tl_y());
    anchors_ptr_[i * 4 + 2] = static_cast<Dtype>(param.anchor(i).br_x());
    anchors_ptr_[i * 4 + 3] = static_cast<Dtype>(param.anchor(i).br_y());
  }
  
  vector<int> roi_indices_shape(1);
  roi_indices_shape[0] = post_nms_topn_;
  roi_indices_.Reshape(roi_indices_shape);

  // rois blob : holds R regions of interest, each is a 5 - tuple
  // (n, x1, y1, x2, y2) specifying an image batch index n and a
  // rectangle(x1, y1, x2, y2)
  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->shape(0) * post_nms_topn_;
  top_shape[1] = 5;
  top[0]->Reshape(top_shape);

  // scores blob : holds scores for R regions of interest
  if (top.size() > 1) {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";

  const Dtype* score = bottom[0]->cpu_data();
  const Dtype* bbox = bottom[1]->cpu_data();
  const Dtype* im_info = bottom[2]->cpu_data();
  Dtype* rois = top[0]->mutable_cpu_data();
  Dtype* top_score = (top.size() > 1) ? top[1]->mutable_cpu_data() : NULL;

  vector<int> proposals_shape(2);
  vector<int> top_shape(2);
  proposals_shape[0] = 0;
  proposals_shape[1] = 5;
  top_shape[0] = 0;
  top_shape[1] = 5;

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    // bottom shape: (2 x num_anchors) x H x W
    const int bottom_H = bottom[0]->height();
    const int bottom_W = bottom[0]->width();
    // input image height & width
    const Dtype img_H = im_info[0];
    const Dtype img_W = im_info[1];
    // scale factor for height & width
    const Dtype scale_H = im_info[2];
    const Dtype scale_W = im_info[3];
    // minimum box width & height
    const Dtype min_box_H = min_size_ * scale_H;
    const Dtype min_box_W = min_size_ * scale_W;
    // number of all proposals = num_anchors * H * W
    const int num_proposals = anchors_.shape(0) * bottom_H * bottom_W;
    // number of top-n proposals before NMS
    const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
    // number of final RoIs
    int num_rois = 0;

    // enumerate all proposals
    //   num_proposals = num_anchors * H * W
    //   (x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    proposals_shape[0] = num_proposals;
    proposals_.Reshape(proposals_shape);
    EnumerateProposals(bbox + num_proposals, score, anchors_.cpu_data(),
                       anchors_.shape(0), proposals_.mutable_cpu_data(),
                       bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
                       feat_stride_h_, feat_stride_w_);

    SortBox(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn_);

    NMS(proposals_.cpu_data(), pre_nms_topn, roi_indices_.mutable_cpu_data(),
        &num_rois, 0, nms_thresh_, post_nms_topn_);

    RetrieveROI(num_rois, n, proposals_.cpu_data(), roi_indices_.cpu_data(),
                rois, top_score);

    top_shape[0] += num_rois;
  }

  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}

#ifndef USE_CUDA
template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {}
#endif

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
