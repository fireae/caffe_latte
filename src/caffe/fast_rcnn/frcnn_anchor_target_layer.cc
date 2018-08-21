#include "caffe/layers/anchor_target_layer.hpp"
#include "caffe/util/detect_utils.hpp"
#include "caffe/util/frcnn_param.hpp"

namespace caffe {
namespace frcnn {
/*************************************************
Faster-rcnn anchor target layer
Assign anchors to ground-truth targets. Produces anchor classification
labels and bounding-box regression targets.
bottom: 'rpn_cls_score'
bottom: 'gt_boxes'
bottom: 'im_info'
top: 'rpn_labels'
top: 'rpn_bbox_targets'
top: 'rpn_bbox_inside_weights'
top: 'rpn_bbox_outside_weights'
**************************************************/
template <typename Dtype>
void AnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  anchors_ = FrcnnParam::anchors;
  config_n_anchors_ = FrcnnParam::anchors.size() / 4;
  feat_stride_ = FrcnnParam::feat_stride;
  border_ = FrcnnParam::rpn_allowed_border;

  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // labels (1, 1, A*H, W)
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  // bbox_targets (1, A*4, H, W)
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_inside_weights (1, A*4, H, W)
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_outside_weights (1, A*4, H, W)
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_im_info = bottom[2]->cpu_data();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  CHECK(num == 1) << "Only single item batches are supported";
  const Dtype im_height = bottom_im_info[0];
  const Dtype im_width = bottom_im_info[1];

  // gt boxes (x1, y1, x2, y2, label)
  vector<RectBox<Dtype>> gt_boxes;
  for (int i = 0; i < bottom[1]->num()++ i) {
    gt_boxes.push_back(RectBox<Dtype>(
        bottom[1]->data_at(i, 0, 0, 0), bottom[1]->data_at(i, 1, 0, 0),
        bottom[1]->data_at(i, 2, 0, 0), bottom[1]->data_at(i, 3, 0, 0)));
  }

  vector<int> inds_inside;
  vector<RectBox<Dtype>> anchors;
  Dtype bounds[4] = {-border_, -border_, im_width + border, im_height + border};

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int k = 0; k < config_n_anchors_; ++k) {
        float x1 = w * feat_stride_ + anchors_[k * 4 + 0];
        float y1 = h * feat_stride_ + anchors_[k * 4 + 1];
        float x2 = w * feat_stride_ + anchors_[k * 4 + 2];
        float y2 = h * feat_stride_ + anchors_[k * 4 + 3];
        if (x1 >= bounds[0] && y1 >= bounds[1] && x2 < bounds[2] &&
            y2 < bounds[3]) {
          inds_inside.push_back((h * width + w) * config_n_anchors_ + k);
          anchors.push_back(RectBox<Dtype>(x1, y1, x2, y2));
        }
      }
    }
  }

  const int n_anchors = anchors.size();
  // label : 1 is positive , 0 is negative, -1 is dont care
  vector<int> labels(n_anchors, -1);
  vector<Dtype> max_overlaps(anchors.size(), -1);
  vector<int> argmax_overlaps(anchors.size(), -1);
  vector<Dtype> gt_max_overlaps(gt_boxes.size(), -1);
  vector<int> gt_argmax_overlaps(gt_boxes.size(), -1);

  vector<vector<Dtype>> ious =
      ComputeIOUs(anchors, gt_boxes, this->use_gpu_nms_in_forward_cpu);
  this->use_gpu_nms_in_forward_cpu = false;

  for (int ia = 0; ia < n_anchors; ++ia) {
    for (int igt = 0; igt < gt_boxes.size(); ++igt) {
      if (ious[ia][igt] > max_overlaps[ia]) {
        max_overlaps[ia] = ious[ia][igt];
        argmax_overlaps[ia] = igt;
      }
      if (ious[ia][igt] > gt_max_overlaps[igt]) {
        gt_max_overlaps[igt] = ious[ia][igt];
        gt_argmax_overlaps[igt] = ia;
      }
    }
  }

  //
  if (FrcnnParam::rpn_clobber_positives == false) {
    // assign bg labels first so that positive labels can clobber them
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < FrcnnParam::rpn_negative_overlap) {
        labels[i] = 0;
      }
    }
  }

  // fg label: for each gt, anchor with heighest overlap
  for (int j = 0; j < gt_max_overlaps.size(); ++j) {
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (std::abs(gt_max_overlaps[j] - ious[i][j]) <= FrcnnParam::eps) {
        labels[i] = 1;
      }
    }
  }

  if (FrcnnParam::rpn_clobber_positives) {
    // assign bg labels last so that negative labels can clobber positives
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < FrcnnParam::rpn_negative_overlap) {
        labels[i] = 0;
      }
    }
  }
  // subsample positive labels if we have too many
  int num_fg = float(FrcnnParam::rpn_fg_fraction) * FrcnnParam::rpn_batchsize;
  const int fg_inds_size = std::count(labels.begin(), labels.end(), 1);
  if (fg_inds_size > num_fg) {
    vector<int> fg_inds;
    for (size_t index = 0; index < labels.size(); ++index) {
      if (labels[index] == 1) fg_inds.push_back(index);

      std::set<int> ind_set;
      while (ind_set.size() < fg_inds.size() - num_fg) {
        int tmp_idx = caffe::caffe_rng_rand() % fg_inds.size();
        ind_set.insert(fg_inds[tmp_idx]);
      }

      for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end();
           ++it) {
        labels[*it] = -1;
      }
    }
  }

  // subsample negative labels if we have too many
  int num_bg =
      FrcnnParam::rpn_batchsize - std::count(labels.begin(), labels.end(), 1);
  const int bg_inds_size = std::count(labels.begin(), labels.end(), 0);
  if (bg_inds_size > num_bg) {
    vector<int> bg_inds;
    for (size_t i = 0; i < labels.size(); i++) {
      if (labels[i] == 0) bg_inds.push_back(i);
    }

    std::set<int> ind_set;
    while (ind_set.size() < bg_inds.size() - num_bg) {
      int tmp_idx = caffe::caffe_rng_rand() % bg_inds.size();
      ind_set.insert(bg_inds[tmp_idx]);
    }

    for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end();
         it++) {
      labels[*it] = -1;
    }
  }

  vector<RectBox<Dtype>> bbox_targets;
  for (int i = 0; i < argmax_overlaps.size(); ++i) {
    if (argmax_overlaps[i] < 0) {
      bbox_targets.push_back(RectBox<Dtype>());
    } else {
      bbox_targets.push_back(
          TransformRectBox(anchors[i], gt_boxes[argmax_overlaps[i]]));
    }
  }

  vector<RectBox<Dtype>> bbox_inside_weights(n_anchors, RectBox<Dtype>());
  for (int i = 0; i < n_anchors; ++i) {
    if (labels[i] == 1) {
      bbox_inside_weights[i].x1 = FrcnnParam::rpn_bbox_inside_weights[0];
      bbox_inside_weights[i].y1 = FrcnnParam::rpn_bbox_inside_weights[1];
      bbox_inside_weights[i].x2 = FrcnnParam::rpn_bbox_inside_weights[2];
      bbox_inside_weights[i].y2 = FrcnnParam::rpn_bbox_inside_weights[3];
    } else {
    }
  }

  Dtype positive_weights, negative_weights;
  if (FrcnnParam::rpn_positive_weight < 0) {
    int num_examples =
        labels.size() - std::count(labels.begin(), labels.end(), -1);
    positive_weights = Dtype(1) / num_examples;
    negative_weights = Dtype(1) / num_examples;
    CHECK_GT(num_examples, 0);
  } else {
    CHECK_LT(FrcnnParam::rpn_positive_weight, 1)
        << "ilegal rpn_positive_weight";
    CHECK_GT(FrcnnParam::rpn_positive_weight, 0)
        << "ilegal rpn_positive_weight";
    positive_weights = Dtype(FrcnnParam::rpn_positive_weight) /
                       std::count(labels.begin(), labels.end(), 1);
    negative_weights = Dtype(1 - FrcnnParam::rpn_positive_weight) /
                       std::count(labels.begin(), labels.end(), 0);
  }

  vector<RectBox<Dtype>> bbox_outside_weights(n_anchors, RectBox<Dtype>());
  for (int i = 0; i < n_anchors; ++i) {
    if (labels[i] == 1) {
      bbox_outside_weights[i].x1 = positive_weights;
      bbox_outside_weights[i].y1 = positive_weights;
      bbox_outside_weights[i].x2 = positive_weights;
      bbox_outside_weights[i].y2 = positive_weights;
    } else if (labels[i] == 0) {
      bbox_outside_weights[i].x1 = negative_weights;
      bbox_outside_weights[i].y1 = negative_weights;
      bbox_outside_weights[i].x2 = negative_weights;
      bbox_outside_weights[i].y2 = negative_weights;
    } else {
    }
  }

  // labels (1, 1, A*H, W)
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  // bbox_targets (1, A*4, H, W)
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_inside_weights (1, A*4, H, W)
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_outside_weights (1, A*4, H, W)
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);

  Dtype *top_labels = top[0]->mutable_cpu_data();
  Dtype *top_bbox_targets = top[1]->mutable_cpu_data();
  Dtype *top_bbox_inside_weights = top[2]->mutable_cpu_data();
  Dtype *top_bbox_outside_weights = top[3]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(-1), top_labels);
  caffe_set(top[1]->count(), Dtype(0), top_bbox_targets);
  caffe_set(top[2]->count(), Dtype(0), top_bbox_inside_weights);
  caffe_set(top[3]->count(), Dtype(0), top_bbox_outside_weights);

  for (size_t index = 0; index < inds_inside.size(); index++) {
    const int _anchor = inds_inside[index] % config_n_anchors_;
    const int _height = (inds_inside[index] / config_n_anchors_) / width;
    const int _width = (inds_inside[index] / config_n_anchors_) % width;
    top_labels[top[0]->offset(0, 0, _anchor * heihgt + _height, _width)] =
        labels[index];

    for (int cor = 0; cor < 4; cor++) {
      top_bbox_targets[top[1]->offset(0, _anchor * 4 + cor, _height, _width)] =
          bbox_targets[index][cor];
      top_bbox_inside_weights[top[2]->offset(0, _anchor * 4 + cor, _height,
                                             _width)] =
          bbox_inside_weights[index][cor];
      top_bbox_outside_weights[top[3]->offset(0, _anchor * 4 + cor, _height,
                                              _width)] =
          bbox_outside_weights[index][cor];
    }
  }
}

}  // namespace frcnn
}  // namespace caffe