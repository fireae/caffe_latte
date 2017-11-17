#include "caffe/layers/anchor_target_layer.hpp"
#include "caffe/net_config.hpp"
#include "caffe/util/frcnn_util.hpp"
#include "caffe/util/nms.hpp"

namespace caffe {

template <typename Dtype>
void AnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  AnchorTargetParameter param = this->layer_param_.anchor_target_param();
  base_size_ = param.base_size();
  feat_stride_ = param.feat_stride();

  vector<Dtype> ratios(param.ratio_size());
  for (int i = 0; i < param.ratio_size(); ++i) {
    ratios[i] = param.ratio(i);
  }
  vector<Dtype> scales(param.scale_size());
  for (int i = 0; i < param.scale_size(); ++i) {
    scales[i] = param.scale(i);
  }

  vector<int> anchors_shape(2);
  anchors_shape[0] = ratios.size() * scales.size();
  anchors_shape[1] = 4;
  anchors_.Reshape(anchors_shape);
  generate_anchors(base_size_, &ratios[0], &scales[0], ratios.size(),
                   scales.size(), anchors_.mutable_cpu_data());
  num_anchors_ = ratios.size() * scales.size();

  int height = bottom[0]->height();
  int width = bottom[0]->width();

  // top[0] -> labels
  vector<int> top0_shape(4);
  top0_shape[0] = 1;
  top0_shape[1] = 1;
  top0_shape[2] = num_anchors_ * height;
  top0_shape[3] = width;
  top[0]->Reshape(top0_shape);

  // top[1] -> bbox_targets
  vector<int> top1_shape(4);
  top1_shape[0] = 1;
  top1_shape[1] = num_anchors_ * 4;
  top1_shape[2] = height;
  top1_shape[3] = width;
  top[1]->Reshape(top1_shape);

  // top[2] -> bbox_inside_weights
  top[2]->Reshape(top1_shape);

  // top[3] -> bbox_outside_weights
  top[3]->Reshape(top1_shape);
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";
  // bottom[0] -> map of shape (..., H, W)
  // bottom[1] -> GT boxes (x1, y1, x2, y2, label)
  // bottom[2] -> im_info
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  int num = bottom[0]->num();

  const Dtype* bottom_gt_boxes = bottom[1]->cpu_data();
  const Dtype* bottom_im_info = bottom[2]->cpu_data();
  const Dtype im_height = bottom_im_info[0];
  const Dtype im_width = bottom_im_info[1];

  vector<Point4f<Dtype>> gt_boxes;
  for (int i = 0; i < bottom[1]->num(); i++) {
    gt_boxes.push_back(
        Point4f<Dtype>(bottom_gt_boxes[i * 5 + 0], bottom_gt_boxes[i * 5 + 1],
                       bottom_gt_boxes[i * 5 + 2], bottom_gt_boxes[i * 5 + 3]));
  }

  vector<int> inds_inside;
  vector<Point4f<Dtype>> anchors;
  int border_ = int(NetConfig::rpn_allowed_border);
  Dtype bounds[4] = {-border_, -border_, im_width + border_,
                     im_height + border_};
  const Dtype* anchors_data = anchors_.cpu_data();
  // Generate anchors
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int k = 0; k < num_anchors_; k++) {
        float x1 = w * feat_stride_ + anchors_data[k * 4 + 0];
        float y1 = h * feat_stride_ + anchors_data[k * 4 + 1];
        float x2 = w * feat_stride_ + anchors_data[k * 4 + 2];
        float y2 = h * feat_stride_ + anchors_data[k * 4 + 3];
        if (x1 >= bounds[0] && y1 >= bounds[1] && x2 < bounds[2] &&
            y2 < bounds[3]) {
          inds_inside.push_back((h * width + w) * num_anchors_ + k);
          anchors.push_back(Point4f<Dtype>(x1, y1, x2, y2));
        }
      }
    }
  }

  //
  const int n_anchors = anchors.size();

  vector<int> labels(n_anchors, -1);
  vector<Dtype> max_overlaps(n_anchors, -1);
  vector<int> argmax_overlaps(n_anchors, -1);
  vector<Dtype> gt_max_overlaps(gt_boxes.size(), -1);
  vector<int> gt_argmax_overlaps(gt_boxes.size(), -1);

  vector<vector<Dtype>> ious = get_ious(anchors, gt_boxes);
  for (int ia = 0; ia < n_anchors; ia++) {
    for (int igt = 0; igt < gt_boxes.size(); igt++) {
      if (ious[ia][igt] > max_overlaps[ia]) {
        max_overlaps[ia] = ious[ia][igt];
        argmax_overlaps[ia] = igt;
      }
      if (ious[ia][igt] > gt_max_overlaps[ia]) {
        gt_max_overlaps[igt] = ious[ia][igt];
        gt_argmax_overlaps[igt] = ia;
      }
    }
  }

  if (NetConfig::rpn_clobber_positives == false) {
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < NetConfig::rpn_negative_overlap) {
        labels[i] = 0;
      }
    }
  }

  // fg label: for each gt, anchors with highest overlap
  int debug_for_highest_over = 0;
  for (int j = 0; j < gt_max_overlaps.size(); ++j) {
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (std::abs(gt_max_overlaps[j] - ious[i][j]) <= NetConfig::eps) {
        labels[i] = 1;
        debug_for_highest_over++;
      }
    }
  }

  // fg label: above thresh IOU
  for (int i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= NetConfig::rpn_positive_overlap) {
      labels[i] = 1;
    }
  }

  if (NetConfig::rpn_clobber_positives) {
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < NetConfig::rpn_negative_overlap) {
        labels[i] = 0;
      }
    }
  }

  // subsample fg labels if we have too many
  int num_fg = float(NetConfig::rpn_fg_fraction) * NetConfig::rpn_batchsize;
  const int fg_inds_size = std::count(labels.begin(), labels.end(), 1);
  if (fg_inds_size > num_fg) {
    vector<int> fg_inds;
    for (size_t index = 0; index < labels.size(); index++) {
      if (labels[index] == 1) fg_inds.push_back(index);

      std::set<int> ind_set;
      while (ind_set.size() < fg_inds.size() - num_fg) {
        int tmp_idx = caffe::caffe_rng_rand() % fg_inds.size();
        ind_set.insert(fg_inds[tmp_idx]);
      }
      for (std::set<int>::iterator it = ind_set.begin(); it != ind_set.end();
           it++) {
        labels[*it] = -1;
      }
    }
  }

  // subsample negative labels if we have too many
  int num_bg =
      NetConfig::rpn_batchsize - std::count(labels.begin(), labels.end(), 1);
  const int bg_inds_size = std::count(labels.begin(), labels.end(), 0);
  if (bg_inds_size > num_bg) {
    vector<int> bg_inds;
    for (int i = 0; i < labels.size(); i++) {
      if (labels[i] == 0) {
        bg_inds.push_back(i);
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
  }

  vector<Point4f<Dtype>> bbox_targets;
  for (int i = 0; i < argmax_overlaps.size(); i++) {
    if (argmax_overlaps[i] < 0) {
      bbox_targets.push_back(Point4f<Dtype>());
    } else {
      bbox_targets.push_back(
          bbox_transform(anchors[i], gt_boxes[argmax_overlaps[i]]));
    }
  }

  vector<Point4f<Dtype>> bbox_inside_weights(n_anchors, Point4f<Dtype>());
  for (int i = 0; i < n_anchors; i++) {
    if (labels[i] == 1) {
      bbox_inside_weights[i][0] = 1.0;
      bbox_inside_weights[i][1] = 1.0;
      bbox_inside_weights[i][2] = 1.0;
      bbox_inside_weights[i][3] = 1.0;
    }
  }

  Dtype positive_weights, negative_weights;
  if (NetConfig::rpn_positive_weight < 0) {
    int num_examples =
        labels.size() - std::count(labels.begin(), labels.end(), -1);
    positive_weights = Dtype(1) / num_examples;
    negative_weights = Dtype(1) / num_examples;
  } else {
    CHECK_LT(NetConfig::rpn_positive_weight, 1) << "ilegal rpn_positive_weight";
    CHECK_GT(NetConfig::rpn_positive_weight, 0) << "ilegal rpn_positive_weight";
    positive_weights = Dtype(NetConfig::rpn_positive_weight) /
                       std::count(labels.begin(), labels.end(), 1);
    negative_weights = Dtype(1 - NetConfig::rpn_positive_weight) /
                       std::count(labels.begin(), labels.end(), 0);
  }

  vector<Point4f<Dtype>> bbox_outside_weights(n_anchors, Point4f<Dtype>());
  for (int i = 0; i < n_anchors; i++) {
    if (labels[i] == 1) {
      bbox_outside_weights[i] =
          Point4f<Dtype>(positive_weights, positive_weights, positive_weights,
                         positive_weights);
    } else if (labels[i] == 0) {
      bbox_outside_weights[i] =
          Point4f<Dtype>(negative_weights, negative_weights, negative_weights,
                         negative_weights);
    }
  }

  Info_Stds_Means_AvePos(bbox_targets, labels);

  // top[0] -> labels
  vector<int> top0_shape(4);
  top0_shape[0] = 1;
  top0_shape[1] = 1;
  top0_shape[2] = num_anchors_ * height;
  top0_shape[3] = width;
  top[0]->Reshape(top0_shape);

  // top[1] -> bbox_targets
  vector<int> top1_shape(4);
  top1_shape[0] = 1;
  top1_shape[1] = num_anchors_ * 4;
  top1_shape[2] = height;
  top1_shape[3] = width;
  top[1]->Reshape(top1_shape);

  // top[2] -> bbox_inside_weights
  top[2]->Reshape(top1_shape);

  // top[3] -> bbox_outside_weights
  top[3]->Reshape(top1_shape);

  Dtype* top_labels = top[0]->mutable_cpu_data();
  Dtype* top_bbox_targets = top[1]->mutable_cpu_data();
  Dtype* top_bbox_inside_weights = top[2]->mutable_cpu_data();
  Dtype* top_bbox_outside_weights = top[3]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());
  caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());
  caffe_set(top[3]->count(), Dtype(0), top[3]->mutable_cpu_data());

  for (size_t i = 0; i < inds_inside.size(); i++) {
    const int _anchor = inds_inside[i] % num_anchors_;
    const int _height = inds_inside[i] / num_anchors_ / width;
    const int _width = (inds_inside[i] / num_anchors_) % width;
    top_labels[top[0]->offset(0, 0, _anchor * height + _height, _width)] =
        labels[i];
    for (int c = 0; c < 4; ++c) {
      top_bbox_targets[top[1]->offset(0, _anchor * 4 + c, _height, _width)] =
          bbox_targets[i][c];
      top_bbox_inside_weights[top[2]->offset(
          0, _anchor * 4 + c, _height, _width)] = bbox_inside_weights[i][c];
      top_bbox_outside_weights[top[3]->offset(
          0, _anchor * 4 + c, _height, _width)] = bbox_outside_weights[i][c];
    }
  }
}

INSTANTIATE_CLASS(AnchorTargetLayer);
REGISTER_LAYER_CLASS(AnchorTarget);

}  // namespace caffe
