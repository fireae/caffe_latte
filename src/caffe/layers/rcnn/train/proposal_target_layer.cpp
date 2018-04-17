#include "caffe/layers/proposal_target_layer.hpp"
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/frcnn_param.hpp"
#include "caffe/util/frcnn_utils.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void ProposalTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  ProposalTargetParameter proposal_target_param =
      this->layer_param_.proposal_target_param();

  this->rng_.reset(new Caffe::RNG(static_cast<unsigned int>(3)));
  this->count_ = 0;
  this->bg_num_ = 0;
  this->fg_num_ = 0;
  fg_thresh_ = FrcnnParam::fg_thresh;
  bg_thresh_hi_ = FrcnnParam::bg_thresh_hi;
  bg_thresh_lo_ = FrcnnParam::bg_thresh_lo;
  bbox_normalize_targets_ = FrcnnParam::bbox_normalize_targets;
  bbox_normalize_means_.resize(4, 0.0);
  bbox_normalize_stds_.resize(4, 0.1);
  bbox_inside_weights_.resize(4, 0.0);
  for (int i = 0; i < 4; i++) {
    bbox_normalize_means_[i] = Dtype(FrcnnParam::bbox_normalize_means[i]);
    bbox_inside_weights_[i] = FrcnnParam::bbox_inside_weights[i];
    bbox_normalize_stds_[i] = FrcnnParam::bbox_normalize_stds[i];
  }
  n_classes_ = FrcnnParam::n_classes;

  LOG(INFO) << "ProposalTargetLayer :: " << n_classes_ << " classes";
  LOG(INFO) << "ProposalTargetLayer :: LayerSetUp";
  // sampled rois (0, x1, y1, x2, y2)
  top[0]->Reshape(1, 5, 1, 1);
  // labels
  top[1]->Reshape(1, 1, 1, 1);
  // bbox_targets
  top[2]->Reshape(1, n_classes_ * 4, 1, 1);
  // bbox_inside_weights
  top[3]->Reshape(1, n_classes_ * 4, 1, 1);
  // bbox_outside_weights
  top[4]->Reshape(1, n_classes_ * 4, 1, 1);
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top) {
  vector<Point4d<Dtype> > all_rois;
  for (int i = 0; i < bottom[0]->num(); i++) {
    all_rois.push_back(Point4d<Dtype>(
        bottom[0]->data_at(i, 1, 0, 0), bottom[0]->data_at(i, 2, 0, 0),
        bottom[0]->data_at(i, 3, 0, 0), bottom[0]->data_at(i, 4, 0, 0)));
    CHECK_EQ(bottom[0]->data_at(i, 0, 0, 0), 0)
        << "Only single item batches are supported";
  }

  vector<Point4d<Dtype> > gt_boxes;
  vector<int> gt_labels;
  for (int i = 0; i < bottom[1]->num(); i++) {
    gt_boxes.push_back(Point4d<Dtype>(
        bottom[1]->data_at(i, 0, 0, 0), bottom[1]->data_at(i, 1, 0, 0),
        bottom[1]->data_at(i, 2, 0, 0), bottom[1]->data_at(i, 3, 0, 0)));
    gt_labels.push_back(bottom[1]->data_at(i, 4, 0, 0));
    CHECK_GT(gt_labels[i], 0) << "Ground Truth Should Be Greater Than 0";
  }

  all_rois.insert(all_rois.end(), gt_boxes.begin(), gt_boxes.end());

  DLOG(ERROR) << "gt boxes size: " << gt_boxes.size();
  const int num_images = 1;
  const int rois_per_image = FrcnnParam::batch_size / num_images;
  const int fg_rois_per_image = rois_per_image * FrcnnParam::fg_fraction;

  // Sample rois with classification labels and bounding box regression
  // targets
  vector<int> labels;
  vector<Point4d<Dtype> > rois;
  vector<vector<Point4d<Dtype> > > bbox_targets, bbox_inside_weights;

  _sample_rois(all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image,
               labels, rois, bbox_targets, bbox_inside_weights);

  CHECK_EQ(labels.size(), rois.size());
  CHECK_EQ(labels.size(), bbox_targets.size());
  CHECK_EQ(labels.size(), bbox_inside_weights.size());
  const int batch_size = rois.size();
  DLOG(ERROR) << "top[0]-> " << batch_size << " , 5, 1, 1";
  // sampled rois
  top[0]->Reshape(batch_size, 5, 1, 1);  // rois
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  Dtype *rois_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    rois_data[top[0]->offset(i, 1, 0, 0)] = rois[i][0];
    rois_data[top[0]->offset(i, 2, 0, 0)] = rois[i][1];
    rois_data[top[0]->offset(i, 3, 0, 0)] = rois[i][2];
    rois_data[top[0]->offset(i, 4, 0, 0)] = rois[i][3];
  }
  // classification labels
  top[1]->Reshape(batch_size, 1, 1, 1);
  Dtype *label_data = top[1]->mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    label_data[top[1]->offset(i, 0, 0, 0)] = labels[i];
  }
  // bbox_targets
  top[2]->Reshape(batch_size, this->n_classes_ * 4, 1, 1);
  caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());
  Dtype *target_data = top[2]->mutable_cpu_data();
  // bbox_inside_weights and bbox_outside_weights
  top[3]->Reshape(batch_size, this->n_classes_ * 4, 1,
                  1);  // bbox_inside_weights
  caffe_set(top[3]->count(), Dtype(0), top[3]->mutable_cpu_data());
  Dtype *bbox_inside_data = top[3]->mutable_cpu_data();
  top[4]->Reshape(batch_size, this->n_classes_ * 4, 1,
                  1);  // bbox_outside_weights
  caffe_set(top[4]->count(), Dtype(0), top[4]->mutable_cpu_data());
  Dtype *bbox_outside_data = top[4]->mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < this->n_classes_; j++) {
      for (int cor = 0; cor < 4; cor++) {
        target_data[top[2]->offset(i, j * 4 + cor, 0, 0)] =
            bbox_targets[i][j][cor];
        bbox_inside_data[top[2]->offset(i, j * 4 + cor, 0, 0)] =
            bbox_inside_weights[i][j][cor];
        bbox_outside_data[top[2]->offset(i, j * 4 + cor, 0, 0)] =
            bbox_inside_weights[i][j][cor] > 0;
      }
    }
  }
  DLOG(INFO) << "ProposalTargetLayer::Forward_cpu End";
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::_sample_rois(
    const vector<Point4d<Dtype> > &all_rois,
    const vector<Point4d<Dtype> > &gt_boxes, const vector<int> &gt_label,
    const int fg_rois_per_image, const int rois_per_image, vector<int> &labels,
    vector<Point4d<Dtype> > &rois,
    vector<vector<Point4d<Dtype> > > &bbox_targets,
    vector<vector<Point4d<Dtype> > > &bbox_inside_weights) {
  // Generate a random sample of RoIs comprising foreground and background
  // examples.

  CHECK_EQ(gt_label.size(), gt_boxes.size());
  // overlaps: (rois x gt_boxes)
  std::vector<std::vector<Dtype> > overlaps = GetIoUs(all_rois, gt_boxes);
  std::vector<Dtype> max_overlaps(all_rois.size(), 0);
  std::vector<int> gt_assignment(all_rois.size(), -1);
  std::vector<int> _labels(all_rois.size());
  for (int i = 0; i < all_rois.size(); ++i) {
    for (int j = 0; j < gt_boxes.size(); ++j) {
      if (max_overlaps[i] <= overlaps[i][j]) {
        max_overlaps[i] = overlaps[i][j];
        gt_assignment[i] = j;
      }
    }
  }
  DLOG(INFO) << "sample_rois : all_rois: " << all_rois.size()
             << ", gt_box: " << gt_boxes.size();
  for (size_t i = 0; i < all_rois.size(); ++i) {
    if (gt_assignment[i] >= 0) {
      CHECK_LT(gt_assignment[i], gt_label.size());
      _labels[i] = gt_label[gt_assignment[i]];
    } else {
      _labels[i] = 0;
    }
  }

  // Select foreground RoIs as those with >= FG_THRESH overlap
  std::vector<int> fg_inds;
  for (int i = 0; i < all_rois.size(); ++i) {
    if (max_overlaps[i] >= fg_thresh_) {
      fg_inds.push_back(i);
    }
  }
  // Guard against the case when an image has fewer than fg_rois_per_image
  // foreground RoIs
  const int fg_rois_per_this_image =
      std::min(fg_rois_per_image, int(fg_inds.size()));
  DLOG(INFO) << "fg_inds [PRE,AFT] : [" << fg_inds.size() << ","
             << fg_rois_per_this_image << "] FG_THRESH : " << fg_thresh_;
  // Sample foreground regions without replacement
  if (fg_inds.size() > 0) {
    shuffle(fg_inds.begin(), fg_inds.end(),
            (caffe::rng_t *)this->rng_->generator());
    fg_inds.resize(fg_rois_per_this_image);
  }

  // Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
  std::vector<int> bg_inds;
  for (int i = 0; i < all_rois.size(); ++i) {
    if (max_overlaps[i] >= bg_thresh_lo_ && max_overlaps[i] < bg_thresh_hi_) {
      bg_inds.push_back(i);
    }
  }
  // Compute number of background RoIs to take from this image (guarding against
  // there being fewer than desired)
  const int bg_rois_per_this_image =
      std::min(rois_per_image - fg_rois_per_this_image, int(bg_inds.size()));
  DLOG(INFO) << "bg_inds [PRE,AFT] : [" << bg_inds.size() << ","
             << bg_rois_per_this_image << "] BG_THRESH : [" << bg_thresh_lo_
             << ", " << bg_thresh_hi_ << ")";
  // Sample background regions without replacement
  if (bg_inds.size() > 0) {
    shuffle(bg_inds.begin(), bg_inds.end(),
            (caffe::rng_t *)this->rng_->generator());
    bg_inds.resize(bg_rois_per_this_image);
  }

  // The indices that we're selecting (both fg and bg)
  std::vector<int> keep_inds(fg_inds);
  keep_inds.insert(keep_inds.end(), bg_inds.begin(), bg_inds.end());
  // Select sampled values from various arrays:
  labels.resize(keep_inds.size());
  rois.resize(keep_inds.size());
  std::vector<Point4d<Dtype> > _gt_boxes(keep_inds.size());
  for (size_t i = 0; i < keep_inds.size(); ++i) {
    labels[i] = _labels[keep_inds[i]];
    rois[i] = all_rois[keep_inds[i]];
    _gt_boxes[i] = gt_assignment[keep_inds[i]] >= 0
                       ? gt_boxes[gt_assignment[keep_inds[i]]]
                       : Point4d<Dtype>();
    // Clamp labels for the background RoIs to 0
    if (i >= fg_rois_per_this_image) labels[i] = 0;
  }

#ifdef DEBUG
  DLOG(INFO) << "num fg : "
             << labels.size() - std::count(labels.begin(), labels.end(), 0);
  DLOG(INFO) << "num bg : " << std::count(labels.begin(), labels.end(), 0);
  CHECK_EQ(std::count(labels.begin(), labels.end(), -1), 0);
  this->_count_ += 1;
  this->_fg_num_ += labels.size() - std::count(labels.begin(), labels.end(), 0);
  this->_bg_num_ += std::count(labels.begin(), labels.end(), 0);
  DLOG(INFO) << "num fg avg : " << this->_fg_num_ * 1. / this->_count_;
  DLOG(INFO) << "num bg avg : " << this->_bg_num_ * 1. / this->_count_;
  DLOG(INFO) << "ratio : "
             << float(this->_fg_num_) / float(this->_bg_num_ + kEPS);
  DLOG(INFO) << "bbox_normalize_targets : "
             << (bbox_normalize_targets ? "True" : "False");
  DLOG(INFO) << "bbox_normalize_means : " << bbox_normalize_means_[0] << ", "
             << bbox_normalize_means_[1] << ", " << bbox_normalize_means_[2]
             << ", " << bbox_normalize_means_[3];
  DLOG(INFO) << "bbox_normalize_stds : " << bbox_normalize_stds_[0] << ", "
             << bbox_normalize_stds_[1] << ", " << bbox_normalize_stds_[2]
             << ", " << bbox_normalize_stds_[3];
#endif

  // def _compute_targets(ex_rois, gt_rois, labels):
  CHECK_EQ(rois.size(), _gt_boxes.size());
  CHECK_EQ(rois.size(), labels.size());
  std::vector<Point4d<Dtype> > bbox_targets_data;
  for (int ii = 0; ii < rois.size(); ii++) {
    bbox_targets_data.push_back(TransformBox(rois[ii], _gt_boxes[ii]));
  }

  if (bbox_normalize_targets_) {
    // Optionally normalize targets by a precomputed mean and stdev
    for (size_t index = 0; index < bbox_targets_data.size(); index++) {
      for (int j = 0; j < 4; j++) {
        bbox_targets_data[index][j] =
            (bbox_targets_data[index][j] - bbox_normalize_means_[j]) /
            bbox_normalize_stds_[j];
      }
    }
  }

  // Compute boxes target
  bbox_targets = std::vector<std::vector<Point4d<Dtype> > >(
      keep_inds.size(), std::vector<Point4d<Dtype> >(this->n_classes_));
  bbox_inside_weights = std::vector<std::vector<Point4d<Dtype> > >(
      keep_inds.size(), std::vector<Point4d<Dtype> >(this->n_classes_));
  for (size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] > 0) {
      int cls = labels[i];
      // get bbox_targets and bbox_inside_weights
      bbox_targets[i][cls] = bbox_targets_data[i];
      bbox_inside_weights[i][cls] = Point4d<Dtype>(bbox_inside_weights_);
    }
  }
}
INSTANTIATE_CLASS(ProposalTargetLayer);
REGISTER_LAYER_CLASS(ProposalTarget);

}  // namespace caffe
