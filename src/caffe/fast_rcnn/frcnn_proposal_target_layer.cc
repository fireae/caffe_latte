#include "caffe/layers/proposal_target_layer.hpp"

namespace caffe {
namespace frcnn {

template <typename Dtype>
void ProposalTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  this->rng_.reset(Caffe::RNG(static_cast<unsigned int>(FrcnnParam::rng_seed)));
  this->count_ = this->bg_num_ = this->fg_num_ = 0;

  config_n_classes_ = FrcnnParam::n_classes;

  // sampled rois (0, x1, y1, x2, y2)
  top[0]->Reshape(1, 5, 1, 1);

  // labels;
  top[1]->Reshape(1, 1, 1, 1);

  // bbox_targets
  top[2]->Reshape(1, config_n_classes_ * 4, 1, 1);

  // bbox_inside_weights
  top[3]->Reshape(1, config_n_classes_ * 4, 1, 1);

  // bbox_outside_weights
  top[4]->Reshape(1, config_n_classes_ * 4, 1, 1);
}

template <typename Dtype>
void ProposalTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  vector<RectBox<Dtype>> all_rois;
  for (int i = 0; i < bottom[0]->num(); ++i) {
    all_rois.push_back(RectBox<Dtype>(
        bottom[0]->data_at(i, 1, 0, 0), bottom[0]->data_at(i, 2, 0, 0),
        bottom[0]->data_at(i, 3, 0, 0), bottom[0]->data_at(i, 4, 0, 0)));
  }

  vector<RectBox<Dtype>> gt_boxes;
  vector<int> gt_labels;
  for (int i = 0; i < bottom[1]->num(); ++i) {
    gt_boxes.push_back(RectBox<Dtype>(
        bottom[1]->data_at(i, 0, 0, 0), bottom[1]->data_at(i, 1, 0, 0),
        bottom[1]->data_at(i, 2, 0, 0), bottom[1]->data_at(i, 3, 0, 0)));
    gt_labels.push_back(bottom[1]->data_at(i, 4, 0, 0));
  }

  const int staget = this->layer_param_.proposal_target_param().stage();
  if (stage != 0) {
    const float gt_iou_thr =
        this->layer_param_.proposal_target_param().gt_iou_thr();
    std::vector<std::vector<Dtype>> overlaps =
        ComputeIOUs(all_rois, gt_boxes, false);
    std::vector<Dtype> max_overlaps(all_rois.size(), 0);
    for (int i = 0; i < all_rois.size(); ++i) {
      for (int j = 0; j < gt_boxes.size(); ++j) {
        if (max_overlaps[i] < overlaps[i][j]) {
          max_overlaps[i] = overlaps[i][j];
        }
      }
    }

    vector<RectBox<Dtype>> valid_rois;
    for (int i = 0; i < all_rois.size(); ++i) {
      if (max_overlaps[i] < gt_iou_thr) {
        valid_rois.push_back(all_rois[i]);
      }
    }
    all_rois = valid_rois;
  }

  all_rois.insert(all_roi.end(), gt_boxes.begin(), gt_boxes.end());

  const int num_images = 1;
  const int rois_per_image = FrcnnParam::batch_size / num_images;
  const int fg_rois_per_image = rois_per_image * FrcnnParam::fg_fraction;

  // Sample rois with classification labels and bounding box regression
  vector<int> labels;
  vector<RectBox<Dtype>> rois;
  vector<vector<RectBox<Dtype>>> bbox_targets, bbox_inside_weights;

  SampledROIs(all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image,
              labels, rois, bbox_targets, bbox_inside_weights);

  const int batch_size = rois.size();
  top[0]->Reshape(batch_size, 5, 1, 1);
  caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());
  Dtype* rois_data = top[0]->mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    rois_data[top[0]->offset(i, 1, 0, 0)] = rois[i].x1;
    rois_data[top[0]->offset(i, 2, 0, 0)] = rois[i].y1;
    rois_data[top[0]->offset(i, 3, 0, 0)] = rois[i].x2;
    rois_data[top[0]->offset(i, 4, 0, 0)] = rois[i].y2;
  }

  // classification labels
  top[1]->Reshape(batch_size, 1, 1, 1);
  Dtype* label_data = top[1]->mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    label_data[top[1]->offset(i, 0, 0, 0)] = labels[i];
  }

  // bbox_targets
  top[2]->Reshape(batch_size, config_n_classes_ * 4, 1, 1);
  caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());

  // bbox_inside_weights
  top[3]->Reshape(batch_size, config_n_classes_ * 4, 1, 1);
  caffe_set(top[3]->count(), Dtype(0), top[3]->mutable_cpu_data());

  // bbox_outside_weights
  top[4]->Reshape(batch_size, config_n_classes_ * 4, 1, 1);
  caffe_set(top[4]->count(), Dtype(0), top[4]->mutable_cpu_data());

  Dtype* target_data = top[2]->mutable_cpu_data();
  Dtype* bbox_inside_data = top[3]->mutable_cpu_data();
  Dtype* bbox_outside_data = top[4]->mutable_cpu_data();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < config_n_classes_; j++) {
      for (int cor = 0; cor < 4; cor++) {
        target_data[top[2]->offset(i, j * 4 + cor, 0, 0)] =
            bbox_targets[i][j][cor];
        bbox_inside_data[top[3]->offset(i, j * 4 + cor, 0, 0)] =
            bbox_inside_weights[i][j][cor];
        bbox_outside_data[top[4]->offset(i, j * 4 + cor, 0, 0)] =
            bbox_outside_weights[i][j][cor];
      }
    }
  }
}

}  // namespace frcnn
}  // namespace caffe