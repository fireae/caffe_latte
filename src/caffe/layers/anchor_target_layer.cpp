#include "caffe/layers/anchor_target_layer.hpp"
#include "caffe/util/nms.hpp"

#define ROUND(x) ((int)((x) + (Dtype)0.5))

namespace caffe {

template <typename Dtype>
struct RPoint {
  RPoint(Dtype x1_, Dtype y1_, Dtype x2_, Dtype y2_)
      : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
  Dtype x1;
  Dtype y1;
  Dtype x2;
  Dtype y2;
};

template <typename Dtype>
static int transform_box(Dtype box[], const Dtype dx, const Dtype dy,
                         const Dtype d_log_w, const Dtype d_log_h,
                         const Dtype img_W, const Dtype img_H,
                         const Dtype min_box_W, const Dtype min_box_H) {
  // width & height of box
  const Dtype w = box[2] - box[0] + (Dtype)1;
  const Dtype h = box[3] - box[1] + (Dtype)1;
  // center location of box
  const Dtype ctr_x = box[0] + (Dtype)0.5 * w;
  const Dtype ctr_y = box[1] + (Dtype)0.5 * h;

  // new center location according to gradient (dx, dy)
  const Dtype pred_ctr_x = dx * w + ctr_x;
  const Dtype pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const Dtype pred_w = exp(d_log_w) * w;
  const Dtype pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;
  box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + (Dtype)0.5 * pred_w;
  box[3] = pred_ctr_y + (Dtype)0.5 * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = std::max((Dtype)0, std::min(box[0], img_W - (Dtype)1));
  box[1] = std::max((Dtype)0, std::min(box[1], img_H - (Dtype)1));
  box[2] = std::max((Dtype)0, std::min(box[2], img_W - (Dtype)1));
  box[3] = std::max((Dtype)0, std::min(box[3], img_H - (Dtype)1));

  // recompute new width & height
  const Dtype box_w = box[2] - box[0] + (Dtype)1;
  const Dtype box_h = box[3] - box[1] + (Dtype)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
}

template <typename Dtype>
static void sort_box(Dtype list_cpu[], const int start, const int end,
                     const int num_top) {
  const Dtype pivot_score = list_cpu[start * 5 + 4];
  int left = start + 1, right = end;
  Dtype temp[5];
  while (left <= right) {
    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score) ++left;
    while (right > start && list_cpu[right * 5 + 4] <= pivot_score) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        temp[i] = list_cpu[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        list_cpu[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start) {
    for (int i = 0; i < 5; ++i) {
      temp[i] = list_cpu[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      list_cpu[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1) {
    sort_box(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end) {
    sort_box(list_cpu, right + 1, end, num_top);
  }
}

template <typename Dtype>
static void generate_anchors(int base_size, const Dtype ratios[],
                             const Dtype scales[], const int num_ratios,
                             const int num_scales, Dtype anchors[]) {
  // base box's width & height & center location
  const Dtype base_area = (Dtype)(base_size * base_size);
  const Dtype center = (Dtype)0.5 * (base_size - (Dtype)1);

  // enumerate all transformed boxes
  Dtype* p_anchors = anchors;
  for (int i = 0; i < num_ratios; ++i) {
    // transformed width & height for given ratio factors
    const Dtype ratio_w = (Dtype)ROUND(sqrt(base_area / ratios[i]));
    const Dtype ratio_h = (Dtype)ROUND(ratio_w * ratios[i]);

    for (int j = 0; j < num_scales; ++j) {
      // transformed width & height for given scale factors
      const Dtype scale_w = (Dtype)0.5 * (ratio_w * scales[j] - (Dtype)1);
      const Dtype scale_h = (Dtype)0.5 * (ratio_h * scales[j] - (Dtype)1);

      // (x1, y1, x2, y2) for transformed box
      p_anchors[0] = center - scale_w;
      p_anchors[1] = center - scale_h;
      p_anchors[2] = center + scale_w;
      p_anchors[3] = center + scale_h;
      p_anchors += 4;
    }  // endfor j
  }
}

template <typename Dtype>
static void enumerate_proposals_cpu(
    const Dtype bottom4d[], const Dtype d_anchor4d[], const Dtype anchors[],
    Dtype proposals[], const int num_anchors, const int bottom_H,
    const int bottom_W, const Dtype img_H, const Dtype img_W,
    const Dtype min_box_H, const Dtype min_box_W, const int feat_stride) {
  Dtype* p_proposal = proposals;
  const int bottom_area = bottom_H * bottom_W;

  for (int h = 0; h < bottom_H; ++h) {
    for (int w = 0; w < bottom_W; ++w) {
      const Dtype x = w * feat_stride;
      const Dtype y = h * feat_stride;
      const Dtype* p_box = d_anchor4d + h * bottom_W + w;
      const Dtype* p_score = bottom4d + h * bottom_W + w;
      for (int k = 0; k < num_anchors; ++k) {
        const Dtype dx = p_box[(k * 4 + 0) * bottom_area];
        const Dtype dy = p_box[(k * 4 + 1) * bottom_area];
        const Dtype d_log_w = p_box[(k * 4 + 2) * bottom_area];
        const Dtype d_log_h = p_box[(k * 4 + 3) * bottom_area];

        p_proposal[0] = x + anchors[k * 4 + 0];
        p_proposal[1] = y + anchors[k * 4 + 1];
        p_proposal[2] = x + anchors[k * 4 + 2];
        p_proposal[3] = y + anchors[k * 4 + 3];
        p_proposal[4] = transform_box(p_proposal, dx, dy, d_log_w, d_log_h,
                                      img_W, img_H, min_box_W, min_box_H) *
                        p_score[k * bottom_area];
        p_proposal += 5;
      }  // endfor k
    }    // endfor w
  }      // endfor h
}

template <typename Dtype>
static void retrieve_rois_cpu(const int num_rois, const int item_index,
                              const Dtype proposals[], const int roi_indices[],
                              Dtype rois[], Dtype roi_scores[]) {
  for (int i = 0; i < num_rois; ++i) {
    const Dtype* const proposals_index = proposals + roi_indices[i] * 5;
    rois[i * 5 + 0] = item_index;
    rois[i * 5 + 1] = proposals_index[0];
    rois[i * 5 + 2] = proposals_index[1];
    rois[i * 5 + 3] = proposals_index[2];
    rois[i * 5 + 4] = proposals_index[3];
    if (roi_scores) {
      roi_scores[i] = proposals_index[4];
    }
  }
}

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

  vector<RPoint<Dtype> > gt_boxes;
  for (int i = 0; i < bottom[1]->num(); i++) {
    gt_boxes.push_back(RPoint<Dtype>(
        bottom_gt_boxes[i * 4 + 0], bottom_gt_boxes[i * 4 + 1],
        bottom_gt_boxes[i * 4 + 2], bottom_gt_boxes[i * 4 + 3], ));
  }

  vector<int> inds_inside;
  vector<RPoint<Dtype> > anchors;
  int border_ = 0;
  Dtype bounds[4] = {-border_, -border_, im_width + border_,
                     im_height + border_};
  // Generate anchors
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int k = 0; k < num_anchors_; k++) {
        float x1 = w * feat_stride_ + anchors_[k * 4 + 0];
        float y1 = h * feat_stride_ + anchors_[k * 4 + 1];
        float x2 = w * feat_stride_ + anchors_[k * 4 + 2];
        float y2 = h * feat_stride_ + anchors_[k * 4 + 3];
        if (x1 >= bounds[0] && y1 >= bounds[1] && x2 < bounds[2] &&
            y2 < bounds[3]) {
          inds_inside.push_back((h * width + w) * num_anchors_ + k);
          anchors.push_back(RPoint(x1, y1, x2, y2));
        }
      }
    }
  }

  //

  const int n_anchors = anchors.size();

  vector<int> labels(n_anchors, -1);
  vector<Dtype> max_overlaps(n_anchors, -1);
  vector<int> argmax_max_overlaps(n_anchors, -1);
  vector<Dtype> gt_max_overlaps(gt_boxes.size(), -1);
  vector<int> gt_argmax_overlaps(gt_boxes.size(), -1);

  vector<vector<Dtype >> ious = get_ious(anchors, gt_boxes);
  for (int ia = 0; ia < n_anchors; ia++) {
      for (int igt = 0; igt < gt_boxes.size(); igt++) {
            if (ious[ia][igt] > max_overlaps[ia]) {
                max_overlaps[ia] = ious[ia][igt];
                argmax_overlaps[ia] = igt;
            } 
            if (ious[ia][igt] > gt_max_overlaps[ia]) {
                gt_max_overlaps[igt] = ious[ia][igt];
                argmax_overlaps[igt] = ia;
            } 
      }
  }
}

#ifndef USE_CUDA
template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {}
#endif

INSTANTIATE_CLASS(AnchorTargetLayer);
REGISTER_LAYER_CLASS(AnchorTarget);

}  // namespace caffe
