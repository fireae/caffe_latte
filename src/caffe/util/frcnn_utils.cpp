#include "caffe/util/frcnn_utils.hpp"
#include "caffe/util/nms_util.hpp"
namespace caffe {

template struct Point4d<float>;
template struct Point4d<double>;

template <typename Dtype>
vector<vector<Dtype>> GetIoUs(const vector<Point4d<Dtype>>& A,
                              const vector<Point4d<Dtype>>& B) {
  vector<vector<Dtype>> ious;
  for (int i = 0; i < A.size(); i++) {
    vector<Dtype> iou;
    for (int j = 0; j < B.size(); j++) {
      iou.push_back(IoU<Dtype>(A[i].points, B[i].points));
    }
    ious.push_back(iou);
  }
  return ious;
}
template <typename Dtype>
Point4d<Dtype> TransformBox(const Point4d<Dtype>& ex_roi,
                            const Point4d<Dtype>& gt_roi) {
  Dtype ex_width = ex_roi[2] - ex_roi[0] + 1;
  Dtype ex_height = ex_roi[3] - ex_roi[1] + 1;
  Dtype ex_ctr_x = ex_roi[0] + 0.5 * ex_width;
  Dtype ex_ctr_y = ex_roi[1] + 0.5 * ex_height;
  Dtype gt_widths = gt_roi[2] - gt_roi[0] + 1;
  Dtype gt_heights = gt_roi[3] - gt_roi[1] + 1;
  Dtype gt_ctr_x = gt_roi[0] + 0.5 * gt_widths;
  Dtype gt_ctr_y = gt_roi[1] + 0.5 * gt_heights;
  Dtype targets_dx = (gt_ctr_x - ex_ctr_x) / ex_width;
  Dtype targets_dy = (gt_ctr_y - ex_ctr_y) / ex_height;
  Dtype targets_dw = log(gt_widths / ex_width);
  Dtype targets_dh = log(gt_heights / ex_height);
  return Point4d<Dtype>(targets_dx, targets_dy, targets_dw, targets_dh);
}
template Point4d<float> TransformBox(const Point4d<float>& ex_roi,
                                     const Point4d<float>& gt_roi);
template Point4d<double> TransformBox(const Point4d<double>& ex_roi,
                                      const Point4d<double>& gt_roi);

template vector<vector<float>> GetIoUs(const vector<Point4d<float>>& A,
                                       const vector<Point4d<float>>& B);
template vector<vector<double>> GetIoUs(const vector<Point4d<double>>& A,
                                        const vector<Point4d<double>>& B);

template <typename Dtype>
int TransformBox(Dtype* box, const Dtype dx, const Dtype dy,
                 const Dtype d_log_w, const Dtype d_log_h, const Dtype img_w,
                 const Dtype img_h, const Dtype min_box_w,
                 const Dtype min_box_h) {
  // width & height of box
  Dtype box_w = box[2] - box[0] + (Dtype)1;
  Dtype box_h = box[3] - box[1] + (Dtype)1;
  // center location of box
  const Dtype ctr_x = box[0] + (Dtype)0.5 * box_w;
  const Dtype ctr_y = box[1] + (Dtype)0.5 * box_h;

  // new center location according to gradient (dx, dy)
  const Dtype pred_ctr_x = dx * box_w + ctr_x;
  const Dtype pred_ctr_y = dy * box_h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const Dtype pred_w = exp(d_log_w) * box_w;
  const Dtype pred_h = exp(d_log_h) * box_h;

  // update upper-left corner location
  box[0] = pred_ctr_x - (Dtype)0.5 * pred_w;
  box[1] = pred_ctr_y - (Dtype)0.5 * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + (Dtype)0.5 * pred_w;
  box[3] = pred_ctr_y + (Dtype)0.5 * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = std::max((Dtype)0, std::min(box[0], img_w - (Dtype)1));
  box[1] = std::max((Dtype)0, std::min(box[1], img_h - (Dtype)1));
  box[2] = std::max((Dtype)0, std::min(box[2], img_w - (Dtype)1));
  box[3] = std::max((Dtype)0, std::min(box[3], img_h - (Dtype)1));

  // recompute new width & height
  box_w = box[2] - box[0] + (Dtype)1;
  box_h = box[3] - box[1] + (Dtype)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_w) * (box_h >= min_box_h);
}

template <typename Dtype>
void EnumerateProposals(const Dtype* scores, const Dtype* bboxes,
                        const Dtype* anchors, const int num_anchors,
                        Dtype* proposals, const int bottom_h,
                        const int bottom_w, const Dtype img_h,
                        const Dtype img_w, const Dtype min_box_h,
                        const Dtype min_box_w, const int feat_stride_h,
                        const int feat_stride_w) {
  const int bottom_area = bottom_h * bottom_w;
  Dtype* proposal = proposals;
  for (int h = 0; h < bottom_h; h++) {
    for (int w = 0; w < bottom_w; w++) {
      const Dtype x = w * feat_stride_w;
      const Dtype y = h * feat_stride_h;
      const Dtype* bbox = bboxes + h * bottom_w + w;
      const Dtype* score = scores + h * bottom_w + w;
      for (int k = 0; k < num_anchors; k++) {
        const Dtype dx = bbox[(k * 4 + 0) * bottom_area];
        const Dtype dy = bbox[(k * 4 + 1) * bottom_area];
        const Dtype d_log_w = bbox[(k * 4 + 2) * bottom_area];
        const Dtype d_log_h = bbox[(k * 4 + 3) * bottom_area];
        proposal[0] = x + anchors[k * 4 + 0];
        proposal[1] = y + anchors[k * 4 + 1];
        proposal[2] = x + anchors[k * 4 + 2];
        proposal[3] = y + anchors[k * 4 + 3];
        proposal[4] = TransformBox(proposal, dx, dy, d_log_w, d_log_h, img_w,
                                   img_h, min_box_w, min_box_h) *
                      score[k * bottom_area];
        proposal += 5;
      }
    }
  }
}

template <typename Dtype>
void SortBox(Dtype* boxes, const int start, const int end, const int num_top) {
  const Dtype pivot_score = boxes[start * 5 + 4];
  int left = start + 1, right = end;
  Dtype temp[5];
  while (left <= right) {
    while (left <= end && boxes[left * 5 + 4] >= pivot_score) ++left;
    while (right > start && boxes[right * 5 + 4] <= pivot_score) --right;
    if (left <= right) {
      for (int i = 0; i < 5; ++i) {
        temp[i] = boxes[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        boxes[left * 5 + i] = boxes[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i) {
        boxes[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start) {
    for (int i = 0; i < 5; ++i) {
      temp[i] = boxes[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      boxes[start * 5 + i] = boxes[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i) {
      boxes[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1) {
    SortBox(boxes, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end) {
    SortBox(boxes, right + 1, end, num_top);
  }
}

template <typename Dtype>
void RetrieveROI(const int num_rois, const int item_index,
                 const Dtype* proposals, const int* roi_indices, Dtype* rois,
                 Dtype* roi_scores) {
  for (int i = 0; i < num_rois; i++) {
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

template int TransformBox<float>(float* box, const float dx, const float dy,
                                 const float d_log_w, const float d_log_h,
                                 const float img_w, const float img_h,
                                 const float min_box_w, const float min_box_h);

template int TransformBox<double>(double* box, const double dx, const double dy,
                                  const double d_log_w, const double d_log_h,
                                  const double img_w, const double img_h,
                                  const double min_box_w,
                                  const double min_box_h);
template void EnumerateProposals<float>(
    const float* scores, const float* bboxes, const float* anchors,
    const int num_anchors, float* proposals, const int bottom_h,
    const int bottom_w, const float img_h, const float img_w,
    const float min_box_h, const float min_box_w, const int feat_stride_h,
    const int feat_stride_w);
template void EnumerateProposals<double>(
    const double* scores, const double* bboxes, const double* anchors,
    const int num_anchors, double* proposals, const int bottom_h,
    const int bottom_w, const double img_h, const double img_w,
    const double min_box_h, const double min_box_w, const int feat_stride_h,
    const int feat_stride_w);

template void SortBox<float>(float* boxes, const int start, const int end,
                             const int num_top);
template void SortBox<double>(double* boxes, const int start, const int end,
                              const int num_top);
template void RetrieveROI<float>(const int num_rois, const int item_index,
                                 const float* proposals, const int* roi_indices,
                                 float* rois, float* roi_scores);
template void RetrieveROI<double>(const int num_rois, const int item_index,
                                  const double* proposals,
                                  const int* roi_indices, double* rois,
                                  double* roi_scores);

float GetScaleFactor(int width, int height, int short_size, int max_long_size) {
  float im_size_min = std::min(width, height);
  float im_size_max = std::max(width, height);

  float scale_factor = static_cast<float>(short_size) / im_size_min;
  // Prevent the biggest axis from being more than max_size
  if (scale_factor * im_size_max > max_long_size) {
    scale_factor = static_cast<float>(max_long_size) / im_size_max;
  }
  return scale_factor;
}
}  // namespace caffe