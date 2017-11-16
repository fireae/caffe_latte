#include "caffe/util/frcnn_util.hpp"
#include "caffe/common.hpp"

namespace caffe {
#define ROUND(x) ((int)((x) + (Dtype)0.5))
void BBoxTransformInv(int box_count, const float* box_deltas,
                      const float* pred_cls, const float* boxes, float* pred,
                      int image_height, int image_width, int class_num) {
  float width, height, center_x, center_y;
  float dx, dy, dw, dh;
  float pred_center_x, pred_center_y, pred_width, pred_height;
  for (int n = 0; n < box_count; n++) {
    width = boxes[n * 4 + 2] - boxes[n * 4 + 0] + 1.0;
    height = boxes[n * 4 + 3] - boxes[n * 4 + 1] + 1.0;
    center_x = boxes[n * 4 + 0] + width * 0.5;
    center_y = boxes[n * 4 + 1] + height * 0.5;

    for (int cls = 1; cls < class_num; cls++) {
      dx = box_deltas[(n * class_num + cls) * 4 + 0];
      dy = box_deltas[(n * class_num + cls) * 4 + 1];
      dw = box_deltas[(n * class_num + cls) * 4 + 2];
      dh = box_deltas[(n * class_num + cls) * 4 + 3];
      pred_center_x = center_x + width * dx;
      pred_center_y = center_y + height * dy;
      pred_width = width * std::exp(dw);
      pred_height = height * std::exp(dh);

      pred[(cls * box_count + n) * 5 + 0] =
          std::max(std::min(float(pred_center_x - 0.5 * pred_width),
                            float(image_width - 1)),
                   0.0f);
      pred[(cls * box_count + n) * 5 + 1] =
          std::max(std::min(float(pred_center_y - 0.5 * pred_height),
                            float(image_height - 1)),
                   0.0f);
      pred[(cls * box_count + n) * 5 + 2] =
          std::max(std::min(float(pred_center_x + 0.5 * pred_width),
                            float(image_width - 1)),
                   0.0f);
      pred[(cls * box_count + n) * 5 + 3] =
          std::max(std::min(float(pred_center_y + 0.5 * pred_height),
                            float(image_height - 1)),
                   0.0f);
      pred[(cls * box_count + n) * 5 + 4] = pred_cls[n * class_num + cls];
    }
  }
}

void ApplyNMS(vector<vector<float>>& pred_boxes, vector<float>& confidence,
              float nms_thresh) {
  for (int i = 0; i < pred_boxes.size() - 1; i++) {
    float s1 = (pred_boxes[i][2] - pred_boxes[i][0] + 1.0) *
               (pred_boxes[i][3] - pred_boxes[i][1] + 1.0);
    for (int j = i + 1; j < pred_boxes.size(); j++) {
      float s2 = (pred_boxes[j][2] - pred_boxes[j][0] + 1.0) *
                 (pred_boxes[j][3] - pred_boxes[j][1] + 1.0);

      float x1 = std::max(pred_boxes[i][0], pred_boxes[j][0]);
      float y1 = std::max(pred_boxes[i][1], pred_boxes[j][1]);
      float x2 = std::max(pred_boxes[i][2], pred_boxes[j][2]);
      float y2 = std::max(pred_boxes[i][3], pred_boxes[j][3]);
      float width = x2 - x1;
      float height = y2 - y1;
      if (width > 0 && height > 0) {
        float IOU = (width * height) / (s1 + s2 - width * height);
        if (IOU > nms_thresh) {
          if (confidence[i] >= confidence[j]) {
            pred_boxes.erase(pred_boxes.begin() + j);
            confidence.erase(confidence.begin() + j);
            j--;
          } else {
            pred_boxes.erase(pred_boxes.begin() + i);
            confidence.erase(confidence.begin() + i);
            i--;
            break;
          }
        }
      }
    }
  }
}

template <typename Dtype>
Dtype get_iou(const Point4f<Dtype>& A, const Point4f<Dtype>& B) {
  const Dtype xx1 = std::max(A[0], B[0]);
  const Dtype yy1 = std::max(A[1], B[1]);
  const Dtype xx2 = std::min(A[2], B[2]);
  const Dtype yy2 = std::min(A[3], B[3]);
  Dtype inter =
      std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
  Dtype areaB = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
  return inter / (areaA + areaB - inter);
}
template float get_iou(const Point4f<float>& A, const Point4f<float>& B);
template double get_iou(const Point4f<double>& A, const Point4f<double>& B);

template <typename Dtype>
vector<vector<Dtype>> get_ious(const vector<Point4f<Dtype>>& A,
                               const vector<Point4f<Dtype>>& B) {
  vector<vector<Dtype>> ious;
  for (size_t i = 0; i < A.size(); i++) {
    ious.push_back(get_ious(A[i], B));
  }
  return ious;
}
template vector<vector<float>> get_ious(const vector<Point4f<float>>& A,
                                        const vector<Point4f<float>>& B);
template vector<vector<double>> get_ious(const vector<Point4f<double>>& A,
                                         const vector<Point4f<double>>& B);

template <typename Dtype>
vector<Dtype> get_ious(const Point4f<Dtype>& A,
                       const vector<Point4f<Dtype>>& B) {
  vector<Dtype> ious;
  for (size_t i = 0; i < B.size(); i++) {
    ious.push_back(get_iou(A, B[i]));
  }
  return ious;
}

template vector<float> get_ious(const Point4f<float>& A,
                                const vector<Point4f<float>>& B);
template vector<double> get_ious(const Point4f<double>& A,
                                 const vector<Point4f<double>>& B);

float get_scale_factor(int width, int height, int short_size,
                       int max_long_size) {
  float im_size_min = std::min(width, height);
  float im_size_max = std::max(width, height);

  float scale_factor = static_cast<float>(short_size) / im_size_min;
  // Prevent the biggest axis from being more than max_size
  if (scale_factor * im_size_max > max_long_size) {
    scale_factor = static_cast<float>(max_long_size) / im_size_max;
  }
  return scale_factor;
}

template <typename Dtype>
Point4f<Dtype> bbox_transform_inv(const Point4f<Dtype>& box,
                                  const Point4f<Dtype>& delta) {
  Dtype src_w = box[2] - box[0] + 1;
  Dtype src_h = box[3] - box[1] + 1;
  Dtype src_ctr_x = box[0] + 0.5 * src_w;  // box[0] + 0.5*src_w;
  Dtype src_ctr_y = box[1] + 0.5 * src_h;  // box[1] + 0.5*src_h;
  Dtype pred_ctr_x = delta[0] * src_w + src_ctr_x;
  Dtype pred_ctr_y = delta[1] * src_h + src_ctr_y;
  Dtype pred_w = exp(delta[2]) * src_w;
  Dtype pred_h = exp(delta[3]) * src_h;
  return Point4f<Dtype>(pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                        pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h);
  // return Point4f<Dtype>(pred_ctr_x - 0.5*(pred_w-1) , pred_ctr_y -
  // 0.5*(pred_h-1) ,
  // pred_ctr_x + 0.5*(pred_w-1) , pred_ctr_y + 0.5*(pred_h-1));
}
template Point4f<float> bbox_transform_inv(const Point4f<float>& box,
                                           const Point4f<float>& delta);
template Point4f<double> bbox_transform_inv(const Point4f<double>& box,
                                            const Point4f<double>& delta);

template <typename Dtype>
vector<Point4f<Dtype>> bbox_transform_inv(
    const Point4f<Dtype>& box, const vector<Point4f<Dtype>>& deltas) {
  vector<Point4f<Dtype>> ans;
  for (size_t index = 0; index < deltas.size(); index++) {
    ans.push_back(bbox_transform_inv(box, deltas[index]));
  }
  return ans;
}
template vector<Point4f<float>> bbox_transform_inv(
    const Point4f<float>& box, const vector<Point4f<float>>& deltas);
template vector<Point4f<double>> bbox_transform_inv(
    const Point4f<double>& box, const vector<Point4f<double>>& deltas);

template <typename Dtype>
Point4f<Dtype> bbox_transform(const Point4f<Dtype>& ex_roi,
                              const Point4f<Dtype>& gt_roi) {
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
  return Point4f<Dtype>(targets_dx, targets_dy, targets_dw, targets_dh);
}
template Point4f<float> bbox_transform(const Point4f<float>& ex_roi,
                                       const Point4f<float>& gt_roi);
template Point4f<double> bbox_transform(const Point4f<double>& ex_roi,
                                        const Point4f<double>& gt_roi);

template <typename Dtype>
vector<Point4f<Dtype>> bbox_transform(const vector<Point4f<Dtype>>& ex_rois,
                                      const vector<Point4f<Dtype>>& gt_rois) {
  CHECK_EQ(ex_rois.size(), gt_rois.size());
  vector<Point4f<Dtype>> transformed_bbox;
  for (size_t i = 0; i < gt_rois.size(); i++) {
    transformed_bbox.push_back(bbox_transform(ex_rois[i], gt_rois[i]));
  }
  return transformed_bbox;
}
template vector<Point4f<float>> bbox_transform(
    const vector<Point4f<float>>& ex_rois,
    const vector<Point4f<float>>& gt_rois);
template vector<Point4f<double>> bbox_transform(
    const vector<Point4f<double>>& ex_rois,
    const vector<Point4f<double>>& gt_rois);

template <typename Dtype>
int transform_box(Dtype box[], const Dtype dx, const Dtype dy,
                  const Dtype d_log_w, const Dtype d_log_h, const Dtype img_W,
                  const Dtype img_H, const Dtype min_box_W,
                  const Dtype min_box_H) {
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
void sort_box(Dtype list_cpu[], const int start, const int end,
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
void generate_anchors(int base_size, const Dtype ratios[], const Dtype scales[],
                      const int num_ratios, const int num_scales,
                      Dtype anchors[]) {
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
void enumerate_proposals_cpu(const Dtype bottom4d[], const Dtype d_anchor4d[],
                             const Dtype anchors[], Dtype proposals[],
                             const int num_anchors, const int bottom_H,
                             const int bottom_W, const Dtype img_H,
                             const Dtype img_W, const Dtype min_box_H,
                             const Dtype min_box_W, const int feat_stride) {
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
void retrieve_rois_cpu(const int num_rois, const int item_index,
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

template <>
int transform_box<float>(float box[], const float dx, const float dy,
                         const float d_log_w, const float d_log_h,
                         const float img_W, const float img_H,
                         const float min_box_W, const float min_box_H);

template <>
void sort_box<float>(float list_cpu[], const int start, const int end,
                     const int num_top);

template void generate_anchors<float>(int base_size, const float ratios[],
                                      const float scales[],
                                      const int num_ratios,
                                      const int num_scales, float anchors[]);

template <>
void enumerate_proposals_cpu<float>(
    const float bottom4d[], const float d_anchor4d[], const float anchors[],
    float proposals[], const int num_anchors, const int bottom_H,
    const int bottom_W, const float img_H, const float img_W,
    const float min_box_H, const float min_box_W, const int feat_stride);
template <>
void retrieve_rois_cpu<float>(const int num_rois, const int item_index,
                              const float proposals[], const int roi_indices[],
                              float rois[], float roi_scores[]);

template <>
int transform_box<double>(double box[], const double dx, const double dy,
                          const double d_log_w, const double d_log_h,
                          const double img_W, const double img_H,
                          const double min_box_W, const double min_box_H);

template <>
void sort_box<double>(double list_cpu[], const int start, const int end,
                      const int num_top);

template void generate_anchors<double>(int base_size, const double ratios[],
                                       const double scales[],
                                       const int num_ratios,
                                       const int num_scales, double anchors[]);

template <>
void enumerate_proposals_cpu<double>(
    const double bottom4d[], const double d_anchor4d[], const double anchors[],
    double proposals[], const int num_anchors, const int bottom_H,
    const int bottom_W, const double img_H, const double img_W,
    const double min_box_H, const double min_box_W, const int feat_stride);
template <>
void retrieve_rois_cpu<double>(const int num_rois, const int item_index,
                               const double proposals[],
                               const int roi_indices[], double rois[],
                               double roi_scores[]);

}  // namespace caffe