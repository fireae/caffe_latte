#ifndef FRCNN_UTIL_HPP_
#define FRCNN_UTIL_HPP_
#include <algorithm>  // std::max
#include <cmath>
#include <vector>
namespace caffe {
using std::vector;

const double kEPS = 1e-14;
template <typename Dtype>
struct Point4f {
  Dtype point[4];
  Point4f(Dtype x1 = 0, Dtype y1 = 0, Dtype x2 = 0, Dtype y2 = 0) {
    point[0] = x1;
    point[1] = y1;
    point[2] = x2;
    point[3] = y2;
  }

  Point4f(const float data[4]) {
    for (int i = 0; i < 4; i++) {
      point[i] = data[i];
    }
  }

  Point4f(const double data[4]) {
    for (int i = 0; i < 4; i++) {
      point[i] = data[i];
    }
  }

  Point4f(const Point4f& other) {
    for (int i = 0; i < 4; i++) {
      point[i] = other.point[i];
    }
  }

  Dtype& operator[](const int id) { return point[id]; }
  const Dtype& operator[](const int id) const { return point[id]; }
};

void BBoxTransformInv(int box_count, const float* box_deltas,
                      const float* pred_cls, const float* boxes, float* pred,
                      int image_height, int image_width, int class_num);
void ApplyNMS(vector<vector<float> >& pred_boxes, vector<float>& confidence,
              float nms_thresh);

template <typename Dtype>
Point4f<Dtype> bbox_transform(const Point4f<Dtype>& ex_rois,
                              const Point4f<Dtype>& gt_rois);

template <typename Dtype>
Dtype get_iou(const Point4f<Dtype>& A, const Point4f<Dtype>& B);

template <typename Dtype>
vector<vector<Dtype> > get_ious(const vector<Point4f<Dtype> >& A,
                                const vector<Point4f<Dtype> >& B);

template <typename Dtype>
vector<Dtype> get_ious(const Point4f<Dtype>& A,
                       const vector<Point4f<Dtype> >& B);

template <typename Dtype>
int transform_box(Dtype box[], const Dtype dx, const Dtype dy,
                  const Dtype d_log_w, const Dtype d_log_h, const Dtype img_W,
                  const Dtype img_H, const Dtype min_box_W,
                  const Dtype min_box_H);

template <typename Dtype>
void sort_box(Dtype list_cpu[], const int start, const int end,
              const int num_top);

template <typename Dtype>
void generate_anchors(int base_size, const Dtype ratios[], const Dtype scales[],
                      const int num_ratios, const int num_scales,
                      Dtype anchors[]);

template <typename Dtype>
void enumerate_proposals_cpu(const Dtype bottom4d[], const Dtype d_anchor4d[],
                             const Dtype anchors[], Dtype proposals[],
                             const int num_anchors, const int bottom_H,
                             const int bottom_W, const Dtype img_H,
                             const Dtype img_W, const Dtype min_box_H,
                             const Dtype min_box_W, const int feat_stride);
template <typename Dtype>
void retrieve_rois_cpu(const int num_rois, const int item_index,
                       const Dtype proposals[], const int roi_indices[],
                       Dtype rois[], Dtype roi_scores[]);

}  // namespace caffe
#endif  // FRCNN_UTIL_HPP_