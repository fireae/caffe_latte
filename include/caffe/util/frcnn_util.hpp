#ifndef FRCNN_UTIL_HPP_
#define FRCNN_UTIL_HPP_
#include <algorithm>  // std::max
#include <cmath>
#include <string>
#include <vector>
#include "caffe/common.hpp"
namespace caffe {
using std::vector;

const double kEPS = 1e-14;

class DataPrepare {
 public:
  DataPrepare() {
    rois.clear();
    ok = false;
  }
  inline string GetImagePath(string root = "") {
    CHECK(this->ok) << "illegal status(ok=" << ok << ")";
    return root + image_path;
  }
  inline int GetImageIndex() {
    CHECK(this->ok) << "illegal status(ok=" << ok << ")";
    return image_index;
  }
  inline vector<vector<float> > GetRois(bool include_diff = false) {
    CHECK(this->ok) << "illegal status(ok=" << ok << ")";
    CHECK_EQ(this->rois.size(), this->diff.size());
    vector<vector<float> > _rois;
    for (size_t index = 0; index < this->rois.size(); index++) {
      if (include_diff == false && this->diff[index] == 1) continue;
      _rois.push_back(this->rois[index]);
    }
    return _rois;
  }
  inline bool load_WithDiff(std::ifstream& infile) {
    string hashtag;
    if (!(infile >> hashtag)) return ok = false;
    CHECK_EQ(hashtag, "#");
    CHECK(infile >> this->image_index >> this->image_path);
    int num_roi;
    CHECK(infile >> num_roi);
    rois.clear();
    diff.clear();
    for (int index = 0; index < num_roi; index++) {
      int label, x1, y1, x2, y2;
      int diff_;
      CHECK(infile >> label >> x1 >> y1 >> x2 >> y2 >> diff_);
      // x1 --; y1 --; x2 --; y2 --;
      // CHECK LABEL
      CHECK_GE(x2, x1) << "illegal coordinate : " << x1 << ", " << x2 << " : "
                       << this->image_path;
      CHECK_GE(y2, y1) << "illegal coordinate : " << y1 << ", " << y2 << " : "
                       << this->image_path;
      vector<float> roi(DataPrepare::NUM);
      roi[DataPrepare::LABEL] = label;
      roi[DataPrepare::X1] = x1;
      roi[DataPrepare::Y1] = y1;
      roi[DataPrepare::X2] = x2;
      roi[DataPrepare::Y2] = y2;
      rois.push_back(roi);
      diff.push_back(diff_);
    }
    return ok = true;
  }
  enum RoiDataField { LABEL, X1, Y1, X2, Y2, NUM };

 private:
  vector<vector<float> > rois;
  vector<int> diff;
  string image_path;
  int image_index;
  bool ok;
};

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
  Point4f(const vector<Dtype>& data) {
    for (int i = 0; i < data.size(); i++) {
      point[i] = data[i];
    }
  }

  Dtype& operator[](const int id) { return point[id]; }
  const Dtype& operator[](const int id) const { return point[id]; }
};

CAFFE_API void BBoxTransformInv(int box_count, const float* box_deltas,
                      const float* pred_cls, const float* boxes, float* pred,
                      int image_height, int image_width, int class_num);
CAFFE_API void ApplyNMS(vector<vector<float> >& pred_boxes, vector<float>& confidence,
              float nms_thresh);

template <typename Dtype>
Point4f<Dtype> bbox_transform(const Point4f<Dtype>& ex_rois,
                              const Point4f<Dtype>& gt_rois);
template <typename Dtype>
std::vector<Point4f<Dtype> > bbox_transform(
    const std::vector<Point4f<Dtype> >& ex_rois,
    const std::vector<Point4f<Dtype> >& gt_rois);
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
float get_scale_factor(int width, int height, int short_size,
                       int max_long_size);

}  // namespace caffe
#endif  // FRCNN_UTIL_HPP_