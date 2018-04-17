#ifndef CAFFE_UTIL_FRCNN_UTILS_HPP_
#define CAFFE_UTIL_FRCNN_UTILS_HPP_
#include <algorithm>  // std::max
#include <cmath>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/frcnn_param.hpp"

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
  inline bool load_WithDiff(std::ifstream &infile) {
    string hashtag;
    if (!(infile >> hashtag)) return ok = false;
    CHECK_EQ(hashtag, "#");
    CHECK(infile >> this->image_index >> this->image_path);
    int num_roi;
    CHECK(infile >> num_roi);
    rois.clear(); diff.clear();
    for (int index = 0; index < num_roi; index++) {
      int label, x1, y1, x2, y2;
      int diff_;
      CHECK(infile >> label >> x1 >> y1 >> x2 >> y2 >> diff_);
      //x1 --; y1 --; x2 --; y2 --;
      // CHECK LABEL
      CHECK(label>0 && label<FrcnnParam::n_classes) << "illegal label : " << label << ", should >= 1 and < " << FrcnnParam::n_classes;
      CHECK_GE(x2, x1) << "illegal coordinate : " << x1 << ", " << x2 << " : " << this->image_path;
      CHECK_GE(y2, y1) << "illegal coordinate : " << y1 << ", " << y2 << " : " << this->image_path;
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
class Point4d {
 public:
  Point4d(Dtype p0 = 0, Dtype p1 = 0, Dtype p2 = 0, Dtype p3 = 0) {
    points[0] = p0;
    points[1] = p1;
    points[2] = p2;
    points[3] = p3;
  }
  Point4d(std::vector<Dtype>& values) {
    for (int i = 0; i < 4; i++) {
      points[i] = values[i];
    }
  }
  Dtype& operator[](const unsigned int idx) { return points[idx]; }
  const Dtype& operator[](const unsigned int idx) const { return points[idx]; }
  Dtype points[4];
};

template <typename Dtype>
Point4d<Dtype> TransformBox(const Point4d<Dtype>& ex_roi,
                            const Point4d<Dtype>& gt_roi);

template <typename Dtype>
vector<vector<Dtype> > GetIoUs(const vector<Point4d<Dtype> >& A,
                               const vector<Point4d<Dtype> >& B);

template <typename Dtype>
void EnumerateProposals(const Dtype* scores, const Dtype* bboxes,
                        const Dtype* anchors, const int num_anchors,
                        Dtype* proposals, const int bottom_h,
                        const int bottom_w, const Dtype img_h,
                        const Dtype img_w, const Dtype min_box_h,
                        const Dtype min_box_w, const int feat_stride_h,
                        const int feat_stride_w);
template <typename Dtype>
void SortBox(Dtype* boxes, const int start, const int end, const int num_top);

template <typename Dtype>
void RetrieveROI(const int num_rois, const int item_index,
                 const Dtype* proposals, const int* roi_indices, Dtype* rois,
                 Dtype* roi_scores);

float GetScaleFactor(int width, int height, int short_size, int max_long_size);

}  // namespace caffe

#endif  // CAFFE_UTIL_FRCNN_UTILS_HPP_