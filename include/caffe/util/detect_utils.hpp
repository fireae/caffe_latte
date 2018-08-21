#ifndef CAFFE_UTIL_DETECT_UTILS_HPP_
#define CAFFE_UTIL_DETECT_UTILS_HPP_
//#include "caffe/util/frcnn_param.hpp"
#include <vector>
#include <string>

namespace caffe {
using std::vector;
using std::string;

struct BoxDataInfo {
  public:
    vector<vector<float> > rois;
    vector<int> difficult;
    string image_path;
    int image_index;
    bool is_ok;
    enum BoxDataType {X1, Y1, X2, Y2, LABEL, NUM};
  public:
    BoxDataInfo() {
      rois.clear();
      is_ok = false;
    }

    inline string GetImagePath(string root = "") {
      CHECK(is_ok) << "illegal status(ok=" << is_ok << ")";
      return root+"/"+image_path;
    }
    inline int GetImageIndex() {
      CHECK(is_ok) << "illegal status(ok=" << is_ok << ")";
      return image_index;
    }

    inline vector<vector<float> > GetRois(bool include_diff = false) {
      CHECK(is_ok) << "illegal status(ok=" << is_ok << ")";
      CHECK_EQ(this->rois.size(), this->difficult.size());
      vector<vector<float> > _rois;
      for (size_t index = 0; index < this->rois.size(); index++) {
        if (!include_diff && this->difficult[index] == 1) continue;
        _rois.push_back(this->rois[index]);
      }
      return _rois;
    }

    inline bool LoadwithDifficult(std::ifstream & infile) {
      string hashtag;
      if (!(infile >> hashtag)) return is_ok = false;
      CHECK_EQ(hashtag , "#");
      CHECK(infile >> this->image_index >> this->image_path);
      int num_roi;
      CHECK(infile >> num_roi);
      rois.clear();
      difficult.clear();
      for (int i = 0; i < num_roi; ++i) {
        int label, x1, y1, x2, y2;
        int diff_;
        CHECK(infile >> label >> x1 >> y1 >> x2 >> y2 >> diff_) << "illegal line of " << image_path;
       // CHECK(label > 0 && label < FrcnnParam::n_classes) << "illegal label: " << label << ", should >= 1 and < "<< FrcnnParam::n_classes;
        CHECK_GT(x2, x1) << "illegal coordinate : " << x1 << ", " << x2 << " : " << this->image_path;
        CHECK_GT(y2, y1) << "illegal coordinate : " << y1 << ", " << y2 << " : " << this->image_path;
        vector<float> roi(BoxDataInfo::NUM);
        roi[BoxDataInfo::LABEL] = label;
        roi[BoxDataInfo::X1] = x1;
        roi[BoxDataInfo::Y1] = y1;
        roi[BoxDataInfo::X2] = x2;
        roi[BoxDataInfo::Y2] = y2;
        rois.push_back(roi);
        difficult.push_back(diff_);
      }
      is_ok = true;
      return is_ok;
    }
};

template <typename Dtype>
struct RectBox {
  Dtype x1, y1, x2, y2;
  RectBox() : x1(0), y1(0), x2(0), y2(0) {}
  RectBox(Dtype x1_, Dtype y1_, Dtype x2_, Dtype y2_)
      : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
  RectBox(const RectBox& other) {
    x1 = other.x1;
    y1 = other.y1;
    x2 = other.x2;
    y2 = other.y2;
  }

  Dtype& operator[](const int idx) {
    switch (idx) {
      case 0:
        return x1;
      case 1:
        return y1;
      case 2:
        return x2;
      case 3:
      default:
        return y2;
    }
  }
  const Dtype& operator[](const int idx) const {
    switch (idx) {
      case 0:
        return x1;
      case 1:
        return y1;
      case 2:
        return x2;
      case 3:
      default:
        return y2;
    }
  }
};

template <typename Dtype>
struct AngleBox {
 public:
  Dtype cx1, cy1, w, h, theta;
};

template <typename Dtype>
struct BBox : RectBox<Dtype> {
 public:
  Dtype confidence;
  int id;
  BBox(Dtype x1 = 0, Dtype y1 = 0, Dtype x2 = 0, Dtype y2 = 0,
       Dtype confidence_ = 0, int id_ = 0)
      : RectBox<Dtype>(x1, y1, x2, y2), confidence(confidence_), id(id_) {}

  BBox(RectBox<Dtype> box, Dtype confidence_, Dtype id_)
      : RectBox<Dtype>(box), confidence(confidence_), id(id_) {}
};

template <typename Dtype>
RectBox<Dtype> TransformBbox(RectBox<Dtype> r1, RectBox<Dtype> r2);

template <typename Dtype>
Dtype ComputeIOU(const RectBox<Dtype>& a, const RectBox<Dtype>& b);

template <typename Dtype>
Dtype ComputeIOU(const BBox<Dtype>& a, const BBox<Dtype>& b);

template <typename Dtype>
std::vector<std::vector<Dtype> > ComputeIOUs(
    const std::vector<RectBox<Dtype> >& a,
    const std::vector<RectBox<Dtype> >& b, bool use_gpu);

template <typename Dtype>
RectBox<Dtype> TransformRectBox(const RectBox<Dtype>& a,
                                const RectBox<Dtype>& b);

}  // namespace caffe

#endif  // CAFFE_UTIL_DETECT_UTILS_HPP_