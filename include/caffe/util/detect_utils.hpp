#ifndef CAFFE_UTIL_DETECT_UTILS_HPP_
#define CAFFE_UTIL_DETECT_UTILS_HPP_
#include <vector>

namespace caffe {

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