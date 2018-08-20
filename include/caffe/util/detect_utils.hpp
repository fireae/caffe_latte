#ifndef CAFFE_UTIL_DETECT_UTILS_HPP_
#define CAFFE_UTIL_DETECT_UTILS_HPP_

namespace caffe {

template <typename Dtype> struct RectBox {
  Dtype x1, y1, x2, y2;
  RectBox() : x1(0), y1(0), x2(0), y2(0) {}
  RectBox(Dtype x1_, Dtype y1_, Dtype x2_, Dtype y2_)
      : x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
};

RectBox TransformBbox(RectBox r1, RectBox r2);
}

#endif // CAFFE_UTIL_DETECT_UTILS_HPP_