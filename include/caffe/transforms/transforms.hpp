#pragma once
#include <assert.h>
#include <vector>
#include "caffe/common.hpp"

namespace caffe {
/****************************************/
template <typename T>
CImg<T> rgb2gray(CImg<T> src) {
  CImg<T> dst(src.width(), src.height(), 1, 1);
  cimg_forXY(src, x, y) {
    T r = src(x, y, 0, 0);
    T g = src(x, y, 0, 1);
    T b = src(x, y, 0, 2);
    dst(x, y, 0, 0) = (2989 * r + 5870 * g + 1140 * b) / 10000;
  }
  return dst;
}

template <typename T>
CImg<T> rgb2bgr(CImg<T> src) {
  cimg_forXY(src, x, y) {
    T r = src(x, y, 0, 0);
    // T g = src(x, y, 0, 1);
    T b = src(x, y, 0, 2);
    src(x, y, 0, 0) = b;
    src(x, y, 0, 2) = r;
  }
  return src;
}

/****************************************/

class Transformation {
 public:
  virtual ~Transformation() {}
  virtual CImg<unsigned char> Transform(CImg<unsigned char>& image) = 0;
};

class Mirror : public Transformation {
 public:
  Mirror() {}
  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    int width = image.width();
    cimg_forXYC(image, x, y, c) {
      unsigned char t = image(x, y, 0, c);
      image(x, y, 0, c) = image(width - x, y, 0, c);
      image(width - x, y, 0, c) = t;
    }
    return image;
  }
};

class Resize : public Transformation {
 public:
  Resize(int width, int height, int interpolation = 1)
      : width_(width), height_(height), interpolation_(interpolation) {}

  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    return image.get_resize(width_, height_, -100, -100, interpolation_);
  }
  int width_;
  int height_;
  int interpolation_;
};

class Grayscale : public Transformation {
 public:
  Grayscale(int num_output_channels = 1)
      : num_output_channels_(num_output_channels) {}

  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    if (image.spectrum() == 1) {
      return image;
    }
    if (image.spectrum() == 3) {
      return rgb2gray(image);
    }
  }
  int num_output_channels_;
};

class RGB2BGR : public Transformation {
 public:
  RGB2BGR() {}

  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    if (image.spectrum() == 3) {
      return rgb2bgr(image);
    }
  }
};

class Normalize : public Transformation {
 public:
  Normalize(vector<float> mean, vector<float> stddev)
      : mean_(mean), stddev_(stddev) {}
  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    assert(image.spectrum() == mean_.size() &&
           image.spectrum() == stddev_.size());

    cimg_forXYC(image, x, y, c) {
      unsigned char data = image(x, y, 0, c);
      image(x, y, 0, c) = (data - mean_[c]) / stddev_[c];
    }
    return image;
  }
  vector<float> mean_;
  vector<float> stddev_;
};

class Mean : public Transformation {
 public:
  Mean(vector<float> mean) : mean_(mean) {}
  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    assert(image.spectrum() == mean_.size());

    cimg_forXYC(image, x, y, c) {
      unsigned char data = image(x, y, 0, c);
      image(x, y, 0, c) = (data - mean_[c]);
    }
    return image;
  }
  vector<float> mean_;
};

class Scale : public Transformation {
 public:
  Scale(vector<float> stddev) : stddev_(stddev) {}
  CImg<unsigned char> Transform(CImg<unsigned char>& image) {
    assert(image.spectrum() == stddev_.size());

    cimg_forXYC(image, x, y, c) {
      unsigned char data = image(x, y, 0, c);
      image(x, y, 0, c) = (data) / stddev_[c];
    }
    return image;
  }
  vector<float> stddev_;
};

}  // namespace caffe