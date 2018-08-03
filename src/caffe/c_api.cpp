#include "caffe/c_api.h"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
using namespace caffe;
#define API_BEGIN() try {
#define API_END()            \
  }                          \
  catch (std::exception e) { \
  }                          \
  return 0;

int CaffeBlobNum(BlobHandle blob) {
  return static_cast<Blob<real_t>*>(blob)->num();
}

int CaffeBlobChannels(BlobHandle blob) {
  return static_cast<Blob<real_t>*>(blob)->channels();
}

int CaffeBlobWidth(BlobHandle blob) {
  return static_cast<Blob<real_t>*>(blob)->width();
}

int CaffeBlobHeight(BlobHandle blob) {
  return static_cast<Blob<real_t>*>(blob)->height();
}

real_t* CaffeBlobData(BlobHandle blob) {
  return static_cast<Blob<real_t>*>(blob)->mutable_cpu_data();
}

int CaffeBlobCount(BlobHandle blob) {
  return static_cast<Blob<real_t>*>(blob)->count();
}

int CaffeBlobReshape(BlobHandle blob, int shape_size, int* shape) {
  API_BEGIN();
  std::vector<int> shape_data(shape, shape + shape_size);
  static_cast<Blob<real_t>*>(blob)->Reshape(shape_data);
  API_END();
}