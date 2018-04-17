#include "caffe/c_api.h"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"

CAFFE_API int CaffeBlobNum(BlobHandle blob) {
  return static_cast<caffe::Blob<float>*>(blob)->num();
}
CAFFE_API int CaffeBlobChannels(BlobHandle blob) {
  return static_cast<caffe::Blob<float>*>(blob)->channels();
}
CAFFE_API int CaffeBlobHeight(BlobHandle blob) {
  return static_cast<caffe::Blob<float>*>(blob)->height();
}
CAFFE_API int CaffeBlobWidth(BlobHandle blob) {
  return static_cast<caffe::Blob<float>*>(blob)->width();
}
CAFFE_API int CaffeBlobCount(BlobHandle blob) {
  return static_cast<caffe::Blob<float>*>(blob)->count();
}
CAFFE_API int CaffeBlobReshape(BlobHandle blob, int shape_size, int* shape) {
  std::vector<int> shape_data(shape, shape + shape_size);
  static_cast<caffe::Blob<float>*>(blob)->Reshape(shape_data);
  return 0;
}
CAFFE_API int CaffeBlobShape(BlobHandle blob, int* shape_size, int** shape) {
  vector<int> s = static_cast<caffe::Blob<float>*>(blob)->shape();
  *shape_size = s.size();
  for (int i = 0; i < *shape_size; i++) {
    (*shape)[i] = s[i];
  }
  return 0;
}

// Net API
CAFFE_API int CaffeNetCreate(const char* net_path, const char* model_path,
                             NetHandle* net) {
  caffe::Net<float>* net_ = new caffe::Net<float>(net_path);
  net_->CopyTrainedLayersFrom(model_path);
  *net = static_cast<NetHandle>(net_);
  return 0;
}
CAFFE_API int CaffeNetCreateFromBuffer(const char* net_buffer, int nb_len,
                                       const char* model_buffer, int mb_len,
                                       NetHandle* net) {
  return 0;
}
CAFFE_API int CaffeNetDestroy(NetHandle net) {
  delete static_cast<caffe::Net<float>*>(net);
}
CAFFE_API int CaffeNetMarkOutput(NetHandle net, const char* name) { return 0; }
CAFFE_API int CaffeNetForward(NetHandle net) {
  static_cast<caffe::Net<float>*>(net)->Forward();
  return 0;
}
CAFFE_API int CaffeNetGetBlob(NetHandle net, const char* name,
                              BlobHandle* blob) {
  std::shared_ptr<caffe::Blob<float>> blob_ =
      static_cast<caffe::Net<float>*>(net)->blob_by_name(name);
  *blob = static_cast<BlobHandle>(blob_.get());
  return 0;
}
CAFFE_API int CaffeNetListBlob(NetHandle net, int* n, const char*** names,
                               BlobHandle** blobs) {
  return 0;
}

CAFFE_API int CaffeNetListParam(NetHandle net, int* n, const char*** names,
                                BlobHandle** params) {
  return 0;
}
