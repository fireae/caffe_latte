#ifndef CAFFE_C_API_H_
#define CAFFE_C_API_H_

#ifdef _MSC_VER
#ifdef CAFFE_EXPORTS
#define CAFFE_API __declspec(dllexport)
#else
#define CAFFE_API __declspec(dllimport)
#endif
#else
#define CAFFE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif  //__cplusplus

typedef float real_t;
typedef void* BlobHandle;
typedef void* NetHandle;

CAFFE_API int CaffeBlobNum(BlobHandle blob);
CAFFE_API int CaffeBlobChannels(BlobHandle blob);
CAFFE_API int CaffeBlobWidth(BlobHandle blob);
CAFFE_API int CaffeBlobHeight(BlobHandle blob);
CAFFE_API real_t* CaffeBlobData(BlobHandle blob);
CAFFE_API int CaffeBlobCount(BlobHandle blob);

CAFFE_API int CaffeBlobReshape(BlobHandle blob, int shape_size, int* shape);
CAFFE_API int CaffeBlobShape(BlobHandle blob, int* shape_size, int** shape);

CAFFE_API int CaffeNetCreaet(const char* net_path, const char* model_path,
                             NetHandle* net);

CAFFE_API int CaffeNetCreateFromBuffer(const char* net_buf, int net_buf_len,
                                       const char* model_buf, int model_buf_len,
                                       NetHandle* net);

CAFFE_API int CaffeNetDestroy(NetHandle net);

CAFFE_API int CaffeNetForward(NetHandle net);
CAFFE_API int CaffeNetBackward(NetHandle net);
CAFFE_API int CaffeNetGetBlob(NetHandle net, const char* blob_name,
                              BlobHandle* blob);

CAFFE_API int CaffeNetListBlobs(NetHandle net, int* n, const char*** names,
                                BlobHandle** blobs);

CAFFE_API int CaffeNetListParams(NetHandle net, int* n, const char*** names,
                                 BlobHandle** params);

CAFFE_API int CaffeSetMode(int mode, int device);

#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  // CAFFE_C_API_H_