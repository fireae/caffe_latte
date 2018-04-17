#ifndef CAFFE_C_API_H_
#define CAFFE_C_API_H_

#ifdef _MSC_VER
	#ifdef CAFFE_EXPORTS
		#define CAFFE_API __declspec(dllexport)
	#else
		#define CAFFE_API __declspec(dllimport)
	#endif // CAFFE_EXPORTS
#else
	#define CAFFE_API
#endif

#ifdef __cpluscplus 
extern "C" {
#endif 

typedef float real_t;
typedef void* BlobHandle;
typedef void* NetHandle;

// Blob API
CAFFE_API int CaffeBlobNum(BlobHandle blob);
CAFFE_API int CaffeBlobChannels(BlobHandle blob);
CAFFE_API int CaffeBlobHeight(BlobHandle blob);
CAFFE_API int CaffeBlobWidth(BlobHandle blob);
CAFFE_API int CaffeBlobCount(BlobHandle blob);
CAFFE_API int CaffeBlobReshape(BlobHandle blob, int shape_size, int* shape);
CAFFE_API int CaffeBlobShape(BlobHandle blob, int* shape_size, int** shape);

//Net API
CAFFE_API int CaffeNetCreate(const char* net_path, const char* model_path, NetHandle* net);
CAFFE_API int CaffeNetCreateFromBuffer(const char* net_buffer, 
	int nb_len, const char*model_buffer, int mb_len, NetHandle* net);
CAFFE_API int CaffeNetDestroy(NetHandle net);
CAFFE_API int CaffeNetMarkOutput(NetHandle net, const char* name);
CAFFE_API int CaffeNetForward(NetHandle net);
CAFFE_API int CaffeNetGetBlob(NetHandle net, const char* name, BlobHandle* blob);
CAFFE_API int CaffeNetListBlob(NetHandle net, int *n, const char*** names, BlobHandle** blobs);
CAFFE_API int CaffeNetListParam(NetHandle net, int*n, const char*** names, BlobHandle** params);

#ifdef __cpluscplus 
}
#endif 

#endif // CAFFE_C_API_H_