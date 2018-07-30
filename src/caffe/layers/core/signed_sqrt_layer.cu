#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/signed_sqrt_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}  
  
template <typename Dtype>
__global__ void caffe_gpu_signed_sqrt(const int nthreads,
        const Dtype* src, Dtype* dst) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        if (src[index] >= 0)
            dst[index] = sqrt(src[index]);
        else
            dst[index] = -sqrt(-src[index]);
    }
}


template <typename Dtype>
void SignedSqrtLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_gpu_signed_sqrt<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, top_data);
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SignedSqrtLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[0]) {
        const int count = bottom[0]->count();

        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* top_data = top[0]->gpu_data();

        caffe_gpu_abs(count, top_data, bottom_diff);
        caffe_gpu_add_scalar(count, epsilon, bottom_diff);
        caffe_gpu_div(count, top_diff, bottom_diff, bottom_diff);
        caffe_gpu_scal(count, Dtype(0.5), bottom_diff);
      }
      
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
      CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SignedSqrtLayer);


}  // namespace caffe
