#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_pixel_max(const int pre_spatial_dim,
    const int spatial_dim, const int p2spatial_dim, Dtype* data,
    Dtype* out, unsigned int step) {
    int scaled_spatial_dim = p2spatial_dim/(2*step);
    for (int index = blockIdx.x*blockDim.x + threadIdx.x;
         index < pre_spatial_dim * scaled_spatial_dim;
         index += blockDim.x * gridDim.x) {
        int n = index / scaled_spatial_dim;
        int s = (index % scaled_spatial_dim) * step * 2;
        int index_data = n * spatial_dim + s;
        
        // Max.
        if (s + step < spatial_dim) {
            data[index_data] = max(data[index_data], data[index_data + step]);
            if (step * 2 >= spatial_dim)
                out[n] = data[index_data];
        }
    }
}

template <typename Dtype>
__global__ void kernel_pixel_subtract(const int pre_spatial_dim,
    const int spatial_dim, Dtype* data, const Dtype* pixel_max) {
  CUDA_KERNEL_LOOP(index, pre_spatial_dim * spatial_dim) {
    int n = index / spatial_dim;
    data[index] -= pixel_max[n];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_pixel_sum(const int pre_spatial_dim,
    const int spatial_dim, const int p2spatial_dim, Dtype* data,
    Dtype* pixel_sum, unsigned int step) {
    int scaled_spatial_dim = p2spatial_dim/(2*step);
    for (int index = blockIdx.x*blockDim.x + threadIdx.x;
         index < pre_spatial_dim * scaled_spatial_dim;
         index += blockDim.x * gridDim.x) {
        int n = index / scaled_spatial_dim;
        int s = (index % scaled_spatial_dim) * step * 2;
        int index_data = n * spatial_dim + s;
        
        // Add
        if (s + step < spatial_dim) {
            data[index_data] += data[index_data + step];
            if (step * 2 >= spatial_dim)
                pixel_sum[n] = data[index_data];
        }
    }
}

template <typename Dtype>
__global__ void kernel_pixel_div(const int pre_spatial_dim,
    const int spatial_dim, Dtype* data, const Dtype* pixel_sum) {
  CUDA_KERNEL_LOOP(index, pre_spatial_dim * spatial_dim) {
    int n = index / spatial_dim;
    data[index] /= pixel_sum[n];
  }
}

template <typename Dtype>
__global__ void kernel_pixel_dot(const int pre_spatial_dim,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* pixel_dot) {
  CUDA_KERNEL_LOOP(index, pre_spatial_dim) {
    Dtype dot = 0;
    for (int s = 0; s < spatial_dim; ++s) {
      dot += (data_1[index * spatial_dim + s]
          * data_2[index * spatial_dim + s]);
    }
    pixel_dot[index] = dot;
  }
}
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
void SpatialSoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num_axes = bottom[0]->num_axes();
  int pre_spatial_dim = bottom[0]->count(0, num_axes - 2);
  int spatial_dim = bottom[0]->count(num_axes - 2);
  int p2spatial_dim = pow(2,(int)ceil(log2((float)spatial_dim)));
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  Dtype* temp_data = temp_data_.mutable_gpu_data();
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_copy(bottom[0]->count(), top_data, temp_data);
  for (unsigned int step=1; step < spatial_dim; step *= 2) {
      int scaled_spatial_dim = p2spatial_dim/(2*step);
      kernel_pixel_max<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim * scaled_spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim, spatial_dim, p2spatial_dim,
          temp_data, scale_data, step);
  }
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_pixel_subtract<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim, spatial_dim, top_data,
      scale_data);
  // divide by temperature
  caffe_gpu_scal<Dtype>(pre_spatial_dim * spatial_dim, temp_, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim * spatial_dim, top_data,
      top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  caffe_copy(bottom[0]->count(), top_data, temp_data);
  for (unsigned int step=1; step < spatial_dim; step *= 2) {
      int scaled_spatial_dim = p2spatial_dim/(2*step);
      kernel_pixel_sum<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim * scaled_spatial_dim),
          CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim, spatial_dim, p2spatial_dim,
          temp_data, scale_data, step);
  }
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_pixel_div<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim, spatial_dim, top_data,
      scale_data);
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SpatialSoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num_axes = bottom[0]->num_axes();
  int pre_spatial_dim = bottom[0]->count(0, num_axes - 2);
  int spatial_dim = bottom[0]->count(num_axes - 2);
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_pixel_dot<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim, spatial_dim, top_diff, top_data,
      scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_pixel_subtract<Dtype><<<CAFFE_GET_BLOCKS(pre_spatial_dim * spatial_dim),
    CAFFE_CUDA_NUM_THREADS>>>(pre_spatial_dim, spatial_dim, bottom_diff,
    scale_data);
  // scale by temperature and elementwise multiplication
  caffe_gpu_scal<Dtype>(top[0]->count(), temp_, bottom_diff);
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialSoftmaxLayer);

}  // namespace caffe
