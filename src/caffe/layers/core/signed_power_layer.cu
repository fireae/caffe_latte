#include <vector>

#include "caffe/layers/signed_power_layer.hpp"
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
void caffe_gpu_signed_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
__global__ void signed_powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      if (a[index] >= 0) {
        y[index] = pow(a[index], alpha);
      } else {
        y[index] = -pow(-a[index], alpha);
      }
  }
}

template <>
void caffe_gpu_signed_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  signed_powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_signed_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  signed_powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <typename Dtype>
__global__ void calc_top_kernel(const int n, const Dtype* x,
                                  const Dtype shift,
                                  const Dtype scale,
                                  const Dtype* power,
                                  const Dtype eps,
                                  Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      Dtype ax_b = scale * x[index] + shift;
      if (ax_b >= 0) {
        y[index] = pow(ax_b + eps, power[0]);
      } else {
        y[index] = - pow( - ax_b + eps, power[0]);
      }
  }
}

template <typename Dtype>
__global__ void calc_diff_kernel(const int n,
                                  const Dtype* x,
                                  const Dtype shift,
                                  const Dtype scale,
                                  const Dtype* power,
                                  const Dtype eps,
                                  Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      Dtype ax_b = scale * x[index] + shift;
      y[index] = power[0] * scale * pow(abs(ax_b) + eps, power[0] - 1);
  }
}


template <typename Dtype>
__global__ void prepare_power_diff_kernel(const int n, 
                                  const Dtype* x,
                                  const Dtype shift,
                                  const Dtype scale,
                                  const Dtype eps,
                                  Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
      y[index] = log(abs(scale * x[index] + shift) + eps);
  }
}

    
template <typename Dtype>
void SignedPowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* power_param = this->blobs_[0].get()->gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  calc_top_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
                                bottom_data,
                                shift_,
                                scale_,
                                power_param,
                                eps_,
                                top_data);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

template <typename Dtype>
void SignedPowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  const int count = bottom[0]->count();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* power_diff = this->blobs_[0].get()->mutable_cpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    prepare_power_diff_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
                                bottom_data,
                                shift_,
                                scale_,
                                eps_,
                                bottom_diff);
    caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
    caffe_gpu_dot(count, bottom_diff, top_data, power_diff);
  }
  if (propagate_down[0]) {
    const Dtype power_param_cpu = this->blobs_[0].get()->cpu_data()[0];
    if (diff_scale_ == Dtype(0) || power_param_cpu == Dtype(1)) {
      caffe_gpu_set(count, diff_scale_, bottom_diff);
    } else {
        const Dtype* power_param_gpu = this->blobs_[0].get()->gpu_data();
        // Compute dy/dx = scale * power * (abs(shift + scale * x) + eps)^(power - 1)
        // NOLINT_NEXT_LINE(whitespace/operators)
        calc_diff_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
                                  bottom_data,
                                  shift_,
                                  scale_,
                                  power_param_gpu,
                                  eps_,
                                  bottom_diff);
    }
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
  }
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
}

INSTANTIATE_LAYER_GPU_FUNCS(SignedPowerLayer);


}  // namespace caffe
