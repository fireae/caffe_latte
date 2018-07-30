#include <algorithm>
#include <vector>

#include "caffe/layers/gaussian_attention_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void CreateMask(const int nthreads,
    const Dtype* const attention_locs,
    const int num, const int num_locs,
    const int height, const int width, 
    const Dtype sigma, 
    Dtype* const mask) {
  // For each mask entry
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int l = (index / width / height) % num_locs;
    const int n = index / width / height / num_locs;
    // dim=0 is along width and dim=1 along height, with -inf towards the top left
    Dtype cur_attention[2] = { attention_locs[n*(2*num_locs) + 2*l], attention_locs[n*(2*num_locs) +  2*l+1]};
    Dtype cur_loc[2] = { Dtype(w) / Dtype(width-1), Dtype(h) / Dtype(height-1)};
    
    mask[index] = Dtype(1.0);
    Dtype y;
    for (int dim = 0; dim < 2; dim++)
    {
      // Make cure everything is within valid ranged
      cur_attention[dim] = min(Dtype(1.0),max(Dtype(-1.0),cur_attention[dim]));
      // Transform the current location to be in [-1,1]
      cur_loc[dim] = cur_loc[dim]*Dtype(2.0)-Dtype(1.0);
      // The input location for scipy.stats.norm.pdf(x,loc=loc,scale=sigma)
      y = (cur_loc[dim] - cur_attention[dim]) / sigma;
      mask[index] = mask[index] * ( exp(-y * y / Dtype(2.0) ) / sigma / sqrt(Dtype(2.0 * 3.141592653589793)) );
    }
  }
}   

template <typename Dtype>
__global__ void CreateOffsets(const int nthreads,
    const Dtype* const attention_locs,
    const int num, const int num_locs,
    const int height, const int width, 
    const int dim, 
    const Dtype sigma, 
    Dtype* const offset) {
  // For each offset entry
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const int l = (index / width / height) % num_locs;
    const int n = index / width / height / num_locs;
    Dtype cur_attention[2] = { attention_locs[n*(2*num_locs) + 2*l], attention_locs[n*(2*num_locs) +  2*l + 1]};
    Dtype cur_loc[2] = { Dtype(w) / Dtype(width-1), Dtype(h) / Dtype(height-1)};
    
    // Make cure everything is within valid ranged
    cur_attention[dim] = min(Dtype(1.0),max(Dtype(-1.0),cur_attention[dim]));
    // Transform the current location to be in [-1,1]
    cur_loc[dim] = cur_loc[dim]*Dtype(2.0)-Dtype(1.0);
    // The input location for scipy.stats.norm.pdf(x,loc=loc,scale=sigma)
    offset[index] = (cur_loc[dim] - cur_attention[dim]) / (sigma * sigma);
  }
}   
  
  
template <typename Dtype>
__global__ void MultiplyMask(const int nthreads,
    const Dtype* const bottom_data,
    const Dtype* const mask,
    const int channels,
    const int height, const int width, 
    const int num_locs, const int channel_stride,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    
    const int mask_id = c / channel_stride;
    const int mask_index = n*num_locs*width*height + mask_id*width*height + (index%(width*height));
    
    top_data[index] = bottom_data[index] * mask[mask_index];
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
void GaussianAttentionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* attention_locs = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  Dtype* mask_data = mask_.mutable_gpu_data();
  
  // NOLINT_NEXT_LINE(whitespace/operators)
  CreateMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      mask_.count(), attention_locs,
      mask_.num(), num_locs_,
      height_, width_, 
      sigma_, mask_data);
  
  const Dtype* const_mask_data = mask_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MultiplyMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, mask_data,
      channels_,
      height_, width_, 
      // the number of locs
      num_locs_, channel_stride_,
      top_data);
  
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void GaussianAttentionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* attention_locs = bottom[1]->gpu_data();
  const Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* top_diff = top[0]->mutable_gpu_diff();
  int count = top[0]->count();
  const Dtype* const_mask_data = mask_.gpu_data();
  const Dtype* ones = ones_.gpu_data();
  // The result needs to be on CPU as required by caffe_gpu_dot
  Dtype* loc_diff = bottom[1]->mutable_cpu_diff();
  
  // Diff wrt to attention locs
  if (propagate_down[1]) {
      for (int dim = 0; dim<2; dim++)
      {		
	  Dtype* tmp_data = tmp_.mutable_gpu_data();
	  // Create the offsets for this dim
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  CreateOffsets<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	      tmp_.count(), attention_locs, 
	      tmp_.num(), num_locs_,
	      height_, width_, 
	      dim, sigma_, tmp_data);
	  
	  
	  
	  // Multiply with top data and top_diff
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  MultiplyMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	      count, top_data, tmp_data,
	      channels_,
	      height_, width_, 
	      // the number of locs
	      num_locs_, channel_stride_,
	      bottom_diff);
	  caffe_gpu_mul<Dtype>(count,bottom_diff,top_diff,bottom_diff);
	  	  
	  // Reduce the blocks to the part diff 
	  const int n = channel_stride_*width_*height_;
	  for (int i=0;i <num_*num_locs_; i++) 
	  {
	      caffe_gpu_dot<Dtype>(n,ones,bottom_diff+i*n,loc_diff+2*i+dim);
	  }
      }
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
  }
  
  if (propagate_down[0]) {
      bottom_diff = bottom[0]->mutable_gpu_diff();
      // The diff wrt to input data 
      // NOLINT_NEXT_LINE(whitespace/operators)
      MultiplyMask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
	  count, top_diff, const_mask_data,
	  channels_,
	  height_, width_, 
	  // the number of locs
	  num_locs_, channel_stride_,
	  bottom_diff);
  }

  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(GaussianAttentionLayer);


}  // namespace caffe
