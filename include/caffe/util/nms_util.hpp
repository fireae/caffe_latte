#ifndef _CAFFE_UTIL_NMS_UTIL_HPP_
#define _CAFFE_UTIL_NMS_UTIL_HPP_

namespace caffe {
template <typename Dtype>
Dtype IoU(const Dtype* A, const Dtype* B);

template <typename Dtype>
void NMS(const Dtype* boxes, const int num_boxes, int* index_out, int* num_out,
         const int base_index, const Dtype nms_thresh, const int max_num_out);

}  // namespace caffe

#endif  //_CAFFE_UTIL_NMS_UTIL_HPP_