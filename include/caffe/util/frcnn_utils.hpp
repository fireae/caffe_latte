#ifndef CAFFE_UTIL_FRCNN_UTILS_HPP_
#define CAFFE_UTIL_FRCNN_UTILS_HPP_
#include <algorithm>  // std::max
#include <cmath>
#include <string>
#include <vector>
#include "caffe/common.hpp"

namespace caffe {
using std::vector;

const double kEPS = 1e-14;

template <typename Dtype>
void EnumerateProposals(const Dtype* scores, const Dtype* bboxes,
                        const Dtype* anchors, const int num_anchors,
                        Dtype* proposals, const int bottom_h,
                        const int bottom_w, const Dtype img_h,
                        const Dtype img_w, const Dtype min_box_h,
                        const Dtype min_box_w, const int feat_stride_h,
                        const int feat_stride_w);
template <typename Dtype>
void SortBox(Dtype* boxes, const int start, const int end, const int num_top);

template <typename Dtype>
void RetrieveROI(const int num_rois, const int item_index,
  const Dtype *proposals, const int * roi_indices, Dtype* rois, Dtype* roi_scores);

}  // namespace caffe

#endif  // CAFFE_UTIL_FRCNN_UTILS_HPP_