#include "caffe/util/nms_util.hpp"
#include <algorithm>
#include <vector>
namespace caffe {

template <typename Dtype>
Dtype IoU(const Dtype* A, const Dtype* B) {
  if (A[0] > B[2] || A[1] > B[3] || A[2] < B[0] || A[3] < B[1]) {
    return 0;
  }
  // overlapped region (= box)
  const Dtype x1 = std::max(A[0], B[0]);
  const Dtype y1 = std::max(A[1], B[1]);
  const Dtype x2 = std::min(A[2], B[2]);
  const Dtype y2 = std::min(A[3], B[3]);

  // intersection area
  const Dtype width = std::max((Dtype)0, x2 - x1 + (Dtype)1);
  const Dtype height = std::max((Dtype)0, y2 - y1 + (Dtype)1);
  const Dtype area = width * height;

  // area of A, B
  const Dtype A_area = (A[2] - A[0] + (Dtype)1) * (A[3] - A[1] + (Dtype)1);
  const Dtype B_area = (B[2] - B[0] + (Dtype)1) * (B[3] - B[1] + (Dtype)1);

  // IoU
  return area / (A_area + B_area - area);
}

template float IoU<float>(const float* A, const float* B);
template double IoU<double>(const double* A, const double* B);

template <typename Dtype>
void NMS(const Dtype* boxes, const int num_boxes, int* index_out, int* num_out,
         const int base_index, const Dtype nms_thresh, const int max_num_out) {
  int count = 0;
  std::vector<char> is_dead(num_boxes);
  for (int i = 0; i < num_boxes; ++i) {
    is_dead[i] = 0;
  }

  for (int i = 0; i < num_boxes; ++i) {
    if (is_dead[i]) {
      continue;
    }

    index_out[count++] = base_index + i;
    if (count == max_num_out) {
      break;
    }

    for (int j = i + 1; j < num_boxes; ++j) {
      if (!is_dead[j] && IoU(&boxes[i * 5], &boxes[j * 5]) > nms_thresh) {
        is_dead[j] = 1;
      }
    }
  }

  *num_out = count;
  is_dead.clear();
}

template void NMS<float>(const float* boxes, const int num_boxes,
                         int* index_out, int* num_out, const int base_index,
                         const float nms_thresh, const int max_num_out);

template void NMS<double>(const double* boxes, const int num_boxes,
                          int* index_out, int* num_out, const int base_index,
                          const double nms_thresh, const int max_num_out);
}  // namespace caffe