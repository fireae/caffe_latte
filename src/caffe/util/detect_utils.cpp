#include "caffe/util/detect_utils.hpp"

namespace caffe {

template <typename Dtype>
Dtype ComputeIOU(const RectBox<Dtype>& a, const RectBox<Dtype>& b) {
  const Dtype xx1 = std::max(a[0], b[0]);
  const Dtype yy1 = std::max(a[1], b[1]);
  const Dtype xx2 = std::min(a[2], b[2]);
  const Dtype yy2 = std::min(a[3], b[3]);
  Dtype inter =
      std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  Dtype areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return inter / (areaA + areaB - inter);
}

template float ComputeIOU(const RectBox<float>& a, const RectBox<float>& b);
template double ComputeIOU(const RectBox<double>& a, const RectBox<double>& b);

template <typename Dtype>
Dtype ComputeIOU(const BBox<Dtype>& a, const BBox<Dtype>& b) {
  const Dtype xx1 = std::max(a[0], b[0]);
  const Dtype yy1 = std::max(a[1], b[1]);
  const Dtype xx2 = std::min(a[2], b[2]);
  const Dtype yy2 = std::min(a[3], b[3]);
  Dtype inter =
      std::max(Dtype(0), xx2 - xx1 + 1) * std::max(Dtype(0), yy2 - yy1 + 1);
  Dtype areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  Dtype areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return inter / (areaA + areaB - inter);
}

template float ComputeIOU(const BBox<float>& a, const BBox<float>& b);
template double ComputeIOU(const BBox<double>& a, const BBox<double>& b);

template <typename Dtype>
std::vector<Dtype> ComputeIOUs(const RectBox<Dtype>& a,
                               const std::vector<RectBox<Dtype>>& b) {
  vector<Dtype> ious;
  for (int i = 0; i < b.size(); i++) {
    ious.push_back(ComputeIOU(a, b[i]));
  }
  return ious;
}

template std::vector<float> ComputeIOUs(const RectBox<float>& a,
                                        const std::vector<RectBox<float>>& b);
template std::vector<double> ComputeIOUs(const RectBox<double>& a,
                                         const std::vector<RectBox<double>>& b);

template <typename Dtype>
std::vector<std::vector<Dtype>> ComputeIOUs(
    const std::vector<RectBox<Dtype>>& a, const std::vector<RectBox<Dtype>>& b,
    bool use_gpu) {
  vector<vector<Dtype>> ious;
  for (size_t i = 0; i < a.size(); i++) {
    ious.push_back(ComputeIOUs(a[i], b));
  }
  return ious;
}

template std::vector<std::vector<float>> ComputeIOUs(
    const std::vector<RectBox<float>>& a, const std::vector<RectBox<float>>& b,
    bool use_gpu);
template std::vector<std::vector<double>> ComputeIOUs(
    const std::vector<RectBox<double>>& a,
    const std::vector<RectBox<double>>& b, bool use_gpu);

template <typename Dtype>
vector<BBox<Dtype>> VoteBbox(const vector<BBox<Dtype>>& dets_NMS,
                             const vector<BBox<Dtype>>& dets_all,
                             Dtype iou_thresh, Dtype add_val) {
  unsigned int N = dets_NMS.size();
  unsigned int M = dets_all.size();
  vector<BBox<Dtype>> dets_voted(N);

  for (size_t i = 0; i < N; i++) {
    Dtype acc_score = 1e-8;
    BBox<Dtype> acc_box;
    for (size_t j = 0; j < M; j++) {
      if (ComputeIOU(dets_NMS[i], dets_all[j]) < iou_thresh) continue;
      Dtype score = dets_all[j].confidence + add_val;  //[4]
      // score = std::max(Dtype(0), score); // neighbor score
      // acc_box += dets_all[4] * dets_all[0:4]
      acc_box[0] += score * dets_all[j][0];
      acc_box[1] += score * dets_all[j][1];
      acc_box[2] += score * dets_all[j][2];
      acc_box[3] += score * dets_all[j][3];
      acc_score += score;
    }
    dets_voted[i][0] = acc_box[0] / acc_score;
    dets_voted[i][1] = acc_box[1] / acc_score;
    dets_voted[i][2] = acc_box[2] / acc_score;
    dets_voted[i][3] = acc_box[3] / acc_score;
    dets_voted[i].confidence =
        dets_NMS[i].confidence;         // Keep the original score
    dets_voted[i].id = dets_NMS[i].id;  //[5] class id
  }
  return dets_voted;
}

template vector<BBox<float>> VoteBbox(const vector<BBox<float>>&,
                                      const vector<BBox<float>>&,
                                      float iou_thresh, float add_val);
template vector<BBox<double>> VoteBbox(const vector<BBox<double>>&,
                                       const vector<BBox<double>>&,
                                       double iou_thresh, double add_val);

float get_scale_factor(int width, int height, int short_size,
                       int max_long_size) {
  float im_size_min = std::min(width, height);
  float im_size_max = std::max(width, height);

  float scale_factor = static_cast<float>(short_size) / im_size_min;
  // Prevent the biggest axis from being more than max_size
  if (scale_factor * im_size_max > max_long_size) {
    scale_factor = static_cast<float>(max_long_size) / im_size_max;
  }
  return scale_factor;
}
}  // namespace caffe