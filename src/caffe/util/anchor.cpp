#include "caffe/util/anchor.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
vector<vector<int> > AnchorText<Dtype>::generate_basic_anchors(
    vector<vector<int> > sizes, int base_size) {
  vector<int> base_anchor = {0, 0, base_size - 1, base_size - 1};
  vector<vector<int> > anchors(sizes.size());
  for (int i = 0; i < sizes.size(); ++i) {
    anchors[i] = scale_anchor(base_anchor, sizes[i][0], sizes[i][1]);
  }

  return anchors;
}

template <typename Dtype>
vector<int> AnchorText<Dtype>::scale_anchor(vector<int> anchor, int h, int w) {
  int x_ctr = (anchor[0] + anchor[2]) * 0.5;
  int y_ctr = (anchor[1] + anchor[3]) * 0.5;
  vector<int> scaled_anchor = anchor;
  scaled_anchor[0] = x_ctr - w / 2 + 0.5;
  scaled_anchor[2] = x_ctr + w / 2;
  scaled_anchor[1] = y_ctr - h / 2 + 0.5;
  scaled_anchor[3] = y_ctr + h / 2;
  return scaled_anchor;
}

template <typename Dtype>
vector<vector<int> > AnchorText<Dtype>::basic_anchors() {
  vector<int> heights = {11, 16, 23, 33, 48, 68, 97, 139, 198, 283};
  vector<int> widths = {16};
  vector<vector<int> > sizes;
  for (int h = 0; h < heights.size(); h++) {
    for (int w = 0; w < widths.size(); w++) {
      vector<int> sz;
      sz.push_back(heights[h]);
      sz.push_back(widths[w]);
      sizes.push_back(sz);
    }
  }

  return generate_basic_anchors(sizes);
}

template <typename Dtype>
vector<vector<int> > AnchorText<Dtype>::locate_anchors(
    vector<int> feat_map_size, int feat_stride) {
  vector<vector<int> > basic_anchors_ = basic_anchors();
  vector<vector<int> > anchors(basic_anchors_.size() * feat_map_size[0] *
                               feat_map_size[1]);
  int index = 0;
  for (int y = 0; y < feat_map_size[0]; y++) {
    for (int x = 0; x < feat_map_size[1]; x++) {
      vector<int> shift = {x * feat_stride, y * feat_stride, x * feat_stride,
                           y * feat_stride};
      for (int i = 0; i < basic_anchors_.size(); i++) {
        vector<int> shift_anchor;
        shift_anchor.push_back(basic_anchors_[i][0] + shift[0]);
        shift_anchor.push_back(basic_anchors_[i][1] + shift[1]);
        shift_anchor.push_back(basic_anchors_[i][2] + shift[2]);
        shift_anchor.push_back(basic_anchors_[i][3] + shift[3]);
        anchors[index + i] = shift_anchor;
      }
      index += basic_anchors_.size();
    }
  }

  return anchors;
}

template <typename Dtype>
vector<vector<Dtype> > AnchorText<Dtype>::apply_deltas_to_anchors(
    vector<vector<Dtype> > boxes_delta, vector<vector<int> > anchors) {
  vector<vector<Dtype> > global_coords;
  for (int i = 0; i < anchors.size(); i++) {
    int anchor_y_ctr = (anchors[i][1] + anchors[i][3]) * 0.5;
    int anchor_h = (anchors[i][3] - anchors[i][1]) + 1;
    vector<Dtype> global_coord(4);
    global_coord[1] = std::exp(boxes_delta[i][1]) * anchor_h;
    global_coord[0] =
        boxes_delta[i][0] * anchor_h + anchor_y_ctr - global_coord[1] / 2.0;
    vector<Dtype> ret_coord(4);
    ret_coord[0] = anchors[i][0];
    ret_coord[1] = global_coord[0];
    ret_coord[2] = anchors[i][2];
    ret_coord[3] = global_coord[0] + global_coord[1];
    global_coords.push_back(ret_coord);
  }

  return global_coords;
}

template class AnchorText<float>;
template class AnchorText<double>;
}  // namespace caffe
