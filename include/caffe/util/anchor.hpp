#ifndef ANCHOR_TEXT_HPP_
#define ANCHOR_TEXT_HPP_
#include <vector>
#include "caffe/common.hpp"
namespace caffe {
using std::vector;

template <typename Dtype>
class AnchorText {
 public:
  AnchorText() {}
  ~AnchorText() {}

  vector<vector<int> > generate_basic_anchors(vector<vector<int> > sizes,
                                              int base_size = 16);
  vector<int> scale_anchor(vector<int> anchor, int h, int w);

  vector<vector<int> > basic_anchors();

  vector<vector<int> > locate_anchors(vector<int> feat_map_size,
                                      int feat_stride);

  vector<vector<Dtype> > apply_deltas_to_anchors(
      vector<vector<Dtype> > boxes_delta, vector<vector<int> > anchors);
};

}  // namespace caffe

#endif  // ANCHOR_TEXT_HPP_
