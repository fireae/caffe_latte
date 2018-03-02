#include "caffe/net_config.hpp"

namespace caffe {

vector<float> NetConfig::scales = {600.0};
float NetConfig::max_size = 1000;
float NetConfig::batch_size = 1;

float NetConfig::fg_fraction = 0.25;
float NetConfig::fg_thresh = 0.5;
// Overlap threshold for a ROI to be considered background (class = 0
// ifoverlap in [LO, HI))
float NetConfig::bg_thresh_hi = 0.5;
float NetConfig::bg_thresh_lo = 0.1;
bool NetConfig::use_flipped = true;

// Train bounding-box regressors
bool NetConfig::bbox_reg = true;  // Unuse
float NetConfig::bbox_thresh = 0.5;
std::string NetConfig::snapshot_infix = "";
bool NetConfig::bbox_normalize_targets = true;
float NetConfig::bbox_inside_weights[4] = {1.0, 1.0, 1.0, 1.0};
float NetConfig::bbox_normalize_means[4] = {0.0, 0.0, 0.0, 0.0};
float NetConfig::bbox_normalize_stds[4] = {0.1, 0.1, 0.2, 0.2};

// RPN to detect objects
float NetConfig::rpn_positive_overlap = 0.7;
float NetConfig::rpn_negative_overlap = 0.3;
// If an anchor statisfied by positive and negative conditions set to negative
bool NetConfig::rpn_clobber_positives = false;
float NetConfig::rpn_fg_fraction = 0.5;
int NetConfig::rpn_batchsize = 256;
float NetConfig::rpn_nms_thresh = 0.7;
int NetConfig::rpn_pre_nms_top_n = 12000;
int NetConfig::rpn_post_nms_top_n = 2000;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float NetConfig::rpn_min_size = 16;
// // Deprecated (outside weights)
// float NetConfig::rpn_bbox_inside_weights[4] = {1.0, 1.0, 1.0, 1.0};
// Give the positive RPN examples weight of p * 1 / {num positives}
// and give negatives a weight of (1 - p)
// Set to -1.0 to use uniform example weighting
float NetConfig::rpn_positive_weight = -1.0;
float NetConfig::rpn_allowed_border = 0.0;

// ======================================== Test
std::vector<float> NetConfig::test_scales = {600.0};
float NetConfig::test_max_size = 1000;
float NetConfig::test_nms = 0.3;

bool NetConfig::test_bbox_reg = true;
// RPN to detect objects
float NetConfig::test_rpn_nms_thresh = 0.7;
int NetConfig::test_rpn_pre_nms_top_n = 6000;
int NetConfig::test_rpn_post_nms_top_n = 300;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float NetConfig::test_rpn_min_size = 16;

// ========================================
// Means PIXEL
float NetConfig::pixel_means[3] = {102.9801, 115.9465, 122.7717};  // BGR
int NetConfig::rng_seed = 3;
float NetConfig::eps = 1e-14;
float NetConfig::inf;

// ========================================
int NetConfig::feat_stride = 16;
std::vector<float> NetConfig::anchors;
float NetConfig::test_score_thresh = 0.7;
int NetConfig::n_classes = 21;
int NetConfig::iter_test = 1000;

void NetConfig::LoadParam(const NetParameter& net_param) {
#if 0
  const NetConfigParameter& config = net_param.net_config_param();
  max_size = config.max_size();
  batch_size = config.batch_size();
  fg_fraction = config.fg_fraction();
  fg_thresh = config.fg_thresh();
  bg_thresh_hi = config.bg_thresh_hi();
  bg_thresh_lo = config.bg_thresh_lo();
  use_flipped = config.use_flipped();
  bbox_reg = config.bbox_reg();
  bbox_thresh = config.bbox_thresh();
  bbox_normalize_targets = config.bbox_normalize_targets();
  rpn_positive_overlap = config.rpn_positive_overlap();
  rpn_negative_overlap = config.rpn_negative_overlap();
  rpn_clobber_positives = config.rpn_clobber_positives();
  rpn_fg_fraction = config.rpn_fg_fraction();
  rpn_batchsize = config.rpn_batchsize();
  rpn_nms_thresh = config.rpn_nms_thresh();
  rpn_pre_nms_top_n = config.rpn_pre_nms_top_n();
  rpn_post_nms_top_n = config.rpn_post_nms_top_n();
  rpn_min_size = config.rpn_min_size();
  rpn_positive_weight = config.rpn_positive_weight();
  rpn_allowed_border = config.rpn_allowed_border();

  for (int i = 0; i < 4; i++) {
    if (i < config.bbox_normalize_mean_size()) {
      bbox_normalize_means[i] = config.bbox_normalize_mean(i);
    }
    if (i < config.bbox_normalize_std_size()) {
      bbox_normalize_stds[i] = config.bbox_normalize_std(i);
    }
    if (i < config.bbox_inside_weight_size()) {
      bbox_inside_weights[i] = config.bbox_inside_weight(i);
    }
  }
#endif
}

}  // namespace caffe