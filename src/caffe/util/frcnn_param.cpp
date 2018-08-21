#include "caffe/util/frcnn_param.hpp"
#include "caffe/common.hpp"
#include "json.hpp"

using json = nlohmann::json;

namespace caffe {
namespace frcnn {

std::string float_to_string(const std::vector<float> data) {
  char buff[200];
  std::string ans;
  for (size_t index = 0; index < data.size(); index++) {
    snprintf(buff, sizeof(buff), "%.2f", data[index]);
    if (index == 0)
      ans = std::string(buff);
    else
      ans += ", " + std::string(buff);
  }
  return ans;
}

std::string float_to_string(const float *data) {
  const int n = sizeof(data) / sizeof(data[0]);
  return float_to_string(std::vector<float>(data, data + n));
}

template <typename Dtype>
Dtype Extract(const json &jfile, std::string key) {
  LOG(INFO) << "key is " << key;
  auto v = jfile.find(key);
  if (v == jfile.end()) {
    LOG(FATAL) << "Can not find key" << key;
  }
  return jfile[key].get<Dtype>();
}

template <typename Dtype>
Dtype Extract(const json &jfile, std::string key, Dtype default_value) {
  LOG(INFO) << "key is " << key;
  auto v = jfile.find(key);
  if (v == jfile.end()) {
    LOG(INFO) << "Can not find key" << key;
    return default_value;
  }

  return jfile[key].get<Dtype>();
}

// FrcnnParam
std::vector<float> FrcnnParam::scales;
float FrcnnParam::max_size;
float FrcnnParam::batch_size;

float FrcnnParam::fg_fraction;
float FrcnnParam::fg_thresh;
// Overlap threshold for a ROI to be considered background (class = 0
// ifoverlap in [LO, HI))
float FrcnnParam::bg_thresh_hi;
float FrcnnParam::bg_thresh_lo;
bool FrcnnParam::use_flipped;
// fyk
int FrcnnParam::use_hist_equalize;
bool FrcnnParam::use_haze_free;
bool FrcnnParam::use_retinex;
float FrcnnParam::data_jitter;
float FrcnnParam::data_rand_scale;
bool FrcnnParam::data_rand_rotate;
float FrcnnParam::data_saturation;
float FrcnnParam::data_hue;
float FrcnnParam::data_exposure;

int FrcnnParam::im_size_align;
int FrcnnParam::roi_canonical_scale;
int FrcnnParam::roi_canonical_level;

int FrcnnParam::test_soft_nms;
bool FrcnnParam::test_use_gpu_nms;
bool FrcnnParam::test_bbox_vote;
bool FrcnnParam::test_decrypt_model;

// Train bounding-box regressors
bool FrcnnParam::bbox_reg;  // Unuse
float FrcnnParam::bbox_thresh;
std::string FrcnnParam::snapshot_infix;
bool FrcnnParam::bbox_normalize_targets;
vector<float> FrcnnParam::bbox_inside_weights;   //[4]
vector<float> FrcnnParam::bbox_normalize_means;  //[4]
vector<float> FrcnnParam::bbox_normalize_stds;   //[4]

// RPN to detect objects
float FrcnnParam::rpn_positive_overlap;
float FrcnnParam::rpn_negative_overlap;
// If an anchor statisfied by positive and negative conditions set to negative
bool FrcnnParam::rpn_clobber_positives;
float FrcnnParam::rpn_fg_fraction;
int FrcnnParam::rpn_batchsize;
float FrcnnParam::rpn_nms_thresh;
int FrcnnParam::rpn_pre_nms_top_n;
int FrcnnParam::rpn_post_nms_top_n;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float FrcnnParam::rpn_min_size;
// Deprecated (outside weights)
vector<float> FrcnnParam::rpn_bbox_inside_weights;  //[4]
// Give the positive RPN examples weight of p * 1 / {num positives}
// and give negatives a weight of (1 - p)
// Set to -1.0 to use uniform example weighting
float FrcnnParam::rpn_positive_weight;
float FrcnnParam::rpn_allowed_border;

// ======================================== Test
std::vector<float> FrcnnParam::test_scales;
float FrcnnParam::test_max_size;
float FrcnnParam::test_nms;

bool FrcnnParam::test_bbox_reg;
// RPN to detect objects
float FrcnnParam::test_rpn_nms_thresh;
int FrcnnParam::test_rpn_pre_nms_top_n;
int FrcnnParam::test_rpn_post_nms_top_n;
// Proposal height and width both need to be greater than RPN_MIN_SIZE (at
// orig image scale)
float FrcnnParam::test_rpn_min_size;

// ========================================
// Means PIXEL
vector<float> FrcnnParam::pixel_means;  // [3]BGR
int FrcnnParam::rng_seed;
float FrcnnParam::eps;
float FrcnnParam::inf;

// ========================================
int FrcnnParam::feat_stride;
std::vector<float> FrcnnParam::anchors;
float FrcnnParam::test_score_thresh;
float FrcnnParam::test_rpn_score_thresh;  // fyk speed up for NMS
int FrcnnParam::n_classes;
int FrcnnParam::iter_test;

void FrcnnParam::LoadParam(const std::string default_config_path) {
  std::vector<float> v_tmp;
  // read a JSON file
  std::ifstream json_file(default_config_path);
  json default_map;
  json_file >> default_map;
  FrcnnParam::scales = Extract<std::vector<float>>(default_map, "scales");
  LOG(INFO) << FrcnnParam::scales[0];
  FrcnnParam::max_size = Extract<float>(default_map, "max_size");
  FrcnnParam::batch_size = Extract<float>(default_map, "batch_size", 1);

  FrcnnParam::fg_fraction = Extract<float>(default_map, "fg_fraction");
  FrcnnParam::fg_thresh = Extract<float>(default_map, "fg_thresh");
  FrcnnParam::bg_thresh_hi = Extract<float>(default_map, "bg_thresh_hi");
  FrcnnParam::bg_thresh_lo = Extract<float>(default_map, "bg_thresh_lo");
  FrcnnParam::use_flipped = Extract<bool>(default_map, "use_flipped", false);
  // fyk: data enhancement & augmentation
  FrcnnParam::use_retinex = Extract<bool>(default_map, "use_retinex", false);
  FrcnnParam::use_haze_free =
      Extract<bool>(default_map, "use_haze_free", false);
  FrcnnParam::use_hist_equalize =
      Extract<int>(default_map, "use_hist_equalize", 0);
  FrcnnParam::data_jitter = Extract<float>(default_map, "data_jitter", -1);
  FrcnnParam::data_rand_scale =
      Extract<float>(default_map, "data_rand_scale", 1);
  FrcnnParam::data_rand_rotate =
      Extract<bool>(default_map, "data_rand_rotate", false);
  FrcnnParam::data_hue = Extract<float>(default_map, "data_hue", 0);
  FrcnnParam::data_saturation =
      Extract<float>(default_map, "data_saturation", 0);
  FrcnnParam::data_exposure = Extract<float>(default_map, "data_exposure", 0);
  FrcnnParam::im_size_align = Extract<int>(default_map, "im_size_align", 1);
  FrcnnParam::roi_canonical_scale =
      Extract<int>(default_map, "roi_canonical_scale", 224);
  FrcnnParam::roi_canonical_level =
      Extract<int>(default_map, "roi_canonical_level", 4);
  FrcnnParam::test_soft_nms = Extract<int>(default_map, "test_soft_nms", 0);
  FrcnnParam::test_use_gpu_nms =
      Extract<bool>(default_map, "test_use_gpu_nms", false);
  FrcnnParam::test_bbox_vote =
      Extract<bool>(default_map, "test_bbox_vote", false);
  FrcnnParam::test_decrypt_model =
      Extract<bool>(default_map, "test_decrypt_model", false);

  FrcnnParam::bbox_reg = Extract<bool>(default_map, "bbox_reg");
  FrcnnParam::bbox_thresh = Extract<float>(default_map, "bbox_thresh");
  FrcnnParam::snapshot_infix =
      Extract<bool>(default_map, "test_decrypt_model", false);
  FrcnnParam::bbox_normalize_targets =
      Extract<bool>(default_map, "bbox_normalize_targets");
  FrcnnParam::bbox_inside_weights =
      Extract<std::vector<float>>(default_map, "bbox_inside_weights");
  FrcnnParam::bbox_normalize_means =
      Extract<std::vector<float>>(default_map, "bbox_normalize_means");
  FrcnnParam::bbox_normalize_stds =
      Extract<std::vector<float>>(default_map, "bbox_normalize_stds");

  FrcnnParam::rpn_positive_overlap =
      Extract<float>(default_map, "rpn_positive_overlap");
  FrcnnParam::rpn_negative_overlap =
      Extract<float>(default_map, "rpn_negative_overlap");
  FrcnnParam::rpn_clobber_positives =
      Extract<bool>(default_map, "rpn_clobber_positives", false);
  FrcnnParam::rpn_fg_fraction = Extract<float>(default_map, "rpn_fg_fraction");
  FrcnnParam::rpn_batchsize = Extract<int>(default_map, "rpn_batchsize");
  FrcnnParam::rpn_nms_thresh = Extract<float>(default_map, "rpn_nms_thresh");
  FrcnnParam::rpn_pre_nms_top_n =
      Extract<int>(default_map, "rpn_pre_nms_top_n");
  FrcnnParam::rpn_post_nms_top_n =
      Extract<int>(default_map, "rpn_post_nms_top_n");
  FrcnnParam::rpn_min_size = Extract<float>(default_map, "rpn_min_size");
  FrcnnParam::rpn_positive_weight =
      Extract<float>(default_map, "rpn_positive_weight");
  FrcnnParam::rpn_allowed_border =
      Extract<float>(default_map, "rpn_allowed_border");
  FrcnnParam::rpn_bbox_inside_weights =
      Extract<std::vector<float>>(default_map, "rpn_bbox_inside_weights");

  // ======================================== Test
  FrcnnParam::test_scales =
      Extract<std::vector<float>>(default_map, "test_scales");
  FrcnnParam::test_max_size = Extract<float>(default_map, "test_max_size");
  FrcnnParam::test_nms = Extract<float>(default_map, "test_nms");

  FrcnnParam::test_bbox_reg = Extract<bool>(default_map, "test_bbox_reg");
  FrcnnParam::test_rpn_nms_thresh =
      Extract<float>(default_map, "test_rpn_nms_thresh");
  FrcnnParam::test_rpn_pre_nms_top_n =
      Extract<int>(default_map, "test_rpn_pre_nms_top_n");
  FrcnnParam::test_rpn_post_nms_top_n =
      Extract<int>(default_map, "test_rpn_post_nms_top_n");
  FrcnnParam::test_rpn_min_size =
      Extract<float>(default_map, "test_rpn_min_size");

  // ========================================
  FrcnnParam::pixel_means =
      Extract<std::vector<float>>(default_map, "pixel_means");
  FrcnnParam::rng_seed = Extract<int>(default_map, "rng_seed");
  FrcnnParam::eps = Extract<float>(default_map, "eps");
  FrcnnParam::inf = Extract<float>(default_map, "inf");

  // ========================================
  FrcnnParam::feat_stride = Extract<int>(default_map, "feat_stride");
  FrcnnParam::anchors = Extract<std::vector<float>>(default_map, "anchors");
  FrcnnParam::test_score_thresh =
      Extract<float>(default_map, "test_score_thresh");
  FrcnnParam::test_rpn_score_thresh =
      Extract<float>(default_map, "test_rpn_score_thresh", 0);
  FrcnnParam::n_classes = Extract<int>(default_map, "n_classes");
  FrcnnParam::iter_test = Extract<int>(default_map, "iter_test");
}

void FrcnnParam::PrintParam() {
  LOG(INFO) << "== Train  Parameters ==";
  LOG(INFO) << "scale             : " << float_to_string(FrcnnParam::scales);
  LOG(INFO) << "max_size          : " << FrcnnParam::max_size;
  LOG(INFO) << "batch_size        : " << FrcnnParam::batch_size;

  LOG(INFO) << "fg_fraction       : " << FrcnnParam::fg_fraction;
  LOG(INFO) << "fg_thresh         : " << FrcnnParam::fg_thresh;
  LOG(INFO) << "bg_thresh_hi      : " << FrcnnParam::bg_thresh_hi;
  LOG(INFO) << "bg_thresh_lo      : " << FrcnnParam::bg_thresh_lo;
  LOG(INFO) << "use_flipped       : "
            << (FrcnnParam::use_flipped ? "yes" : "no");

  LOG(INFO) << "use_bbox_reg      : " << (FrcnnParam::bbox_reg ? "yes" : "no");
  LOG(INFO) << "bbox_thresh       : " << FrcnnParam::bbox_thresh;
  LOG(INFO) << "snapshot_infix    : " << FrcnnParam::snapshot_infix;
  LOG(INFO) << "normalize_targets : "
            << (FrcnnParam::bbox_normalize_targets ? "yes" : "no");

  LOG(INFO) << "rpn_pos_overlap   : " << FrcnnParam::rpn_positive_overlap;
  LOG(INFO) << "rpn_neg_overlap   : " << FrcnnParam::rpn_negative_overlap;
  LOG(INFO) << "clobber_positives : "
            << (FrcnnParam::rpn_clobber_positives ? "yes" : "no");
  LOG(INFO) << "rpn_fg_fraction   : " << FrcnnParam::rpn_fg_fraction;
  LOG(INFO) << "rpn_batchsize     : " << FrcnnParam::rpn_batchsize;
  LOG(INFO) << "rpn_nms_thresh    : " << FrcnnParam::rpn_nms_thresh;
  LOG(INFO) << "rpn_pre_nms_top_n : " << FrcnnParam::rpn_pre_nms_top_n;
  LOG(INFO) << "rpn_post_nms_top_n: " << FrcnnParam::rpn_post_nms_top_n;
  LOG(INFO) << "rpn_min_size      : " << FrcnnParam::rpn_min_size;
  LOG(INFO) << "rpn_bbox_inside_weights :"
            << float_to_string(FrcnnParam::rpn_bbox_inside_weights);
  LOG(INFO) << "rpn_positive_weight     :" << FrcnnParam::rpn_positive_weight;
  LOG(INFO) << "rpn_allowed_border      :" << FrcnnParam::rpn_allowed_border;

  LOG(INFO) << "== Test   Parameters ==";
  LOG(INFO) << "test_scales          : "
            << float_to_string(FrcnnParam::test_scales);
  LOG(INFO) << "test_max_size        : " << FrcnnParam::test_max_size;
  LOG(INFO) << "test_nms             : " << FrcnnParam::test_nms;
  LOG(INFO) << "test_bbox_reg        : "
            << (FrcnnParam::test_bbox_reg ? "yes" : "no");
  LOG(INFO) << "test_rpn_nms_thresh  : " << FrcnnParam::test_rpn_nms_thresh;
  LOG(INFO) << "rpn_pre_nms_top_n    : " << FrcnnParam::test_rpn_pre_nms_top_n;
  LOG(INFO) << "rpn_post_nms_top_n   : " << FrcnnParam::test_rpn_post_nms_top_n;
  LOG(INFO) << "test_rpn_min_sizen   : " << FrcnnParam::test_rpn_min_size;

  LOG(INFO) << "== Global Parameters ==";
  LOG(INFO) << "pixel_means[BGR]     : " << FrcnnParam::pixel_means[0] << " , "
            << FrcnnParam::pixel_means[1] << " , "
            << FrcnnParam::pixel_means[2];
  LOG(INFO) << "rng_seed             : " << FrcnnParam::rng_seed;
  LOG(INFO) << "eps                  : " << FrcnnParam::eps;
  LOG(INFO) << "inf                  : " << FrcnnParam::inf;
  LOG(INFO) << "feat_stride          : " << FrcnnParam::feat_stride;
  LOG(INFO) << "anchors_size         : " << FrcnnParam::anchors.size();
  LOG(INFO) << "test_score_thresh    : " << FrcnnParam::test_score_thresh;
  LOG(INFO) << "n_classes            : " << FrcnnParam::n_classes;
  LOG(INFO) << "iter_test            : " << FrcnnParam::iter_test;
}

}  // namespace frcnn
}  // namespace caffe
