#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "caffe/layers/roi_data_layer.hpp"
#include "caffe/util/detect_utils.hpp"
#include "caffe/util/frcnn_param.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {
namespace frcnn {

template <typename Dtype>
RoiDataLayer<Dtype>::~RoiDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void RoiDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                         const vector<Blob<Dtype> *> &top) {
  std::string default_config_file =
      this->layer_param_.roi_data_param().config();
  FrcnnParam::LoadParam(default_config_file);
  FrcnnParam::PrintParam();
  std::ifstream infile(this->layer_param_.roi_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open roi_data file"
                       << this->layer_param_.roi_data_param().source();
  std::map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));
  roi_database_.clear();

  BoxDataInfo box_data;
  while (box_data.LoadwithDifficult(infile)) {
    string image_path = box_data.GetImagePath(
        this->layer_param_.roi_data_param().root_folder());
    image_database_.push_back(image_path);
    lines_.push_back(image_database_.size() - 1);
    if (cache_images_) {
      Datum datum;
      if (!ReadFileToDatum(image_path, &datum)) {
        LOG(ERROR) << "Could not open or find file " << image_path;
        return;
      }
      image_database_cache_.push_back(std::make_pair(image_path, datum));
    }

    vector<vector<float>> rois = box_data.GetRois(true);
    for (int i = 0; i < rois.size(); i++) {
      int label = rois[i][BoxDataInfo::LABEL];
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }
    roi_database_.push_back(rois);
    if (lines_.size() % 1000 == 0) {
      LOG(INFO) << "Num: " << lines_.size() << " " << image_path
                << " rois to process: " << rois.size();
    }
  }

  CHECK_GT(lines_.size(), 0) << "No Image in ground truth file";
  LOG(INFO) << "Number of Images : " << lines_.size();

  for (auto it : label_hist) {
    LOG(INFO) << "class " << it.first << " has " << it.second << " samples";
  }

  // image
  vector<float> scales = FrcnnParam::scales;
  max_short_ = *max_element(scales.begin(), scales.end());
  max_long_ = FrcnnParam::max_size;
  const int batch_size = 1;

  // data mean
  for (int i = 0; i < 3; i++) {
    mean_values_[i] = FrcnnParam::pixel_means[i];
  }

  max_short_ = int(std::ceil(max_short_ / float(FrcnnParam::im_size_align)) *
                   FrcnnParam::im_size_align);
  max_long_ = int(std::ceil(max_long_ / float(FrcnnParam::im_size_align)) *
                  FrcnnParam::im_size_align);
  top[0]->Reshape(batch_size, 3, max_short_, max_long_);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(batch_size, 3, max_short_, max_long_);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
            << top[0]->channels() << "," << top[0]->height() << ","
            << top[0]->width();

  // im_info: height width scale_factor
  top[1]->Reshape(1, 3, 1, 1);
  // gt_boxes: label x1 y1 x2 y2
  top[2]->Reshape(batch_size, 5, 1, 1);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(batch_size + 1, 5, 1, 1);
  }

  LOG(INFO) << "Shuffling data";
  // if we use multi GPU
  // int solver_count = Caffe::solver_count();
  // Caffe::solver_rank() is set different value of each device by parallel.cpp
  const unsigned int prefetch_rng_seed =
      FrcnnParam::rng_seed + Caffe::solver_rank();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  lines_id_ = 0;  // First Shuffle
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  std::shuffle(lines_.begin(), lines_.end(), *prefetch_rng);
}

template <typename Dtype>
void RoiDataLayer<Dtype>::ShuffleImages() {
  lines_id_++;
  if (lines_id_ >= lines_.size()) {
    LOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
    CHECK(prefetch_rng_);
    caffe::rng_t *prefetch_rng =
        static_cast<caffe::rng_t *>(prefetch_rng_->generator());
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(lines_.begin(), lines_.end(), g);
  }
}

template <typename Dtype>
unsigned int RoiDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void RoiDataLayer<Dtype>::CheckResetRois(vector<vector<float>> &rois,
                                         const string image_path,
                                         const float cols, const float rows,
                                         const float im_scale) {
  for (int i = 0; i < rois.size(); ++i) {
    bool ok = rois[i][BoxDataInfo::X1] >= 0 && rois[i][BoxDataInfo::Y1] >= 0 &&
              rois[i][BoxDataInfo::X2] < cols &&
              rois[i][BoxDataInfo::Y2] < cols;
    if (!ok) {
      rois[i][BoxDataInfo::X1] = std::max(0.f, rois[i][BoxDataInfo::X1]);
      rois[i][BoxDataInfo::Y1] = std::max(0.f, rois[i][BoxDataInfo::Y1]);
      rois[i][BoxDataInfo::X2] = std::min(cols - 1, rois[i][BoxDataInfo::X2]);
      rois[i][BoxDataInfo::Y2] = std::min(rows - 1, rois[i][BoxDataInfo::Y2]);
    }
  }
}

template <typename Dtype>
void RoiDataLayer<Dtype>::FlipRois(vector<vector<float>> &rois,
                                   const float cols) {
  for (int i = 0; i < rois.size(); i++) {
    CHECK_GT(rois[i][BoxDataInfo::X1], 0)
        << "rois[" << i << "][x1]: " << rois[i][BoxDataInfo::X1];
    CHECK_LT(rois[i][BoxDataInfo::X2], cols)
        << "rois[" << i << "][x2]: " << rois[i][BoxDataInfo::X2];
    float old_x1 = rois[i][BoxDataInfo::X1];
    float old_x2 = rois[i][BoxDataInfo::X2];
    rois[i][BoxDataInfo::X1] = cols - old_x2 - 1;
    rois[i][BoxDataInfo::X2] = cols - old_x1 - 1;
  }
}

template <typename Dtype>
void RoiDataLayer<Dtype>::load_batch(Batch<Dtype> *batch) {
  const vector<float> scales = FrcnnParam::scales;
  const bool use_flipped = FrcnnParam::use_flipped;
  const int batch_size = 1;
  ShuffleImages();
}

template <typename Dtype>
void RoiDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  Batch<Dtype> *batch =
      this->prefetch_full_.pop("Data Layer prefetch queue empty");

  // Reshape load data
  top[0]->ReshapeLike(batch->data_);
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());

  if (this->output_labels_) {
    caffe_copy(3, batch->label_.cpu_data(), top[1]->mutable_cpu_data());

    top[2]->Reshape(batch->label_.num() - 1, batch->label_.channels(),
                    batch->label_.height(), batch->label_.width());
    caffe_copy(batch->label_.count() - 5, batch->label_.cpu_data() + 5,
               top[2]->mutable_cpu_data());
  }

  this->prefetch_free_.push(batch);
}

// #ifndef USE_CUDA
// STUB_GPU(RoiDataLayer);
// #endif

INSTANTIATE_CLASS(RoiDataLayer);
REGISTER_LAYER_CLASS(RoiData);
}  // namespace frcnn
}  // namespace caffe