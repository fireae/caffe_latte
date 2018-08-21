#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/detect_utils.hpp"
#include "caffe/util/frcnn_param.hpp"

namespace caffe {
namespace frcnn {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(1, 5, 1, 1);
  if (top.size() > 1) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_rpn_score =
      bottom[0]->cpu_data();                             // rpn_cls_prob_reshape
  const Dtype* bottom_rpn_bbox = bottom[1]->cpu_data();  // rpn_bbox_pred
  const Dtype* bottom_im_info = bottom[2]->cpu_data();   // im_info

  const int num = bottom[1]->num();
  const int channels = bottom[1]->channels();
  const int height = bottom[1]->height();
  const int width = bottom[1]->width();
  CHECK(num == 1) << "Only batch_size==1 support";
  CHECK(channels % 4 == 0) << "rpn bbox pred channels should be divided by 4";

  const float im_height = bottom_im_info[0];
  const float im_width = bottom_im_info[1];

  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float rpn_nms_thresh;
  int rpn_min_size;
  if (this->phase_ == TRAIN) {
    rpn_pre_nms_top_n = FrcnnParam::rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::rpn_nms_thresh;
    rpn_min_size = FrcnnParam::rpn_min_size;
  } else {
    rpn_pre_nms_top_n = FrcnnParam::test_rpn_pre_nms_top_n;
    rpn_post_nms_top_n = FrcnnParam::test_rpn_post_nms_top_n;
    rpn_nms_thresh = FrcnnParam::test_rpn_nms_thresh;
    rpn_min_size = FrcnnParam::test_rpn_min_size;
  }

  const int config_n_anchors = FrcnnParam::anchors.size() / 4;
  typedef pair<Dtype, int> sort_pair;
  std::vector<sort_pair> sort_vector;
  const Dtype bounds[4] = {im_width - 1, im_height - 1, im_width - 1,
                           im_height - 1};
  const Dtype min_size = bottom_im_info[2] * rpn_min_size;

  int feat_stride = this->layer_param_.proposal_param().feat_stride();
  if (feat_stride == 0) {
    feat_stride = FrcnnParam::feat_stride;
  }

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int k = 0; k < config_n_anchors; ++k) {
        Dtype score = bottom_rpn_score[config_n_anchors * height * width +
                                       k * height * width + h * width + w];
        if (this->phase_ == TEST && score < FrcnnParam::test_rpn_score_thresh &&
            k > 0) {
          continue;
        }

        RectBox<Dtype> anchor(
            FrcnnParam::anchors[k * 4 + 0] + w * feat_stride,  // shift_x[i][j]
            FrcnnParam::anchors[k * 4 + 1] + h * feat_stride,  // shift_y[i][j]
            FrcnnParam::anchors[k * 4 + 2] + w * feat_stride,  // shift_x[i][j]
            FrcnnParam::anchors[k * 4 + 3] + h * feat_stride   // shift_y[i][j]
        );

        RectBox<Dtype> box_delta(
            bottom_rpn_bbox[(k * 4 + 0) * height * width + h * width + w],
            bottom_rpn_bbox[(k * 4 + 1) * height * width + h * width + w],
            bottom_rpn_bbox[(k * 4 + 2) * height * width + h * width + w],
            bottom_rpn_bbox[(k * 4 + 3) * height * width + h * width + w]);

        RectBox<Dtype> cbox = TransformInvRectBox(anchor, box_delta);

        // clip predicted boxes to image
        // for (int q = 0; q < 4; ++q) {
        //     cbox.x1 = std::max(Dtype(0), std::min(cbox[0], bounds[0]));
        // }
        cbox.x1 = std::max(Dtype(0), std::min(cbox[0], bounds[0]));
        cbox.y1 = std::max(Dtype(1), std::min(cbox[1], bounds[1]));
        cbox.x2 = std::max(Dtype(2), std::min(cbox[2], bounds[2]));
        cbox.y2 = std::max(Dtype(3), std::min(cbox[3], bounds[3]));

        // remove predicted boxes with either height or width < threshold
        if ((cbox[2] - cbox[0] + 1) >= min_size &&
            (cbox[3] - cbox[1] + 1) >= min_size) {
          const now_index = sort_vector.size();
          sort_vector.push_back(sort_pair(score, now_index));
          anchors.push_back(cbox);
        }
      }
    }
  }

  std::sort(sort_vector.begin(), sort_vector.end(), std::greater<sort_pair>());
  const int n_anchors = std::min((int)sort_vector.size(), rpn_pre_nms_top_n);
  sort_vector.erase(sort_vector.begin() + n_anchors, sort_vector.end());
  std::vector<bool> select(n_anchors, true);

  // apply nms
  std::vector<RectBox<Dtype> > box_final;
  std::vector<Dtype> scores;

// cuda nms
#ifdef defined(USE_CUDA) && defined(USE_GPU_NMS)

#endif
  if (FrcnnParam::test_soft_nms) {  // soft nms
  } else {                          // naive nms
    for (int i = 0; i < n_anchors && box_final.size() < rpn_post_nms_top_n;
         ++i) {
      if (select[i]) {
        const int cur_i = sort_vector[i].second;
        for (int j = i + 1; j < n_anchors; ++j) {
          if (select[j]) {
            const int cur_j = sort_vector[j].second;
            if (ComputeIOU(anchors[cur_i], anchors[cur_j]) > rpn_nms_thresh) {
              select[j] = false;
            }
          }
        }

        box_final.push_back(anchors[cur_i]);
        scores.push_back(sort_vector[i].first);
      }
    }
  }

  top[0]->Reshape(box_final.size(), 5, 1, 1);
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (size_t i = 0; i < box_final.size(); i++) {
    RectBox<Dtype>& box = box_final[i];
    top_data[i * 5 + 0] = 0;
    top_data[i * 5 + 1] = box.x1;
    top_data[i * 5 + 2] = box.y1;
    top_data[i * 5 + 3] = box.x2;
    top_data[i * 5 + 4] = box.y2;
  }

  if (top.size() > 1) {
    top[1]->Reshape(box_final.size(), 1, 1, 1);
    for (size_t i = 0; i < scores.size(); i++) {
      top[1]->mutable_cpu_data()[i] = scores[i];
    }
  }
}

}  // namespace frcnn
}  // namespace caffe