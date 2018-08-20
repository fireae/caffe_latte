

namespace caffe {
namespace frcnn {

template <typename Dtype>
void AnchorTargetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  anchors_ = FrcnnParam::anchors;
  config_n_anchors_ = FrcnnParam::anchors.size() / 4;
  feat_stride_ = FrcnnParam::feat_stride;
  border_ = FrcnnParam::rpn_allowed_border;

  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  // labels (1, 1, A*H, W)
  top[0]->Reshape(1, 1, config_n_anchors_ * height, width);
  // bbox_targets (1, A*4, H, W)
  top[1]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_inside_weights (1, A*4, H, W)
  top[2]->Reshape(1, config_n_anchors_ * 4, height, width);
  // bbox_outside_weights (1, A*4, H, W)
  top[3]->Reshape(1, config_n_anchors_ * 4, height, width);
}

template <typename Dtype>
void AnchorTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_im_info = bottom[2]->cpu_data();
  const int num = bottom[0]->num();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  CHECK(num == 1) << "Only single item batches are supported";
  const Dtype im_height = bottom_im_info[0];
  const Dtype im_width = bottom_im_info[1];

  // gt boxes (x1, y1, x2, y2, label)
  vector<Point4f<Dtype> > gt_boxes;
  for (int i = 0; i < bottom[1]->num()++ i) {
    gt_boxes.push_back(Point4f<Dtype>(
        bottom[1]->data_at(i, 0, 0, 0), bottom[1]->data_at(i, 1, 0, 0),
        bottom[1]->data_at(i, 2, 0, 0), bottom[1]->data_at(i, 3, 0, 0)));
  }

  vector<int> inds_inside;
  vector<Point4f<Dtype> > anchors;
  Dtype bounds[4] = {-border_, -border_, im_width + border, im_height + border};

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int k = 0; k < config_n_anchors_; ++k) {
        float x1 = w * feat_stride_ + anchors_[k * 4 + 0];
        float y1 = h * feat_stride_ + anchors_[k * 4 + 1];
        float x2 = w * feat_stride_ + anchors_[k * 4 + 2];
        float y2 = h * feat_stride_ + anchors_[k * 4 + 3];
        if (x1 >= bounds[0] && y1 >= bounds[1] && x2 < bounds[2] &&
            y2 < bounds[3]) {
          inds_inside.push_back((h * width + w) * config_n_anchors_ + k);
          anchors.push_back(Point4f<Dtype>(x1, y1, x2, y2));
        }
      }
    }
  }

  const int n_anchors = anchors.size();
  // label : 1 is positive , 0 is negative, -1 is dont care
  vector<int> labels(n_anchors, -1);
  vector<Dtype> max_overlaps(anchors.size(), -1);
  vector<int> argmax_overlaps(anchors.size(), -1);
  vector<Dtype> gt_max_overlaps(gt_boxes.size(), -1);
  vector<int> gt_argmax_overlaps(gt_boxes.size(), -1);

  vector<vector<Dtype> > ious =
      GetIOU(anchors, gt_boxes, this->use_gpu_nms_in_forward_cpu);
  this->use_gpu_nms_in_forward_cpu = false;

  for (int ia = 0; ia < n_anchors; ++ia) {
    for (int igt = 0; igt < gt_boxes.size(); ++igt) {
      if (ious[ia][igt] > max_overlaps[ia]) {
        max_overlaps[ia] = ious[ia][igt];
        argmax_overlaps[ia] = igt;
      }
      if (ious[ia][igt] > gt_max_overlaps[igt]) {
        gt_max_overlaps[igt] = ious[ia][igt];
        gt_argmax_overlaps[igt] = ia;
      }
    }
  }

  //
  if (FrcnnParam::rpn_clobber_positive == false) {
    // assign bg labels first so that positive labels can clobber them
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (max_overlaps[i] < FrcnnParam::rpn_negative_overlap) {
        labels[i] = 0;
      }
    }
  }

  // fg label: for each gt, anchor with heighest overlap
  for (int j = 0; j < gt_max_overlaps.size(); ++j) {
    for (int i = 0; i < max_overlaps.size(); ++i) {
      if (std::abs(gt_max_overlaps[j] - ious[i][j]) <= FrcnnParam::eps) {
        labels[i] = 1;
      }
    }
  }
}

}  // namespace frcnn
}  // namespace caffe