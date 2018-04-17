#include "caffe/layers/text_proposal_layer.hpp"
#include "caffe/util/anchor.hpp"

#define ROUND(x) ((int)((x) + (Dtype)0.5))

namespace caffe {

template <typename Dtype>
void TextProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void TextProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";

  int num_anchors = 10;

  Blob<Dtype>* scores = bottom[0];

  const Dtype* bbox_deltas = bottom[1]->cpu_data();

  AnchorText<Dtype> anchor_text;
  vector<int> image_size;
  image_size.push_back(bottom[0]->height());
  image_size.push_back(bottom[0]->width());
  vector<vector<int> > anchors = anchor_text.locate_anchors(image_size, 16);

  Blob<Dtype> scores_trans;
  scores_trans.Reshape(scores->num(), scores->height(), scores->width(),
                       scores->channels() - num_anchors);

  Dtype* scores_trans_data = scores_trans.mutable_cpu_data();
  for (int n = 0; n < scores_trans.num(); ++n) {
    for (int c = 0; c < scores_trans.channels(); ++c) {
      for (int h = 0; h < scores_trans.height(); ++h) {
        for (int w = 0; w < scores_trans.width(); ++w) {
          scores_trans_data[scores_trans.offset(n, c, h, w)] =
              scores->data_at(n, w + num_anchors, c, h);
        }
      }
    }
  }

  Blob<Dtype> bbox_delta_trans;
  bbox_delta_trans.Reshape(bottom[1]->num(), bottom[1]->height(),
                           bottom[1]->width(), bottom[1]->channels());
  Dtype* bbox_delta_trans_data = bbox_delta_trans.mutable_cpu_data();
  for (int n = 0; n < bbox_delta_trans.num(); ++n) {
    for (int c = 0; c < bbox_delta_trans.channels(); ++c) {
      for (int h = 0; h < bbox_delta_trans.height(); ++h) {
        for (int w = 0; w < bbox_delta_trans.width(); ++w) {
          bbox_delta_trans_data[bbox_delta_trans.offset(n, c, h, w)] =
              bottom[1]->data_at(n, w, c, h);
        }
      }
    }
  }
  vector<int> bbox_shape;
  bbox_shape.push_back(bbox_delta_trans.count() / 2);
  bbox_shape.push_back(2);
  bbox_delta_trans.Reshape(bbox_shape);
  vector<vector<Dtype> > bbox_delta_data;
  bbox_delta_trans_data = bbox_delta_trans.mutable_cpu_data();
  for (int n = 0; n < bbox_delta_trans.count() / 2; n++) {
    vector<Dtype> bbox_item;
    for (int k = 0; k < 2; k++) {
      bbox_item.push_back(*(bbox_delta_trans_data + k));
    }
    bbox_delta_data.push_back(bbox_item);
    bbox_delta_trans_data += 2;
  }
  vector<vector<Dtype> > proposals =
      anchor_text.apply_deltas_to_anchors(bbox_delta_data, anchors);

  vector<int> top0_shape;
  top0_shape.push_back(proposals.size());
  top0_shape.push_back(4);
  top[0]->Reshape(top0_shape);
  Dtype* top0_data = top[0]->mutable_cpu_data();
  int index = 0;
  for (int i = 0; i < proposals.size(); i++) {
    for (int k = 0; k < 4; k++) {
      top0_data[index] = proposals[i][k];
      index++;
    }
  }

  top[1]->Reshape(scores_trans.shape());
  top[1]->CopyFrom(scores_trans);
  vector<int> scores_shape;
  scores_shape.push_back(scores_trans.count());
  scores_shape.push_back(1);
  top[1]->Reshape(scores_shape);
}

template <typename Dtype>
void TextProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(TextProposalLayer);
REGISTER_LAYER_CLASS(TextProposal);

}  // namespace caffe
