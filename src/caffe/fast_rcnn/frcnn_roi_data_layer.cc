

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
      this->layer_param_.window_data_param().config();
  FrcnnParam::LoadParam(default_config_file);
  FrcnnParam::PrintParam();
}
}  // namespace frcnn
}  // namespace caffe