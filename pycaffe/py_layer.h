#ifndef CAFFE_PYTHON_PY_LAYER_H_
#define CAFFE_PYTHON_PY_LAYER_H_
#include <pybind11/pybind11.h>
#include <caffe/layer.hpp>

namespace py = pybind11;
namespace caffe {
typedef float Dtype;

class PyLayer : public Layer<Dtype> {
 public:
  using Layer::Layer;
  const char* type() const {
    py::print("Python");
    PYBIND11_OVERLOAD_PURE(const char*, Layer<Dtype>, type);
  }
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                  const vector<Blob<Dtype>*>& top) {
    PYBIND11_OVERLOAD_PURE(void, Layer<Dtype>, LayerSetup, bottom, top);
  }

  void Reshape(const vector<Blob<Dtype>*>& bottom,
               const vector<Blob<Dtype>*>& top) {
    PYBIND11_OVERLOAD_PURE(void, Layer<Dtype>, LayerReshape, bottom, top);
  }

  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                   const vector<Blob<Dtype>*>& top) {
    PYBIND11_OVERLOAD_PURE(void, Layer<Dtype>, Forward_cpu, bottom, top);
  }

  void Backward_cpu(const vector<Blob<Dtype>*>& top,
                    const vector<bool>& propagate_down,
                    const vector<Blob<Dtype>*>& bottom) {
    PYBIND11_OVERLOAD_PURE(void, Layer<Dtype>, Backward_cpu, top,
                           propagate_down, bottom);
  }
};
}  // namespace caffe

#endif  // CAFFE_PYTHON_PY_LAYER_H_