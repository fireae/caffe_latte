#ifndef CAFFE_PYTHON_PY_LAYER_H_
#define CAFFE_PYTHON_PY_LAYER_H_
#include <pybind11/pybind11.h>
#include <caffe/layer.hpp>

namespace py = pybind11;
namespace caffe {
	typedef float Dtype;

class PyLayer : public Layer<Dtype> {
public:
	// using caffe::Layer<Dtype>;
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
};
}

#endif //CAFFE_PYTHON_PY_LAYER_H_