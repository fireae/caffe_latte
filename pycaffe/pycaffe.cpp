#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe/blob.hpp"
#include <memory>
namespace py = pybind11;
using namespace caffe;
typedef float Dtype;

PYBIND11_MODULE(pycaffe, m) {
	//m.doc("pycaffe pybind11");
    py::class_<Blob<Dtype>, shared_ptr<Blob<Dtype>> > blob(m, "Blob");
	blob.def("data", [](Blob<Dtype>* self) -> py::object{
		return py::cast(self->mutable_cpu_data());
	});
}