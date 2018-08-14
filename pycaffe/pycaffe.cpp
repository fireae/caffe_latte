#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include "caffe/blob.hpp"
namespace py = pybind11;
typedef float Dtype;
//const int NPY_DTYPE = NPY_FLOAT32;
namespace caffe {

Dtype _add(shared_ptr<Blob<Dtype>> a) {
  int count = a->count();
  Dtype* d = a->mutable_cpu_data();
  Dtype s = 0.0;
  for (int i = 0; i < count; i++) {
    d[i] *= 2;
    s += d[i];
  }
  return s;
}


void Numpy2BlobData(Blob<Dtype>& m, py::array_t<Dtype> info) {
  py::print("Numpy2BlobData");
  m.Reshape(info.shape()[0], info.shape()[1], info.shape()[2], info.shape()[3]);
  m.set_cpu_data(const_cast<Dtype*>(info.data()));
}

py::array_t<Dtype> BlobData2Numpy(Blob<Dtype>& m) {
  py::print("BlobData2Numpy");
  py::array_t<Dtype> a({m.num(), m.channels(), m.height(), m.width()},
                       {sizeof(Dtype) * m.channels() * m.height() * m.width(),
                        sizeof(Dtype) * m.height() * m.width(),
                        sizeof(Dtype) * m.width(), sizeof(Dtype)},
                       m.mutable_cpu_data());
  return a;
}

void Numpy2BlobDiff(Blob<Dtype>& m, py::array_t<Dtype> info) {
  py::print("Numpy2BlobDiff");
  m.Reshape(info.shape()[0], info.shape()[1], info.shape()[2], info.shape()[3]);
  Dtype* diff = static_cast<Dtype*>m.mutable_cpu_diff();
  diff = const_cast<Dtype*>(info.data());
  //m.set_cpu_diff(const_cast<Dtype*>(info.data()));
}

py::array_t<Dtype> BlobDiff2Numpy(Blob<Dtype>& m) {
  py::print("BlobDiff2Numpy");
  py::array_t<Dtype> a({m.num(), m.channels(), m.height(), m.width()},
                       {sizeof(Dtype) * m.channels() * m.height() * m.width(),
                        sizeof(Dtype) * m.height() * m.width(),
                        sizeof(Dtype) * m.width(), sizeof(Dtype)},
                       m.mutable_cpu_diff());
  return a;
}


PYBIND11_MODULE(pycaffe, m) {
  // m.doc("pycaffe pybind11");
  py::class_<Blob<Dtype>, shared_ptr<Blob<Dtype>>> blob(m, "Blob",
                                                        py::buffer_protocol());
  blob.def(py::init<>());
  blob.def(py::init<const int, const int, const int, const int>());
//   blob.def("set_data", [](Blob<Dtype>& m, py::buffer b) {
//     py::buffer_info info = b.request();
//     m.Reshape(info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
//     m.set_cpu_data(static_cast<Dtype*>(info.ptr));
//   });
//   blob.def_buffer([](Blob<Dtype>& m) -> py::buffer_info {
//     return py::buffer_info(
//         m.mutable_cpu_data(), sizeof(Dtype),
//         py::format_descriptor<Dtype>::format(), 4,
//         {m.num(), m.channels(), m.height(), m.width()},
//         {sizeof(Dtype) * m.channels() * m.height() * m.width(),
//          sizeof(Dtype) * m.height() * m.width(), sizeof(Dtype) * m.width(),
//          sizeof(Dtype)});
//   });
//   blob.def("data", [](Blob<Dtype>& m) -> py::buffer_info {
//     return py::buffer_info(
//         m.mutable_cpu_data(), sizeof(Dtype),
//         py::format_descriptor<Dtype>::format(), 4,
//         {m.num(), m.channels(), m.height(), m.width()},
//         {sizeof(Dtype) * m.channels() * m.height() * m.width(),
//          sizeof(Dtype) * m.height() * m.width(), sizeof(Dtype) * m.width(),
//          sizeof(Dtype)});
//   });

  blob.def_property("data", &BlobData2Numpy, &Numpy2BlobData);
  blob.def_property("diff", &BlobDiff2Numpy, &Numpy2BlobDiff);
  m.def("add", &_add);
}
