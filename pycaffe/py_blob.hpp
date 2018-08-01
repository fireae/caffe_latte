#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
using namespace caffe;
namespace py = pybind11;

namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<Blob<T> > {
 public:
  PYBIND11_TYPE_CASTER(Blob<T>, _("Blob<T>"));

  // conversion part1 (python -> c++)
  bool load(py::handle src, bool convert) {
    if (!convert && !py::array_t<T>::check_(src)) {
      return false;
    }

    auto buf =
        py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(src);

    if (!buf) {
      return false;
    }

    auto dims = buf.ndim();
    if (dims < 1) {
      return false;
    }

    std::vector<int> shape(buf.ndim());

    for (int i = 0; i < buf.ndim(); i++) {
      shape[i] = buf.shape()[i];
    }

    Blob<T> tmp(shape);
    tmp.set_cpu_data(const_cast<T*>(buf.data()));
    value.CopyFrom(tmp, false, true);
    return true;
  }

  // conversion part 2 (c++-> python)
  static py::handle cast(const Blob<T>& src, py::return_value_policy policy,
                         py::handle parent) {
    std::vector<size_t> shape(src.shape().size());
    std::vector<size_t> strides(src.shape().size());
    py::print("src.shape().size() ", src.shape().size());
    for (int i = 0; i < src.shape().size(); i++) {
      shape[i] = src.shape(i);
    }
    strides[0] = shape[1] * shape[2] * shape[3];
    strides[1] = shape[2] * shape[3];
    strides[2] = shape[3];
    strides[3] = 1;
    const T* v = src.cpu_data();
    for (int i = 0; i < src.count(); i++) {
      py::print(v[i]);
    }
    float data[24];
    for (int k = 0; k < 24; k++) {
      data[k] = float(k);
    }
    strides.clear();
    py::array a(std::move(shape), std::move(strides), data);
    py::print("shape ", a.shape());
    py::print("strides ", a.strides());
    py::print("a[11] ", a.data(1, 1, 1, 1));
    return a.release();
  }
};
}  // namespace detail
}  // namespace pybind11