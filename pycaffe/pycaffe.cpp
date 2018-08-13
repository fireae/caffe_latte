#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
namespace py = pybind11;
typedef float Dtype;
//const int NPY_DTYPE = NPY_FLOAT32;
namespace caffe {

    PYBIND11_MODULE(pycaffe, m) {
        py::class_<Blob<Dtype>, shared_ptr<Blob<Dtype>>> blob(m, "Blob", py::buffer_protocol());
        blob.def_buffer([](Blob<Dtype> &m) -> py::buffer_info {
        return py::buffer_info(
            m.mutable_cpu_data(),                               /* Pointer to buffer */
            sizeof(Dtype),                          /* Size of one scalar */
            py::format_descriptor<Dtype>::format(), /* Python struct-style format descriptor */
            4,                                      /* Number of dimensions */
            { m.num(), m.channels(), m.height(), m.width() },                 /* Buffer dimensions */
            {}
            // { sizeof(Dtype) * m.rows(),             /* Strides (in bytes) for each index */
            //   sizeof(Dtype) }
        );
    });

    }
}