#include <Python.h>  // NOLINT(build/include_alpha)
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include <functional>
#include <string>
#include <vector>
//#include "py_blob.hpp"

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"

namespace py = pybind11;
typedef float Dtype;
const int NPY_DTYPE = NPY_FLOAT32;
namespace caffe {
using namespace std;

void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }

void InitLog() { ::caffe::InitLogging(""); }
void Log(const string& s) { LOG(INFO) << s; }
void set_random_seed(unsigned int seed) { Caffe::set_random_seed(seed); }

static void CheckFile(const std::string& filename) {
  std::ifstream f(filename.c_str());
  if (!f.good()) {
    f.close();
    throw std::runtime_error("Could not open file " + filename);
  }
  f.close();
}

shared_ptr<Net<Dtype>> Net_Init(std::string network_file, int phase,
                                const int level) {
  CheckFile(network_file);
  shared_ptr<Net<Dtype>> net(
      new Net<Dtype>(network_file, static_cast<Phase>(phase), level));

  return net;
}

shared_ptr<Net<Dtype>> Net_Init_Load(std::string param_file,
                                     std::string pretrained_param_file,
                                     int phase) {
  shared_ptr<Net<Dtype>> net(
      new Net<Dtype>(param_file, static_cast<Phase>(phase)));
  CheckFile(param_file);
  CheckFile(pretrained_param_file);
  net->CopyTrainedLayersFrom(pretrained_param_file);
  return net;
}

void Net_Save(const Net<Dtype>& net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, filename);
}

py::object Blob_Reshape(Blob<Dtype>& self, py::args args, py::kwargs kwargs) {
  if (py::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }

  // Blob<Dtype>* self = args[0].cast<Blob<Dtype>*>();

  vector<int> shape(py::len(args));
  for (int i = 0; i < py::len(args); ++i) {
    shape[i] = args[i].cast<int>();
  }
  self.Reshape(shape);
  return py::object();
}

py::object BlobVec_add_blob(py::tuple args, py::dict kwargs) {
  if (py::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }
  //   typedef vector<shared_ptr<Blob<Dtype>>> BlobVec;
  //   BlobVec* self = args[0].cast<BlobVec*>();
  //   vector<int> shape(py::len(args) - 1);
  //   for (int i = 1; i < py::len(args); i++) {
  //     shape[i - 1] = args[i].cast<int>();
  //   }
  //   self->push_back(shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
  return py::object();
}

template <typename Dtype>
class NetCallback : public Net<Dtype>::Callback {
 public:
  explicit NetCallback(py::object run) : run_(run) {}

 protected:
  virtual void run(int layer) { run_(layer); }
  py::object run_;
};

void Net_before_forward(Net<Dtype>* net, py::object run) {
  net->add_before_forward(new NetCallback<Dtype>(run));
}
void Net_after_forward(Net<Dtype>* net, py::object run) {
  net->add_after_forward(new NetCallback<Dtype>(run));
}
void Net_before_backward(Net<Dtype>* net, py::object run) {
  net->add_before_backward(new NetCallback<Dtype>(run));
}
void Net_after_backward(Net<Dtype>* net, py::object run) {
  net->add_after_backward(new NetCallback<Dtype>(run));
}

Solver<Dtype>* GetSolverFromFile(const string& filename) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(filename, &param);
  return SolverFloatRegistry()->Create(param.type(), param).get();
}

void shared_weights(Solver<Dtype>* solver, Net<Dtype>* net) {
  net->ShareTrainedLayersWith(solver->net().get());
}

struct NdarrayConverterGenerator {
  template <typename T>
  struct apply;
};

template <>
struct NdarrayConverterGenerator::apply<Dtype*> {
  struct type {
    PyObject* operator()(Dtype* data) const {
      // Just store the data pointer, and add the shape information in postcall.
      return PyArray_SimpleNewFromData(0, NULL, NPY_DTYPE, data);
    }
    const PyTypeObject* get_pytype() { return &PyArray_Type; }
  };
};

struct NdarrayCallPolicies {
  typedef NdarrayConverterGenerator result_converter;
  PyObject* postcall(PyObject* pyargs, PyObject* result) {
    py::object pyblob = py::cast<py::tuple>(pyargs)[0];
    shared_ptr<Blob<Dtype>> blob = (pyblob).cast<shared_ptr<Blob<Dtype>>>();
    // Free the temporary pointer-holding array, and construct a new one with
    // the shape information from the blob.
    void* data = PyArray_DATA(reinterpret_cast<PyArrayObject*>(result));
    Py_DECREF(result);
    const int num_axes = blob->num_axes();
    vector<npy_intp> dims(blob->shape().begin(), blob->shape().end());
    PyObject* arr_obj =
        PyArray_SimpleNewFromData(num_axes, dims.data(), NPY_FLOAT32, data);
    // SetBaseObject steals a ref, so we need to INCREF.
    Py_INCREF(pyblob.ptr());
    PyArray_SetBaseObject(reinterpret_cast<PyArrayObject*>(arr_obj),
                          pyblob.ptr());
    return arr_obj;
  }
};

void Add2(int a) {
  //   py::scoped_interpreter guard{};
  py::module calc = py::module::import("calc");
  py::object res = calc.attr("add22")(a, 1);
  int n = res.cast<int>();
  py::print(n);
}

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

PYBIND11_MODULE(pycaffe, m) {
  m.doc() = "pybind11 pycaffe plugin";
  m.def("init_log", &InitLog, "init log function");
  m.def("log", &Log, "a function to log out");
  m.def("set_mode_cpu", &set_mode_cpu);
  m.def("set_mode_gpu", &set_mode_gpu);
  m.def("set_random_seed", &set_random_seed);
  m.def("set_device", &Caffe::SetDevice);
  m.def("solver_count", &Caffe::solver_count);
  m.def("set_solver_count", &Caffe::set_solver_count);
  m.def("solver_rank", &Caffe::solver_rank);
  m.def("set_solver_rank", &Caffe::set_solver_rank);
  m.def("set_multiprocess", &Caffe::set_multiprocess);
  m.def("layer_type_list", &LayerRegistry<Dtype>::LayerTypeList);
  m.def("add2", &Add2);
  py::class_<Net<Dtype>, shared_ptr<Net<Dtype>>> net(m, "Net");
  net.def(py::init(&Net_Init));
  net.def("forward", &Net<Dtype>::ForwardFromTo);
  net.def("backward", &Net<Dtype>::BackwardFromTo);
  net.def("reshape", &Net<Dtype>::Reshape);
  net.def("clear_param_diffs", &Net<Dtype>::ClearParamDiffs);
  net.def("copy_from", static_cast<void (Net<Dtype>::*)(const string)>(
                           &Net<Dtype>::CopyTrainedLayersFrom));
  net.def("share_with", &Net<Dtype>::ShareTrainedLayersWith);
  net.def("save", &Net_Save);

  net.def("blob_loss_weights", &Net<Dtype>::blob_loss_weights);
  net.def("bottom_ids", &Net<Dtype>::bottom_ids);
  net.def("top_ids", &Net<Dtype>::top_ids);
  net.def("blobs", &Net<Dtype>::blobs);
  net.def("layers", &Net<Dtype>::layers);
  net.def("blob_names", &Net<Dtype>::blob_names);
  net.def("layer_names", &Net<Dtype>::layer_names);
  net.def("inputs", &Net<Dtype>::input_blob_indices);
  net.def("outputs", &Net<Dtype>::output_blob_indices);
  //   net.def("set_input_arrays", &Net<Dtype>::bottom_ids,
  //                     py::return_value_policy::reference_internal);

  py::class_<Layer<Dtype>, shared_ptr<Layer<Dtype>>> layer(m, "Layer");
  // layer.def(py::init<const caffe::LayerParameter&>());
  layer.def("setup", &Layer<Dtype>::LayerSetUp);
  layer.def("reshape", &Layer<Dtype>::Reshape);
  layer.def_property_readonly("type", &Layer<Dtype>::type);

  py::class_<SolverParameter>(m, "SolverParameter")
      .def_property_readonly("max_iter", &SolverParameter::max_iter)
      .def_property_readonly("display", &SolverParameter::display)
      .def_property_readonly("layer_wise_reduce",
                             &SolverParameter::layer_wise_reduce);

  py::class_<Solver<Dtype>, shared_ptr<Solver<Dtype>>>(m, "Solver")
      .def_property_readonly("net", &Solver<Dtype>::net)
      .def_property_readonly("test_nets", &Solver<Dtype>::test_nets,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("iter", &Solver<Dtype>::iter)
      .def("step", &Solver<Dtype>::Step)
      .def("restore", &Solver<Dtype>::Restore)
      .def("snapshot", &Solver<Dtype>::Snapshot)
      .def("shared_weights", &shared_weights);

  py::class_<Timer, shared_ptr<Timer>>(m, "Timer")
      .def(py::init<>())
      .def("start", &Timer::Start)
      .def("stop", &Timer::Stop)
      .def_property_readonly("ms", &Timer::MilliSeconds);
}

}  // namespace caffe