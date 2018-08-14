#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/solver.hpp"
#include "py_layer.h"

namespace py = pybind11;
typedef float Dtype;
namespace caffe {

void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }

void InitLog() { ::caffe::InitLogging(""); }
void Log(const string &s) { LOG(INFO) << s; }
void set_random_seed(unsigned int seed) { Caffe::set_random_seed(seed); }

void Numpy2BlobData(Blob<Dtype> &m, py::array_t<Dtype> info) {
  py::print("Numpy2BlobData");
  m.Reshape(info.shape()[0], info.shape()[1], info.shape()[2], info.shape()[3]);
  m.set_cpu_data(const_cast<Dtype *>(info.data()));
}

py::array_t<Dtype> BlobData2Numpy(Blob<Dtype> &m) {
  py::print("BlobData2Numpy");
  py::array_t<Dtype> a({m.num(), m.channels(), m.height(), m.width()},
                       {sizeof(Dtype) * m.channels() * m.height() * m.width(),
                        sizeof(Dtype) * m.height() * m.width(),
                        sizeof(Dtype) * m.width(), sizeof(Dtype)},
                       m.mutable_cpu_data());
  return a;
}

void Numpy2BlobDiff(Blob<Dtype> &m, py::array_t<Dtype> info) {
  py::print("Numpy2BlobDiff");
  m.Reshape(info.shape()[0], info.shape()[1], info.shape()[2], info.shape()[3]);
  Dtype *diff = static_cast<Dtype *>(m.mutable_cpu_diff());
  diff = const_cast<Dtype *>(info.data());
  // m.set_cpu_diff(const_cast<Dtype*>(info.data()));
}

py::array_t<Dtype> BlobDiff2Numpy(Blob<Dtype> &m) {
  py::print("BlobDiff2Numpy");
  py::array_t<Dtype> a({m.num(), m.channels(), m.height(), m.width()},
                       {sizeof(Dtype) * m.channels() * m.height() * m.width(),
                        sizeof(Dtype) * m.height() * m.width(),
                        sizeof(Dtype) * m.width(), sizeof(Dtype)},
                       m.mutable_cpu_diff());
  return a;
}

void Blob_Reshape(Blob<Dtype> &self, py::args args, py::kwargs kwargs) {
  if (py::len(kwargs) > 0) {
    throw std::runtime_error("Blob.reshape takes no kwargs");
  }

  vector<int> shape(py::len(args));
  for (int i = 0; i < py::len(args); ++i) {
    shape[i] = args[i].cast<int>();
  }
  self.Reshape(shape);
}

static void CheckFile(const std::string &filename) {
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

void Net_Save(const Net<Dtype> &net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  WriteProtoToBinaryFile(net_param, filename);
}

void share_weights(Solver<Dtype> *solver, Net<Dtype> *net) {
  net->ShareTrainedLayersWith(solver->net().get());
}

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

  py::class_<Net<Dtype>, shared_ptr<Net<Dtype>>> net(m, "Net",
                                                     py::dynamic_attr());
  // net.def(py::init(&Net_Init, py::arg("phase") = 0, py::arg("level") = 0));
  net.def(py::init<>(&Net_Init_Load));
  net.def("_forward", &Net<Dtype>::ForwardFromTo);
  net.def("_backward", &Net<Dtype>::BackwardFromTo);
  net.def("reshape", &Net<Dtype>::Reshape);
  net.def("clear_param_diffs", &Net<Dtype>::ClearParamDiffs);
  net.def("copy_from", static_cast<void (Net<Dtype>::*)(const string)>(
                           &Net<Dtype>::CopyTrainedLayersFrom));
  net.def("share_with", &Net<Dtype>::ShareTrainedLayersWith);
  net.def("save", &Net_Save);
  net.def_property_readonly("_blob_loss_weights",
                            &Net<Dtype>::blob_loss_weights);
  net.def_property_readonly("_bottom_ids", &Net<Dtype>::bottom_ids);
  net.def_property_readonly("_top_ids", &Net<Dtype>::top_ids);
  net.def("layers", &Net<Dtype>::layers,
          py::return_value_policy::reference_internal);
  net.def_property_readonly("_blob_names", &Net<Dtype>::blob_names);
  net.def_property_readonly("_layer_names", &Net<Dtype>::layer_names);
  net.def_property_readonly("_inputs", &Net<Dtype>::input_blob_indices);
  net.def_property_readonly("_outputs", &Net<Dtype>::output_blob_indices);
  net.def_property_readonly("_blobs", &Net<Dtype>::blobs);

  py::class_<Blob<Dtype>, shared_ptr<Blob<Dtype>>> blob(m, "Blob",
                                                        py::buffer_protocol());
  blob.def(py::init<>());
  blob.def(py::init<const int, const int, const int, const int>());
  blob.def("shape", static_cast<const vector<int> &(Blob<Dtype>::*)() const>(
                        &Blob<Dtype>::shape));
  blob.def("reshape", &Blob_Reshape);
  blob.def("num", &Blob<Dtype>::num);
  blob.def("channels", &Blob<Dtype>::channels);
  blob.def("height", &Blob<Dtype>::height);
  blob.def("width", &Blob<Dtype>::width);
  blob.def("count",
           static_cast<int (Blob<Dtype>::*)() const>(&Blob<Dtype>::count));
  blob.def_property("data", &BlobData2Numpy, &Numpy2BlobData);
  blob.def_property("diff", &BlobDiff2Numpy, &Numpy2BlobDiff);

  py::class_<LayerParameter>(m, "LayerParameter").def(py::init<>());

  py::class_<Layer<Dtype>, PyLayer> layer(m, "Layer");
  layer.def("setup", &Layer<Dtype>::LayerSetUp);
  layer.def("reshape", &Layer<Dtype>::Reshape);
  layer.def("blobs", &Layer<Dtype>::blobs);
  layer.def_property_readonly("type", &Layer<Dtype>::type);

  py::class_<SolverParameter>(m, "SolverParameter")
      .def_property_readonly("max_iter", &SolverParameter::max_iter)
      .def_property_readonly("display", &SolverParameter::display)
      .def_property_readonly("layer_wise_reduce",
                             &SolverParameter::layer_wise_reduce);

  py::class_<Solver<Dtype>, shared_ptr<Solver<Dtype>>> solver(m, "Solver");
  solver.def_property_readonly("net", &Solver<Dtype>::net);
  solver.def_property_readonly("test_nets", &Solver<Dtype>::test_nets,
                               py::return_value_policy::reference_internal);
  solver.def_property_readonly("iter", &Solver<Dtype>::iter);
  solver.def("step", &Solver<Dtype>::Step);
  solver.def("restore", &Solver<Dtype>::Restore);
  solver.def("snapshot", &Solver<Dtype>::Snapshot);
  solver.def("share_weights", &share_weights);

  py::class_<SGDSolver<Dtype>, shared_ptr<SGDSolver<Dtype>>>(m, "SGDSolver",
                                                             solver)
      .def(py::init<string>());

  py::class_<NesterovSolver<Dtype>, shared_ptr<NesterovSolver<Dtype>>>(
      m, "NesterovSolver", solver)
      .def(py::init<string>());

  py::class_<AdaGradSolver<Dtype>, shared_ptr<AdaGradSolver<Dtype>>>(
      m, "AdaGradSolver", solver)
      .def(py::init<string>());

  py::class_<RMSPropSolver<Dtype>, shared_ptr<RMSPropSolver<Dtype>>>(
      m, "RMSPropSolver", solver)
      .def(py::init<string>());

  py::class_<AdaDeltaSolver<Dtype>, shared_ptr<AdaDeltaSolver<Dtype>>>(
      m, "AdaDeltaSolver", solver)
      .def(py::init<string>());

  py::class_<AdamSolver<Dtype>, shared_ptr<AdamSolver<Dtype>>>(m, "AdamSolver",
                                                               solver)
      .def(py::init<string>());

  py::class_<Timer, shared_ptr<Timer>>(m, "Timer")
      .def(py::init<>())
      .def("start", &Timer::Start)
      .def("stop", &Timer::Stop)
      .def_property_readonly("ms", &Timer::MilliSeconds);
}

}  // namespace caffe