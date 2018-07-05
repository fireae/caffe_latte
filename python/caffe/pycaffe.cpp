#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// these need to be included after boost on OS X
#include <fstream>  // NOLINT
#include <string>   // NOLINT(build/include_order)
#include <vector>   // NOLINT(build/include_order)
#if 1
#include "caffe/caffe.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/sgd_solvers.hpp"

namespace py = pybind11;
typedef float Dtype;

namespace caffe {

// Selecting mode.
void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }

void InitLog() { ::caffe::InitLogging(""); }
void InitLogLevel(int level) { InitLog(); }
void InitLogLevelPipe(int level, bool stderr) {
  // FLAGS_minloglevel = level;
  // FLAGS_logtostderr = stderr;
  InitLog();
}
void Log(const string& s) { LOG(INFO) << s; }

void set_random_seed(unsigned int seed) { Caffe::set_random_seed(seed); }

// For convenience, check that input files can be opened, and raise an
// exception that will send to Python if not (caffe could still crash
// later if the input files are disturbed before they are actually used, but
// this saves frustration in most cases).
static void CheckFile(const string& filename) {
  std::ifstream f(filename.c_str());
  if (!f.good()) {
    f.close();
    throw std::runtime_error("Could not open file " + filename);
  }
  f.close();
}

void Net_Save(const Net<Dtype>& net, string filename) {
  NetParameter net_param;
  net.ToProto(&net_param, false);
  // WriteProtoToBinaryFile(net_param, filename.c_str());
}

Solver<Dtype>* GetSolverFromFile(const string& filename) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(filename, &param);
  return SolverFloatRegistry()->Create(param.type(), param).get();
}

PYBIND11_MODULE(pycaffe, m) {
  m.doc() = "pybind11 pycaffe plugin";
  m.def("init_log", &InitLog, "a function init log");
  m.def("init_log", &InitLogLevel, "a function init log");
  m.def("init_log", &InitLogLevelPipe, "a function init log");
  m.def("log", &Log, "a function init log");
  m.def("set_mode_cpu", &set_mode_cpu);
  m.def("set_mode_gpu", &set_mode_gpu);
  m.def("set_random_seed", &set_random_seed);
  m.def("set_device", &Caffe::SetDevice);
  m.def("solver_count", &Caffe::solver_count);
  m.def("set_solver_count", &Caffe::set_solver_count);
  m.def("solver_rank", &Caffe::solver_rank);
  m.def("set_solver_rank", &Caffe::set_solver_rank);
  m.def("layer_type_list", &LayerRegistry<Dtype>::LayerTypeList);
}

}  // namespace caffe
#endif
