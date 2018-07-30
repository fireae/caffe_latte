#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"

namespace py = pybind11;
typedef float Dtype;

namespace caffe {

void set_mode_cpu() { Caffe::set_mode(Caffe::CPU);}
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU);}

void InitLog() { ::caffe::InitLogging("");}
void Log(const string& s) { LOG(INFO) << s; }

PYBIND11_MODULE(pycaffe, m) {
    m.doc()  = "pybind11 pycaffe plugin";
    m.def("init_log", &InitLog, "init log function");
    m.def("log", &Log, "a function to log out");
    m.def("set_mode_cpu", &set_mode_cpu);
    m.def("set_mode_gpu", &set_mode_gpu);
    
}

}