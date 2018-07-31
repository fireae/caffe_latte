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
	using namespace std;

void set_mode_cpu() { Caffe::set_mode(Caffe::CPU);}
void set_mode_gpu() { Caffe::set_mode(Caffe::GPU);}

void InitLog() { ::caffe::InitLogging("");}
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

shared_ptr<Net<Dtype> > Net_Init(std::string network_file,
	int phase, const int level) {
	CheckFile(network_file);
	shared_ptr<Net<Dtype> > net(new Net<Dtype>(
		network_file, static_cast<Phase>(phase), level));


	return net;
}

shared_ptr<Net<Dtype> > Net_Init_Load(std::string param_file,
	std::string pretrained_param_file, int phase) {
	shared_ptr<Net<Dtype> > net( new Net<Dtype>(param_file, 
		static_cast<Phase>(phase)));
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

Solver<Dtype> * GetSolverFromFile(const string& filename) {
	SolverParameter param;
	ReadSolverParamsFromTextFileOrDie(filename, &param);
	return SolverFloatRegistry()->Create(param.type(), param).get();
}

void shared_weights(Solver<Dtype>* solver, Net<Dtype>* net) {
	net->ShareTrainedLayersWith(solver->net().get());
}

PYBIND11_MODULE(pycaffe, m) {
    m.doc()  = "pybind11 pycaffe plugin";
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

	py::class_<Net<Dtype> > net(m, "Net");
	net.def("log", &InitLog);
	
}

}