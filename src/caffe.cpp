#include <caffe/flags.hpp>
#include <caffe/logging.hpp>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include "caffe/util/string.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using namespace caffe;  // NOLINT(build/namespaces)


void CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
	std::vector<std::string> model_names = caffe::SplitString(model_list, ",");
	for (int i = 0; i < model_names.size(); ++i) {
		solver->net()->CopyTrainedLayersFrom(model_names[i]);
		for (int j = 0; j < solver->test_nets().size(); ++j) {
			solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
		}
	}
}

int main(int argc, char* argv[]) {
	/*ItemRepository g;
	InitItemRepository(&g, 10);
	std::thread t0(ProduceTask, &g);
	std::thread c0(ConsumeTask, &g);
	std::thread t1(ProduceTask, &g);
	std::thread c1(ConsumeTask, &g);
	c0.join();
	t0.join();
	c1.join();
	t1.join();*/
	caffe::SolverParameter solver_param;
	std::string FLAGS_solver  = "lenet_solver.prototxt";
	std::string FLAGS_weights;
	caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
	shared_ptr<caffe::Solver<float> > solver(
		caffe::SolverFloatRegistry()->Create(solver_param.type(), solver_param));
	
	//CopyLayers(solver.get(), FLAGS_weights);
	solver->Solve();
}	