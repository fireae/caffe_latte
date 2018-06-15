#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <climits>
#include <cmath>
#include <fstream>   // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include "caffe/flags.hpp"
#include "caffe/logging.hpp"
#define cimg_display 0
#define cimg_use_jpeg
#include <CImg.h>

#include "caffe/util/device_alternate.hpp"
#ifdef _WIN32
#ifdef SIMPLE_EXPORT
#define CAFFE_API __declspec(dllimport)
#else
#define CAFFE_API __declspec(dllexport)
#endif
#else
#define CAFFE_API
#endif
// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
                                           \
 private:                                  \
  classname(const classname&);             \
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)             \
  CAFFE_API char gInstantiationGuard##classname; \
  template CAFFE_API class classname<float>;     \
  template CAFFE_API class classname<double>

#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::Forward_gpu(   \
      const std::vector<Blob<float>*>& bottom,   \
      const std::vector<Blob<float>*>& top);     \
  template void classname<double>::Forward_gpu(  \
      const std::vector<Blob<double>*>& bottom,  \
      const std::vector<Blob<double>*>& top);

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::Backward_gpu(   \
      const std::vector<Blob<float>*>& top,       \
      const std::vector<bool>& propagate_down,    \
      const std::vector<Blob<float>*>& bottom);   \
  template void classname<double>::Backward_gpu(  \
      const std::vector<Blob<double>*>& top,      \
      const std::vector<bool>& propagate_down,    \
      const std::vector<Blob<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname);    \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace caffe {

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isinf;
using std::isnan;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class CAFFE_API Caffe {
 public:
  ~Caffe();

  static Caffe& Get();

  enum Brew { CPU, GPU };

  // This random number generator facade hides  and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  CAFFE_API class RNG {
   public:
    CAFFE_API RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();

   private:
    class Generator;
    shared_ptr<Generator> generator_;
  };

  // Getters for rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
#ifdef USE_CUDA
  inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
  inline static curandGenerator_t curand_generator() {
    return Get().curand_generator_;
  }
#endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both  and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);
  // Parallel training
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static int solver_rank() { return Get().solver_rank_; }
  inline static void set_solver_rank(int val) { Get().solver_rank_ = val; }
  inline static bool multiprocess() { return Get().multiprocess_; }
  inline static void set_multiprocess(bool val) { Get().multiprocess_ = val; }
  inline static bool root_solver() { return Get().solver_rank_ == 0; }

 protected:
#ifdef USE_CUDA
  cublasHandle_t cublas_handle_;
  curandGenerator_t curand_generator_;
#endif
  shared_ptr<RNG> random_generator_;

  Brew mode_;

  // Parallel training
  int solver_count_;
  int solver_rank_;
  bool multiprocess_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
