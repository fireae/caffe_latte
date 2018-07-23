#ifdef USE_EIGEN
#include "caffe/util/math_functions.hpp"
#include <Eigen/Dense>
#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <string.h> // memset

#include "caffe/common.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

#define MAP_SVECTOR(name, ptr, N) Eigen::Map<Eigen::VectorXf> name(ptr, N)
#define MAP_CONST_SVECTOR(name, ptr, N) Eigen::Map<const Eigen::VectorXf> name(ptr, N)
#define MAP_DVECTOR(name, ptr, N) Eigen::Map<Eigen::VectorXd> name(ptr, N)
#define MAP_CONST_DVECTOR(name, ptr, N) Eigen::Map<const Eigen::VectorXd> name(ptr, N)
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXf;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXd;

#define MAP_SMATRIX(name, ptr, M, N) Eigen::Map<MatXf> name(ptr, M, N)
#define MAP_CONST_SMATRIX(name, ptr, M, N) Eigen::Map<const MatXf> name(ptr, M, N)
#define MAP_DMATRIX(name, ptr, M, N) Eigen::Map<MatXd> name(ptr, M, N)
#define MAP_CONST_DMATRIX(name, ptr, M, N) Eigen::Map<const MatXd> name(ptr, M, N)

template<>
void caffe_cpu_gemm<float>(const CAFFE_BLAS_TRANSPOSE TransA, 
    const CAFFE_BLAS_TRANSPOSE TransB, const int M, const int N, 
    const int K, const float alpha, const float* A, const float* B, 
    const float beta, float* C) {
    MAP_SMATRIX(eC, C, M, N);
    eC *= beta;
    if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
        MAP_CONST_SMATRIX(eA, A, M, K);
        MAP_CONST_SMATRIX(eB, B, K, N);
        eC.noalias() += alpha * (eA * eB);
    } else if (TransA == CblasNoTrans && TransB == CblasTrans) {
        MAP_CONST_SMATRIX(eA, A, M, K);
        MAP_CONST_SMATRIX(eB, B, N, K);
        eC.noalias() += alpha * (eA * eB.transpose());
    } else if (TransA == CblasTrans && TransB == CblasNoTrans) {
        MAP_CONST_SMATRIX(eA, A, K, M);
        MAP_CONST_SMATRIX(eB, B, K, N);
        eC.noalias() += alpha * (eA.transpose() * eB);
    } else {
        MAP_CONST_SMATRIX(eA, A, K, M);
        MAP_CONST_SMATRIX(eB, B, N, K);
        eC.noalias() += alpha * (eA.transpose() * eB.transpose());
    }
}

template<>
void caffe_cpu_gemm<double>(const CAFFE_BLAS_TRANSPOSE TransA,
    const CAFFE_BLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
	MAP_DMATRIX(eC, C, M, N);
	eC *= beta;
	if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
		MAP_CONST_DMATRIX(eA, A, M, K);
		MAP_CONST_DMATRIX(eB, B, K, N);
		eC.noalias() += alpha * (eA * eB);
	} else if(TransA == CblasNoTrans && TransB == CblasTrans) {
		MAP_CONST_DMATRIX(eA, A, M, K);
		MAP_CONST_DMATRIX(eB, B, N, K);
		eC.noalias() += alpha * (eA * eB.transpose());
	} else if (TransA == CblasTrans && TransB == CblasNoTrans) {
		MAP_CONST_DMATRIX(eA, A, K, M);
		MAP_CONST_DMATRIX(eB, B, K, N);
		eC.noalias() += alpha * (eA.transpose() * eB);
	} else {
		MAP_CONST_DMATRIX(eA, A, K, M);
		MAP_CONST_DMATRIX(eB, B, N, K);
		eC.noalias() += alpha * (eA.transpose() * eB.transpose());
	}
}

template <>
void caffe_cpu_gemv<float>(const CAFFE_BLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
    MAP_CONST_SMATRIX(eA, A, M, N);
    if(TransA == CblasNoTrans){
        MAP_SVECTOR(eY, y, M);
        eY *= beta;
        MAP_CONST_SVECTOR(eX, x, N);
        eY.noalias() += alpha * (eA * eX);
    }else{
        MAP_SVECTOR(eY, y, N);
        eY *= beta;
        MAP_CONST_SVECTOR(eX, x, M);
        eY.noalias() += alpha * (eA.transpose() * eX);
    }
}

template <>
void caffe_cpu_gemv<double>(const CAFFE_BLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  MAP_CONST_DMATRIX(eA, A, M, N);
	if(TransA == CblasNoTrans){
		MAP_DVECTOR(eY, y, M);
		eY *= beta;
		MAP_CONST_DVECTOR(eX, x, N);
		eY.noalias() += alpha * (eA * eX);
	}else{
		MAP_DVECTOR(eY, y, N);
		eY *= beta;
		MAP_CONST_DVECTOR(eX, x, M);
		eY.noalias() += alpha * (eA.transpose() * eX);
	}
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { 
   MAP_SVECTOR(eY, Y, N);
	MAP_CONST_SVECTOR(eX, X, N);
	eY = alpha * eX + eY;
}

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { 
    MAP_DVECTOR(eY, Y, N);
	MAP_CONST_DVECTOR(eX, X, N);
	eY = alpha * eX + eY;
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifdef USE_CUDA
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  MAP_SVECTOR(eX, X, N);
	eX *= alpha;
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  MAP_DVECTOR(eX, X, N);
	eX *= alpha;
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  MAP_SVECTOR(eY, Y, N);
	MAP_CONST_SVECTOR(eX, X, N);
	eY = alpha * eX + beta * eY;
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  MAP_DVECTOR(eY, Y, N);
	MAP_CONST_DVECTOR(eX, X, N);
	eY = alpha * eX + beta * eY;
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return 0;//(*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return nextafter(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(gen);
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<Dtype> random_distribution(a, sigma);
  
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(gen);
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution random_distribution(p);
  
  for (int i = 0; i < n; ++i) {
    r[i] = random_distribution(gen);
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution random_distribution(p);
  
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(random_distribution(gen));
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
        MAP_CONST_SVECTOR(eX, x, n);
	MAP_CONST_SVECTOR(eY, y, n);
	return eX.dot(eY);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  MAP_CONST_DVECTOR(eX, x, n);
	MAP_CONST_DVECTOR(eY, y, n);
	return eX.dot(eY);
}

// template <typename Dtype>
// Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {

// }

template<>
float caffe_cpu_dot<float>(const int n, const float* x, const float* y) {
    MAP_CONST_SVECTOR(eX, x, n);
	MAP_CONST_SVECTOR(eY, y, n);
	return eX.dot(eY);
}

template<>
double caffe_cpu_dot<double>(const int n, const double* x, const double* y) {
    MAP_CONST_DVECTOR(eX, x, n);
	MAP_CONST_DVECTOR(eY, y, n);
	return eX.dot(eY);
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  float sum = 0.0;
  for (int i = 0; i < n; i++) {
      sum += std::abs(x[i]);
  }
  return sum;
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
      sum += std::abs(x[i]);
  }
  return sum;
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

} // namespace caffe
#endif // USE_EIGEN