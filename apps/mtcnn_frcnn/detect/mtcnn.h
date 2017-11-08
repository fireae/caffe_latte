#ifndef MTCNN_H_
#define MTCNN_H_

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

// c++
#include <fstream>
#include <string>
#include <vector>
// opencv
#include <opencv2/opencv.hpp>
// boost
#include "boost/make_shared.hpp"

//#define CPU_ONLY
#define INTER_FAST
using namespace caffe;

typedef struct FaceRect {
  float x1;
  float y1;
  float x2;
  float y2;
  float score; /**< Larger score should mean higher confidence. */

} FaceRect;

typedef struct FacePts { float x[5], y[5]; } FacePts;

typedef struct FaceInfo {
  FaceRect bbox;
  cv::Vec4f regression;
  FacePts facePts;
  double roll;
  double pitch;
  double yaw;
} FaceInfo;

class MTCNN {
 public:
  MTCNN(const string& proto_model_dir);

  void Detect(const cv::Mat& image, vector<FaceInfo>& face_infos);
  void Detect(const vector<cv::Mat>& images,
              vector<vector<FaceInfo>>& face_info_vec);

 private:
  void PutImageToInputLayer(const cv::Mat& image, Blob<float>* input_layer,
                            int width, int height);
  void PreprocessImage(const cv::Mat& image, cv::Mat& rgb_image);
  void PNetDetect(const cv::Mat& rgb_image, double thresh, double factor,
                  int min_size, vector<FaceInfo>& pnet_bboxes);
  void RNetDetect(const cv::Mat& rgb_image, double thresh,
                  vector<FaceInfo>& candidate_bboxes,
                  vector<FaceInfo>& rnet_bboxes);
  void ONetDetect(const cv::Mat& rgb_image, double thresh,
                  vector<FaceInfo>& candidate_bboxes,
                  vector<FaceInfo>& onet_bboxes);
  void PredictImage(boost::shared_ptr<Net<float>> net, const cv::Mat& rgb_image,
                    int width, int height, vector<Blob<float>*>& output_blobs);
  void DetectFace(const cv::Mat& rgb_image, double thresh,
                  vector<FaceInfo>& candidate_bboxes, int type,
                  vector<FaceInfo>& rnet_bboxes);

 private:
  boost::shared_ptr<Net<float>> PNet_;
  boost::shared_ptr<Net<float>> RNet_;
  boost::shared_ptr<Net<float>> ONet_;
  float pnet_thresh_;
  float rnet_thresh_;
  float onet_thresh_;
  float factor_;
  float min_size_;
};

#endif  // MTCNN_H_