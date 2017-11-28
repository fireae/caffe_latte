#include "frcnn.h"
#include <string>
#include <vector>
#include "caffe/caffe.hpp"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace frcnn;
using namespace caffe;
int main(int argc, char* argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  cv::Mat image = cv::imread(argv[1]);
  string model_file =
      "/home/wencc/Myplace/caffe_latte/apps/rcnn/model/"
      "vgg16_faster_rcnn_face.prototxt";
  string weights_file =
      "/home/wencc/Myplace/caffe_latte/apps/rcnn/model/"
      "vgg16_faster_rcnn_face.caffemodel";
  // int GPUID = 0;
  // Caffe::SetDevice(GPUID);
  // Caffe::set_mode(Caffe::GPU);
  Caffe::set_mode(Caffe::CPU);
  FasterRCNN det;
  det.Init(model_file, weights_file);
  vector<FrcnnBox> box;
  det.Detect(image, box, 5);
  return 0;
}
