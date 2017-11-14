#include "detect/frcnn.h"
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
      "/home/wencc/Myplace/caffe_latte/apps/mtcnn_frcnn/model/"
      "vgg16_faster_rcnn_face.prototxt";
  string weights_file =
      "/home/wencc/Myplace/caffe_latte/apps/mtcnn_frcnn/model/"
      "vgg16_faster_rcnn_face.caffemodel";
  Caffe::set_mode(Caffe::CPU);
  FasterRCNN det;
  det.Init(model_file, weights_file);
  vector<FrcnnBox> box;
  det.Detect(image, box, 5);
  string lbf_regress_model = model_dir + "/LBF.model";
  shared_ptr<LbfCascador> lbf_cascador;
  lbf_cascador.reset(new LbfCascador());
  FILE* fd = fopen(lbf_regress_model.c_str(), "rb");
  lbf_cascador->Read(fd);
  fclose(fd);

  Rect face_box = box[0];
  BBox facebox(std::max<int>(0, face_box.x), std::max<int>(0, face_box.y),
               face_box.width, face_box.height);
  Mat face_shape = lbf_cascador->Predict(gray, facebox);
  cv::rectangle(show_image, face_box, cv::Scalar(0, 255, 0), 1);
  vector<cv::Point> face_points;
  for (int ipt = 0; ipt < 68; ipt++) {
    int xpt = face_shape.at<double>(ipt, 0);
    int ypt = face_shape.at<double>(ipt, 1);
    face_points.push_back(cv::Point(xpt, ypt));
    cv::circle(show_image, cv::Point(xpt, ypt), 2, cv::Scalar(255));
  }
  // cv::imwrite("det_align.jpg", show_image);
  //   for (int i = 0; i < face_infos.size(); i++) {
  //     cv::Rect face_bbox = face_infos[i].face_bbox;
  //     int max_span = std::max(face_bbox.width, face_bbox.height);
  //     int x = (max_span - face_bbox.width) / 2;
  //     int y = (max_span - face_bbox.height) / 2;
  //     face_bbox.x -= x;
  //     face_bbox.y -= y;
  //     cv::Rect crop_box = cv::Rect(std::max(0, face_bbox.x),
  //                                  std::max(0, face_bbox.y), max_span,
  //                                  max_span);
  //     cv::Mat img_patch = image(crop_box);
  //     if (img_patch.empty()) continue;
  //     char new_name_[512];
  //     sprintf(new_name_, "%s_%d.jpg", new_name.c_str(), i);
  //     cv::imwrite(new_name_, img_patch);
  //   }
  return 0;
}
