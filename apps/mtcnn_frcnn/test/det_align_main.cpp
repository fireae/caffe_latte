#include <glog/logging.h>
#include "mtcnn_detect_align.h"

int main(int argc, char* argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  string model_dir = argv[1];
  string image_path = argv[2];
  string new_name = argv[3];
  MTCNNDetectAlign mt_detect_align;
  mt_detect_align.Init(model_dir);

  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cout << "can not open image file " << image_path;
    return 0;
  }

  vector<JDFaceInfo> face_infos;
  mt_detect_align.Detect(image, face_infos);

  cv::Mat show_image = image.clone();
  for (int i = 0; i < face_infos.size(); i++) {
    cv::rectangle(show_image, face_infos[i].face_bbox, cv::Scalar(255, 0, 0),
                  1);
    for (int j = 0; j < face_infos[i].face_points.size(); j++) {
      cv::circle(show_image, face_infos[i].face_points[j], 1,
                 cv::Scalar(255, 0, 255), 1);
    }
    char new_name_[512];
    sprintf(new_name_, "%s_%d.jpg", new_name.c_str(), i);
    cv::imwrite(new_name_, show_image);
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