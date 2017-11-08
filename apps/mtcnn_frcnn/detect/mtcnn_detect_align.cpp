
#include "mtcnn_detect_align.h"

MTCNNDetectAlign::MTCNNDetectAlign(const string& model_dir) { Init(model_dir); }

bool MTCNNDetectAlign::Init(const string& model_dir) {
  mtcnn.reset(new MTCNN(model_dir));
  frcnn_detector.reset(new FRCNNDetector(model_dir));

  string lbf_regress_model = model_dir + "/LBF.model";
  lbf_cascador.reset(new LbfCascador());
  FILE* fd = fopen(lbf_regress_model.c_str(), "rb");
  lbf_cascador->Read(fd);
  fclose(fd);
  return true;
}

int MTCNNDetectAlign::Detect(const cv::Mat& image,
                             vector<JDFaceInfo>& face_infos) {
  cv::Mat image2 = image;
  cv::Mat image_left = image2.clone();
  cv::transpose(image2, image_left);
  cv::flip(image_left, image_left, 1);

  cv::Mat image_b = image2.clone();
  cv::transpose(image_left, image_b);
  cv::flip(image_b, image_b, 1);

  cv::Mat image_r = image2.clone();
  cv::transpose(image_b, image_r);
  cv::flip(image_r, image_r, 1);

  vector<Mat> images;
  images.push_back(image2);
  images.push_back(image_left);
  images.push_back(image_b);
  images.push_back(image_r);

  vector<vector<FaceInfo> > faceinfos;
  mtcnn->Detect(images, faceinfos);

  int face_count = 0;
  for (int t = 0; t < faceinfos.size(); t++) {
    vector<FaceInfo>& faceInfo = faceinfos[t];
    for (int i = 0; i < faceInfo.size(); i++) {
      float x = faceInfo[i].bbox.x1;
      float y = faceInfo[i].bbox.y1;
      float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
      float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
      JDFaceInfo jd_face_info;
      jd_face_info.face_bbox = cv::Rect(y, x, w, h);
      jd_face_info.score = faceInfo[i].bbox.score;
      jd_face_info.type = t;
      face_infos.push_back(jd_face_info);
    }

    face_count += faceinfos[t].size();
  }
  if (face_count == 0) {
    vector<FrcnnFaceInfo> frcnn_face_infos;
    frcnn_detector->Detect(image, frcnn_face_infos);

    for (int k = 0; k < frcnn_face_infos.size(); k++) {
      float x = frcnn_face_infos[k].x1;
      float y = frcnn_face_infos[k].y1;
      float h = frcnn_face_infos[k].x2 - frcnn_face_infos[k].x1 + 1;
      float w = frcnn_face_infos[k].y2 - frcnn_face_infos[k].y1 + 1;
      JDFaceInfo jd_face_info;
      jd_face_info.face_bbox = cv::Rect(y, x, w, h);
      jd_face_info.score = frcnn_face_infos[k].score;
      jd_face_info.type = frcnn_face_infos[k].type;
    }
  }

  //   cv::Mat gray;
  //   cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  //   for (int i = 0; i < faceInfo.size(); i++) {
  //     float x = faceInfo[i].bbox.x1;
  //     float y = faceInfo[i].bbox.y1;
  //     float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
  //     float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
  //     JDFaceInfo jd_face_info;
  //     jd_face_info.face_bbox = cv::Rect(y, x, w, h);
  //     jd_face_info.score = faceInfo[i].bbox.score;

  //     BBox facebox(std::max<int>(0, y), std::max<int>(0, x), w, h);
  //     Mat face_shape = lbf_cascador->Predict(gray, facebox);
  //     vector<cv::Point> face_points;
  //     for (int ipt = 0; ipt < 68; ipt++) {
  //       int xpt = face_shape.at<double>(ipt, 0);
  //       int ypt = face_shape.at<double>(ipt, 1);
  //       face_points.push_back(cv::Point(xpt, ypt));
  //     }

  //     jd_face_info.face_points = face_points;
  //     face_infos.push_back(jd_face_info);
  //   }
}