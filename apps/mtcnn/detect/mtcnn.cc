#include "mtcnn.h"

MTCNN::MTCNN(const std::string& proto_model_dir) {
#ifndef USE_CUDA
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  pnet_thresh_ = 0.3;  // 0.6;
  rnet_thresh_ = 0.4;  // 0.7;
  onet_thresh_ = 0.4;  // 0.7;
  factor_ = 0.709;
  min_size_ = 40;
  /* Load the network. */
  PNet_.reset(new Net<float>((proto_model_dir + "/det1.prototxt"), TEST));
  PNet_->CopyTrainedLayersFrom(proto_model_dir + "/det1.caffemodel");

  CHECK_EQ(PNet_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(PNet_->num_outputs(), 2)
      << "Network should have exactly two output, one"
         " is bbox and another is confidence.";

#ifndef USE_CUDA
  RNet_.reset(new Net<float>((proto_model_dir + "/det2.prototxt"), TEST));
#else
  RNet_.reset(new Net<float>((proto_model_dir + "/det2_input.prototxt"), TEST));
#endif
  RNet_->CopyTrainedLayersFrom(proto_model_dir + "/det2.caffemodel");

//  CHECK_EQ(RNet_->num_inputs(), 0) << "Network should have exactly one
//  input.";
//  CHECK_EQ(RNet_->num_outputs(),3) << "Network should have exactly two output,
//  one"
//                                     " is bbox and another is confidence.";

#ifndef USE_CUDA
  ONet_.reset(new Net<float>((proto_model_dir + "/det3.prototxt"), TEST));
#else
  ONet_.reset(new Net<float>((proto_model_dir + "/det3_input.prototxt"), TEST));
#endif
  ONet_->CopyTrainedLayersFrom(proto_model_dir + "/det3.caffemodel");

  //  CHECK_EQ(ONet_->num_inputs(), 1) << "Network should have exactly one
  //  input.";
  //  CHECK_EQ(ONet_->num_outputs(),3) << "Network should have exactly three
  //  output, one"
  //                                     " is bbox and another is confidence.";
}

///////////////////////////////////////////////

void GetBoundingBox(Blob<float>* confidence, Blob<float>* regression,
                    float scale, float thresh, int image_width,
                    int image_height, vector<FaceInfo>& candidate_rects) {
  int stride = 2;
  int cell_size = 12;

  int curr_feature_map_w_ =
      std::ceil((image_width - cell_size) * 1.0 / stride) + 1;
  int curr_feature_map_h_ =
      std::ceil((image_height - cell_size) * 1.0 / stride) + 1;

  int regression_offset = curr_feature_map_w_ * curr_feature_map_h_;

  int count = confidence->count() / 2;
  const float* confidence_data = confidence->cpu_data();
  confidence_data += count;
  const float* regression_data = regression->cpu_data();

  for (int i = 0; i < count; i++) {
    if (*(confidence_data + i) >= thresh) {
      int y = i / curr_feature_map_w_;
      int x = i - curr_feature_map_w_ * y;

      float x_top = (int)((x * stride + 1) / scale);
      float y_top = (int)((y * stride + 1) / scale);
      float x_bottom = (int)((x * stride + cell_size - 1 + 1) / scale);
      float y_bottom = (int)((y * stride + cell_size - 1 + 1) / scale);

      FaceRect face_rect;
      face_rect.x1 = x_top;
      face_rect.y1 = y_top;
      face_rect.x2 = x_bottom;
      face_rect.y2 = y_bottom;
      face_rect.score = *(confidence_data + i);
      FaceInfo face_info;
      face_info.bbox = face_rect;
      face_info.regression =
          cv::Vec4f(regression_data[i + 0 * regression_offset],
                    regression_data[i + 1 * regression_offset],
                    regression_data[i + 2 * regression_offset],
                    regression_data[i + 3 * regression_offset]);
      candidate_rects.push_back(face_info);
    }
  }
}

bool CompareBBox(const FaceInfo& a, const FaceInfo& b) {
  return a.bbox.score > b.bbox.score;
}

float IoU(const FaceInfo& a, const FaceInfo& b) {
  FaceRect a_rect = a.bbox;
  FaceRect b_rect = b.bbox;

  float x = std::max<float>(static_cast<float>(a_rect.x1),
                            static_cast<float>(b_rect.x1));
  float y = std::max<float>(static_cast<float>(a_rect.y1),
                            static_cast<float>(b_rect.y1));
  float w = std::min<float>(static_cast<float>(a_rect.x2),
                            static_cast<float>(b_rect.x2)) -
            x + 1;
  float h = std::min<float>(static_cast<float>(a_rect.y2),
                            static_cast<float>(b_rect.y2)) -
            y + 1;
  if (w <= 0 || h <= 0) {
    return 0.0;
  }

  float area1 = (a_rect.x2 - a_rect.x1 + 1) * (a_rect.y2 - a_rect.y1 + 1);
  float area2 = (b_rect.x2 - b_rect.x1 + 1) * (b_rect.y2 - b_rect.y1 + 1);
  float area_intersect = w * h;

  return (area_intersect / (area1 + area2 - area_intersect));
}

float IoM(const FaceInfo& a, const FaceInfo& b) {
  FaceRect a_rect = a.bbox;
  FaceRect b_rect = b.bbox;

  float x = std::max<float>(static_cast<float>(a_rect.x1),
                            static_cast<float>(b_rect.x1));
  float y = std::max<float>(static_cast<float>(a_rect.y1),
                            static_cast<float>(b_rect.y1));
  float w = std::min<float>(static_cast<float>(a_rect.x2),
                            static_cast<float>(b_rect.x2)) -
            x + 1;
  float h = std::min<float>(static_cast<float>(a_rect.y2),
                            static_cast<float>(b_rect.y2)) -
            y + 1;
  if (w <= 0 || h <= 0) {
    return 0.0;
  }

  float area1 = (a_rect.x2 - a_rect.x1 + 1) * (a_rect.y2 - a_rect.y1 + 1);
  float area2 = (b_rect.x2 - b_rect.x1 + 1) * (b_rect.y2 - b_rect.y1 + 1);
  float area_intersect = w * h;

  return static_cast<float>(area_intersect / std::max<float>(area1, area2));
}

void NMS(vector<FaceInfo>& face_infos, double thresh,
         vector<FaceInfo>& face_infos_nms, int method = 0) {
  std::sort(face_infos.begin(), face_infos.end(), CompareBBox);

  vector<bool> merge_mask(face_infos.size(), false);
  for (int i = 0; i < face_infos.size(); i++) {
    if (merge_mask[i]) {
      continue;
    }

    for (int j = i + 1; j < face_infos.size(); j++) {
      if (merge_mask[j] || i == j) {
        continue;
      }

      double iou = IoU(face_infos[i], face_infos[j]);
      if (method == 1) {
        iou = IoM(face_infos[i], face_infos[j]);
      }
      if (iou > thresh) {
        merge_mask[j] = true;
      }
    }
  }

  for (int i = 0; i < face_infos.size(); i++) {
    if (!merge_mask[i]) {
      face_infos_nms.push_back(face_infos[i]);
    }
  }
}

void BoxRegression(vector<FaceInfo>& face_infos,
                   vector<FaceInfo>& face_infos_reg, int stage) {
  for (int i = 0; i < face_infos.size(); i++) {
    FaceRect face_rect;
    FaceInfo temp_face_info;
    float regw = face_infos[i].bbox.y2 - face_infos[i].bbox.y1;
    float regh = face_infos[i].bbox.x2 - face_infos[i].bbox.x1;
    regw += (stage == 1) ? 0 : 1;
    regh += (stage == 1) ? 0 : 1;

    face_rect.y1 = face_infos[i].bbox.y1 + regw * face_infos[i].regression[0];
    face_rect.x1 = face_infos[i].bbox.x1 + regh * face_infos[i].regression[1];
    face_rect.y2 = face_infos[i].bbox.y2 + regw * face_infos[i].regression[2];
    face_rect.x2 = face_infos[i].bbox.x2 + regh * face_infos[i].regression[3];
    face_rect.score = face_infos[i].bbox.score;
    temp_face_info.bbox = face_rect;
    temp_face_info.regression = face_infos[i].regression;
    if (stage == 3) {
      temp_face_info.facePts = face_infos[i].facePts;
    }

    face_infos_reg.push_back(temp_face_info);
  }
}

void MTCNN::PreprocessImage(const cv::Mat& image, cv::Mat& rgb_image) {
  image.convertTo(rgb_image, CV_32FC3);
  cvtColor(rgb_image, rgb_image, cv::COLOR_BGR2RGB);
  rgb_image = rgb_image.t();
}

void MTCNN::Detect(const cv::Mat& image, vector<FaceInfo>& face_infos) {
  if (image.empty()) {
    return;
  }

  cv::Mat rgb_image;
  PreprocessImage(image, rgb_image);

  vector<FaceInfo> pnet_bboxes;
  PNetDetect(rgb_image, pnet_thresh_, factor_, min_size_, pnet_bboxes);
  if (pnet_bboxes.size() <= 0) {
    return;
  }

  vector<FaceInfo> rnet_bboxes;
  DetectFace(rgb_image, rnet_thresh_, pnet_bboxes, 0, rnet_bboxes);
  if (rnet_bboxes.size() <= 0) {
    return;
  }

  vector<FaceInfo> onet_bboxes;
  DetectFace(rgb_image, onet_thresh_, rnet_bboxes, 1, onet_bboxes);
  // std::cout << "onet box num: " << onet_bboxes.size() << std::endl;
  // for (int o = 0; o < onet_bboxes.size(); o++) {
  //   std::cout << "onet : " <<onet_bboxes[o].facePts.x[0] << std::endl;
  // }

  vector<FaceInfo> onet_bboxes_reg;
  BoxRegression(onet_bboxes, onet_bboxes_reg, 3);
  vector<FaceInfo> onet_nms;
  NMS(onet_bboxes_reg, 0.7, onet_nms, 1);
  face_infos = onet_nms;
  // cv::Mat image_show = image.clone();
  // for (int om = 0; om < onet_nms.size(); om++) {
  //   std::cout << "onet nms: " << onet_nms[om].facePts.x[1] << std::endl;
  //   FaceRect face_rect = onet_nms[om].bbox;
  //   int x = face_rect.x1;
  //   int y = face_rect.y1;
  //   int w = face_rect.x2 - face_rect.x1;
  //   int h = face_rect.y2 - face_rect.y1;

  //   //cv::rectangle(image_show,cv::Rect(y,x,h, w),cv::Scalar(255,0,0),2);
  //   //FacePts facePts = onet_nms[om].facePts;
  //   // for(int j=0;j<5;j++)
  //   //
  //   cv::circle(image_show,cv::Point(facePts.y[j],facePts.x[j]),1,cv::Scalar(255,255,0),2);
  // }

  // cv::imwrite("d.jpg", image_show);
}

void MTCNN::Detect(const vector<cv::Mat>& images,
                   vector<vector<FaceInfo>>& face_info_vec) {
  face_info_vec.resize(images.size());
  for (int i = 0; i < images.size(); i++) {
    Detect(images[i], face_info_vec[i]);
  }
}

void MTCNN::PutImageToInputLayer(const cv::Mat& image, Blob<float>* input_layer,
                                 int width, int height) {
  float* input_data = input_layer->mutable_cpu_data();
  int input_idx = 0;
  int stride = width * height;
  int stride2 = 2 * width * height;
  for (int h = 0; h < height; h++) {
    const float* ptr = image.ptr<float>(h);
    for (int w = 0; w < width; w++) {
      input_data[input_idx] = ptr[w * 3];
      input_data[input_idx + stride] = ptr[w * 3 + 1];
      input_data[input_idx + stride2] = ptr[w * 3 + 2];
      input_idx++;
    }
  }
}

void MTCNN::PredictImage(boost::shared_ptr<Net<float>> net,
                         const cv::Mat& rgb_image, int width, int height,
                         vector<Blob<float>*>& output_blobs) {
  Blob<float>* input_layer = net->input_blobs()[0];
  cv::Mat resized_image;
  cv::resize(rgb_image, resized_image, cv::Size(width, height), 0, 0,
             cv::INTER_AREA);
#ifdef INTER_FAST
  cv::resize(rgb_image, resized_image, cv::Size(width, height), 0, 0,
             cv::INTER_NEAREST);
#else
  cv::resize(rgb_image, resized_image, cv::Size(width, height), 0, 0,
             cv::INTER_AREA);
#endif
  resized_image.convertTo(resized_image, CV_32FC3, 0.0078125,
                          -127.5 * 0.0078125);

  input_layer->Reshape(1, input_layer->channels(), height, width);
  net->Reshape();

  PutImageToInputLayer(resized_image, input_layer, width, height);
  net->Forward();

  output_blobs = net->output_blobs();
}

void MTCNN::PNetDetect(const cv::Mat& rgb_image, double thresh, double factor,
                       int min_size, vector<FaceInfo>& pnet_bboxes) {
  boost::shared_ptr<Net<float>> net = PNet_;
  int min_wh = std::min(rgb_image.rows, rgb_image.cols);
  int factor_count = 0;
  double m = 12.0 / (1.0 * (min_size));
  min_wh *= m;
  std::vector<double> scales;
  while (min_wh >= 12) {
    scales.push_back(m * std::pow(factor, factor_count));
    min_wh *= factor;
    ++factor_count;
  }

  Blob<float>* input_layer = net->input_blobs()[0];

  for (int i = 0; i < factor_count; i++) {
    double scale = scales[i];
    int width_scale = std::ceil(rgb_image.cols * scale);
    int height_scale = std::ceil(rgb_image.rows * scale);
    // std::cout << width_scale << "--" << height_scale << std::endl;

    vector<Blob<float>*> output_blobs;
    PredictImage(net, rgb_image, width_scale, height_scale, output_blobs);
    Blob<float>* regression = output_blobs[0];
    Blob<float>* confidence = output_blobs[1];
    vector<FaceInfo> candidate_rects;
    GetBoundingBox(confidence, regression, scale, 0.6, width_scale,
                   height_scale, candidate_rects);
    vector<FaceInfo> candidate_bboxes_nms;
    NMS(candidate_rects, 0.5, candidate_bboxes_nms);
    pnet_bboxes.insert(pnet_bboxes.end(), candidate_bboxes_nms.begin(),
                       candidate_bboxes_nms.end());
  }
}

void BBox2Square(std::vector<FaceInfo>& face_infos) {
  for (int i = 0; i < face_infos.size(); i++) {
    float w = face_infos[i].bbox.x2 - face_infos[i].bbox.x1;
    float h = face_infos[i].bbox.y2 - face_infos[i].bbox.y1;
    float side = h > w ? h : w;
    face_infos[i].bbox.x1 += (w - side) * 0.5;
    face_infos[i].bbox.y1 += (h - side) * 0.5;

    face_infos[i].bbox.x2 = (int)(face_infos[i].bbox.x1 + side);
    face_infos[i].bbox.y2 = (int)(face_infos[i].bbox.y1 + side);
    face_infos[i].bbox.x1 = (int)(face_infos[i].bbox.x1);
    face_infos[i].bbox.y1 = (int)(face_infos[i].bbox.y1);
  }
}

void FacePadding(std::vector<FaceInfo>& face_infos,
                 vector<FaceInfo>& face_infos_padding, int image_width,
                 int image_height) {
  for (int i = 0; i < face_infos.size(); i++) {
    FaceInfo temp_face_info;
    temp_face_info = face_infos[i];
    temp_face_info.bbox.x1 = face_infos[i].bbox.x1;
    temp_face_info.bbox.x2 = face_infos[i].bbox.x2;
    temp_face_info.bbox.y1 = face_infos[i].bbox.y1;
    temp_face_info.bbox.y2 = face_infos[i].bbox.y2;
    if (temp_face_info.bbox.x1 <= 1) {
      temp_face_info.bbox.x1 = 1;
    }
    if (temp_face_info.bbox.x1 >= (image_width - 1)) {
      temp_face_info.bbox.x1 = image_width - 1;
    }
    if (temp_face_info.bbox.y1 <= 1) {
      temp_face_info.bbox.y1 = 1;
    }
    if (temp_face_info.bbox.y1 >= (image_height - 1)) {
      temp_face_info.bbox.y1 = image_height - 1;
    }
    if (temp_face_info.bbox.x2 <= 1) {
      temp_face_info.bbox.x2 = 1;
    }
    if (temp_face_info.bbox.x2 >= (image_width - 1)) {
      temp_face_info.bbox.x2 = image_width - 1;
    }
    if (temp_face_info.bbox.y2 <= 1) {
      temp_face_info.bbox.y2 = 1;
    }
    if (temp_face_info.bbox.y2 >= (image_height - 1)) {
      temp_face_info.bbox.y2 = image_height - 1;
    }

    if (temp_face_info.bbox.x1 > temp_face_info.bbox.x2) {
      float t = temp_face_info.bbox.x1;
      temp_face_info.bbox.x1 = temp_face_info.bbox.x2;
      temp_face_info.bbox.x2 = t;
    }
    if (temp_face_info.bbox.y1 > temp_face_info.bbox.y2) {
      float t = temp_face_info.bbox.y1;
      temp_face_info.bbox.y1 = temp_face_info.bbox.y2;
      temp_face_info.bbox.y2 = t;
    }
    // std::cout << "temp face bbox " << temp_face_info.bbox.x1 << " " <<
    // temp_face_info.bbox.x2 << " " << temp_face_info.bbox.y1 << " " <<
    // temp_face_info.bbox.y2 << std::endl;
    // temp_face_info.bbox.x1 = (face_infos[i].bbox.x1 < 1) ? 1 :
    // face_infos[i].bbox.x1;
    // temp_face_info.bbox.y1 = (face_infos[i].bbox.y1 < 1) ? 1 :
    // face_infos[i].bbox.y1;
    // temp_face_info.bbox.x2 = (face_infos[i].bbox.x2 > image_width) ?
    // (image_width - 1) : face_infos[i].bbox.x2;
    // temp_face_info.bbox.y2 = (face_infos[i].bbox.y2 > image_height) ?
    // (image_height - 1) : face_infos[i].bbox.y2;

    face_infos_padding.push_back(temp_face_info);
  }
}

void MTCNN::DetectFace(const cv::Mat& rgb_image, double thresh,
                       vector<FaceInfo>& candidate_bboxes, int type,
                       vector<FaceInfo>& rnet_bboxes) {
  boost::shared_ptr<Net<float>> net = RNet_;
  if (type == 1) {
    net = ONet_;
  }
  int image_width = rgb_image.cols;
  int image_height = rgb_image.rows;
  int num_boxes = candidate_bboxes.size();
  if (num_boxes > 0) {
    vector<FaceInfo> candidate_boxes_nms;
    NMS(candidate_bboxes, 0.7, candidate_boxes_nms);
    vector<FaceInfo> face_infos_reg;
    BoxRegression(candidate_boxes_nms, face_infos_reg, 1);
    BBox2Square(face_infos_reg);
    vector<FaceInfo> face_infos_padding;
    FacePadding(face_infos_reg, face_infos_padding, image_width, image_height);

    Blob<float>* input_layer = net->input_blobs()[0];
    int input_width = input_layer->width();
    int input_height = input_layer->height();

    for (int i = 0; i < face_infos_padding.size(); i++) {
      // std::cout << "image width height " << image_width << " " <<
      // image_height << std::endl;
      // std::cout << "y1 y2 " << int(face_infos_padding[i].bbox.y1 - 1) << " "
      // << int(face_infos_padding[i].bbox.y2) << std::endl;
      // std::cout << "x1 x2 " << int(face_infos_padding[i].bbox.x1 - 1) << " "
      // << int(face_infos_padding[i].bbox.x2) << std::endl;
      cv::Range rng1 = cv::Range(
          std::max(0, int(face_infos_padding[i].bbox.y1 - 1)),
          std::min(int(face_infos_padding[i].bbox.y2), rgb_image.rows - 1));
      cv::Range rng2 = cv::Range(
          std::max(0, int(face_infos_padding[i].bbox.x1 - 1)),
          std::min(int(face_infos_padding[i].bbox.x2), rgb_image.cols - 1));
      if (rng1.start > rng1.end || rng2.start > rng2.end) {
        continue;
      }
      // cv::Mat crop_image = rgb_image(cv::Range(face_infos_padding[i].bbox.y1
      // - 1, face_infos_padding[i].bbox.y2),
      //     cv::Range(face_infos_padding[i].bbox.x1 - 1,
      //     face_infos_padding[i].bbox.x2));
      cv::Mat crop_image = rgb_image(rng1, rng2);
      int pad_top =
          std::abs(face_infos_padding[i].bbox.x1 - face_infos_reg[i].bbox.x1);
      int pad_left =
          std::abs(face_infos_padding[i].bbox.y1 - face_infos_reg[i].bbox.y1);
      int pad_right =
          std::abs(face_infos_padding[i].bbox.y2 - face_infos_reg[i].bbox.y2);
      int pad_bottom =
          std::abs(face_infos_padding[i].bbox.x2 - face_infos_reg[i].bbox.x2);
      cv::copyMakeBorder(crop_image, crop_image, pad_left, pad_right, pad_top,
                         pad_bottom, cv::BORDER_CONSTANT, cv::Scalar(0));

      vector<Blob<float>*> output_blobs;
      PredictImage(net, crop_image, input_width, input_height, output_blobs);

      const Blob<float>* regression;
      const Blob<float>* confidence;
      const float* regression_data;
      const float* confidence_data;

      const Blob<float>* points;
      const float* points_data;
      if (type == 0) {
        regression = output_blobs[0];
        confidence = output_blobs[1];
        confidence_data = confidence->cpu_data() + confidence->count() / 2;
        regression_data = regression->cpu_data();
      } else {
        regression = output_blobs[0];
        confidence = output_blobs[2];
        points = output_blobs[1];
        confidence_data = confidence->cpu_data() + confidence->count() / 2;
        regression_data = regression->cpu_data();
        points_data = points->cpu_data();
      }

      if (*(confidence_data) > thresh) {
        FaceRect face_rect;
        face_rect.x1 = face_infos_reg[i].bbox.x1;
        face_rect.y1 = face_infos_reg[i].bbox.y1;
        face_rect.x2 = face_infos_reg[i].bbox.x2;
        face_rect.y2 = face_infos_reg[i].bbox.y2;
        face_rect.score = *confidence_data;
        FaceInfo face_info;
        face_info.bbox = face_rect;
        face_info.regression =
            cv::Vec4f(regression_data[0], regression_data[1],
                      regression_data[2], regression_data[3]);
        if (type == 1) {
          FacePts face_pts;
          float w = face_rect.y2 - face_rect.y1 + 1;
          float h = face_rect.x2 - face_rect.x1 + 1;
          for (int j = 0; j < 5; j++) {
            face_pts.y[j] = face_rect.y1 + *(points_data + j) * h - 1;
            face_pts.x[j] = face_rect.x1 + *(points_data + j + 5) * w - 1;
          }
          face_info.facePts = face_pts;
        }
        // std::cout << "rnet: " << regression_data[0] << " " <<
        // regression_data[1] << " " << regression_data[2] << std::endl;
        rnet_bboxes.push_back(face_info);
      }
    }
  }
}

/////////////////////////////////////////////////////