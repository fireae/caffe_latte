// caffe
#include "mtcnn.h"
//#include "LBFRegressor.h"
#include "lbf/lbf.hpp"
using namespace cv;
using namespace std;
using namespace lbf;

// // parameters
// Params global_params;
// string modelPath ="../model/";
// string dataPath = "./../../Datasets/";
// string cascadeName = "../model/haarcascade_frontalface_alt.xml";

// // set the parameters when training models.
// void InitializeGlobalParam(){
//     global_params.bagging_overlap = 0.4;
//     global_params.max_numtrees = 10;
//     global_params.max_depth = 5;
//     global_params.landmark_num = 68;
//     global_params.initial_num = 5;

//     global_params.max_numstage = 7;
//     double m_max_radio_radius[10] = {0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08,
//     0.06, 0.06,0.05};
//     double m_max_numfeats[10] = {500, 500, 500, 300, 300, 200,
//     200,200,100,100};
//     for (int i=0;i<10;i++){
//         global_params.max_radio_radius[i] = m_max_radio_radius[i];
//     }
//     for (int i=0;i<10;i++){
//         global_params.max_numfeats[i] = m_max_numfeats[i];
//     }
//     global_params.max_numthreshs = 500;
// }

// void ReadGlobalParamFromFile(string path){
//     cout << "Loading GlobalParam..." << endl;
//     ifstream fin;
//     fin.open(path);
//     fin >> global_params.bagging_overlap;
//     fin >> global_params.max_numtrees;
//     fin >> global_params.max_depth;
//     fin >> global_params.max_numthreshs;
//     fin >> global_params.landmark_num;
//     fin >> global_params.initial_num;
//     fin >> global_params.max_numstage;

//     for (int i = 0; i< global_params.max_numstage; i++){
//         fin >> global_params.max_radio_radius[i];
//     }

//     for (int i = 0; i < global_params.max_numstage; i++){
//         fin >> global_params.max_numfeats[i];
//     }
//     cout << "Loading GlobalParam end"<<endl;
//     fin.close();
// }
// void FaceAlignment(cv::Mat& image, vector<cv::Rect>& faces) {
//     ReadGlobalParamFromFile("../model/LBF.model");
//     LBFRegressor regressor;
//     regressor.Load("../model/LBF.model");
//     const static Scalar colors[] =  { CV_RGB(0,0,255),
//         CV_RGB(0,128,255),
//         CV_RGB(0,255,255),
//         CV_RGB(0,255,0),
//         CV_RGB(255,128,0),
//         CV_RGB(255,255,0),
//         CV_RGB(255,0,0),
//         CV_RGB(255,0,255)} ;

//     double scale = 1;
//     cv::Mat gray;
//     cvtColor( image, gray, CV_BGR2GRAY );
//     double t = (double)cvGetTickCount();
//     int i = 0;
//     for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end();
//     r++, i++) {
//         Point center;
//         Scalar color = colors[i%8];
//         BoundingBox boundingbox;

//         boundingbox.start_x = r->x*scale;
//         boundingbox.start_y = r->y*scale;
//         boundingbox.width   = (r->width-1)*scale;
//         boundingbox.height  = (r->height-1)*scale;
//         boundingbox.centroid_x = boundingbox.start_x + boundingbox.width/2.0;
//         boundingbox.centroid_y = boundingbox.start_y +
//         boundingbox.height/2.0;

//         t =(double)cvGetTickCount();
//         Mat_<double> current_shape = regressor.Predict(gray,boundingbox,1);
//         t = (double)cvGetTickCount() - t;
//         printf( "alignment time = %g ms\n",
//         t/((double)cvGetTickFrequency()*1000.) );
//         for(int i = 0;i < global_params.landmark_num;i++){
//              circle(image,Point2d(current_shape(i,0),current_shape(i,1)),3,Scalar(255,255,255),-1,8,0);
//         }
//     }
//     imwrite("121.jpg", image);

// }

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "MTMain.bin [model dir] [imagePath]" << std::endl;
    return 0;
  }
  ::caffe::InitLogging(argv[0]);
  double threshold[3] = {0.6, 0.7, 0.7};
  double factor = 0.709;
  int minSize = 40;
  std::string proto_model_dir = argv[1];
  MTCNN detector(proto_model_dir);

  std::string imageName = argv[2];
  cv::Mat image = cv::imread(imageName);
  cv::Mat image2 = image.clone();
  std::vector<FaceInfo> faceInfo;
  clock_t t1 = clock();
  std::cout << "Detect " << image.rows << "X" << image.cols;

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
  detector.Detect(images, faceinfos);

  detector.Detect(image2, faceInfo);
#ifndef USE_CUDA
  std::cout << " Time Using CPU: " << (clock() - t1) * 1.0 / 1000 << std::endl;
#else
  std::cout << " Time Using GPU-CUDNN: " << (clock() - t1) * 1.0 / 1000
            << std::endl;
#endif
  LbfCascador lbf_cascador;
  FILE *fd = fopen("../model/LBF.model", "rb");
  lbf_cascador.Read(fd);

  for (int j = 0; j < faceinfos.size(); j++) {
    vector<FaceInfo> &faceInfo = faceinfos[j];
    cv::Mat image = images[j];

    cv::Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    vector<Rect> face_rects;
    for (int i = 0; i < faceInfo.size(); i++) {
      float x = faceInfo[i].bbox.x1;
      float y = faceInfo[i].bbox.y1;
      float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
      float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
      cv::rectangle(image, cv::Rect(y, x, w, h), cv::Scalar(255, 0, 0), 1);
      face_rects.push_back(cv::Rect(y, x, w, h));
      FacePts facePts = faceInfo[i].facePts;
      for (int j = 0; j < 5; j++)
        cv::circle(image, cv::Point(facePts.y[j], facePts.x[j]), 1,
                   cv::Scalar(255, 255, 0), 1);

      BBox facebox(std::max<int>(0, y), std::max<int>(0, x), w, h);
      Mat myshape = lbf_cascador.Predict(gray, facebox);
      for (int ipt = 0; ipt < 68; ipt++) {
        int xpt = myshape.at<double>(ipt, 0);
        int ypt = myshape.at<double>(ipt, 1);

        cv::circle(image, cv::Point(xpt, ypt), 1, cv::Scalar(255, 0, 255), 1);
      }
    }

    char new_name_[512];
    sprintf(new_name_, "a_%d.jpg", j);
    cv::imwrite(new_name_, image);
  }

  cv::Mat gray;
  cvtColor(image, gray, CV_BGR2GRAY);
  vector<Rect> face_rects;
  for (int i = 0; i < faceInfo.size(); i++) {
    float x = faceInfo[i].bbox.x1;
    float y = faceInfo[i].bbox.y1;
    float h = faceInfo[i].bbox.x2 - faceInfo[i].bbox.x1 + 1;
    float w = faceInfo[i].bbox.y2 - faceInfo[i].bbox.y1 + 1;
    cv::rectangle(image, cv::Rect(y, x, w, h), cv::Scalar(255, 0, 0), 1);
    face_rects.push_back(cv::Rect(y, x, w, h));
    FacePts facePts = faceInfo[i].facePts;
    for (int j = 0; j < 5; j++)
      cv::circle(image, cv::Point(facePts.y[j], facePts.x[j]), 1,
                 cv::Scalar(255, 255, 0), 1);

    BBox facebox(std::max<int>(0, y), std::max<int>(0, x), w, h);
    Mat myshape = lbf_cascador.Predict(gray, facebox);
    for (int ipt = 0; ipt < 68; ipt++) {
      int xpt = myshape.at<double>(ipt, 0);
      int ypt = myshape.at<double>(ipt, 1);

      cv::circle(image, cv::Point(xpt, ypt), 1, cv::Scalar(255, 0, 255), 1);
    }
  }
  // FaceAlignment(image, face_rects);
  cv::imwrite("c.jpg", image);
  // cv::imshow("a",image);
  // cv::waitKey(0);

  return 1;
}
