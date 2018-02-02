#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "text_detection.hpp"
#include "text_ocr.hpp"
using namespace cv;
using namespace std;
using namespace caffe;
using namespace jdcn;

int main(int argc, char* argv[]) {
#if 1
  string model_file = "D:\\BaiduNetdiskDownload\\ctpn/deploy.prototxt";
  string weight_file =
      "D:\\BaiduNetdiskDownload\\ctpn/ctpn_trained_model.caffemodel";
  string image_name = "D:\\tests\\b5.png";
  // std::string model_file = "";
  // std::string weight_file = "";
  // std::string image_name = "";

  string ocr_model_path =
      "D:\\workspace\\caffe_ocr\\examples\\ocr\\inception-bn";
  string label_file = ocr_model_path + "/label.txt";
  TextOCR text_ocr;
  text_ocr.Init(ocr_model_path, false);
  vector<string> vocabulary = text_ocr.GetLabels(label_file);
  TextDetector detector;
  bool is_init = detector.Init(model_file, weight_file);
  if (!is_init) {
    return 0;
  }
  cv::Mat image = cv::imread(image_name);
  detector.PrecessImage(image);
  vector<vector<float>> text_lines;
  detector.DetectTextLines(text_lines);
  cv::Mat show_image = image.clone();
  for (int i = 0; i < text_lines.size(); ++i) {
    vector<float>& text_line = text_lines[i];

    cv::rectangle(
        show_image,
        cv::Rect(text_line[0], text_line[1], text_line[2] - text_line[0],
                 text_line[3] - text_line[1]),
        cv::Scalar(0, 255, 0), 2);

    cv::Mat ocr_image =
        image(cv::Rect(text_line[0], text_line[1], text_line[2] - text_line[0],
                       text_line[3] - text_line[1]));
    int w = ocr_image.cols, h = ocr_image.rows;
    if (2 * w <= h) {
      cv::transpose(ocr_image, ocr_image);
      cv::flip(ocr_image, ocr_image, 1);
      w = ocr_image.cols, h = ocr_image.rows;
    }
    int hstd = 32;
    int wstd = 280;
    int w1 = hstd * w / h;
    if (w1 != w && h != hstd)
      cv::resize(ocr_image, ocr_image, cv::Size(w1, hstd));
    vector<int> outshape;
    vector<float> pred = text_ocr.GetPrediction(ocr_image, outshape);
    for (int k = 0; k < pred.size(); ++k) {
      if (pred[k] > 0) {
        cout << vocabulary[int(pred[k])] << " ";
      }
    }
    cout << "\n";
  }
  cv::imwrite("show.png", show_image);

#endif

#if 0
  string model_path = "D:\\workspace\\caffe_ocr\\examples\\ocr\\inception-bn";
  string image_name = "d:\\1.jpg";
  cv::Mat ocr_image = cv::imread(image_name);
  int w = ocr_image.cols, h = ocr_image.rows;
  if (2 * w <= h) {
    cv::transpose(ocr_image, ocr_image);
    cv::flip(ocr_image, ocr_image, 1);
    w = ocr_image.cols, h = ocr_image.rows;
  }
  int hstd = 32;
  int wstd = 280;
  int w1 = hstd * w / h;
  if (w1 != w && h != hstd)
    cv::resize(ocr_image, ocr_image, cv::Size(w1, hstd));
  TextOCR text_ocr;
  text_ocr.Init(model_path, false);
  
  vector<int> outshape;
  vector<float> pred = text_ocr.GetPrediction(ocr_image, outshape);
#endif

  return 0;
}