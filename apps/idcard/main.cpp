#define SIMPLE_EXPORT
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "classification.hpp"
#include "graph.hpp"
#include "ltp/ner_dll.h"
#include "ltp/postag_dll.h"
#include "ltp/segment_dll.h"
#include "ployfit.hpp"

using namespace cv;
using namespace caffe;
using namespace std;

std::string ws2s(const std::wstring& ws) {
  std::string curLocale = setlocale(LC_ALL, NULL);  // curLocale = "C";
  setlocale(LC_ALL, "chs");
  const wchar_t* _Source = ws.c_str();
  size_t _Dsize = 2 * ws.size() + 1;
  char* _Dest = new char[_Dsize];
  memset(_Dest, 0, _Dsize);
  wcstombs(_Dest, _Source, _Dsize);
  std::string result = _Dest;
  delete[] _Dest;
  setlocale(LC_ALL, curLocale.c_str());
  return result;
}

std::wstring s2ws(const std::string& s) {
  setlocale(LC_ALL, "chs");
  const char* _Source = s.c_str();
  size_t _Dsize = s.size() + 1;
  wchar_t* _Dest = new wchar_t[_Dsize];
  wmemset(_Dest, 0, _Dsize);
  mbstowcs(_Dest, _Source, _Dsize);
  std::wstring result = _Dest;
  delete[] _Dest;
  setlocale(LC_ALL, "C");
  return result;
}

void ocr_init(const string& model_folder, Classifier** cnn) {
  bool use_gpu = false;

  // load model
  Classifier* pCNN = new Classifier();
  if (!pCNN->Init(model_folder, use_gpu)) {
    LOG(INFO) << "init error";
    delete pCNN;
    pCNN = NULL;
    return;
  }
  *cnn = pCNN;
}

void recognize_textline(const cv::Mat& image, string& result,
                        Classifier* pCNN) {
  int wstd = 0, hstd = 0;
  pCNN->GetInputImageSize(wstd, hstd);

  // get alphabet
  vector<string> alphabets = pCNN->GetLabels();

  int idxBlank = 0;
  vector<string>::const_iterator it =
      find(alphabets.begin(), alphabets.end(), "blank");
  if (it != alphabets.end()) idxBlank = (int)(it - alphabets.begin());

  for (size_t i = 0; i < alphabets.size(); i++) {
    wchar_t c = 0;
    if (alphabets[i] == "blank") continue;
    // wstring wlabel = string2wstring(alphabets[i], true);
    // mapLabel2IDs.insert(make_pair(wlabel[0], i));
  }

  int sumspend = 0;
  int nok_lexicon = 0;
  int nok_nolexicon = 0;

  cv::Mat img = image.clone();  // cv::imread(imgfile, CV_LOAD_IMAGE_COLOR);
  int w = img.cols, h = img.rows;
  if (2 * w <= h) {
    cv::transpose(img, img);
    cv::flip(img, img, 1);
    w = img.cols, h = img.rows;
  }

  int w1 = hstd * w / h;
  if (w1 != w && h != hstd) cv::resize(img, img, cv::Size(w1, hstd));

  int start = clock();

  vector<int> shape;
  vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);
  for (int i = 0; i < pred.size(); i++) {
    if (pred[i] >= 0) {
      result += alphabets[pred[i]];
    }
  }

  int end = clock();
  sumspend += (end - start);
}

void test_ocr_chinese(const string& imgfile, const string& model_folder) {
  bool use_gpu = false;

  // load model
  Classifier* pCNN = new Classifier();
  if (!pCNN->Init(model_folder, use_gpu)) {
    LOG(INFO) << "init error";
    delete pCNN;
    pCNN = NULL;
    return;
  }

  int wstd = 0, hstd = 0;
  pCNN->GetInputImageSize(wstd, hstd);

  // get alphabet
  vector<string> alphabets = pCNN->GetLabels();

  int idxBlank = 0;
  vector<string>::const_iterator it =
      find(alphabets.begin(), alphabets.end(), "blank");
  if (it != alphabets.end()) idxBlank = (int)(it - alphabets.begin());

  map<wchar_t, int> mapLabel2IDs;
  for (size_t i = 0; i < alphabets.size(); i++) {
    wchar_t c = 0;
    if (alphabets[i] == "blank") continue;
    // wstring wlabel = string2wstring(alphabets[i], true);
    // mapLabel2IDs.insert(make_pair(wlabel[0], i));
  }

  int sumspend = 0;
  int nok_lexicon = 0;
  int nok_nolexicon = 0;

  cv::Mat img = cv::imread(imgfile, CV_LOAD_IMAGE_COLOR);
  int w = img.cols, h = img.rows;
  if (2 * w <= h) {
    cv::transpose(img, img);
    cv::flip(img, img, 1);
    w = img.cols, h = img.rows;
  }

  int w1 = hstd * w / h;
  if (w1 != w && h != hstd) cv::resize(img, img, cv::Size(w1, hstd));

  int start = clock();

  vector<int> shape;
  vector<float> pred = pCNN->GetOutputFeatureMap(img, shape);
  for (int i = 0; i < pred.size(); i++) {
    if (pred[i] >= 0) {
      LOG(INFO) << alphabets[pred[i]];
    }
  }
  int end = clock();
  sumspend += (end - start);

  // string strpredict0 = GetPredictString(pred, idxBlank, alphabets);
}

void clipBoxes(vector<float>& box, int height, int width) {
  if (box[0] < 0) {
    box[0] = 0;
  }
  if (box[0] > width - 1) {
    box[0] = width - 1;
  }
  if (box[2] < 0) {
    box[2] = 0;
  }
  if (box[2] > width - 1) {
    box[2] = width - 1;
  }
  if (box[1] < 0) {
    box[1] = 0;
  }
  if (box[1] > height - 1) {
    box[1] = height - 1;
  }
  if (box[3] < 0) {
    box[3] = 0;
  }
  if (box[3] > height - 1) {
    box[3] = height - 1;
  }
}

int main(int argc, char* argv[]) {
  Caffe::set_mode(Caffe::CPU);
  string model_file = "/home/wencc/Myplace/CTPN/models/deploy.prototxt";
  string trained_file =
      "/home/wencc/Myplace/CTPN/models/ctpn_trained_model.caffemodel";
  string image_name = argv[1];  //"/home/wencc/Myplace/CTPN/demo_images/4.jpg";
  shared_ptr<Net<float>> net(new Net<float>(model_file, TEST));

  net->CopyTrainedLayersFrom(trained_file);

  shared_ptr<Blob<float>> blob_data = net->blob_by_name("data");
  shared_ptr<Blob<float>> im_info = net->blob_by_name("im_info");
  // 102.9801, 115.9465, 122.7717

  cv::Mat im_org = cv::imread(image_name);
  cv::Mat im;
  float ratio = 800.0 / std::max(im_org.rows, im_org.cols);
  cv::resize(im_org, im, cv::Size(), ratio, ratio);
  cv::Mat image;
  im.convertTo(image, CV_32FC3);

  image -= cv::Scalar(102.9801, 115.9465, 122.7717);
  int width = image.cols;
  int height = image.rows;
  blob_data->Reshape(1, 3, height, width);
  float* blob_data_ptr = blob_data->mutable_cpu_data();
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      blob_data_ptr[(0 * height + h) * width + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[0]);
      blob_data_ptr[(1 * height + h) * width + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[1]);
      blob_data_ptr[(2 * height + h) * width + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[2]);
    }
  }
  float* im_data = im_info->mutable_cpu_data();
  im_data[0] = height;
  im_data[1] = width;
  net->Forward();
  shared_ptr<Blob<float>> rois = net->blob_by_name("rois");
  shared_ptr<Blob<float>> scores = net->blob_by_name("scores");
  float min_score = 0.7;
  float* scores_data = scores->mutable_cpu_data();
  float* rois_data = rois->mutable_cpu_data();
  vector<vector<float>> text_proposals;
  vector<float> vec_scores;
  for (int i = 0; i < scores->shape()[0]; i++) {
    if (scores_data[i] > min_score) {
      vector<float> vec_roi;
      vec_roi.push_back(rois_data[i * 4 + 0]);
      vec_roi.push_back(rois_data[i * 4 + 1]);
      vec_roi.push_back(rois_data[i * 4 + 2]);
      vec_roi.push_back(rois_data[i * 4 + 3]);
      clipBoxes(vec_roi, height, width);
      text_proposals.push_back(vec_roi);
      vec_scores.push_back(scores_data[i]);
    }
  }

  Classifier* ocr_cls;
  string model_folder = "/home/wencc/models/ChineseOCR/inception-bn-res-blstm/";
  ocr_init(model_folder, &ocr_cls);

  string seg_model = "/home/wencc/Downloads/ltp_data_v3.4.0/cws.model";
  string tag_model = "/home/wencc/Downloads/ltp_data_v3.4.0/pos.model";
  void* seg_engine = segmentor_create_segmentor(seg_model.c_str());
  void* tag_engine = postagger_create_postagger(tag_model.c_str(), NULL);

  vector<int> im_size(2);
  im_size[0] = height;
  im_size[1] = width;
  TextProposalGraphBuilder builder;
  Graph g = builder.build_graph(text_proposals, vec_scores, im_size);
  vector<vector<int>> tp_groups = g.sub_graphs_connected();

  cv::Mat show_image = im.clone();

  vector<vector<float>> text_lines;
  for (int i = 0; i < tp_groups.size(); i++) {
    vector<int> tp_group = tp_groups[i];

    vector<vector<float>> text_line_boxes;

    float x0 = 100000.0;
    float x1 = 0.0;
    vector<float> lt_x0;
    vector<float> rt_y0;
    vector<float> lb_x0;
    vector<float> rb_y0;
    for (int j = 0; j < tp_group.size(); j++) {
      int index = tp_group[j];
      text_line_boxes.push_back(text_proposals[index]);
      if (text_proposals[index][0] < x0) {
        x0 = text_proposals[index][0];
      }
      if (text_proposals[index][2] > x1) {
        x1 = text_proposals[index][2];
      }
      lt_x0.push_back(text_proposals[index][0]);
      rt_y0.push_back(text_proposals[index][1]);

      lb_x0.push_back(text_proposals[index][0]);
      rb_y0.push_back(text_proposals[index][3]);
    }
    float offset = (text_line_boxes[0][2] - text_line_boxes[0][0]) * 0.5;
    czy::Fit fitline;
    fitline.polyfit(lt_x0, rt_y0, 1);
    vector<double> factor;
    fitline.getFactor(factor);
    float lt_y = factor[1] * (x0 + offset) + factor[0];
    float rt_y = factor[1] * (x1 - offset) + factor[0];

    fitline.polyfit(lb_x0, rb_y0, 1);
    fitline.getFactor(factor);
    float lb_y = factor[1] * (x0 + offset) + factor[0];
    float rb_y = factor[1] * (x1 - offset) + factor[0];

    vector<float> text_line(5);
    text_line[0] = (x0);
    text_line[1] = std::min((lt_y), (rt_y));
    text_line[2] = (x1);
    text_line[3] = std::max((lb_y), (rb_y));
    cv::rectangle(
        show_image,
        cv::Rect(text_line[0], text_line[1], text_line[2] - text_line[0],
                 text_line[3] - text_line[1]),
        cv::Scalar(0, 255, 0), 2);

    cv::Mat line_image =
        im(cv::Rect(text_line[0], text_line[1], text_line[2] - text_line[0],
                    text_line[3] - text_line[1]));
    string result;
    recognize_textline(line_image, result, ocr_cls);
    LOG(INFO) << result;

    vector<string> words;
    int len = segmentor_segment(seg_engine, result, words);
    vector<string> tags;
    postagger_postag(tag_engine, words, tags);
    for (int t = 0; t < tags.size(); t++) {
      std::cout << words[t] << " / " << tags[t] << std::endl;
      if (t == tags.size() - 1)
        std::cout << std::endl;
      else
        std::cout << " ";
    }

    double sum = 0.0;
    for (int j = 0; j < tp_group.size(); j++) {
      sum += vec_scores[tp_group[j]];
    }
    text_line[4] = (sum / tp_group.size());

    text_lines.push_back(text_line);
  }

  cv::imwrite("show.png", show_image);
  return 0;
}