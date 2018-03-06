#include "text_ocr.hpp"

#ifdef _WIN32
#include "windows.h"

string GBKToUTF8(const std::string& strGBK)
{
	int nLen = MultiByteToWideChar(CP_ACP, 0, strGBK.c_str(), -1, NULL, 0);
	WCHAR * wszUTF8 = new WCHAR[nLen];
	MultiByteToWideChar(CP_ACP, 0, strGBK.c_str(), -1, wszUTF8, nLen);

	nLen = WideCharToMultiByte(CP_UTF8, 0, wszUTF8, -1, NULL, 0, NULL, NULL);
	char * szUTF8 = new char[nLen];
	WideCharToMultiByte(CP_UTF8, 0, wszUTF8, -1, szUTF8, nLen, NULL, NULL);

	std::string strTemp(szUTF8);
	delete[]wszUTF8;
	delete[]szUTF8;
	return strTemp;
}

string UTF8ToGBK(const std::string& strUTF8)
{
	int nLen = MultiByteToWideChar(CP_UTF8, 0, strUTF8.c_str(), -1, NULL, 0);
	unsigned short * wszGBK = new unsigned short[nLen + 1];
	memset(wszGBK, 0, nLen * 2 + 2);
	MultiByteToWideChar(CP_UTF8, 0, strUTF8.c_str(), -1, (LPWSTR)wszGBK, nLen);

	nLen = WideCharToMultiByte(CP_ACP, 0, (LPWSTR)wszGBK, -1, NULL, 0, NULL, NULL);
	char *szGBK = new char[nLen + 1];
	memset(szGBK, 0, nLen + 1);
	WideCharToMultiByte(CP_ACP, 0, (LPWSTR)wszGBK, -1, szGBK, nLen, NULL, NULL);

	std::string strTemp(szGBK);
	delete[]szGBK;
	delete[]wszGBK;
	return strTemp;
}

#endif

namespace jdcn {
bool TextOCR::Init(const string& model_path, bool gpu_mode) {
  string model_file = model_path + "/deploy.prototxt";
  string weight_file = model_path + "/ocr.caffemodel";
  string label_file = model_path + "/label.txt";
  return Init(model_file, weight_file, label_file, gpu_mode);
}
bool TextOCR::Init(const string& model_file, const string& weight_file,
                   const string& label_file, bool gpu_mode) {
  if (!gpu_mode)
    Caffe::set_mode(Caffe::CPU);
  else
    Caffe::set_mode(Caffe::GPU);
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weight_file);
  labels_ = GetLabels(label_file);
  return true;
}

vector<string> TextOCR::GetLabels(const string& label_file) {
  vector<string> vlabels;
  if (label_file.size() > 0) {
    std::ifstream labels(label_file.c_str());
    string line;
    while (std::getline(labels, line)) {
#ifdef _WIN32
      string sgbk = UTF8ToGBK(line);
      vlabels.push_back(sgbk);
#endif 
      vlabels.push_back(string(line.substr(0, line.size() - 1)));
    }

  } else {
    Blob<float>* output_layer = net_->output_blobs()[0];
    char szlabel[100];
    printf("output ch=%d\n", output_layer->channels());
    for (int i = 0; i < output_layer->channels(); i++) {
      sprintf(szlabel, "%d", i);
      vlabels.push_back(szlabel);
    }
  }

  return vlabels;
}

void TextOCR::PrecessImage(const cv::Mat& im_org) {
  cv::Mat im;
  ratio_ = 1.0;
  cv::resize(im_org, im, cv::Size(), ratio_, ratio_);
  cv::Mat image;
  im.convertTo(image, CV_32FC3);
  // mean value (152.0, 152.0, 152.0)
  image -= cv::Scalar(152.0, 152.0, 152.0);
  image_width_ = image.cols;
  image_height_ = image.rows;
  image_channels_ = image.channels();

  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, image_channels_, image_height_, image_width_);

  float* blob_data_ptr = input_layer->mutable_cpu_data();
  for (int h = 0; h < image_height_; ++h) {
    for (int w = 0; w < image_width_; ++w) {
      blob_data_ptr[(0 * image_height_ + h) * image_width_ + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[0]);
      blob_data_ptr[(1 * image_height_ + h) * image_width_ + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[1]);
      blob_data_ptr[(2 * image_height_ + h) * image_width_ + w] =
          float(image.at<cv::Vec3f>(cv::Point(w, h))[2]);
    }
  }
}

vector<float> TextOCR::GetPrediction(const cv::Mat& image,
                                     std::vector<int>& outshape) {
  PrecessImage(image);
  net_->Forward();

  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->count();

  outshape = output_layer->shape();
  vector<float> pred = std::vector<float>(begin, end);
  //for (int i = 0; i < pred.size(); i++) {
  //  if (pred[i] >= 0) {
  //    // result += alphabets[pred[i]];
  //    printf("%s\n", labels_[pred[i]]);
  //  }
  //}
  return std::vector<float>(begin, end);
}
}  // namespace jdcn
