#include "classification.hpp"

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
    LOG(INFO) << pred[i] << " ";
  }
  int end = clock();
  sumspend += (end - start);

  // string strpredict0 = GetPredictString(pred, idxBlank, alphabets);
}

int main(int argc, char* argv[]) {
  string imgfolder = argv[2];
  string modelfolder = argv[1];
  test_ocr_chinese(imgfolder, modelfolder);
}
