#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "CImg.h"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/filesystem.hpp"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {
using caffe::filesystem::path;
using ::google::protobuf::Message;

CAFFE_API inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  const path& model =
      caffe::filesystem::temp_directory_path() / "caffe_test.%%%%-%%%%";
  for (int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++) {
    const path& dir = caffe::filesystem::unique_path(model).str();
    bool done = caffe::filesystem::create_directory(dir);
    if (done) {
      *temp_dirname = dir.str();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

CAFFE_API inline void MakeTempFilename(string* temp_filename) {
  static path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if (temp_files_subpath.empty()) {
    string path_string = "";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
      (temp_files_subpath / caffe::format_int(next_temp_file++, 9)).str();
}

CAFFE_API bool ReadProtoFromTextFile(const char* filename, Message* proto);

CAFFE_API inline bool ReadProtoFromTextFile(const string& filename,
                                            Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

CAFFE_API inline void ReadProtoFromTextFileOrDie(const char* filename,
                                                 Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

CAFFE_API inline void ReadProtoFromTextFileOrDie(const string& filename,
                                                 Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

CAFFE_API void WriteProtoToTextFile(const Message& proto, const char* filename);
CAFFE_API inline void WriteProtoToTextFile(const Message& proto,
                                           const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

CAFFE_API bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

CAFFE_API inline bool ReadProtoFromBinaryFile(const string& filename,
                                              Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

CAFFE_API inline void ReadProtoFromBinaryFileOrDie(const char* filename,
                                                   Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

CAFFE_API inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                                   Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}

CAFFE_API void WriteProtoToBinaryFile(const Message& proto,
                                      const char* filename);
CAFFE_API inline void WriteProtoToBinaryFile(const Message& proto,
                                             const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

CAFFE_API bool ReadFileToDatum(const string& filename,
                               const std::vector<int>& labels, Datum* datum);

CAFFE_API inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, std::vector<int>(1, -1), datum);
}

CAFFE_API bool ReadImageToDatum(const string& filename,
                                const std::vector<int>& labels,
                                const int height, const int width,
                                const bool is_color,
                                const std::string& encoding, Datum* datum);

CAFFE_API inline bool ReadImageToDatum(const string& filename,
                                       const std::vector<int>& labels,
                                       const int height, const int width,
                                       const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, labels, height, width, is_color, "", datum);
}

CAFFE_API inline bool ReadImageToDatum(const string& filename,
                                       const std::vector<int>& labels,
                                       const int height, const int width,
                                       Datum* datum) {
  return ReadImageToDatum(filename, labels, height, width, true, datum);
}

CAFFE_API inline bool ReadImageToDatum(const string& filename,
                                       const std::vector<int>& labels,
                                       const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, labels, 0, 0, is_color, datum);
}

CAFFE_API inline bool ReadImageToDatum(const string& filename,
                                       const std::vector<int>& labels,
                                       Datum* datum) {
  return ReadImageToDatum(filename, labels, 0, 0, true, datum);
}

CAFFE_API inline bool ReadImageToDatum(const string& filename,
                                       const std::vector<int>& labels,
                                       const std::string& encoding,
                                       Datum* datum) {
  return ReadImageToDatum(filename, labels, 0, 0, true, encoding, datum);
}

CAFFE_API bool DecodeDatumNative(Datum* datum);
CAFFE_API bool DecodeDatum(Datum* datum, bool is_color);
CAFFE_API CImg<unsigned char> ReadImage(const string& filename,
                                        const int height, const int width,
                                        const bool is_color, int* img_height,
                                        int* img_width);
CAFFE_API CImg<unsigned char> ReadImage(const string& filename,
                                        const int height, const int width,
                                        const bool is_color);
#if 0
cv::Mat ReadImageToCVMat(const string& filename, const int height,
  const int width, const bool is_color, int* img_height,
  int* img_width);
cv::Mat ReadImageToCVMat(const string& filename, const int height,
                         const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename, const int height,
                         const int width);

cv::Mat ReadImageToCVMat(const string& filename, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // CAFFE_UTIL_IO_H_
