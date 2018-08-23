// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/flags.hpp"
#include "caffe/logging.hpp"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;

CAFFE_DEFINE_bool(gray, false,
                  "When this option is on, treat images as grayscale ones");
CAFFE_DEFINE_bool(shuffle, false,
                  "Randomly shuffle the order of images and their labels");
CAFFE_DEFINE_string(backend, "lmdb",
                    "The backend {lmdb, leveldb} for storing the result");
CAFFE_DEFINE_int32(resize_width, 0, "Width images are resized to");
CAFFE_DEFINE_int32(resize_height, 0, "Height images are resized to");
CAFFE_DEFINE_bool(
    check_size, false,
    "When this option is on, check that all the datum have the same size");
CAFFE_DEFINE_bool(
    encoded, false,
    "When this option is on, the encoded image will be save in datum");
CAFFE_DEFINE_string(
    encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV

  caffe::SetUsageMessage(
      "Convert a set of images to the leveldb/lmdb\n"
      "format used as input for Caffe.\n"
      "Usage:\n"
      "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
      "The ImageNet dataset for the training demo is at\n"
      "    http://www.image-net.org/download-images\n");
  caffe::ParseCommandLineFlags(&argc, &argv);

  if (argc < 4) {
    caffe::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, vector<int> > > lines;
  std::string filename;
  int label;
  string line;

  while (getline(infile, line)) {
    size_t postfixpos = line.rfind(".");
    if (postfixpos == string::npos) {
      LOG(INFO) << "wrong format:" << line;
      continue;
    }
    size_t firstblank = line.find(' ', postfixpos + 1);
    filename = line.substr(0, firstblank);
    string strlabels = line.substr(firstblank + 1);
    std::istringstream iss(strlabels);
    vector<int> labels;
    while (iss >> label) labels.push_back(label);
    lines.push_back(std::make_pair(filename, labels));
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  shared_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  shared_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = lines[line_id].first;
      size_t p = fn.rfind('.');
      if (p == fn.npos)
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    status = ReadImageToDatum(root_folder + lines[line_id].first,
                              lines[line_id].second, resize_height,
                              resize_width, is_color, enc, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size)
            << "Incorrect data field size " << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
