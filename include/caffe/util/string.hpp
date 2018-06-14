#ifndef CAFFE_UTIL_STRING_H_
#define CAFFE_UTIL_STRING_H_

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace caffe {

inline std::vector<std::string> SplitString(const std::string& str,
                                            const std::string& c) {
  std::vector<std::string> ret;
  std::string temp(str);
  size_t pos;
  while (pos = temp.find(c), pos != std::string::npos) {
    ret.push_back(temp.substr(0, pos));
    temp.erase(0, pos + 1);
  }
  ret.push_back(temp);
  return ret;
}
}  // namespace caffe

#endif  // CAFFE_UTIL_STRING_H_