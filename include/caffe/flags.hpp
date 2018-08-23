#ifndef CAFFE_CORE_FLAGS_H_
#define CAFFE_CORE_FLAGS_H_
// A lightweighted commandline flags tool for caffe2, so we do not need to rely
// on gflags. If you have gflags installed, set the macro CAFFE2_USE_GFLAGS will
// seamlessly route everything to gflags.

#include "caffe/registry.hpp"

namespace caffe {

/**
 * Sets the usage message when a commandline tool is called with "--help".
 */
void SetUsageMessage(const string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
const char* UsageMessage();

bool ShowUsageWithFlagsRestrict(const char* str, const char* str2);

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that caffe2 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
bool ParseCommandLineFlags(int* pargc, char*** pargv);
/**
 * Checks if the commandline flags has already been passed.
 */
bool CommandLineFlagsHasBeenParsed();

class CaffeFlagParser {
 public:
  CaffeFlagParser() {}
  bool success() { return success_; }

 protected:
  template <typename T>
  bool Parse(const string& content, T* value);
  bool success_;
};

CAFFE_DECLARE_REGISTRY(CaffeFlagsRegistry, CaffeFlagParser, const string&);

}  // namespace caffe

// The macros are defined outside the caffe namespace. In your code, you should
// write the CAFFE_DEFINE_* and CAFFE_DECLARE_* macros outside any namespace
// as well.

#define CAFFE_DEFINE_typed_var(type, name, default_value, help_str)         \
  type FLAGS_##name = default_value;                                        \
  namespace {                                                               \
  class CaffeFlagParser_##name : public caffe::CaffeFlagParser {            \
   public:                                                                  \
    explicit CaffeFlagParser_##name(const string& content) {                \
      success_ = CaffeFlagParser::Parse<type>(content, &FLAGS_##name);      \
    }                                                                       \
  };                                                                        \
  }                                                                         \
  RegistererCaffeFlagsRegistry g_CaffeFlagsRegistry_##name(                 \
      #name, CaffeFlagsRegistry(),                                          \
      RegistererCaffeFlagsRegistry::DefaultCreator<CaffeFlagParser_##name>, \
      "(" #type ", default " #default_value ") " help_str);

#define CAFFE_DEFINE_int(name, default_value, help_str) \
  CAFFE_DEFINE_typed_var(int, name, default_value, help_str)
#define CAFFE_DEFINE_int32(name, default_value, help_str) \
  CAFFE_DEFINE_typed_var(int, name, default_value, help_str)
#define CAFFE_DEFINE_int64(name, default_value, help_str) \
  CAFFE_DEFINE_typed_var(int64_t, name, default_value, help_str)
#define CAFFE_DEFINE_double(name, default_value, help_str) \
  CAFFE_DEFINE_typed_var(double, name, default_value, help_str)
#define CAFFE_DEFINE_bool(name, default_value, help_str) \
  CAFFE_DEFINE_typed_var(bool, name, default_value, help_str)
#define CAFFE_DEFINE_string(name, default_value, help_str) \
  CAFFE_DEFINE_typed_var(string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE_DECLARE_typed_var(type, name) \
  namespace caffe {                         \
  extern type FLAGS_##name;                 \
  }  // namespace caffe

#define CAFFE_DECLARE_int(name) CAFFE_DECLARE_typed_var(int, name)
#define CAFFE_DECLARE_int64(name) CAFFE_DECLARE_typed_var(int64_t, name)
#define CAFFE_DECLARE_double(name) CAFFE_DECLARE_typed_var(double, name)
#define CAFFE_DECLARE_bool(name) CAFFE_DECLARE_typed_var(bool, name)
#define CAFFE_DECLARE_string(name) CAFFE_DECLARE_typed_var(string, name)

#endif  //
