#ifndef CAFFE2_CORE_FLAGS_H_
#define CAFFE2_CORE_FLAGS_H_
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

}  // namespace caffe2

////////////////////////////////////////////////////////////////////////////////
// Below are gflags and non-gflags specific implementations.
////////////////////////////////////////////////////////////////////////////////

#ifdef CAFFE2_USE_GFLAGS

#define CAFFE2_GFLAGS_DEF_WRAPPER(type, name, default_value, help_str) \
  DEFINE_##type(name, default_value, help_str);                        \
  namespace caffe2 {                                                   \
  using ::FLAGS_##name;                                                \
  }

#define CAFFE2_DEFINE_int(...) CAFFE2_GFLAGS_DEF_WRAPPER(int32, __VA_ARGS__)
#define CAFFE2_DEFINE_int64(...) CAFFE2_GFLAGS_DEF_WRAPPER(int64, __VA_ARGS__)
#define CAFFE2_DEFINE_double(...) CAFFE2_GFLAGS_DEF_WRAPPER(double, __VA_ARGS__)
#define CAFFE2_DEFINE_bool(...) CAFFE2_GFLAGS_DEF_WRAPPER(bool, __VA_ARGS__)
#define CAFFE2_DEFINE_string(name, default_value, help_str) \
  CAFFE2_GFLAGS_DEF_WRAPPER(string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE2_GFLAGS_DECLARE_WRAPPER(type, name) \
  DECLARE_##type(name);                           \
  namespace caffe2 {                              \
  using ::FLAGS_##name;                           \
  }  // namespace caffe2

#define CAFFE2_DECLARE_int(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(int32, name)
#define CAFFE2_DECLARE_int64(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(int64, name)
#define CAFFE2_DECLARE_double(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(double, name)
#define CAFFE2_DECLARE_bool(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(bool, name)
#define CAFFE2_DECLARE_string(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(string, name)

#else  // CAFFE2_USE_GFLAGS

namespace caffe {

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

// The macros are defined outside the caffe2 namespace. In your code, you should
// write the CAFFE2_DEFINE_* and CAFFE2_DECLARE_* macros outside any namespace
// as well.

#define CAFFE_DEFINE_typed_var(type, name, default_value, help_str)         \
  namespace caffe {                                                         \
  type FLAGS_##name = default_value;                                        \
  namespace {                                                               \
  class CaffeFlagParser_##name : public CaffeFlagParser {                   \
   public:                                                                  \
    explicit CaffeFlagParser_##name(const string& content) {                \
      success_ = CaffeFlagParser::Parse<type>(content, &FLAGS_##name);      \
    }                                                                       \
  };                                                                        \
  }                                                                         \
  RegistererCaffeFlagsRegistry g_CaffeFlagsRegistry_##name(                 \
      #name, CaffeFlagsRegistry(),                                          \
      RegistererCaffeFlagsRegistry::DefaultCreator<CaffeFlagParser_##name>, \
      "(" #type ", default " #default_value ") " help_str);                 \
  }

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
  }  // namespace caffe2

#define CAFFE_DECLARE_int(name) CAFFE_DECLARE_typed_var(int, name)
#define CAFFE_DECLARE_int64(name) CAFFE_DECLARE_typed_var(int64_t, name)
#define CAFFE_DECLARE_double(name) CAFFE_DECLARE_typed_var(double, name)
#define CAFFE_DECLARE_bool(name) CAFFE_DECLARE_typed_var(bool, name)
#define CAFFE_DECLARE_string(name) CAFFE_DECLARE_typed_var(string, name)
#endif

#endif  // CAFFE2_CORE_FLAGS_H_