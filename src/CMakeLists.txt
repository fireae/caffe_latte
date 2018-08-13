set(src_dir ${PROJECT_SOURCE_DIR}/src/caffe/)
# generate protobuf sources

add_library(caffe_proto STATIC ${src_dir}/proto/caffe.pb.cc)
target_link_libraries(caffe_proto libprotobuf)
file(GLOB_RECURSE cpps ${src_dir}/*.cpp)
file(GLOB_RECURSE hpps ${src_dir}/*.hpp)
file(GLOB_RECURSE hdrs ${src_dir}/*.h)
add_library(caffe ${cpps} ${hpps} ${hdrs})
target_link_libraries(caffe caffe_proto libopenblas  pybind11::embed)

add_executable(caffe_bin caffe.cpp ${cpps} ${hpps} ${hdrs})
target_link_libraries(caffe_bin caffe_proto libopenblas  pybind11::embed)