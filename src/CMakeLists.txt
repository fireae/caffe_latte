set(src_dir ${PROJECT_SOURCE_DIR}/src/caffe/)
# generate protobuf sources
if(WIN32)
    set(Caffe_protobuf libprotobuf)
    set(Caffe_blas libopenblas)
else() 
    set(Caffe_protobuf protobuf)
    set(Caffe_blas openblas pthread)
endif()

add_library(caffe_proto STATIC ${src_dir}/proto/caffe.pb.cc)
target_link_libraries(caffe_proto ${Caffe_protobuf})
file(GLOB_RECURSE cpps ${src_dir}/*.cpp)
<<<<<<< HEAD
file(GLOB_RECURSE hpps ${src_dir}/*.hpp)
file(GLOB_RECURSE hdrs ${src_dir}/*.h)
add_library(caffe ${cpps} ${hpps} ${hdrs})
target_link_libraries(caffe caffe_proto libopenblas  pybind11::embed)

add_executable(caffe_bin caffe.cpp ${cpps} ${hpps} ${hdrs})
target_link_libraries(caffe_bin caffe_proto libopenblas  pybind11::embed)
=======
file(GLOB_RECURSE hpps ${PROJECT_SOURCE_DIR}/include/caffe/*.hpp)
add_library(caffe ${cpps} ${hpps})
target_link_libraries(caffe caffe_proto ${Caffe_blas}  pybind11::embed)

add_executable(caffe_bin caffe.cpp ${cpps} ${hpps})
target_link_libraries(caffe_bin caffe_proto ${Caffe_blas}  pybind11::embed)
>>>>>>> 0e5c50a8e0f1479c033bdefbc29fc55528748178
