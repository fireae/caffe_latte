set(src_dir ${PROJECT_SOURCE_DIR}/src/caffe/)
# generate protobuf sources
if(WIN32)
    set(Caffe_protobuf libprotobuf)
    set(Caffe_blas libopenblas)
else() 
    set(Caffe_protobuf protobuf)
    set(Caffe_blas openblas pthread boost_system opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs dl)
endif()

add_library(caffe_proto STATIC ${src_dir}/proto/caffe.pb.cc)
target_link_libraries(caffe_proto ${Caffe_protobuf})
file(GLOB_RECURSE cpps ${src_dir}/*.cpp)
file(GLOB_RECURSE hpps ${PROJECT_SOURCE_DIR}/include/caffe/*.hpp)

add_library(caffe ${cpps} ${hpps})
target_link_libraries(caffe ${Caffe_LINKER_LIBS})
target_include_directories(caffe ${Caffe_INCLUDE_DIRS}
                                 PUBLIC
                                 $<BUILD_INTERFACE:${Caffe_INCLUDE_DIR}>
                                 $<INSTALL_INTERFACE:include>)
#target_compile_definitions(caffe ${Caffe_DEFINITIONS})
if(Caffe_COMPILE_OPTIONS)
  target_compile_options(caffe ${Caffe_COMPILE_OPTIONS})
endif()

# add_executable(caffe_bin caffe.cpp ${cpps} ${hpps})
# target_link_libraries(caffe_bin caffe_proto ${Caffe_LINKER_LIBS}  pybind11::embed)
