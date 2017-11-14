cmake_minimum_required(VERSION 2.8)
project(mtcnn_frcnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CAFFE_ROOT /home/wencc/Myplace/caffe_latte/)
include_directories(${PROJECT_SOURCE_DIR}/detect)
include_directories(${PROJECT_SOURCE_DIR}/alignment/include)
link_directories(${PROJECT_SOURCE_DIR}/alignment/build)
include_directories(${CAFFE_ROOT}/include ${CAFFE_ROOT}/build/include)

link_directories(${CAFFE_ROOT}/build/lib)
link_directories(/home/wencc/workspace/face-alignment-at-3000fps/build)

add_library(mtfr_det_align SHARED ${PROJECT_SOURCE_DIR}/detect/mtcnn_detect_align.cpp ${PROJECT_SOURCE_DIR}/detect/faster_rcnn.cpp ${PROJECT_SOURCE_DIR}/detect/frcnn.cpp ${PROJECT_SOURCE_DIR}/detect/mtcnn.cpp)

add_executable(mtfr_detect_alignment ${PROJECT_SOURCE_DIR}/test/detect_alignment.cpp)
add_executable(mtfr_det_align_main ${PROJECT_SOURCE_DIR}/test/det_align_main.cpp)
add_executable(frcnn_align ${PROJECT_SOURCE_DIR}/test/frcnn_align.cpp)

#target_link_libraries(mtmain face_detect caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc)

target_link_libraries(mtfr_detect_alignment mtfr_det_align lib3000fps liblinear caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_ml opencv_objdetect)
target_link_libraries(mtfr_det_align_main mtfr_det_align lib3000fps liblinear caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_ml opencv_objdetect)

target_link_libraries(frcnn_align mtfr_det_align lib3000fps liblinear caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_ml opencv_objdetect)
