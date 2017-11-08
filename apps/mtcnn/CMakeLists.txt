cmake_minimum_required(VERSION 2.8)
project(FaceDetectAlign)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CAFFE_ROOT /home/wencc/Myplace/caffe_latte/)
include_directories(${PROJECT_SOURCE_DIR}/detect)
include_directories(${PROJECT_SOURCE_DIR}/alignment/include)
link_directories(${PROJECT_SOURCE_DIR}/alignment/build)
include_directories(${CAFFE_ROOT}/include ${CAFFE_ROOT}/build/include)

link_directories(${CAFFE_ROOT}/build/lib)
link_directories(/home/wencc/workspace/face-alignment-at-3000fps/build)
add_library(face_detect SHARED ${PROJECT_SOURCE_DIR}/detect/mtcnn.cc)

# add_library(face_alignment SHARED ${PROJECT_SOURCE_DIR}/alignment/Tree.cpp 
#     ${PROJECT_SOURCE_DIR}/alignment/Utils.cpp ${PROJECT_SOURCE_DIR}/alignment/Facedetect.cpp
#     ${PROJECT_SOURCE_DIR}/alignment/LBFRegressor.cpp 
#     ${PROJECT_SOURCE_DIR}/alignment/RandomForest.cpp
#     ${PROJECT_SOURCE_DIR}/alignment/TestModel.cpp ${PROJECT_SOURCE_DIR}/alignment/TrainModel.cpp
#     ${PROJECT_SOURCE_DIR}/alignment/liblinear/linear.cpp ${PROJECT_SOURCE_DIR}/alignment/liblinear/tron.cpp
#     ${PROJECT_SOURCE_DIR}/alignment/liblinear/blas/daxpy.c ${PROJECT_SOURCE_DIR}/alignment/liblinear/blas/ddot.c 
#     ${PROJECT_SOURCE_DIR}/alignment/liblinear/blas/dnrm2.c ${PROJECT_SOURCE_DIR}/alignment/liblinear/blas/dscal.c)

add_library(det_align SHARED ${PROJECT_SOURCE_DIR}/detect/mtcnn_detect_align.cc)

# add_executable(lbfmain ${PROJECT_SOURCE_DIR}/test/LBF.cpp)
add_executable(mtmain ${PROJECT_SOURCE_DIR}/test/MTMain.cpp)
add_executable(da ${PROJECT_SOURCE_DIR}/test/detect_alignment.cpp)
add_executable(det_align_main ${PROJECT_SOURCE_DIR}/test/det_align_main.cc)

target_link_libraries(mtmain face_detect caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc)
# target_link_libraries(lbfmain face_alignment opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_ml opencv_objdetect)

target_link_libraries(da face_detect lib3000fps liblinear caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_ml opencv_objdetect)
target_link_libraries(det_align_main det_align lib3000fps liblinear face_detect caffe boost_system boost_thread glog gflags openblas opencv_core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_ml opencv_objdetect)
