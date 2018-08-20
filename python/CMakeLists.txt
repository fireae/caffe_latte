#if(NOT HAVE_PYTHON)
#  message(STATUS "Python interface is disabled or not all required dependencies found. Building without it...")
#  return()
#endif()
find_package(Boost)
message("${Boost_INCLUDE_DIRS}")
message("${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(D:\\Apps\\Anaconda3\\Lib\\site-packages\\numpy\\core\\include)
file(GLOB_RECURSE python_srcs ${PROJECT_SOURCE_DIR}/python/*.cpp)
link_directories(D:\\Apps\\Anaconda3\\Library\\lib)
set(PYTHON_LIBRARIES boost_python)
add_library(pycaffe SHARED ${python_srcs})
message("${Caffe_LINK}")
set_target_properties(pycaffe PROPERTIES PREFIX "" OUTPUT_NAME "_caffe")
target_include_directories(pycaffe PUBLIC ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_link_libraries(pycaffe PUBLIC caffe)

#if(UNIX OR APPLE)
#    set(__linkname "${PROJECT_SOURCE_DIR}/python/caffe/_caffe.so")
#    add_custom_command(TARGET pycaffe POST_BUILD
#                       COMMAND ln -sf $<TARGET_LINKER_FILE:pycaffe> "${__linkname}"
#                       COMMAND ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/python/caffe/proto
#                       COMMAND touch ${PROJECT_SOURCE_DIR}/python/caffe/proto/__init__.py
#                       COMMAND cp ${proto_gen_folder}/*.py ${PROJECT_SOURCE_DIR}/python/caffe/proto/
#                       COMMENT "Creating symlink ${__linkname} -> ${PROJECT_BINARY_DIR}/lib/_caffe${Caffe_POSTFIX}.so")
#endif()
#
## ---[ Install
## scripts
#file(GLOB python_files *.py requirements.txt)
#install(FILES ${python_files} DESTINATION python)
#
## module
#install(DIRECTORY caffe
#    DESTINATION python
#    FILES_MATCHING
#    PATTERN "*.py"
#    PATTERN "ilsvrc_2012_mean.npy"
#    PATTERN "test" EXCLUDE
#    )
#
## _caffe.so
#install(TARGETS pycaffe  DESTINATION python/caffe)

