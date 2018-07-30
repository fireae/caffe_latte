#pybind11_add_module(pycaffe pycaffe.cpp)
add_library(pycaffe MODULE pycaffe.cpp)
target_link_libraries(pycaffe PRIVATE pybind11::module caffe)