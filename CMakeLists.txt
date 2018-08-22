cmake_minimum_required(VERSION 3.4)

if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "1.0.0" CACHE STRING "Caffe logical version")
set(CAFFE_TARGET_SOVERSION "1.0.0" CACHE STRING "Caffe soname version")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Output/Lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Output/Lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Output/Bin)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)

# ---[ Options
caffe_option(USE_CUDA  "Build Caffe without CUDA support" OFF) 
caffe_option(USE_CUDNN "Build Caffe with cuDNN library support" ON IF USE_CUDA)
caffe_option(USE_NCCL "Build Caffe with NCCL library support" OFF)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF)
caffe_option(BUILD_tools  "Build tools" OFF)
caffe_option(BUILD_examples  "Build examples" OFF)
caffe_option(BUILD_docs   "Build documentation" OFF IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_OPENCV "Build with OpenCV support" ON)
caffe_option(USE_LEVELDB "Build with levelDB" OFF)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(ALLOW_LMDB_NOLOCK "Allow MDB_NOLOCK when reading LMDB files (only if necessary)" OFF)
caffe_option(USE_OPENMP "Link with OpenMP (when your BLAS wants OpenMP and you get linker errors)" OFF)
caffe_option(protobuf_MODULE_COMPATIBLE "Make the protobuf-config.cmake compatible with the module mode" ON IF MSVC)
caffe_option(COPY_PREREQUISITES "Copy the prerequisites next to each executable or shared library directory" ON IF MSVC)
caffe_option(INSTALL_PREREQUISITES "Install the prerequisites next to each executable or shared library directory" ON IF MSVC)
caffe_option(USE_WARP_CTC "Build Caffe with Warp_ctc" ON)
if(MSVC)
  # default to static libs
  caffe_option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
else()
  caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
endif()
if(MSVC AND BUILD_SHARED_LIBS)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    # see issue https://gitlab.kitware.com/cmake/cmake/issues/16552#note_215236
    message(FATAL_ERROR "The Visual Studio generator cannot build a shared library. Use the Ninja generator instead.")
  endif()
  # Some tests (solver tests) fail when caffe is built as a shared library. The problem comes
  # from protobuf that has a global static empty_string_ variable. Since caffe and test.testbin
  # link to a static protobuf library both end up with their own instance of the empty_string_
  # variable. This causes some SEH exception to occur. In practice if the caffe executable does not link
  # to protobuf this problem should not happen. Use at your own risk.
  message(WARNING "Some tests (solvers) will fail when building as a shared library with MSVC")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
set(Caffe_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR})
include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR})

if (WIN32)
set(ANACONDA_PATH )
else()
set(ANACONDA_PATH /home/wencc/miniconda2/)
endif()

include_directories(${ANACONDA_PATH}/include)
link_directories(${ANACONDA_PATH}/lib)
include_directories(${PROJECT_SOURCE_DIR}/include/caffe/proto)

if(USE_WARP_CTC)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/warp-ctc/include)
#list(APPEND Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/warp-ctc/include)
endif()
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/)
# ---[ Includes & defines for CUDA
# cuda_compile() does not have per-call dependencies or include pathes
# (cuda_compile() has per-call flags, but we set them here too for clarity)
#
# list(REMOVE_ITEM ...) invocations remove PRIVATE and PUBLIC keywords from collected definitions and include pathes
if(HAVE_CUDA)
  # pass include pathes to cuda_include_directories()
  set(Caffe_ALL_INCLUDE_DIRS ${Caffe_INCLUDE_DIRS})
  list(REMOVE_ITEM Caffe_ALL_INCLUDE_DIRS PRIVATE PUBLIC)
  cuda_include_directories(${Caffe_INCLUDE_DIR} ${Caffe_SRC_DIR} ${Caffe_ALL_INCLUDE_DIRS})

  # add definitions to nvcc flags directly
  set(Caffe_ALL_DEFINITIONS ${Caffe_DEFINITIONS})
  list(REMOVE_ITEM Caffe_ALL_DEFINITIONS PRIVATE PUBLIC)
  list(APPEND CUDA_NVCC_FLAGS ${Caffe_ALL_DEFINITIONS})
endif()

# ---[ Subdirectories
add_subdirectory(3rdparty/pybind11)
include_directories(${PROJECT_SOURCE_DIR}/3rdparty/pybind11/include)
add_subdirectory(src)
add_subdirectory(3rdparty/warp_ctc)
#add_subdirectory(pycaffe)
#add_subdirectory(python)
add_subdirectory(tools)
#add_subdirectory(apps)


