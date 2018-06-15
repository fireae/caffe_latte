# These lists are later turned into target properties on main caffe library target
set(Caffe_LINKER_LIBS "")
set(Caffe_INCLUDE_DIRS "")
set(Caffe_DEFINITIONS "")
set(Caffe_COMPILE_OPTIONS "")


if(DEFINED MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0.40629.0)
  # Required for VS 2013 Update 4 or earlier.
  list(APPEND Caffe_DEFINITIONS PUBLIC -DBOOST_NO_CXX11_TEMPLATE_ALIASES)
endif()
# ---[ Threads
find_package(Threads REQUIRED)
list(APPEND Caffe_LINKER_LIBS PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# ---[ OpenMP
if(USE_OPENMP)
  # Ideally, this should be provided by the BLAS library IMPORTED target. However,
  # nobody does this, so we need to link to OpenMP explicitly and have the maintainer
  # to flick the switch manually as needed.
  #
  # Moreover, OpenMP package does not provide an IMPORTED target as well, and the
  # suggested way of linking to OpenMP is to append to CMAKE_{C,CXX}_FLAGS.
  # However, this naïve method will force any user of Caffe to add the same kludge
  # into their buildsystem again, so we put these options into per-target PUBLIC
  # compile options and link flags, so that they will be exported properly.
  find_package(OpenMP REQUIRED)
  list(APPEND Caffe_LINKER_LIBS PRIVATE ${OpenMP_CXX_FLAGS})
  list(APPEND Caffe_COMPILE_OPTIONS PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# ---[ Google-protobuf
include(cmake/ProtoBuf.cmake)

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${LMDB_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${LMDB_LIBRARIES})
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_LMDB)
  if(ALLOW_LMDB_NOLOCK)
    list(APPEND Caffe_DEFINITIONS PRIVATE -DALLOW_LMDB_NOLOCK)
  endif()
endif()

# ---[ CUDA
include(cmake/Cuda.cmake)
if(NOT HAVE_CUDA)
  if(NOT USE_CUDA)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    if(USE_CUDA)
      message(STATUS "Use Cuda...")
      list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_CUDA)
    else()
      message(WARNING "-- CUDA is not detected by cmake. Building without it...")
    endif()
  endif()
else()
  if(NOT USE_CUDA)
    message(STATUS "-- CUDA is disabled. Building without it...")
  else()
    if(USE_CUDA)
      message(STATUS "Use Cuda...")
      list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_CUDA)
    else()
      message(WARNING "-- CUDA is not detected by cmake. Building without it...")
    endif()
  endif()
endif()

if(USE_NCCL)
  include("cmake/External/nccl.cmake")
  include_directories(SYSTEM ${NCCL_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS ${NCCL_LIBRARIES})
  add_definitions(-DUSE_NCCL)
endif()

# ---[ OpenCV
if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND) # if not OpenCV 3.x, then imgcodecs are not found
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenCV_INCLUDE_DIRS})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
  list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_OPENCV)
endif()

# ---[ BLAS
if(NOT APPLE)
  set(BLAS "open" CACHE STRING "Selected BLAS library")
  set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

  if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
    find_package(Atlas REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${Atlas_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${Atlas_LIBRARIES})
  elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
    find_package(OpenBLAS REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${OpenBLAS_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${OpenBLAS_LIB})
  elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
    find_package(MKL REQUIRED)
    list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${MKL_INCLUDE_DIR})
    list(APPEND Caffe_LINKER_LIBS PUBLIC ${MKL_LIBRARIES})
    list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_MKL)
  endif()
elseif(APPLE)
  find_package(vecLib REQUIRED)
  list(APPEND Caffe_INCLUDE_DIRS PUBLIC ${vecLib_INCLUDE_DIR})
  list(APPEND Caffe_LINKER_LIBS PUBLIC ${vecLib_LINKER_LIBS})

  if(VECLIB_FOUND)
    if(NOT vecLib_INCLUDE_DIR MATCHES "^/System/Library/Frameworks/vecLib.framework.*")
      list(APPEND Caffe_DEFINITIONS PUBLIC -DUSE_ACCELERATE)
    endif()
  endif()
endif()

# ---[ Python
# if(BUILD_python)
#   if(NOT "${python_version}" VERSION_LESS "3.0.0")
#     # use python3
#     find_package(PythonInterp 3.0)
#     find_package(PythonLibs 3.0)
#     find_package(NumPy 1.7.1)
#     # Find the matching boost python implementation
#     set(version ${PYTHONLIBS_VERSION_STRING})

#     STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
#     find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
#     set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

#     while(NOT "${version}" STREQUAL "" AND NOT Boost_PYTHON_FOUND)
#       STRING( REGEX REPLACE "([0-9.]+).[0-9]+" "\\1" version ${version} )

#       STRING( REGEX REPLACE "[^0-9]" "" boost_py_version ${version} )
#       find_package(Boost 1.46 COMPONENTS "python-py${boost_py_version}")
#       set(Boost_PYTHON_FOUND ${Boost_PYTHON-PY${boost_py_version}_FOUND})

#       STRING( REGEX MATCHALL "([0-9.]+).[0-9]+" has_more_version ${version} )
#       if("${has_more_version}" STREQUAL "")
#         break()
#       endif()
#     endwhile()
#     if(NOT Boost_PYTHON_FOUND)
#       find_package(Boost 1.46 COMPONENTS python)
#     endif()
#   else()
#     # disable Python 3 search
#     find_package(PythonInterp 2.7)
#     find_package(PythonLibs 2.7)
#     find_package(NumPy 1.7.1)
#     find_package(Boost 1.46 COMPONENTS python)
#   endif()
#   if(PYTHONLIBS_FOUND AND NUMPY_FOUND AND Boost_PYTHON_FOUND)
#     set(HAVE_PYTHON TRUE)
#     if(Boost_USE_STATIC_LIBS AND MSVC)
#       list(APPEND Caffe_DEFINITIONS PUBLIC -DBOOST_PYTHON_STATIC_LIB)
#     endif()
#     if(BUILD_python_layer)
#       list(APPEND Caffe_DEFINITIONS PRIVATE -DWITH_PYTHON_LAYER)
#       list(APPEND Caffe_INCLUDE_DIRS PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} PUBLIC ${Boost_INCLUDE_DIRS})
#       list(APPEND Caffe_LINKER_LIBS PRIVATE ${PYTHON_LIBRARIES} PUBLIC ${Boost_LIBRARIES})
#     endif()
#   endif()
# endif()


