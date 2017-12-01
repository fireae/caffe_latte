
include_directories(include)

option(WITH_OMP "compile warp-ctc with openmp." OF)

if(NOT WITH_OMP)
    add_definitions(-DCTC_DISABLE_OMP)
endif()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (USE_CUDA)

    MESSAGE(STATUS "Building shared library with GPU support")

    CUDA_ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cu src/ctc_cpp_entrypoint.cu src/reduce.cu)

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp src/ctc_cpp_entrypoint.cpp)
ENDIF()

list(APPEND Caffe_LINKER_LIBS warpctc)
