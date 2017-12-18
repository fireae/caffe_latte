#判断编译器类型
if(MSVC)
#添加一个名为WITH_CRT_DLL的开关选项，
#这样就可以在cmake-gui中或在命令行随时修改该开关选项。
  option(WITH_CRT_DLL
    "Link all libjpeg-turbo libraries and executables with the C run-time DLL (msvcr*.dll) instead of the static C run-time library (libcmt*.lib.)  The default is to use the C run-time DLL only with the libraries and executables that need it."
    FALSE)
  if(NOT WITH_CRT_DLL)
  # for循环修改所有CMAKE_<LANG>_FLAGS开关的编译选项变量，用正则表达式将/MD替换成/MT
    # Use the static C library for all build types
    foreach(var CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${var} MATCHES "/MD")
        #正则表达式替换
        string(REGEX REPLACE "/MD" "/MT" ${var} "${${var}}")
      endif()
    endforeach()
  endif()
  add_definitions(-W3 -wd4996)
endif()