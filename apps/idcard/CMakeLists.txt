set(name idcard)
file(GLOB source_file ${PROJECT_SOURCE_DIR}/apps/${name}/*.cpp)
include_directories(/home/wencc/workspace/ltp/include)
link_directories(/home/wencc/workspace/ltp/lib)

add_executable(${name} ${source_file})
target_link_libraries(${name} ${Caffe_LINK})# segmentor postagger)

# set back RUNTIME_OUTPUT_DIRECTORY
#set_target_properties(${name} PROPERTIES
#RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/apps/${name}")

caffe_set_solution_folder(${name} apps)

# install
install(TARGETS ${name} DESTINATION ${CMAKE_INSTALL_BINDIR})


if(UNIX OR APPLE)
# Funny command to make tutorials work
# TODO: remove in future as soon as naming is standardized everywhere
set(__outname ${PROJECT_BINARY_DIR}/apps/${name}/${name}${Caffe_POSTFIX})
add_custom_command(TARGET ${name} POST_BUILD
               COMMAND ln -sf "${__outname}" "${__outname}.bin")
endif()
