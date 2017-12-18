set(name ctpn)
file(GLOB source_file ${PROJECT_SOURCE_DIR}/apps/${name}/*.cpp)

add_executable(${name} ${source_file})
target_link_libraries(${name} ${Caffe_LINK})
caffe_default_properties(${name})

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
