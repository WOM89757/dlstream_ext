# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

cmake_minimum_required (VERSION 3.1)

include(CMakeParseArguments)
macro(add_demo)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_SAMPLE_SOURCES})
    if(IE_SAMPLE_HEADERS)
        source_group("include" FILES ${IE_SAMPLE_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_SAMPLE_NAME} ${IE_SAMPLE_SOURCES} ${IE_SAMPLE_HEADERS})

    if(WIN32)
        set_target_properties(${IE_SAMPLE_NAME} PROPERTIES COMPILE_PDB_NAME ${IE_SAMPLE_NAME})
    endif()

    if(IE_SAMPLE_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_SAMPLE_NAME} PRIVATE ${IE_SAMPLE_INCLUDE_DIRECTORIES} )
    endif()

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES}
                                                    ${IE_SAMPLE_DEPENDENCIES} ngraph::ngraph oputils gflags)

    if(UNIX)
        target_link_libraries(${IE_SAMPLE_NAME} PRIVATE pthread)
    endif()
endmacro()

find_package(OpenCV REQUIRED COMPONENTS core highgui videoio imgproc imgcodecs gapi)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

add_subdirectory(cpp/draw_face_attributes)
add_subdirectory(cpp/pvd_pipeline)
add_subdirectory(cpp/common)
add_subdirectory(cpp/fd_pipeline)
add_subdirectory(cpp/fire_pipeline)


add_custom_target(copy_model_proc ALL)
add_custom_command(TARGET copy_model_proc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/model_proc ${CMAKE_SAMPLES_OUTPUT_DIRECTORY}/model_proc)

