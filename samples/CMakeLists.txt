# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

cmake_minimum_required (VERSION 3.1)

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

