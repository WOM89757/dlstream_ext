# ==============================================================================
# Copyright (C) 2018-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

cmake_minimum_required(VERSION 3.1)

set (TARGET_NAME "fd_pipeline")

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

find_package(OpenCV REQUIRED core imgproc)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GSTREAMER gstreamer-1.0>=1.16 REQUIRED)
pkg_check_modules(GSTVIDEO gstreamer-video-1.0>=1.16 REQUIRED)
pkg_check_modules(GLIB2 glib-2.0 REQUIRED)

# use pkg-config if sample builds as standalone. Otherwise vars DLSTREAMER_INCLUDE_DIRS/etc set by top level cmake
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR})
    pkg_check_modules(DLSTREAMER dl-streamer REQUIRED)
endif()

# add_subdirectory(common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/oputils/include/utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/oputils/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file (GLOB MAIN_SRC *.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file (GLOB MAIN_HEADERS *.h*
                        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*
)

add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})

set_target_properties(${TARGET_NAME} PROPERTIES CMAKE_CXX_STANDARD 14)

target_include_directories(${TARGET_NAME}
PRIVATE
        ${GSTVIDEO_INCLUDE_DIRS}
        ${GSTREAMER_INCLUDE_DIRS}
        ${GLIB2_INCLUDE_DIRS}
        ${DLSTREAMER_INCLUDE_DIRS}
        ${InferenceEngine_INCLUDE_DIRS}
)

target_link_libraries(${TARGET_NAME}
PRIVATE
        ${OpenCV_LIBS}
        ${GLIB2_LIBRARIES}
        ${GSTREAMER_LDFLAGS}
        ${GSTVIDEO_LDFLAGS}
        oputils
        ngraph::ngraph
    )


    
# file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

# add_demo(NAME fd_pipeline
#     SOURCES ${SOURCES}
#     HEADERS ${HEADERS}
#     INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/"
#     DEPENDENCIES monitors)
    