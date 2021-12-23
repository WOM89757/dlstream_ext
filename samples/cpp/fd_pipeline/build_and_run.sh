#!/bin/bash
# ==============================================================================
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================
# source /opt/intel/openvino_2021/bin/setupvars.sh
BASE_DIR=$PWD
BUILD_DIR=$HOME/intel/dl_streamer/samples/fd_pipeline/build
rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

MODELS_PATH=/home/zy/opvinomodels/intel/

if [ -f /etc/lsb-release ]; then
    cmake ${BASE_DIR}
else
    cmake3 ${BASE_DIR}
fi

make -j $(nproc)

cd ${BASE_DIR}

FILE=${1:-https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female-and-male.mp4}
${BUILD_DIR}/fd_pipeline -i $FILE
