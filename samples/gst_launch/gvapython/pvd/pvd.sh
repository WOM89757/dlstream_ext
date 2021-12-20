#!/bin/bash
# ==============================================================================
# Copyright (C) 2018-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set -e

INPUT=${1:-https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female-and-male.mp4}

DEVICE=${2:-CPU}

if [[ $3 == "display" ]] || [[ -z $3 ]]; then
  SINK_ELEMENT="gvawatermark ! videoconvert ! fpsdisplaysink video-sink=xvimagesink sync=true"
elif [[ $3 == "fps" ]]; then
  SINK_ELEMENT="gvafpscounter ! fakesink async=false "
else
  echo Error wrong value for SINK_ELEMENT parameter
  echo Possible values: display - render, fps - show FPS only
  exit
fi

SCRIPTDIR="$(dirname "$(realpath "$0")")"
PYTHON_SCRIPT1=$SCRIPTDIR/postproc_callbacks/add_roiList.py
PYTHON_SCRIPT2=$SCRIPTDIR/postproc_callbacks/remove_roiList.py
PYTHON_SCRIPT3=$SCRIPTDIR/postproc_callbacks/age_logger.py

if [[ $INPUT == "/dev/video"* ]]; then
  SOURCE_ELEMENT="v4l2src device=${INPUT}"
elif [[ $INPUT == *"://"* ]]; then
  SOURCE_ELEMENT="urisourcebin buffer-size=4096 uri=${INPUT}"
else
  SOURCE_ELEMENT="filesrc location=${INPUT}"
fi

DETECT_MODEL_PATH=${MODELS_PATH}/intel/person-vehicle-bike-detection-2004/FP32/person-vehicle-bike-detection-2004.xml
DETECT_MODEL_PROC=${MODELS_PATH}/model_proc/intel/object_detection/person-vehicle-bike-detection-2004.json
CLASS_MODEL_PATH=${MODELS_PATH}/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml

echo Running sample with the following parameters:
echo GST_PLUGIN_PATH=${GST_PLUGIN_PATH}

PIPELINE="gst-launch-1.0 \
$SOURCE_ELEMENT ! decodebin ! \
gvapython module=$PYTHON_SCRIPT1 ! \
gvadetect model=$DETECT_MODEL_PATH model-proc=$DETECT_MODEL_PROC device=$DEVICE inference-interval=10 threshold=0.6 inference-region=1  object-class=region ! queue ! \
gvatrack tracking-type=short-term ! \
gvapython module=$PYTHON_SCRIPT2 ! \
tee name=t ! queue ! \
$SINK_ELEMENT t. ! queue ! \
gvametaconvert tags={\"steamid\":\ 12} ! \
tee name=p ! queue ! \
gvametapublish method=mqtt address=192.168.56.103:1883 topic=dl ! fakesink p. ! queue ! \
gvametapublish method=file file-path=/home/zy/opvinomodels/re.json file-format=json ! fakesink"


# gvapython module=$PYTHON_SCRIPT1 ! \
# gvainference inference-region=roi-list model=$CLASS_MODEL_PATH device=$DEVICE ! queue ! \
# gvapython module=$PYTHON_SCRIPT2 ! \
# gvapython module=$PYTHON_SCRIPT3 class=AgeLogger function=log_age kwarg={\\\"log_file_path\\\":\\\"/tmp/age_log.txt\\\"} ! \

echo ${PIPELINE}
PYTHONPATH=$PYTHONPATH:$(dirname "$0")/../../../../python \
${PIPELINE}
