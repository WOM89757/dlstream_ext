# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

from gstgva import VideoFrame

DETECT_THRESHOLD = 0.5

Gst.init(sys.argv)

def process_frame(frame: VideoFrame) -> bool:
    width = frame.video_info().width
    height = frame.video_info().height

    x_min = int(0.1 * width + 0.5)
    y_min = int(0.1 * height + 0.5)
    x_max = int(0.5 * width + 0.5)
    y_max = int(0.8 * height + 0.5)
    label = "region"
    roi = frame.add_region(x_min, y_min, x_max - x_min, y_max - y_min, label)
    # frame.remove_region(roi)
    return True