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
    # frame.remove_region(roi)
    for roi in frame.regions():
        label = roi.label()
    #     # print(label)
        # if label == "region":

            # frame.remove_region(roi)

    return True