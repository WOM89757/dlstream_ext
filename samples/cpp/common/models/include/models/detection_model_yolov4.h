/*
// Copyright (C) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include "detection_model.h"


class ModelYolo4 : public DetectionModel {
protected:

    class RegionYolov4 {
    public:
        int num = 0;
        int classes = 0;
        int coords = 0;
        int sides[2];
        std::vector<float> anchors;
        friend std::ostream &operator<< (std::ostream &out, const RegionYolov4 &r) {
            out << r.num << " " << r.classes << " " << r.coords << " " << r.sides[0] << " " << r.anchors.size();
            return out;
        }
    };

public:
    /// Constructor.
    /// @param modelFileName name of model to load
    /// @param confidenceThreshold - threshold to eliminate low-confidence detections.
    /// Any detected object with confidence lower than this threshold will be ignored.
    /// @param useAutoResize - if true, image will be resized by IE.
    /// Otherwise, image will be preprocessed and resized using OpenCV routines.
    /// @param useAdvancedPostprocessing - if true, an advanced algorithm for filtering/postprocessing will be used
    /// (with better processing of multiple crossing objects). Otherwise, classic algorithm will be used.
    /// @param boxIOUThreshold - threshold to treat separate output regions as one object for filtering
    /// during postprocessing (only one of them should stay). The default value is 0.5
    /// @param labels - array of labels for every class. If this array is empty or contains less elements
    /// than actual classes number, default "Label #N" will be shown for missing items.
    ModelYolo4(const std::string& modelFileName, float confidenceThreshold, bool useAutoResize,
        bool useAdvancedPostprocessing = true, float boxIOUThreshold = 0.5, const std::vector<std::string>& labels = std::vector<std::string>());

    std::unique_ptr<ResultBase> postprocess(InferenceResult& infResult) override;

protected:
    void prepareInputsOutputs(InferenceEngine::CNNNetwork& cnnNetwork) override;

    void parseYOLOV4Output(const std::string& output_name, const InferenceEngine::Blob::Ptr& blob,
        const unsigned long resized_im_h, const unsigned long resized_im_w, const unsigned long original_im_h,
        const unsigned long original_im_w, std::vector<DetectedObject>& objects);

    static int calculateEntryIndex(int side, int lcoords, int lclasses, int location, int entry);
    static double intersectionOverUnion(const DetectedObject& o1, const DetectedObject& o2);

    std::map<std::string, RegionYolov4> regions;
    double boxIOUThreshold;
    bool useAdvancedPostprocessing;
};
