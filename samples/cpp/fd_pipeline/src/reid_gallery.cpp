// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "face_reid.hpp"
#include "tracker.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <limits>

#include <opencv2/opencv.hpp>



void split(const std::string& str, 
           std::vector<std::string>& tokens, 
           const std::string delim = " ") {
    tokens.clear();
    
    auto start = str.find_first_not_of(delim, 0);
    auto position = str.find_first_of(delim, start);
    while (position != std::string::npos || start != std::string::npos) {
        tokens.emplace_back(std::move(str.substr(start, position - start)));
        start = str.find_first_not_of(delim, position);
        position = str.find_first_of(delim, start);
    }
}

namespace {
    float ComputeReidDistance(const cv::Mat& descr1, const cv::Mat& descr2) {
        float xy = static_cast<float>(descr1.dot(descr2));
        float xx = static_cast<float>(descr1.dot(descr1));
        float yy = static_cast<float>(descr2.dot(descr2));
        float norm = sqrt(xx * yy) + 1e-6f;
        return 1.0f - xy / norm;
    }

    bool file_exists(const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    inline char separator() {
        #ifdef _WIN32
        return '\\';
        #else
        return '/';
        #endif
    }

    std::string folder_name(const std::string& path) {
        size_t found_pos;
        found_pos = path.find_last_of(separator());
        if (found_pos != std::string::npos)
            return path.substr(0, found_pos);
        return std::string(".") + separator();
    }

}  // namespace

const char EmbeddingsGallery::unknown_label[] = "Unknown";
const int EmbeddingsGallery::unknown_id = TrackedObject::UNKNOWN_LABEL_IDX;

RegistrationStatus EmbeddingsGallery::RegisterIdentity(const std::string& identity_label,
                                                       const cv::Mat& image,
                                                       int min_size_fr, bool crop_gallery,
                                                       detection::FaceDetection& detector,
                                                       const VectorCNN& landmarks_det,
                                                       const VectorCNN& image_reid,
                                                       cv::Mat& embedding) {
    cv::Mat target = image;
    if (crop_gallery) {
      detector.enqueue(image);
      detector.submitRequest();
      detector.wait();
      detection::DetectedObjects faces = detector.fetchResults();
      if (faces.size() == 0) {
        return RegistrationStatus::FAILURE_NOT_DETECTED;
      }
      cv::Mat face_roi = image(faces[0].rect);
      target = face_roi;
    }
    if ((target.rows < min_size_fr) && (target.cols < min_size_fr)) {
      return RegistrationStatus::FAILURE_LOW_QUALITY;
    }
    cv::Mat landmarks;
    landmarks_det.Compute(target, &landmarks, cv::Size(2, 5));
    std::vector<cv::Mat> images = {target};
    std::vector<cv::Mat> landmarks_vec = {landmarks};
    AlignFaces(&images, &landmarks_vec);
    image_reid.Compute(images[0], &embedding);
    return RegistrationStatus::SUCCESS;
}

EmbeddingsGallery::EmbeddingsGallery(const std::string& faceImgPath,
                                     double threshold, int min_size_fr,
                                     bool crop_gallery, const detection::DetectorConfig &detector_config,
                                     const VectorCNN& landmarks_det,
                                     const VectorCNN& image_reid,
                                     bool use_greedy_matcher)
    : reid_threshold(threshold),
      use_greedy_matcher(use_greedy_matcher) {
    if (faceImgPath.empty()) {
        return;
    }

    detection::FaceDetection detector(detector_config);
    
    // std::string faceImgPath = "/home/zy/opvinomodels/fdimgs/";
    std::vector<cv::String> filepaths;
    cv::glob(faceImgPath.c_str(), filepaths, false);
    if(filepaths.size() == 0) {
        std::cout << "No face img be found, init finished!" << std::endl;
        return ;
    }
    int id = 0;
    
    for (const auto& filepath : filepaths) {
        if (filepath.empty() || !(filepath.find(".jpg") == filepath.npos || filepath.find(".png") == filepath.npos)) {
            std::cout << "skip file: " << filepath << std::endl;
            continue;
        }
        std::vector<cv::Mat> embeddings;
        std::cout << "file path: " << filepath << std::endl;
        size_t s = filepath.find_last_of('/') + 1;
        size_t e = filepath.find_last_of('.');
        std::string label = filepath.substr(s, e - s);
        std::vector<std::string> data;
        split(label, data, "-");
        std::string personname = data[0];
        std::string personId = data[1];
        
        std::cout << "label: " << label << std::endl;
        std::cout << "personname: " << personname << std::endl;
        std::cout << "personId: " << personId << std::endl;


        cv::Mat image = cv::imread(filepath);
        CV_Assert(!image.empty());
        cv::Mat emb;
        RegistrationStatus status = RegisterIdentity(label, image, min_size_fr, crop_gallery,  detector, landmarks_det, image_reid, emb);
        if (status == RegistrationStatus::SUCCESS) {
            embeddings.push_back(emb);
            idx_to_id.push_back(id);
            identities.emplace_back(embeddings, label, id, personId);
            ++id;
        }
    }


}

std::vector<int> EmbeddingsGallery::GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const {
    if (embeddings.empty() || idx_to_id.empty())
        return std::vector<int>(embeddings.size(), unknown_id);

    cv::Mat distances(static_cast<int>(embeddings.size()), static_cast<int>(idx_to_id.size()), CV_32F);

    for (int i = 0; i < distances.rows; i++) {
        int k = 0;
        for (size_t j = 0; j < identities.size(); j++) {
            for (const auto& reference_emb : identities[j].embeddings) {
                distances.at<float>(i, k) = ComputeReidDistance(embeddings[i], reference_emb);
                k++;
            }
        }
    }
    KuhnMunkres matcher(use_greedy_matcher);
    auto matched_idx = matcher.Solve(distances);
    std::vector<int> output_ids;
    for (auto col_idx : matched_idx) {
        if (distances.at<float>(output_ids.size(), col_idx) > reid_threshold)
            output_ids.push_back(unknown_id);
        else{
            // identities[idx_to_id[col_idx]].threshold = distances.at<float>(output_ids.size(), col_idx);
            output_ids.push_back(idx_to_id[col_idx]);
        }
            
    }
    return output_ids;
}


std::vector<int> EmbeddingsGallery::GetGalleryObjectByEmbeddings(const std::vector<cv::Mat>& embeddings) {
    if (embeddings.empty() || idx_to_id.empty())
        return std::vector<int>(embeddings.size(), unknown_id);

    cv::Mat distances(static_cast<int>(embeddings.size()), static_cast<int>(idx_to_id.size()), CV_32F);

    for (int i = 0; i < distances.rows; i++) {
        int k = 0;
        for (size_t j = 0; j < identities.size(); j++) {
            for (const auto& reference_emb : identities[j].embeddings) {
                distances.at<float>(i, k) = ComputeReidDistance(embeddings[i], reference_emb);
                k++;
            }
        }
    }
    KuhnMunkres matcher(use_greedy_matcher);
    auto matched_idx = matcher.Solve(distances);
    std::vector<int> output_ids;
    for (auto col_idx : matched_idx) {
        if (distances.at<float>(output_ids.size(), col_idx) > reid_threshold)
            output_ids.push_back(unknown_id);
        else{
            identities[idx_to_id[col_idx]].threshold = distances.at<float>(output_ids.size(), col_idx);
            output_ids.push_back(idx_to_id[col_idx]);
        }
            
    }
    return output_ids;
}

std::string EmbeddingsGallery::GetLabelByID(int id) const {
    if (id >= 0 && id < static_cast<int>(identities.size()))
        return identities[id].label;
    else
        return unknown_label;
}

float EmbeddingsGallery::GetThresholdByID(int id) const {
    if (id >= 0 && id < static_cast<int>(identities.size()))
        return identities[id].threshold;
    else
        return 0;
}


std::string EmbeddingsGallery::GetPersonIdByID(int id) const {
    if (id >= 0 && id < static_cast<int>(identities.size()))
        return identities[id].personId;
    else
        return unknown_label;
}

size_t EmbeddingsGallery::size() const {
    return identities.size();
}

std::vector<std::string> EmbeddingsGallery::GetIDToLabelMap() const  {
    std::vector<std::string> map;
    map.reserve(identities.size());
    for (const auto& item : identities)  {
        map.emplace_back(item.label);
    }
    return map;
}

bool EmbeddingsGallery::LabelExists(const std::string& label) const {
    return identities.end() != std::find_if(identities.begin(), identities.end(),
                                        [label](const GalleryObject& o){return o.label == label;});
}
