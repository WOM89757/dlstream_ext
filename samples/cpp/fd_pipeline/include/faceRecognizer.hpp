#pragma once

#include "cnn.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "logger.hpp"
#include <inference_engine.hpp>

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual bool LabelExists(const std::string &label) const = 0;
    virtual std::string GetLabelByID(int id) const = 0;
    virtual std::string GetPersonIdByID(int id) const = 0;
    virtual std::vector<std::string> GetIDToLabelMap() const = 0;
    virtual std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) = 0;

    virtual float GetThresholdByID(int id) const = 0;
    virtual std::vector<int> GetIDsByEmbeddingsWithThreshold(const std::vector<cv::Mat>& embeddings) = 0;


    virtual std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const = 0;

    virtual void PrintPerformanceCounts(
        const std::string &landmarks_device, const std::string &reid_device) = 0;
};



class FaceRecognizerDefault : public FaceRecognizer {
public:
    FaceRecognizerDefault(
            const CnnConfig& landmarks_detector_config,
            const CnnConfig& reid_config,
            const detection::DetectorConfig& face_registration_det_config,
            const std::string& face_gallery_path,
            double reid_threshold,
            int min_size_fr,
            bool crop_gallery,
            bool greedy_reid_matching
    )
        : landmarks_detector(landmarks_detector_config),
          face_reid(reid_config),
          face_gallery(face_gallery_path, reid_threshold, min_size_fr, crop_gallery,
                       face_registration_det_config, landmarks_detector, face_reid,
                       greedy_reid_matching)
    {
        if (face_gallery.size() == 0) {
            std::cout << "Face reid gallery is empty!" << std::endl;
        } else {
            std::cout << "Face reid gallery size: " << face_gallery.size() << std::endl;
        }
    }

    bool LabelExists(const std::string &label) const override {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const override {
        return face_gallery.GetLabelByID(id);
    }

    std::string GetPersonIdByID(int id) const override {
        return face_gallery.GetPersonIdByID(id);
    }

    float GetThresholdByID(int id) const override {
        return face_gallery.GetThresholdByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const override {
        return face_gallery.GetIDToLabelMap();
    }
    
    std::vector<int> GetIDsByEmbeddings(const std::vector<cv::Mat>& embeddings) const override {
        return face_gallery.GetIDsByEmbeddings(embeddings);
    }
    std::vector<int> GetIDsByEmbeddingsWithThreshold(const std::vector<cv::Mat>& embeddings) override {
        return face_gallery.GetGalleryObjectByEmbeddings(embeddings);
    }
    

    std::vector<int> Recognize(const cv::Mat& frame, const detection::DetectedObjects& faces) override {
        std::vector<cv::Mat> face_rois;

        for (const auto& face : faces) {
            face_rois.push_back(frame(face.rect));
        }

        std::vector<cv::Mat> landmarks, embeddings;

        landmarks_detector.Compute(face_rois, &landmarks, cv::Size(2, 5));
        std::cout << "landmark_points size: " << landmarks.size() << std::endl;
        std::cout << "landmark_points size: " << landmarks[0].rows << std::endl;
        std::cout << "landmark_points size: " << landmarks[0].cols << std::endl;
        AlignFaces(&face_rois, &landmarks);
        face_reid.Compute(face_rois, &embeddings);
        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

    void PrintPerformanceCounts(
            const std::string &landmarks_device, const std::string &reid_device) override {
        landmarks_detector.PrintPerformanceCounts(landmarks_device);
        face_reid.PrintPerformanceCounts(reid_device);
    }

private:
    VectorCNN landmarks_detector;
    VectorCNN face_reid;
    EmbeddingsGallery face_gallery;
};
