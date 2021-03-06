/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "post_processor.h"
#include "converters/to_roi/atss.h"
#include "converters/to_roi/ssd.h"
#include "converters/to_tensor/raw_data_copy.h"

#include "gstgvadetect.h"
#include "gva_base_inference.h"
#include "inference_backend/logger.h"
#include "inference_impl.h"

#include <exception>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

using namespace post_processing;

namespace {
std::set<std::string> getDeclaredLayers(const std::map<std::string, GstStructure *> &model_proc_outputs) {
    std::set<std::string> layers;

    for (const auto &model_proc_output : model_proc_outputs) {
        GstStructure *s = model_proc_output.second;
        if (s == nullptr)
            throw std::runtime_error("Can not get model_proc output information.");

        if (not gst_structure_has_field(s, "layer_name") and not gst_structure_has_field(s, "layer_names"))
            return layers;
        if (gst_structure_has_field(s, "layer_name") and gst_structure_has_field(s, "layer_names"))
            return layers;

        if (gst_structure_has_field(s, "layer_name") and not gst_structure_has_field(s, "layer_names")) {
            layers.emplace(gst_structure_get_string(s, "layer_name"));
        }

        if (gst_structure_has_field(s, "layer_names") and not gst_structure_has_field(s, "layer_name")) {
            GValueArray *arr = nullptr;
            gst_structure_get_array(const_cast<GstStructure *>(s), "layer_names", &arr);
            if (arr and arr->n_values) {
                for (guint i = 0; i < arr->n_values; ++i)
                    layers.emplace(g_value_get_string(g_value_array_get_nth(arr, i)));
                g_value_array_free(arr);
            } else {
                throw std::runtime_error("\"layer_names\" array is null.");
            }
        }
    }

    return layers;
}
std::set<std::string> getDeclaredLayers(const PostProcessor::ModelOutputsInfo &model_outputs_info) {
    std::set<std::string> layers;

    for (const auto &output : model_outputs_info) {
        layers.insert(output.first);
    }

    return layers;
}
} // namespace

PostProcessor::ModelProcOutputsValidationResult
PostProcessor::validateModelProcOutputs(const std::map<std::string, GstStructure *> &model_proc_outputs,
                                        const ModelOutputsInfo &model_outputs_info) const {
    const size_t procs_num = model_proc_outputs.size();
    if (procs_num == 0) {
        return PostProcessor::ModelProcOutputsValidationResult::USE_DEFAULT;
    }

    const auto proc_layers = getDeclaredLayers(model_proc_outputs);
    if (proc_layers.empty() and procs_num == 1) {
        return PostProcessor::ModelProcOutputsValidationResult::USE_DEFAULT;
    }

    if (proc_layers.empty() and procs_num > 1) {
        GVA_ERROR("Number of declared output_postprocs more then 1, but layers are not defined.");
        return PostProcessor::ModelProcOutputsValidationResult::FAIL;
    }

    const auto model_layers = getDeclaredLayers(model_outputs_info);

    for (const auto &proc_layer : proc_layers) {
        if (model_layers.find(proc_layer) == model_layers.cend()) {
            const std::string msg = proc_layer + " is not contained among model's output layers.";
            GVA_ERROR(msg.c_str());
            return PostProcessor::ModelProcOutputsValidationResult::FAIL;
        }
    }

    return PostProcessor::ModelProcOutputsValidationResult::OK;
}

void PostProcessor::setConverterIfNotSpecified(GstStructure *model_proc_output,
                                               const ModelOutputsInfo &model_outputs_info, int inference_type) {
    if (model_proc_output == nullptr)
        throw std::runtime_error("Can not get model_proc output information.");

    if (gst_structure_has_field(model_proc_output, "converter"))
        return;

    switch (inference_type) {
    case GST_GVA_DETECT_TYPE: {
        if (ATSSConverter::isValidModelOutputs(model_outputs_info))
            gst_structure_set(model_proc_output, "converter", G_TYPE_STRING, ATSSConverter::getName().c_str(), NULL);
        else if (SSDConverter::isValidModelOutputs(model_outputs_info))
            gst_structure_set(model_proc_output, "converter", G_TYPE_STRING, SSDConverter::getName().c_str(), NULL);
        else
            throw std::runtime_error("Failed to determine the default detection converter. Please specify it yourself "
                                     "in the 'model-proc' file.");
    } break;
    case GST_GVA_CLASSIFY_TYPE:
    case GST_GVA_INFERENCE_TYPE:
        gst_structure_set(model_proc_output, "converter", G_TYPE_STRING, RawDataCopyConverter::getName().c_str(), NULL);
        break;
    default:
        throw std::runtime_error("Unknown inference type.");
    }
}

PostProcessor::PostProcessor(InferenceImpl *inference_impl, GvaBaseInference *base_inference) {
    try {
        auto inference_type = base_inference->type;
        auto inference_region = base_inference->inference_region;
        const std::string any_layer_name = "ANY";

        const auto &model = inference_impl->GetModel();

        std::map<std::string, std::vector<std::string>> labels = model.labels;
        if (labels.empty())
            labels.insert(std::make_pair(any_layer_name, std::vector<std::string>{}));

        ModelImageInputInfo input_image_info;
        model.inference->GetModelImageInputInfo(input_image_info.width, input_image_info.height,
                                                input_image_info.batch_size, input_image_info.format,
                                                input_image_info.memory_type);

        ModelOutputsInfo model_outputs_info = model.inference->GetModelOutputsInfo();
        const std::string &model_name = model.name;

        auto validation_result = validateModelProcOutputs(model.output_processor_info, model_outputs_info);

        switch (validation_result) {
        case PostProcessor::ModelProcOutputsValidationResult::USE_DEFAULT: {
            std::unordered_set<std::string> layer_names;
            layer_names.reserve(model_outputs_info.size());
            for (const auto &output_info : model_outputs_info) {
                layer_names.insert(output_info.first);
            }

            std::map<std::string, GstStructure *> model_proc_outputs;
            GstStructureUniquePtr model_proc_output_info(nullptr, gst_structure_free);

            if (model.output_processor_info.empty()) {
                model_proc_output_info.reset(gst_structure_new_empty(any_layer_name.c_str()));
                model_proc_outputs.insert(std::make_pair(any_layer_name, model_proc_output_info.get()));
            } else {
                model_proc_outputs = model.output_processor_info;
            }
            setConverterIfNotSpecified(model_proc_outputs.cbegin()->second, model_outputs_info, inference_type);

            if (inference_type == GST_GVA_DETECT_TYPE) {
                GstGvaDetect *gva_detect = reinterpret_cast<GstGvaDetect *>(base_inference);
                gst_structure_set(model_proc_outputs.cbegin()->second, "confidence_threshold", G_TYPE_DOUBLE,
                                  gva_detect->threshold, NULL);
            }

            converters.emplace_back(layer_names, model_proc_outputs.cbegin()->second, inference_type, inference_region,
                                    input_image_info, model_name, labels.at(model_proc_outputs.cbegin()->first));
        } break;
        case PostProcessor::ModelProcOutputsValidationResult::OK: {
            for (const auto &model_proc_output : model.output_processor_info) {
                if (model_proc_output.second == nullptr) {
                    throw std::runtime_error("Can not get model_proc output information.");
                }

                if (inference_type == GST_GVA_DETECT_TYPE) {
                    GstGvaDetect *gva_detect = reinterpret_cast<GstGvaDetect *>(base_inference);
                    gst_structure_set(model_proc_output.second, "confidence_threshold", G_TYPE_DOUBLE,
                                      gva_detect->threshold, NULL);
                }

                converters.emplace_back(model_proc_output.second, inference_type, inference_region, input_image_info,
                                        model_name, labels.at(model_proc_output.first));
            }
        } break;

        default:
            throw std::runtime_error("Cannot create post-processor with current model-proc information for model: " +
                                     model_name);
        }

    } catch (const std::exception &e) {
        GVA_ERROR(e.what());
    }
}

PostProcessor::ExitStatus PostProcessor::process(const OutputBlobs &output_blobs, InferenceFrames &frames) const {
    try {
        for (const auto &converter : converters) {
            converter.convert(output_blobs, frames);
        }
    } catch (const std::exception &e) {
        GVA_ERROR(e.what());
        return ExitStatus::FAIL;
    }

    return ExitStatus::SUCCESS;
}
