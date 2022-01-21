/*
 * INTEL CONFIDENTIAL
 * Copyright 2018-2019 Intel Corporation
 *
 * The source code contained or described herein and all documents related to
 * the source code ("Material") are owned by Intel Corporation or its suppliers
 * or licensors. Title to the Material remains with Intel Corporation or
 * its suppliers and licensors. The Material contains trade secrets and
 * proprietary and confidential information of Intel or its suppliers and
 * licensors. The Material is protected by worldwide copyright and trade secret
 * laws and treaty provisions. No part of the Material may be used, copied,
 * reproduced, modified, published, uploaded, posted, transmitted, distributed,
 * or disclosed in any way without Intel's prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual
 * property right is granted to or conferred upon you by disclosure or delivery
 * of the Materials, either expressly, by implication, inducement, estoppel or
 * otherwise. Any license under such intellectual property rights must be express
 * and approved by Intel in writing.
 */

#ifndef __VAS_COMMON_H__
#define __VAS_COMMON_H__

#include <cstdint>


// #define VAS_STATIC_LIB

#if defined (__linux__) || defined (__linux) || defined (linux)
    #define VAS_EXPORT __attribute__((visibility("default")))
#elif defined(_WIN32) || defined(WIN32)
    #if !defined(VAS_STATIC_LIB)
        #if defined (VAS_WIN_EXPORT)
            #define VAS_EXPORT __declspec(dllexport)
        #else
            #define VAS_EXPORT __declspec(dllimport)
        #endif
    #else
        #define VAS_EXPORT
    #endif   // !defined(VAS_STATIC_LIB)
#else
    #define VAS_EXPORT
#endif


/**
 * @namespace vas
 *
 * Namespace for all the API of Video Analytics Suite.
 */
namespace vas
{

/**
 * @class Version
 *
 * Contains version information.
 */
class Version
{
public:
    /**
     * Constructor.
     *
     * @param[in] major Major version.
     * @param[in] minor Minor version.
     * @param[in] patch Patch version.
     */
    explicit Version(uint32_t major, uint32_t minor, uint32_t patch) : major_(major), minor_(minor), patch_(patch) {}

    /**
     * Returns major version.
     */
    uint32_t GetMajor() const noexcept { return major_; }

    /**
     * Returns minor version.
     */
    uint32_t GetMinor() const noexcept { return minor_; }

    /**
     * Returns patch version.
     */
    uint32_t GetPatch() const noexcept { return patch_; }

private:
    uint32_t major_;
    uint32_t minor_;
    uint32_t patch_;
};


/**
 * @enum BackendType
 *
 * Represents HW backend types.
 */
enum class BackendType
{
    CPU,    /**< CPU */
    GPU,    /**< GPU */
    VPU,    /**< VPU */
    FPGA,   /**< FPGA */
    HDDL    /**< HDDL */
};

/**
 * @enum ColorFormat
 *
 * Represents Color formats.
 */
enum class ColorFormat
{
    BGR,
    NV12,
    BGRX,
    GRAY,
    I420
};

};   // namespace vas


#endif   // __VAS_COMMON_H__
