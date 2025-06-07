/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <TacosGreedy.h>

using namespace Tacos;

inline int count_within_tolerance(const std::vector<Tacos::Time>& samples,
                           Tacos::Time expected,
                           double tolerance) {
    int count = 0;

    for (const auto& time : samples) {
        const auto err = static_cast<double>(time) / expected;
        const auto error_rate = std::abs(err - 1.0);
        if (error_rate <= tolerance) {
            count++;
        }
    }

    return count;
}
