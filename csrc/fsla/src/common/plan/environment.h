// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE RUN-TIME ENVIRONMENT (docs/internals/common/plan/environment.md)
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "plan/planner.h"

namespace fsla::plan {

struct DeviceFacts {
  Named read;
  std::string model;
  int cc;
  int sms;
};

const DeviceFacts& device_facts(int device);

template <typename K>
typename K::Descriptor descriptor_of(int device, const Named& call);

template <typename K>
typename K::Descriptor checked_descriptor(const std::vector<std::int64_t>& words, int device);

}  // namespace fsla::plan
