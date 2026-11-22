// Copyright 2026 Blake Bottum
// SPDX-License-Identifier: Apache-2.0
// THE RUN-TIME ENVIRONMENT (docs/internals/common/plan/environment.md): a device's facts, read once,
// and the descriptors made from them.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "plan/planner.h"

namespace fsla::plan {

struct DeviceFacts {
  Descriptor descriptor;
  std::string model;
  int cc;
  int sms;
};

const DeviceFacts& device_facts(int device);

Descriptor descriptor_of(int device, const Named& call);

Descriptor checked_descriptor(const std::vector<std::int64_t>& words, int device);

}  // namespace fsla::plan
