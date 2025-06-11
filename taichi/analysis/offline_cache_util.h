#pragma once

#include <string>

#include "taichi/rhi/arch.h"

namespace taichi::lang {

struct CompileConfig;
struct DeviceCapabilityConfig;
class SNode;
class Kernel;

std::string get_hashed_offline_cache_key_of_snode(const Kernel *kernel, const SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         const DeviceCapabilityConfig &caps,
                                         const Kernel *kernel);

}  // namespace taichi::lang
