#pragma once

#include <string>
#include <ostream>

#include "taichi/rhi/arch.h"

namespace taichi::lang {

class IRNode;
class Kernel;

void gen_offline_cache_key(const Kernel *kernel, IRNode *ast, std::ostream *os);

}  // namespace taichi::lang
