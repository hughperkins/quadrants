#include "taichi/python/snode_registry.h"

#include "taichi/common/logging.h"
#include "taichi/ir/snode.h"
#include "taichi/program/program.h"

namespace taichi::lang {

SNode *SNodeRegistry::create_root(Program *prog) {
  std::cout << "SNodeRegistry create root" << std::endl;
  TI_ASSERT(prog != nullptr);
  auto n = std::make_unique<SNode>(/*depth=*/0, SNodeType::root,
                                   prog->get_snode_to_fields(),
                                   &prog->get_snode_rw_accessors_bank());
  auto *res = n.get();
  snodes_.push_back(std::move(n));
  return res;
}

std::unique_ptr<SNode> SNodeRegistry::finalize(const SNode *snode) {
  std::cout << "Finalizing SNode: " << snode->get_name() << std::endl;
  auto i = 0;
  for (auto it = snodes_.begin(); it != snodes_.end(); ++it) {
    std::cout << "SNodeRegistry snodes_[" << i << "] = " << (*it)->get_name()
              << std::endl;
    if (it->get() == snode) {
      std::cout << "SNodeRegistry found the node" << std::endl;
      auto res = std::move(*it);
      snodes_.erase(it);
      return res;
    }
    i += 1;
  }
  return nullptr;
}

}  // namespace taichi::lang
