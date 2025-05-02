#include <algorithm>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/ir_builder.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {
namespace analysis {
TEST(BuildCfg, Basic1) {
  auto block = std::make_unique<Block>();
  auto var_a = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  block->push_back<LocalStoreStmt>(var_a, const_123);

  auto const_1 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 1));
  auto const_2 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 2));
  // auto const_3 =
  // block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 3));

  auto load_a = block->push_back<LocalLoadStmt>(var_a);
  auto const_111 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 111));
  auto cmp1 =
      block->push_back<BinaryOpStmt>(BinaryOpType::cmp_gt, load_a, const_111);
  auto if_stmt = static_cast<IfStmt *>(block->push_back<IfStmt>(cmp1));
  // auto if_true_block = std::make_unique<Block>();
  // auto if_false_block = std::make_unique<Block>();
  {
    if_stmt->true_statements = std::make_unique<Block>();
    auto block = if_stmt->true_statements.get();
    auto load_a = block->push_back<LocalLoadStmt>(var_a);
    auto add =
        block->push_back<BinaryOpStmt>(BinaryOpType::add, load_a, const_1);
    block->push_back<LocalStoreStmt>(var_a, add);
  }
  {
    if_stmt->false_statements = std::make_unique<Block>();
    auto block = if_stmt->false_statements.get();
    auto load_a = block->push_back<LocalLoadStmt>(var_a);
    auto add =
        block->push_back<BinaryOpStmt>(BinaryOpType::sub, load_a, const_2);
    block->push_back<LocalStoreStmt>(var_a, add);
  }
  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;
}
}  // namespace analysis
}  // namespace taichi::lang
