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
namespace irpass::analysis {
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

  auto load_a = block->push_back<LocalLoadStmt>(var_a);
  auto const_111 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 111));
  auto cmp1 =
      block->push_back<BinaryOpStmt>(BinaryOpType::cmp_gt, load_a, const_111);
  auto if_stmt = static_cast<IfStmt *>(block->push_back<IfStmt>(cmp1));
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

  auto cfg = build_cfg(block.get());
  cfg->print_graph_structure();
  /*
  Control Flow Graph with 5 nodes:
  Node 0 : empty; next={1}
  Node 1 : $0~$7 (size=8); prev={0}; next={2, 3}
  Node 2 : $9~$11 (size=3); prev={1}; next={4}
  Node 3 : $12~$14 (size=3); prev={1}; next={4}
  Node 4 : empty; prev={2, 3}
  */
  EXPECT_EQ(cfg->size(), 5);
  EXPECT_TRUE(cfg->nodes[0]->empty());
  EXPECT_EQ(cfg->nodes[1]->size(), 8);
  EXPECT_EQ(cfg->nodes[2]->size(), 3);
  EXPECT_EQ(cfg->nodes[3]->size(), 3);
  EXPECT_TRUE(cfg->nodes[4]->empty());
  EXPECT_EQ(cfg->final_node, cfg->size() - 1);
}
}  // namespace irpass::analysis
}  // namespace taichi::lang
