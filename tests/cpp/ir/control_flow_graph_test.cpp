#include "gtest/gtest.h"

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/statements.h"
#include "tests/cpp/program/test_program.h"
#include "tests/cpp/ir/ndarray_kernel.h"
#include "taichi/ir/snode.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

namespace taichi::lang {

std::unique_ptr<Block> create_8675_scenario() {
  auto block = std::make_unique<Block>();
  auto const_true =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::u1, true));
  block->push_back<AssertStmt>(const_true, std::string("assertion failed"),
                               std::vector<Stmt *>());

  TestProgram test_prog;
  test_prog.setup(Arch::x64);
  Program *prog = test_prog.prog();
  prog->materialize_runtime();

  auto root_snode = prog->get_snode_root(0);
  auto global_ptr =
      block->push_back<GlobalPtrStmt>(root_snode, std::vector<Stmt *>(), true);

  auto const_1_23 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f64, 1.23f));
  auto const_2_34 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f64, 2.34f));
  auto const_3_45 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f64, 3.45f));
  std::vector<Stmt *> matrix_elements = {const_1_23, const_2_34, const_3_45};
  auto matrix_init = block->push_back<MatrixInitStmt>(matrix_elements);
  block->push_back<GlobalStoreStmt>(global_ptr, matrix_init);
  auto const_8 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 8));
  auto const_true2 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::u1, true));
  block->push_back<AssertStmt>(const_true2, std::string("assertion failed"),
                               std::vector<Stmt *>());
  auto global_ptr2 =
      block->push_back<GlobalPtrStmt>(root_snode, std::vector<Stmt *>(), false);
  auto shift_ptr = block->push_back<MatrixPtrStmt>(global_ptr2, const_8);
  auto global_load = block->push_back<GlobalLoadStmt>(shift_ptr);
  auto cast_to_f32 = static_cast<UnaryOpStmt *>(
      block->push_back<UnaryOpStmt>(UnaryOpType::cast_value, global_load));
  cast_to_f32->cast_type = PrimitiveType::f32;
  block->push_back<ReturnStmt>(cast_to_f32);
  return block;
}

TEST(ControlFlowGraph, Basic) {
  /*
  Original code we are trying to generate:
  <u1> $1 = const true
  2 : assert $1, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of
  size
  () with indices ()
  "
  <*[Tensor (3) f64]> $3 = global ptr [S2place<f64>], index [] activate=true
  <f64> $4 = const 1.2300000190734863
  <f64> $5 = const 2.3399999141693115
  <f64> $6 = const 3.450000047683716
  <[Tensor (3) f64]> $7 = [$4, $5, $6]
  $8 : global store [$3 <- $7]
  <i32> $9 = const 8
  <u1> $10 = const true
  11 : assert $10, "(kernel=my_kernel_c80_0) Accessing field (S2place<f64>) of
  size () with indices ()
  "
  <*f64> $12 = global ptr [S2place<f64>], index [] activate=false
  <*f64> $13 = shift ptr [$12 + $9]
  <f64> $14 = global load $13
  <f32> $15 = cast_value<f32> $14
  $16 : return tmp15

  (except we start at $0)
  */

  auto block = std::make_unique<Block>();
  auto const_true =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::u1, true));
  block->push_back<AssertStmt>(const_true, std::string("assertion failed"),
                               std::vector<Stmt *>());

  TestProgram test_prog;
  test_prog.setup(Arch::x64);
  Program *prog = test_prog.prog();
  prog->materialize_runtime();

  auto root_snode = prog->get_snode_root(0);
  auto global_ptr =
      block->push_back<GlobalPtrStmt>(root_snode, std::vector<Stmt *>(), true);

  auto const_1_23 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f64, 1.23f));
  auto const_2_34 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f64, 2.34f));
  auto const_3_45 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f64, 3.45f));
  std::vector<Stmt *> matrix_elements = {const_1_23, const_2_34, const_3_45};
  auto matrix_init = block->push_back<MatrixInitStmt>(matrix_elements);
  block->push_back<GlobalStoreStmt>(global_ptr, matrix_init);
  auto const_8 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 8));
  auto const_true2 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::u1, true));
  block->push_back<AssertStmt>(const_true2, std::string("assertion failed"),
                               std::vector<Stmt *>());
  auto global_ptr2 =
      block->push_back<GlobalPtrStmt>(root_snode, std::vector<Stmt *>(), false);
  auto shift_ptr = block->push_back<MatrixPtrStmt>(global_ptr2, const_8);
  auto global_load = block->push_back<GlobalLoadStmt>(shift_ptr);
  auto cast_to_f32 = static_cast<UnaryOpStmt *>(
      block->push_back<UnaryOpStmt>(UnaryOpType::cast_value, global_load));
  cast_to_f32->cast_type = PrimitiveType::f32;
  block->push_back<ReturnStmt>(cast_to_f32);

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  /*
  Control Flow Graph with 3 nodes:
  Node 0 : empty; next={1}
  Node 1 : $0~$15 (size=16); prev={0}; next={2}
  Node 2 : empty; prev={1}
  */
  cfg->store_to_load_forwarding(false, false);
  cfg->print_graph_structure();

  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
}

void dump_reach_definition(ControlFlowGraph *cfg) {
  for (auto i = 0; i < cfg->size(); i++) {
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "  reach_in:";
    for (const auto &stmt : cfg->nodes[i]->reach_in) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
    std::cout << "  reach_gen:";
    for (const auto &stmt : cfg->nodes[i]->reach_gen) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
    std::cout << "  reach_kill:";
    for (const auto &stmt : cfg->nodes[i]->reach_kill) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
    std::cout << "  reach_out:";
    for (const auto &stmt : cfg->nodes[i]->reach_out) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
  }
}

void dump_live_definition(ControlFlowGraph *cfg) {
  for (auto i = 0; i < cfg->size(); i++) {
    std::cout << "Node " << i << ":" << std::endl;
    std::cout << "  live_gen:";
    for (const auto &stmt : cfg->nodes[i]->live_gen) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
    std::cout << "  live_kill:";
    for (const auto &stmt : cfg->nodes[i]->live_kill) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
    std::cout << "  live_in:";
    for (const auto &stmt : cfg->nodes[i]->live_in) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
    std::cout << "  live_out:";
    for (const auto &stmt : cfg->nodes[i]->live_out) {
      std::cout << " " << stmt->name();
    }
    std::cout << std::endl;
  }
}

TEST(ControlFlowGraph, reaching_definition_analysis_basic1_a) {
  auto block = std::make_unique<Block>();
  block->push_back<AllocaStmt>(PrimitiveType::i32);

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  cfg->reaching_definition_analysis(false);
  dump_reach_definition(cfg.get());
  cfg->print_graph_structure();
}

TEST(ControlFlowGraph, reaching_definition_analysis_basic1_b) {
  auto block = std::make_unique<Block>();
  block->push_back<AllocaStmt>(PrimitiveType::i32);
  block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  cfg->reaching_definition_analysis(false);
  dump_reach_definition(cfg.get());
  cfg->print_graph_structure();
}

TEST(ControlFlowGraph, reaching_definition_analysis_basic1_c) {
  auto block = std::make_unique<Block>();
  auto var_a = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  block->push_back<LocalStoreStmt>(var_a, const_123);

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  cfg->reaching_definition_analysis(false);
  dump_reach_definition(cfg.get());
  cfg->print_graph_structure();
}

TEST(ControlFlowGraph, reaching_definition_analysis_8675) {
  auto block = create_8675_scenario();

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  cfg->reaching_definition_analysis(false);
  cfg->print_graph_structure();
}

Block *addCfgIfNode(Block *block) {
  auto const_true =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::u1, true));
  auto if_stmt = static_cast<IfStmt *>(block->push_back<IfStmt>(const_true));
  if_stmt->true_statements = std::make_unique<Block>();
  auto true_block = if_stmt->true_statements.get();
  return true_block;
}

CFGNode *find_node(ControlFlowGraph *cfg, Stmt *stmt) {
  for (auto i = 0; i < cfg->size(); i++) {
    auto node = cfg->nodes[i].get();
    if (!node->empty() && node->block->statements.size() > 0 &&
        node->block->statements[0].get() == stmt) {
      return node;
    }
  }
  return nullptr;
}

TEST(ControlFlowGraph, live_variable_analysis_gen_kill_101) {
  auto block = std::make_unique<Block>();
  // this causes a kill; I'm not sure on what basis 🤔 There is no assignment
  // here
  auto var_a = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));

  Stmt *block_a_first_stmt;
  {  // block a
    auto if_block = addCfgIfNode(block.get());
    block_a_first_stmt = if_block->push_back<LocalLoadStmt>(
        var_a);  // should cause a gen ("used (load) before any assignment, in
                 // same basic block")
  }

  Stmt *block_b_first_stmt;
  {  // block b
    auto if_block = addCfgIfNode(block.get());
    block_b_first_stmt = if_block->push_back<LocalStoreStmt>(
        var_a,
        const_123);  // should cause a kill ("assigned (store) in a block")
  }

  Stmt *block_c_first_stmt;
  {  // block c
    auto if_block = addCfgIfNode(block.get());
    block_c_first_stmt = if_block->push_back<LocalLoadStmt>(
        var_a);  // should cause a gen ("used (load) before any assignment, in
                 // same basic block")
    if_block->push_back<LocalStoreStmt>(
        var_a,
        const_123);  // should cause a kill ("assigned (store) in a block")
  }

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());

  cfg->print_graph_structure();
  ControlFlowGraph::LiveVarAnalysisConfig config_opt;
  cfg->live_variable_analysis(false, config_opt);
  dump_live_definition(cfg.get());
  cfg->print_graph_structure();

  auto block_a_node = find_node(cfg.get(), block_a_first_stmt);
  auto block_b_node = find_node(cfg.get(), block_b_first_stmt);
  auto block_c_node = find_node(cfg.get(), block_c_first_stmt);

  ASSERT_EQ(block_a_node->live_gen.size(), 1);
  ASSERT_EQ(*block_a_node->live_gen.begin(), var_a);
  ASSERT_EQ(block_a_node->live_kill.size(), 0);

  ASSERT_EQ(block_b_node->live_gen.size(), 0);
  ASSERT_EQ(block_b_node->live_kill.size(), 1);
  ASSERT_EQ(*block_b_node->live_kill.begin(), var_a);

  ASSERT_EQ(block_c_node->live_gen.size(), 1);
  ASSERT_EQ(*block_c_node->live_gen.begin(), var_a);
  ASSERT_EQ(block_c_node->live_kill.size(), 1);
  ASSERT_EQ(*block_c_node->live_kill.begin(), var_a);
}

TEST(ControlFlowGraph, live_variable_analysis_progressive_death) {
  auto block = std::make_unique<Block>();
  auto var_a = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto var_b = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto var_c = block->push_back<AllocaStmt>(PrimitiveType::i32);
  // auto var_d = block->push_back<AllocaStmt>(PrimitiveType::i32);
  // auto var_e = block->push_back<AllocaStmt>(PrimitiveType::i32);

  {
    auto if_block = addCfgIfNode(block.get());
    if_block->push_back<LocalLoadStmt>(var_a);
  }

  {
    auto if_block = addCfgIfNode(block.get());
    if_block->push_back<LocalLoadStmt>(var_b);
  }

  {
    auto if_block = addCfgIfNode(block.get());
    if_block->push_back<LocalLoadStmt>(var_c);
  }

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  ControlFlowGraph::LiveVarAnalysisConfig config_opt;
  cfg->live_variable_analysis(false, config_opt);
  dump_live_definition(cfg.get());
  cfg->print_graph_structure();
}

TEST(ControlFlowGraph, live_variable_analysis_basic1_c) {
  auto block = std::make_unique<Block>();
  auto var_a = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto var_c = block->push_back<AllocaStmt>(PrimitiveType::i32);
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  auto const_1 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 1));
  block->push_back<LocalStoreStmt>(var_a, const_123);
  auto const_true =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::u1, true));

  {
    auto if_stmt = static_cast<IfStmt *>(block->push_back<IfStmt>(const_true));
    {
      if_stmt->true_statements = std::make_unique<Block>();
      auto block = if_stmt->true_statements.get();
      auto load_a = block->push_back<LocalLoadStmt>(var_a);
      auto add =
          block->push_back<BinaryOpStmt>(BinaryOpType::add, load_a, const_1);
      block->push_back<LocalStoreStmt>(var_a, add);
    }
  }

  {
    auto if_stmt = static_cast<IfStmt *>(block->push_back<IfStmt>(const_true));
    {
      if_stmt->true_statements = std::make_unique<Block>();
      auto block = if_stmt->true_statements.get();
      auto var_b = block->push_back<AllocaStmt>(PrimitiveType::i32);
      block->push_back<LocalStoreStmt>(var_b, const_1);
      block->push_back<LocalStoreStmt>(var_c, const_1);
    }
  }

  std::string ir_string;
  irpass::print(block->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;

  auto cfg = irpass::analysis::build_cfg(block.get());
  cfg->print_graph_structure();
  ControlFlowGraph::LiveVarAnalysisConfig config_opt;
  cfg->live_variable_analysis(false, config_opt);
  dump_live_definition(cfg.get());
  cfg->print_graph_structure();
}

}  // namespace taichi::lang
