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

TEST(ControlFlowGraph, reaching_definition_analysis_basic1) {
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
  for (auto i = 0; i < cfg->size(); i++) {
    std::cout << "reach_in for node " << i << ":" << std::endl;
    for (const auto &stmt : cfg->nodes[i]->reach_in) {
      std::cout << stmt->name() << std::endl;
    }
    std::cout << "reach_gen for node " << i << ":" << std::endl;
    for (const auto &stmt : cfg->nodes[i]->reach_gen) {
      std::cout << stmt->name() << std::endl;
    }
    std::cout << "reach_kill for node " << i << ":" << std::endl;
    for (const auto &stmt : cfg->nodes[i]->reach_kill) {
      std::cout << stmt->name() << std::endl;
    }
    std::cout << "reach_out for node " << i << ":" << std::endl;
    for (const auto &stmt : cfg->nodes[i]->reach_out) {
      std::cout << stmt->name() << std::endl;
    }
  }
  std::cout << "reach_gen for node 0:" << std::endl;
  for (const auto &stmt : cfg->nodes[0]->reach_gen) {
    std::cout << stmt->name() << std::endl;
  }
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
}  // namespace taichi::lang
