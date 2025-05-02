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

TEST(ControlFlowGraph, Basic) {
  IRBuilder builder;
  auto *tmp1 = builder.get_bool(true);
  builder.create_assert(tmp1, "assertion failed");

  TestProgram test_prog;
  test_prog.setup(Arch::x64);
  Program *prog = test_prog.prog();
  prog->materialize_runtime();

  SNode *root_snode = prog->get_snode_root(0);
  std::vector<Stmt *> indices;
  auto *tmp3 = builder.create_global_ptr(root_snode, indices);

  auto *tmp4 = builder.get_float64(1.23f);
  auto *tmp5 = builder.get_float64(2.34f);
  auto *tmp6 = builder.get_float64(3.45f);

  auto *tmp7 = builder.create_matrix_init({tmp4, tmp5, tmp6});
  builder.create_global_store(tmp3, tmp7);
  builder.get_int32(8);
  auto *tmp9 = builder.get_bool(true);
  builder.create_assert(tmp9, "assertion failed");

  auto ir = builder.extract_ir();
  std::string ir_string;
  irpass::print(ir->get_ir_root(), &ir_string);
  std::cout << ir_string << std::endl;
}

TEST(ControlFlowGraph, BasicV2) {
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
}
}  // namespace taichi::lang
