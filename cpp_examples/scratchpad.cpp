#include <memory>

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

using namespace taichi;
using namespace lang;
void writeResult(Block *block, int idx, Stmt *value) {
  auto arg0LoadStmt = block->push_back<ArgLoadStmt>(
      ArgLoadStmt({0},
                  TypeFactory::get_instance().get_ndarray_struct_type(
                      get_data_type<float>(), 1),
                  /*is_ptr=*/true,
                  /*create_load=*/false,
                  /*arg_depth=*/0));
  auto idx_stmt =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, idx));
  auto extptr = std::unique_ptr<ExternalPtrStmt>(
      new ExternalPtrStmt(arg0LoadStmt, {idx_stmt}, 1, {}, false));
  auto arg0Ptr = block->insert(std::move(extptr));
  auto globalStore0 =
      std::unique_ptr<GlobalStoreStmt>(new GlobalStoreStmt(arg0Ptr, value));
  block->insert(std::move(globalStore0));
}

void writeIR(Block *block) {
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f32, 1.23));
  auto const_555 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f32, 5.55));

  auto const_1_23 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f32, 1.23f));
  auto const_2_34 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f32, 2.34f));
  auto const_3_45 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::f32, 3.45f));

  std::vector<Stmt *> matrix_elements = {const_1_23, const_2_34, const_3_45};
  auto matrixInit = block->push_back<MatrixInitStmt>(matrix_elements);
  auto &type_factory = TypeFactory::get_instance();
  Type *tensor3 = type_factory.get_tensor_type(
      {3}, type_factory.get_primitive_type(PrimitiveTypeID::f32));
  matrixInit->ret_type = tensor3;

  writeResult(block, 1, const_123);
  writeResult(block, 4, const_555);
}

int main() {
  auto program = Program(host_arch());
  program.get_program_impl()->config->opt_level = 0;
  program.get_program_impl()->config->external_optimization_level = 0;
  program.get_program_impl()->config->advanced_optimization = false;
  program.get_program_impl()->config->print_ir = true;
  const auto &config = program.compile_config();

  std::unique_ptr<Kernel> kernel_ret;

  auto block = std::make_unique<Block>();
  writeIR(block.get());

  kernel_ret = std::make_unique<Kernel>(program, block.release(), "ret");
  kernel_ret->insert_ndarray_param(get_data_type<float>(), /*total_dim=*/1);
  kernel_ret->finalize_params();

  auto ctx_ret = kernel_ret->make_launch_context();
  const auto &compiled_kernel_data =
      program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
  program.materialize_runtime();

  const int size = 10;
  auto array = std::make_unique<float[]>(size);

  ctx_ret.set_arg_external_array_with_shape(
      /*arg_id=*/{0}, (uint64)array.get(), size, {size});

  program.launch_kernel(compiled_kernel_data, ctx_ret);
  for (int i = 0; i < size; i++) {
    std::cout << "array[" << i << "] = " << array[i] << std::endl;
  }
}
