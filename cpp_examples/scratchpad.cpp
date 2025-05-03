#include <memory>

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

int main() {
  using namespace taichi;
  using namespace lang;
  auto program = Program(host_arch());
  program.get_program_impl()->config->opt_level = 0;
  program.get_program_impl()->config->external_optimization_level = 0;
  program.get_program_impl()->config->advanced_optimization = false;
  program.get_program_impl()->config->print_ir = true;
  const auto &config = program.compile_config();

  std::unique_ptr<Kernel> kernel_ret;

  auto block = std::make_unique<Block>();
  auto const_2 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 2));
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 555));
  auto arg0LoadStmt = block->push_back<ArgLoadStmt>(
      ArgLoadStmt({0},
                  TypeFactory::get_instance().get_ndarray_struct_type(
                      get_data_type<int>(), 1),
                  /*is_ptr=*/true,
                  /*create_load=*/false,
                  /*arg_depth=*/0));

  auto extptr = std::unique_ptr<ExternalPtrStmt>(
      new ExternalPtrStmt(arg0LoadStmt, {const_2}, 1, {}, false));
  auto arg0Ptr = block->insert(std::move(extptr));
  auto globalStore0 =
      std::unique_ptr<GlobalStoreStmt>(new GlobalStoreStmt(arg0Ptr, const_123));
  block->insert(std::move(globalStore0));

  kernel_ret = std::make_unique<Kernel>(program, block.release(), "ret");
  kernel_ret->insert_ndarray_param(get_data_type<int>(), /*total_dim=*/1);
  kernel_ret->finalize_params();

  auto ctx_ret = kernel_ret->make_launch_context();
  const auto &compiled_kernel_data =
      program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
  program.materialize_runtime();

  const int size = 10;
  auto array = std::make_unique<int[]>(size);

  ctx_ret.set_arg_external_array_with_shape(
      /*arg_id=*/{0}, (uint64)array.get(), size, {size});

  program.launch_kernel(compiled_kernel_data, ctx_ret);
  for (int i = 0; i < size; i++) {
    std::cout << "array[" << i << "] = " << array[i] << std::endl;
  }
}
