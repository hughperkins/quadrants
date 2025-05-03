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
  auto const_type = PrimitiveType::i32;
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  block->push_back<ReturnStmt>(const_123);

  kernel_ret = std::make_unique<Kernel>(program, block.release(), "ret");
  kernel_ret->insert_ret(PrimitiveType::i32);
  kernel_ret->finalize_rets();

  auto ctx_ret = kernel_ret->make_launch_context();
  const auto &compiled_kernel_data =
      program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
  program.materialize_runtime();
  program.launch_kernel(compiled_kernel_data, ctx_ret);
  std::cout << "res " << program.fetch_result<int>(0) << std::endl;
}
