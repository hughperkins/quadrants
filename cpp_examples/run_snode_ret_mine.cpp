#include <memory>

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

std::unique_ptr<taichi::lang::IRNode> get_ir_unique_ptr(
    taichi::lang::Block *block) {
  auto new_block = std::make_unique<taichi::lang::Block>();
  for (auto &stmt : block->statements) {
    auto stmt_clone = stmt->clone();
    new_block->insert(std::move(stmt_clone));
  }

  // Return the new block as an IRNode
  return new_block;
}

int main() {
  using namespace taichi;
  using namespace lang;
  auto program = Program(host_arch());
  program.get_program_impl()->config->opt_level = 0;
  program.get_program_impl()->config->advanced_optimization = false;
  program.get_program_impl()->config->print_ir = true;
  const auto &config = program.compile_config();

  int n = 10;
  program.materialize_runtime();
  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->pointer(Axis(0), n);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/false);

  std::unique_ptr<Kernel> kernel_ret;

  {
    auto block = std::make_unique<Block>();
    auto const_type = PrimitiveType::i32;
    auto const_123 =
        block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
    auto ret_stmt =
        static_cast<ReturnStmt *>(block->push_back<ReturnStmt>(const_123));
    ret_stmt->ret_type = PrimitiveType::i32;

    std::cout << "ret values " << ret_stmt->values.size() << std::endl;
    std::cout << "ret values " << ret_stmt->values.size() << " "
              << ret_stmt->values[0]->name() << std::endl;
    kernel_ret = std::make_unique<Kernel>(program, block.release(), "ret");
  }

  auto ctx_ret = kernel_ret->make_launch_context();
  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
    program.launch_kernel(compiled_kernel_data, ctx_ret);
    std::cout << "res " << program.fetch_result<int>(0) << std::endl;
  }
}
