#include <memory>

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/llvm/kernel_compiler.h"
// #include "taichi/codegen/codegen.h"
#include "taichi/ir/analysis.h"

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

std::unique_ptr<IRNode> minimal_prepare_ir(const Kernel &kernel,
                                           const CompileConfig &config) {
  auto ir = irpass::analysis::clone(kernel.ir.get());

  // Create offloaded task structure
  auto root = ir.get();
  if (root->is<Block>()) {
    auto block = root->as<Block>();
    auto offloaded = std::make_unique<OffloadedStmt>(
        OffloadedStmt::TaskType::serial, config.arch,
        const_cast<Kernel *>(&kernel));
    offloaded->body = std::make_unique<Block>();

    // Move all statements to the offloaded block
    for (int i = 0; i < block->size(); i++) {
      offloaded->body->insert(std::move(block->statements[i]));
    }
    block->statements.clear();
    block->insert(std::move(offloaded));
  }

  // Set return types for all statements to avoid crashes
  irpass::analysis::gather_statements(ir.get(), [](Stmt *stmt) {
    if (stmt->ret_type == DataType()) {
      if (stmt->is<ConstStmt>()) {
        stmt->ret_type = stmt->as<ConstStmt>()->val.dt;
      } else if (stmt->is<UnaryOpStmt>()) {
        auto unary = stmt->as<UnaryOpStmt>();
        if (unary->op_type == UnaryOpType::cast_value) {
          stmt->ret_type = unary->cast_type;
        }
      }
    }
    return false;
  });

  return ir;
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
  writeResult(block, 5, matrixInit);

  auto cast1 = std::unique_ptr<UnaryOpStmt>(
      new UnaryOpStmt(UnaryOpType::cast_value, const_555));
  cast1->cast_type = type_factory.get_primitive_type(PrimitiveTypeID::i32);
  cast1->ret_type = cast1->cast_type;
  auto cast1b = block->insert(std::move(cast1));
  writeResult(block, 9, cast1b);

  auto cast2 = std::unique_ptr<UnaryOpStmt>(
      new UnaryOpStmt(UnaryOpType::cast_value, matrixInit));
  cast2->cast_type = type_factory.get_primitive_type(PrimitiveTypeID::f32);
  cast2->ret_type = cast2->cast_type;
  auto cast2b = block->insert(std::move(cast2));
  writeResult(block, 10, cast2b);
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

  auto program_impl = program.get_program_impl();
  auto llvm_program = static_cast<LlvmProgramImpl *>(program_impl);
  auto tlctx = llvm_program->get_llvm_context();

  LLVM::KernelCompiler::Config compiler_config;
  compiler_config.tlctx = tlctx;
  LLVM::KernelCompiler kernel_compiler(compiler_config);

  auto prepared_ir = minimal_prepare_ir(*kernel_ret, program.compile_config());

  auto compiled_kernel_data = kernel_compiler.compile(
      config, program.get_device_caps(), *kernel_ret, *prepared_ir);

  //   const auto &compiled_kernel_data =
  //       program.compile_kernel(config, program.get_device_caps(),
  //       *kernel_ret);
  program.materialize_runtime();

  const int size = 20;
  auto array = std::make_unique<float[]>(size);

  ctx_ret.set_arg_external_array_with_shape(
      /*arg_id=*/{0}, (uint64)array.get(), size, {size});

  program.launch_kernel(*compiled_kernel_data, ctx_ret);
  for (int i = 0; i < size; i++) {
    std::cout << "array[" << i << "] = " << array[i] << std::endl;
  }
}
