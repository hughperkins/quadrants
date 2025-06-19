#include "taichi/codegen/codegen.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"

#include "taichi/codegen/llvm/kernel_compiler.h"
#include "taichi/codegen/llvm/compiled_kernel_data.h"

#if defined(TI_WITH_CUDA)
#include "taichi/codegen/cuda/codegen_cuda.h"
#endif

namespace taichi::lang {
namespace LLVM {

KernelCompiler::KernelCompiler(Config config) : config_(std::move(config)) {
}

KernelCompiler::IRNodePtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const Kernel &kernel_def) const {
  std::cout << "KernelCompiler::compile(CompileConfig, Kernel) " << std::endl;
  auto ir = irpass::analysis::clone(kernel_def.ir.get());
  bool verbose = compile_config.print_ir;
  if (kernel_def.is_accessor && !compile_config.print_accessor_ir) {
    verbose = false;
  }
  irpass::compile_to_offloads(ir.get(), compile_config, &kernel_def,
                              /*verbose=*/verbose,
                              /*autodiff_mode=*/kernel_def.autodiff_mode,
                              /*ad_use_stack=*/true,
                              /*start_from_ast=*/kernel_def.ir_is_ast());
  return ir;
}

KernelCompiler::CKDPtr KernelCompiler::compile(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def,
    IRNode &chi_ir) const {
  std::cout << "KernelCompiler::compile(CompileConfig, DeviceCapabilityConfig, Kernel) " << std::endl;
  LLVM::CompiledKernelData::InternalData data;
  auto codegen = KernelCodeGen::create(compile_config, &kernel_def, &chi_ir,
                                       *config_.tlctx);
  data.compiled_data = codegen->compile_kernel_to_module();
  data.args.reserve(kernel_def.nested_parameters.size());
  for (const auto &p : kernel_def.nested_parameters)
    data.args.push_back(p);
  data.rets = kernel_def.rets;
  data.args_type = kernel_def.args_type;
  data.args_size = kernel_def.args_size;
  data.ret_type = kernel_def.ret_type;
  data.ret_size = kernel_def.ret_size;
  return std::make_unique<LLVM::CompiledKernelData>(compile_config.arch, data);
}

KernelCompiler::CKDPtr KernelCompiler::compile_with_struct_ptx(
    const CompileConfig &compile_config,
    const DeviceCapabilityConfig &device_caps,
    const Kernel &kernel_def,
    IRNode &chi_ir,
    const std::string &struct_ptx) const {
  std::cout << "KernelCompiler::compile_with_struct_ptx " << std::endl;
  LLVM::CompiledKernelData::InternalData data;
  auto codegen = KernelCodeGen::create(compile_config, &kernel_def, &chi_ir,
                                       *config_.tlctx);
  
  // For CUDA, use the struct PTX during compilation
  if (compile_config.arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    // Get the CUDA codegen and inject struct PTX
    auto cuda_codegen = dynamic_cast<KernelCodeGenCUDA*>(codegen.get());
    if (cuda_codegen) {
      // This would require modifying the CUDA codegen to accept struct PTX
      // For now, we'll use the regular compilation
      data.compiled_data = codegen->compile_kernel_to_module();
    } else {
      data.compiled_data = codegen->compile_kernel_to_module();
    }
#else
    data.compiled_data = codegen->compile_kernel_to_module();
#endif
  } else {
    data.compiled_data = codegen->compile_kernel_to_module();
  }
  
  data.args.reserve(kernel_def.nested_parameters.size());
  for (const auto &p : kernel_def.nested_parameters)
    data.args.push_back(p);
  data.rets = kernel_def.rets;
  data.args_type = kernel_def.args_type;
  data.args_size = kernel_def.args_size;
  data.ret_type = kernel_def.ret_type;
  data.ret_size = kernel_def.ret_size;
  return std::make_unique<LLVM::CompiledKernelData>(compile_config.arch, data);
}

}  // namespace LLVM
}  // namespace taichi::lang
