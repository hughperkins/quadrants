#include "taichi/runtime/cuda/struct_compilation_manager.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/common/core.h"
#include "taichi/util/lang_util.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <sstream>
#include <ctime>

namespace taichi::lang {

StructCompilationManager::StructCompilationManager(TaichiLLVMContext *tlctx,
                                                   const CompileConfig &config)
    : tlctx_(tlctx), config_(config) {
  // Initialize JIT session for CUDA
  auto data_layout = TaichiLLVMContext::get_data_layout(Arch::cuda);
  jit_session_ = std::make_unique<JITSessionCUDA>(tlctx, config, data_layout);
}

std::string StructCompilationManager::compile_struct_to_ptx(SNode *root) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  std::string struct_key = make_struct_key(root);
  
  // Check if already cached
  auto it = struct_cache_.find(struct_key);
  if (it != struct_cache_.end()) {
    it->second.last_used_at = std::time(nullptr);
    return it->second.ptx_code;
  }
  
  // Create struct module and compile to PTX
  auto struct_module = create_struct_module(root);
  std::string ptx_code = jit_session_->compile_struct_to_ptx(struct_module);
  
  // Cache the result
  StructPTXCache cache_entry;
  cache_entry.ptx_code = ptx_code;
  cache_entry.created_at = std::time(nullptr);
  cache_entry.last_used_at = cache_entry.created_at;
  cache_entry.size = ptx_code.size();
  
  struct_cache_[struct_key] = std::move(cache_entry);
  
  TI_DEBUG("Compiled struct to PTX, key: {}, size: {} bytes", 
           struct_key, ptx_code.size());
  
  return ptx_code;
}

std::string StructCompilationManager::get_cached_struct_ptx(SNode *root) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  
  std::string struct_key = make_struct_key(root);
  auto it = struct_cache_.find(struct_key);
  
  if (it != struct_cache_.end()) {
    it->second.last_used_at = std::time(nullptr);
    return it->second.ptx_code;
  }
  
  return "";
}

bool StructCompilationManager::is_struct_cached(SNode *root) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  std::string struct_key = make_struct_key(root);
  return struct_cache_.find(struct_key) != struct_cache_.end();
}

void StructCompilationManager::clear_cache() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  struct_cache_.clear();
}

std::string StructCompilationManager::make_struct_key(SNode *root) const {
  // Create a unique key for the struct based on its structure
  std::stringstream ss;
  ss << "struct_" << root->id << "_" << static_cast<int>(root->type);
  
  // Add more structural information for uniqueness
  for (auto &child : root->ch) {
    if (child) {
      ss << "_" << child->id << "_" << static_cast<int>(child->type);
    }
  }
  
  return ss.str();
}

std::unique_ptr<llvm::Module> StructCompilationManager::create_struct_module(SNode *root) {
  // Create a new LLVM module for the struct with proper target triple
  auto module = tlctx_->new_module("struct_ptx");
  module->setTargetTriple("nvptx64-nvidia-cuda");
  
  // Create field access functions that kernels can call
  create_field_access_functions(module.get(), root);
  
  return module;
}

void StructCompilationManager::create_field_access_functions(llvm::Module *module, SNode *root) {
  // Create individual functions first
  create_linearize_function(module, root);
  create_field_access_function(module, root);
  create_child_access_function(module, root);
  
  // Create function pointer table for efficient access
  create_function_pointer_table(module);
}

void StructCompilationManager::create_linearize_function(llvm::Module *module, SNode *root) {
  auto &ctx = module->getContext();
  
  // Function signature: int linearize_2d_coords(int i, int j, int stride)
  auto int32_type = llvm::Type::getInt32Ty(ctx);
  
  std::vector<llvm::Type*> param_types = {int32_type, int32_type, int32_type};
  auto func_type = llvm::FunctionType::get(int32_type, param_types, false);
  
  auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, 
                                   "linearize_2d_coords", module);
  
  // Create basic block
  auto bb = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> builder(bb);
  
  // Get function parameters
  auto i = func->getArg(0);
  auto j = func->getArg(1);
  auto stride = func->getArg(2);
  
  // Calculate linear index: i * stride + j
  auto i_times_stride = builder.CreateMul(i, stride);
  auto linear_index = builder.CreateAdd(i_times_stride, j);
  
  builder.CreateRet(linear_index);
}

void StructCompilationManager::create_field_access_function(llvm::Module *module, SNode *root) {
  auto &ctx = module->getContext();
  
  // Function signature: void* get_field_ptr(void* root, int i, int j, int stride)
  auto void_ptr_type = llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0);
  auto int32_type = llvm::Type::getInt32Ty(ctx);
  
  std::vector<llvm::Type*> param_types = {void_ptr_type, int32_type, int32_type, int32_type};
  auto func_type = llvm::FunctionType::get(void_ptr_type, param_types, false);
  
  auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, 
                                   "get_field_ptr_2d", module);
  
  // Create basic block
  auto bb = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> builder(bb);
  
  // Get function parameters
  auto root_ptr = func->getArg(0);
  auto i = func->getArg(1);
  auto j = func->getArg(2);
  auto stride = func->getArg(3);
  
  // Calculate offset: (i * stride + j) * sizeof(int32)
  auto i_times_stride = builder.CreateMul(i, stride);
  auto linear_index = builder.CreateAdd(i_times_stride, j);
  auto offset_bytes = builder.CreateMul(linear_index, llvm::ConstantInt::get(int32_type, 4));
  
  // Cast to int64 for pointer arithmetic
  auto offset_64 = builder.CreateSExt(offset_bytes, llvm::Type::getInt64Ty(ctx));
  
  // Calculate final pointer: root_ptr + offset
  auto final_ptr = builder.CreateGEP(llvm::Type::getInt8Ty(ctx), root_ptr, offset_64);
  
  builder.CreateRet(final_ptr);
}

void StructCompilationManager::create_child_access_function(llvm::Module *module, SNode *root) {
  auto &ctx = module->getContext();
  
  // Function signature: void* get_child_ptr(void* parent, int child_id)
  auto void_ptr_type = llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0);
  auto int32_type = llvm::Type::getInt32Ty(ctx);
  
  std::vector<llvm::Type*> param_types = {void_ptr_type, int32_type};
  auto func_type = llvm::FunctionType::get(void_ptr_type, param_types, false);
  
  auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, 
                                   "get_child_ptr", module);
  
  // Create basic block
  auto bb = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> builder(bb);
  
  // Get function parameters
  auto parent_ptr = func->getArg(0);
  // auto child_id = func->getArg(1);  // Unused for now
  
  // For now, just return the parent pointer (simplified implementation)
  // In a full implementation, this would calculate the proper child offset
  builder.CreateRet(parent_ptr);
}

void StructCompilationManager::create_function_pointer_table(llvm::Module *module) {
  auto &ctx = module->getContext();
  
  // Create function pointer types
  auto void_ptr_type = llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0);
  auto int32_type = llvm::Type::getInt32Ty(ctx);
  
  // Function pointer type for get_field_ptr_2d
  std::vector<llvm::Type*> field_access_params = {void_ptr_type, int32_type, int32_type, int32_type};
  auto field_access_func_type = llvm::FunctionType::get(void_ptr_type, field_access_params, false);
  auto field_access_ptr_type = llvm::PointerType::get(field_access_func_type, 0);
  
  // Function pointer type for linearize_2d_coords
  std::vector<llvm::Type*> linearize_params = {int32_type, int32_type, int32_type};
  auto linearize_func_type = llvm::FunctionType::get(int32_type, linearize_params, false);
  auto linearize_ptr_type = llvm::PointerType::get(linearize_func_type, 0);
  
  // Create struct for function pointer table
  std::vector<llvm::Type*> table_members = {field_access_ptr_type, linearize_ptr_type};
  auto table_type = llvm::StructType::create(ctx, table_members, "StructFunctionTable");
  
  // Create global constant with function pointers
  auto field_access_func = module->getFunction("get_field_ptr_2d");
  auto linearize_func = module->getFunction("linearize_2d_coords");
  
  std::vector<llvm::Constant*> table_values = {
    llvm::ConstantExpr::getBitCast(field_access_func, field_access_ptr_type),
    llvm::ConstantExpr::getBitCast(linearize_func, linearize_ptr_type)
  };
  
  auto table_constant = llvm::ConstantStruct::get(table_type, table_values);
  auto table_global = new llvm::GlobalVariable(*module, table_type, true, 
                                              llvm::GlobalValue::ExternalLinkage,
                                              table_constant, "struct_function_table");
  
  // Create function to get the function pointer table
  auto get_table_func_type = llvm::FunctionType::get(llvm::PointerType::get(table_type, 0), {}, false);
  auto get_table_func = llvm::Function::Create(get_table_func_type, llvm::Function::ExternalLinkage,
                                              "get_struct_function_table", module);
  
  auto bb = llvm::BasicBlock::Create(ctx, "entry", get_table_func);
  llvm::IRBuilder<> builder(bb);
  builder.CreateRet(table_global);
}

std::string StructCompilationManager::get_or_compile_struct_ptx(const CompileConfig &config,
                                                               const DeviceCapabilityConfig &device_caps) {
  // For now, return empty string since we need a root SNode to compile
  // In a full implementation, this would be based on the kernel's dependencies
  return "";
}

std::string StructCompilationManager::get_cache_key() const {
  // For now, return empty string since we need a root SNode to get the key
  // In a full implementation, this would return the key for the current struct
  return "";
}

}  // namespace taichi::lang 