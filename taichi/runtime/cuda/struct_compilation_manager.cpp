#include "taichi/runtime/cuda/struct_compilation_manager.h"
#include "taichi/codegen/llvm/struct_llvm.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/common/core.h"
#include "taichi/util/lang_util.h"
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

std::string StructCompilationManager::make_struct_key(SNode *root) {
  // Create a unique key for the struct based on its structure
  std::stringstream ss;
  ss << "struct_" << root->id << "_" << root->type;
  
  // Add child information
  for (auto &child : root->ch) {
    ss << "_" << child->id << "_" << child->type;
  }
  
  return ss.str();
}

std::unique_ptr<llvm::Module> StructCompilationManager::create_struct_module(SNode *root) {
  // Create a new LLVM module for the struct
  auto module = tlctx_->new_module("struct_ptx");
  
  // Use the existing struct compiler to generate the LLVM module
  // This reuses the existing struct compilation infrastructure
  auto struct_compiler = std::make_unique<StructCompilerLLVM>(
      Arch::cuda, nullptr, std::move(module), 0);
  
  // Compile the struct
  struct_compiler->run(*root);
  
  // Return the compiled module
  return struct_compiler->get_module();
}

}  // namespace taichi::lang 