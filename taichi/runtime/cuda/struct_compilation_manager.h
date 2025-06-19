#pragma once

#include <memory>
#include <unordered_map>
#include <string>
#include <mutex>

#include "taichi/ir/snode.h"
#include "taichi/runtime/cuda/jit_cuda.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/program/compile_config.h"

namespace taichi::lang {

struct StructPTXCache {
  std::string ptx_code;
  std::time_t created_at;
  std::time_t last_used_at;
  size_t size;
  
  StructPTXCache() : created_at(0), last_used_at(0), size(0) {}
};

class StructCompilationManager {
 public:
  explicit StructCompilationManager(TaichiLLVMContext *tlctx,
                                   const CompileConfig &config);

  // Compile a struct to PTX and cache it
  std::string compile_struct_to_ptx(SNode *root);

  // Get cached PTX for a struct
  std::string get_cached_struct_ptx(SNode *root);

  // Get or compile struct PTX (for integration with kernel compilation)
  std::string get_or_compile_struct_ptx(const CompileConfig &config,
                                        const DeviceCapabilityConfig &device_caps);

  // Check if struct is cached
  bool is_struct_cached(SNode *root) const;

  // Clear cache
  void clear_cache();

  // Get cache statistics
  size_t get_cache_size() const { return struct_cache_.size(); }

  // Get cache key for the current struct (for integration)
  std::string get_cache_key() const;

  // Field access function creation methods
  void create_field_access_functions(llvm::Module *module, SNode *root);
  void create_linearize_function(llvm::Module *module, SNode *root);
  void create_field_access_function(llvm::Module *module, SNode *root);
  void create_child_access_function(llvm::Module *module, SNode *root);
  void create_function_pointer_table(llvm::Module *module);

 private:
  std::string make_struct_key(SNode *root) const;
  std::unique_ptr<llvm::Module> create_struct_module(SNode *root);
  
  TaichiLLVMContext *tlctx_;
  const CompileConfig &config_;
  std::unique_ptr<JITSessionCUDA> jit_session_;
  mutable std::mutex cache_mutex_;
  std::unordered_map<std::string, StructPTXCache> struct_cache_;
};

}  // namespace taichi::lang 