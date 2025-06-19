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
  create_child_access_functions(module, root);
  
  // Create function pointer table for efficient access
  create_function_table(module, root);
}

void StructCompilationManager::create_linearize_function(llvm::Module *module, SNode *root) {
  auto &ctx = module->getContext();
  
  // Function signature: int linearize_coords(int* coords, int* strides, int num_dims)
  auto coord_ptr_type = llvm::PointerType::get(llvm::Type::getInt32Ty(ctx), 0);
  auto stride_ptr_type = llvm::PointerType::get(llvm::Type::getInt32Ty(ctx), 0);
  auto int32_type = llvm::Type::getInt32Ty(ctx);
  
  std::vector<llvm::Type*> param_types = {coord_ptr_type, stride_ptr_type, int32_type};
  auto func_type = llvm::FunctionType::get(int32_type, param_types, false);
  
  auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, 
                                   "linearize_coords", module);
  
  // Create basic block
  auto bb = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> builder(bb);
  
  // Get function parameters
  auto coords = func->getArg(0);
  auto strides = func->getArg(1);
  auto num_dims = func->getArg(2);
  
  // Implement linearization: result = sum(coords[i] * strides[i])
  auto result = builder.CreateAlloca(int32_type);
  builder.CreateStore(llvm::ConstantInt::get(int32_type, 0), result);
  
  // Create loop
  auto loop_header = llvm::BasicBlock::Create(ctx, "loop_header", func);
  auto loop_body = llvm::BasicBlock::Create(ctx, "loop_body", func);
  auto loop_exit = llvm::BasicBlock::Create(ctx, "loop_exit", func);
  
  builder.CreateBr(loop_header);
  builder.SetInsertPoint(loop_header);
  
  auto i = builder.CreateAlloca(int32_type);
  builder.CreateStore(llvm::ConstantInt::get(int32_type, 0), i);
  
  auto i_val = builder.CreateLoad(int32_type, i);
  auto cond = builder.CreateICmpSLT(i_val, num_dims);
  builder.CreateCondBr(cond, loop_body, loop_exit);
  
  builder.SetInsertPoint(loop_body);
  
  // Load coords[i] and strides[i]
  auto coord_gep = builder.CreateGEP(int32_type, coords, i_val);
  auto stride_gep = builder.CreateGEP(int32_type, strides, i_val);
  auto coord_val = builder.CreateLoad(int32_type, coord_gep);
  auto stride_val = builder.CreateLoad(int32_type, stride_gep);
  
  // Multiply and add to result
  auto product = builder.CreateMul(coord_val, stride_val);
  auto current_result = builder.CreateLoad(int32_type, result);
  auto new_result = builder.CreateAdd(current_result, product);
  builder.CreateStore(new_result, result);
  
  // Increment i
  auto new_i = builder.CreateAdd(i_val, llvm::ConstantInt::get(int32_type, 1));
  builder.CreateStore(new_i, i);
  builder.CreateBr(loop_header);
  
  builder.SetInsertPoint(loop_exit);
  auto final_result = builder.CreateLoad(int32_type, result);
  builder.CreateRet(final_result);
}

void StructCompilationManager::create_field_access_function(llvm::Module *module, SNode *root) {
  auto &ctx = module->getContext();
  
  // Function signature: void* get_field_ptr(void* root, int linear_index)
  auto void_ptr_type = llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0);
  auto int32_type = llvm::Type::getInt32Ty(ctx);
  
  std::vector<llvm::Type*> param_types = {void_ptr_type, int32_type};
  auto func_type = llvm::FunctionType::get(void_ptr_type, param_types, false);
  
  auto func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, 
                                   "get_field_ptr", module);
  
  // Create basic block
  auto bb = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> builder(bb);
  
  // Get function parameters
  auto root_ptr = func->getArg(0);
  // auto linear_index = func->getArg(1);  // Unused for now
  
  // For now, just return the root pointer (simplified implementation)
  // In a full implementation, this would calculate the proper offset based on field layout
  builder.CreateRet(root_ptr);
}

void StructCompilationManager::create_child_access_functions(llvm::Module *module, SNode *root) {
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

void StructCompilationManager::create_function_table(llvm::Module *module, SNode *root) {
  auto &ctx = module->getContext();
  
  // Define function pointer types
  auto linearize_func_type = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(ctx),
      {llvm::PointerType::get(llvm::Type::getInt32Ty(ctx), 0),
       llvm::PointerType::get(llvm::Type::getInt32Ty(ctx), 0),
       llvm::Type::getInt32Ty(ctx)},
      false);
  
  auto field_access_func_type = llvm::FunctionType::get(
      llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0),
      {llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0),
       llvm::Type::getInt32Ty(ctx)},
      false);
  
  auto child_access_func_type = llvm::FunctionType::get(
      llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0),
      {llvm::PointerType::get(llvm::Type::getInt8Ty(ctx), 0),
       llvm::Type::getInt32Ty(ctx)},
      false);
  
  // Create function pointer types
  auto linearize_func_ptr_type = llvm::PointerType::get(linearize_func_type, 0);
  auto field_access_func_ptr_type = llvm::PointerType::get(field_access_func_type, 0);
  auto child_access_func_ptr_type = llvm::PointerType::get(child_access_func_type, 0);
  
  // Create struct type for the function table
  std::vector<llvm::Type*> table_elements = {
      linearize_func_ptr_type,
      field_access_func_ptr_type,
      child_access_func_ptr_type
  };
  
  auto table_type = llvm::StructType::create(ctx, table_elements, "FieldAccessTable");
  
  // Get function references
  auto linearize_func = module->getFunction("linearize_coords");
  auto field_access_func = module->getFunction("get_field_ptr");
  auto child_access_func = module->getFunction("get_child_ptr");
  
  // Create function pointer constants
  auto linearize_func_ptr = llvm::ConstantExpr::getBitCast(linearize_func, linearize_func_ptr_type);
  auto field_access_func_ptr = llvm::ConstantExpr::getBitCast(field_access_func, field_access_func_ptr_type);
  auto child_access_func_ptr = llvm::ConstantExpr::getBitCast(child_access_func, child_access_func_ptr_type);
  
  // Create the table as a global constant
  std::vector<llvm::Constant*> table_values = {
      linearize_func_ptr,
      field_access_func_ptr,
      child_access_func_ptr
  };
  
  auto table_constant = llvm::ConstantStruct::get(table_type, table_values);
  auto table_global = new llvm::GlobalVariable(
      *module, table_type, true, llvm::GlobalValue::ExternalLinkage,
      table_constant, "field_access_table");
  
  // Create a function to get the table pointer
  auto get_table_func_type = llvm::FunctionType::get(
      llvm::PointerType::get(table_type, 0), {}, false);
  
  auto get_table_func = llvm::Function::Create(
      get_table_func_type, llvm::Function::ExternalLinkage,
      "get_field_access_table", module);
  
  auto bb = llvm::BasicBlock::Create(ctx, "entry", get_table_func);
  llvm::IRBuilder<> builder(bb);
  
  builder.CreateRet(table_global);
}

}  // namespace taichi::lang 