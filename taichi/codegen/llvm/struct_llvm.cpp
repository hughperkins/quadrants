#include "taichi/codegen/llvm/struct_llvm.h"

#include "llvm/IR/Verifier.h"
#include "llvm/IR/IRBuilder.h"

#include "taichi/ir/ir.h"
#include "taichi/struct/struct.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/ir/transforms.h"

namespace taichi::lang {

StructCompilerLLVM::StructCompilerLLVM(Arch arch,
                                       const CompileConfig &config,
                                       TaichiLLVMContext *tlctx,
                                       std::unique_ptr<llvm::Module> &&module,
                                       int snode_tree_id)
    : LLVMModuleBuilder(std::move(module), tlctx),
      arch_(arch),
      config_(config),
      tlctx_(tlctx),
      llvm_ctx_(tlctx_->get_this_thread_context()),
      snode_tree_id_(snode_tree_id) {
}

StructCompilerLLVM::StructCompilerLLVM(Arch arch,
                                       LlvmProgramImpl *prog,
                                       std::unique_ptr<llvm::Module> &&module,
                                       int snode_tree_id)
    : StructCompilerLLVM(arch,
                         *prog->config,
                         prog->get_llvm_context(),
                         std::move(module),
                         snode_tree_id) {
}

void StructCompilerLLVM::generate_types(SNode &snode) {
  TI_AUTO_PROF;
  auto type = snode.type;
  if (snode.is_bit_level)
    return;
  llvm::Type *node_type = nullptr;
  std::cout << "StructCompilerLLVM::generate_types: "
            << snode.type_name() << " snode id " << snode.id << " " << snode.type_name() << " " << snode.node_type_name << "\n";

  auto ctx = llvm_ctx_;
  TI_ASSERT(ctx == tlctx_->get_this_thread_context());

  // create children type that supports forking...

  std::vector<llvm::Type *> ch_types;
  size_t curr_offset = 0;
  for (int i = 0; i < snode.ch.size(); i++) {
    std::cout << " ch " << i << std::endl;
    // size_t ch_offset = curr_offset;
    ch_offsets.push_back(curr_offset);
    auto ch_snode = snode.ch[i].get();
    ch_snode_ids.push_back(ch_snode->id);
    if (!snode.ch[i]->is_bit_level) {
      // Bit-level SNodes do not really have a corresponding LLVM type
      auto ch = get_llvm_node_type(module.get(), snode.ch[i].get());
      if (ch->isArrayTy()) {
        std::cout << " ch struct numelements " << ch->getStructNumElements()
                  << " array num elements " << ch->getArrayNumElements()
                  << " is array ty " << ch->isArrayTy() << " "
                  << " is vector ty " << ch->isVectorTy() << std::endl;
        ch->print(llvm::outs());
        llvm::outs() << "\n";
        ch_types.push_back(ch);
        if (ch->isArrayTy()) {
          auto elem_ty = llvm::cast<llvm::ArrayType>(ch)->getElementType();
          std::cout << " array elem type: ";
          elem_ty->print(llvm::outs());
          llvm::outs() << "\n";
          // std::cout << "elem_ty integer bit width " <<
          // elem_ty->getIntegerBitWidth() << " bits\n"; std::cout << elem_ty->
          const llvm::DataLayout &dl = module->getDataLayout();
          std::cout << "elem_ty size in bytes: " << dl.getTypeAllocSize(elem_ty)
                    << "\n";
          curr_offset +=
              dl.getTypeAllocSize(elem_ty) * ch->getArrayNumElements();
        } 
      }
    }
  }
  for(auto i = 0; i < ch_offsets.size(); i++) {
    std::cout << " ch " << i << " " << ch_snode_ids[i] << " offset: " << ch_offsets[i] << std::endl;
  }
  std::cout << "StructCompilerLLVM::generate_types: "
            << "ch_types.size() = " << ch_types.size() << "\n";

  auto ch_type =
      llvm::StructType::create(*ctx, ch_types, snode.node_type_name + "_ch");

  snode.cell_size_bytes = tlctx_->get_type_size(ch_type);

  for (int i = 0; i < snode.ch.size(); i++) {
    if (!snode.ch[i]->is_bit_level) {
      snode.ch[i]->offset_bytes_in_parent_cell =
          tlctx_->get_struct_element_offset(ch_type, i);
    }
  }

  llvm::Type *body_type = nullptr, *aux_type = nullptr;
  if (type == SNodeType::dense || type == SNodeType::bitmasked) {
    TI_ASSERT(snode._morton == false);
    body_type = llvm::ArrayType::get(ch_type, snode.max_num_elements());
    if (type == SNodeType::bitmasked) {
      aux_type = llvm::ArrayType::get(llvm::Type::getInt32Ty(*llvm_ctx_),
                                      (snode.max_num_elements() + 31) / 32);
    }
  } else if (type == SNodeType::root) {
    body_type = ch_type;
  } else if (type == SNodeType::place) {
    body_type = tlctx_->get_data_type(snode.dt);
  } else if (type == SNodeType::bit_struct) {
    if (!arch_is_cpu(arch_)) {
      TI_ERROR_IF(data_type_bits(snode.physical_type) < 32,
                  "bit_struct physical type must be at least 32 bits on "
                  "non-CPU backends.");
    }
    body_type = tlctx_->get_data_type(snode.physical_type);
  } else if (type == SNodeType::quant_array) {
    // A quant array SNode should have only one child
    TI_ASSERT(snode.ch.size() == 1);
    auto &ch = snode.ch[0];
    Type *ch_type = ch->dt;
    if (!arch_is_cpu(arch_)) {
      TI_ERROR_IF(data_type_bits(snode.physical_type) <= 16,
                  "quant_array physical type must be at least 32 bits on "
                  "non-CPU backends.");
    }
    snode.dt = TypeFactory::get_instance().get_quant_array_type(
        snode.physical_type, ch_type, snode.num_cells_per_container);

    DataType container_primitive_type(snode.physical_type);
    body_type = tlctx_->get_data_type(container_primitive_type);
  } else if (type == SNodeType::pointer) {
    // mutex
    aux_type = llvm::ArrayType::get(llvm::PointerType::getInt64Ty(*ctx),
                                    snode.max_num_elements());
    body_type = llvm::ArrayType::get(llvm::PointerType::getInt8PtrTy(*ctx),
                                     snode.max_num_elements());
  } else if (type == SNodeType::dynamic) {
    // mutex and n (number of elements)
    aux_type =
        llvm::StructType::get(*ctx, {llvm::PointerType::getInt32Ty(*ctx),
                                     llvm::PointerType::getInt32Ty(*ctx)});
    body_type = llvm::PointerType::getInt8PtrTy(*ctx);
  } else {
    TI_P(snode.type_name());
    TI_NOT_IMPLEMENTED;
  }
  if (aux_type != nullptr) {
    std::cout << "calling StructType::create with aux_type "
              << aux_type->getStructName().str() << std::endl;
    node_type = llvm::StructType::create(*ctx, {aux_type, body_type}, "");
  } else {
    node_type = body_type;
  }

  TI_ASSERT(node_type != nullptr);
  TI_ASSERT(body_type != nullptr);

  // Here we create a stub holding 4 LLVM types as struct members.
  // The aim is to give a **unique** name to the stub, so that we can look up
  // these types using this name. This decouples them from the LLVM context.
  // Note that body_type might not have a unique name, since literal structs
  // (such as {i32, i32}) cannot be aliased in LLVM.
  auto stub = llvm::StructType::create(
      *ctx,
      {node_type, body_type, aux_type ? aux_type : llvm::Type::getInt8Ty(*ctx),
       // aux_type might be null
       ch_type},
      type_stub_name(&snode));
  // irpass::print(stub);
  std::cout << "StructCompilerLLVM::generate_types: "
            << type_stub_name(&snode) << "\n";
  stub->print(llvm::outs());
  llvm::outs() << "\n";

  // Create a dummy function in the module with the type stub as return type
  // so that the type is referenced in the module
  auto ft = llvm::FunctionType::get(stub, false);
  create_function(ft, type_stub_name(&snode) + "_func");
}

void StructCompilerLLVM::generate_refine_coordinates(SNode *snode) {
  TI_AUTO_PROF;
  auto coord_type = get_runtime_type("PhysicalCoordinates");
  auto coord_type_ptr = llvm::PointerType::get(coord_type, 0);

  auto ft = llvm::FunctionType::get(
      llvm::Type::getVoidTy(*llvm_ctx_),
      {coord_type_ptr, coord_type_ptr, llvm::Type::getInt32Ty(*llvm_ctx_)},
      false);

  auto func = create_function(ft, snode->refine_coordinates_func_name());

  auto bb = llvm::BasicBlock::Create(*llvm_ctx_, "entry", func);

  llvm::IRBuilder<> builder(bb, bb->begin());
  std::vector<llvm::Value *> args;

  for (auto &arg : func->args()) {
    args.push_back(&arg);
  }

  auto inp_coords = args[0];
  auto outp_coords = args[1];
  // auto l = args[2];

  for (int i = 0; i < taichi_max_num_indices; i++) {
    // auto addition = tlctx_->get_constant(0);
    if (snode->extractors[i].shape > 1) {
      // std::cout << "Refining coordinates for " << snode->node_type_name
      //           << " at index " << i << std::endl;
      // // auto extr = snode->extractors[i];
      // std::cout << "  extr acc_shape: " << extr.acc_shape
      //           << ", shape: " << extr.shape << std::endl;
      // auto prev = tlctx_->get_constant(snode->extractors[i].acc_shape *
      //                                  snode->extractors[i].shape);
      // auto next = tlctx_->get_constant(snode->extractors[i].acc_shape);
      // Use UDiv/URem instead of SDiv/SRem so that LLVM can optimize them
      // into bitwise operations when the divisor is a power of two.
      // std::cout << "   urem " << (snode->extractors[i].acc_shape *
      //                                  snode->extractors[i].shape) << " udiv " << (snode->extractors[i].acc_shape) << std::endl;
      // addition = builder.CreateUDiv(builder.CreateURem(l, prev), next);
    }
    auto in = call(&builder, "PhysicalCoordinates_get_val", inp_coords,
                   tlctx_->get_constant(i));
    //   std::cout << "   mul " << snode->extractors[i].shape << std::endl;
    // in =
    //     builder.CreateMul(in, tlctx_->get_constant(snode->extractors[i].shape));
    // auto added = builder.CreateAdd(in, addition);
    auto added = in;
    call(&builder, "PhysicalCoordinates_set_val", outp_coords,
         tlctx_->get_constant(i), added);
  }
  builder.CreateRetVoid();
}

// define ptr @get_ch_S4_to_S5(ptr %0) {
// entry:
//   %getch = getelementptr %S4_ch, ptr %0, i32 0, i32 0
//   ret ptr %getch
// }
void StructCompilerLLVM::generate_child_accessors(SNode &snode) {
  TI_AUTO_PROF;
  auto type = snode.type;
  stack.push_back(&snode);

  bool is_leaf = type == SNodeType::place;

  if (!is_leaf) {
    generate_refine_coordinates(&snode);
  }

  if (snode.parent != nullptr) {
    // create the get ch function
    auto parent = snode.parent;

    auto ft =
        llvm::FunctionType::get(llvm::Type::getInt8PtrTy(*llvm_ctx_),
                                {llvm::Type::getInt8PtrTy(*llvm_ctx_)}, false);

    // auto funcfoo = create_function(ft, "foobar" + snode.get_ch_from_parent_func_name());
    auto funcfoo = create_function(ft, snode.get_ch_from_parent_func_name());

      std::cout << "StructCompilerLLVM::generate_child_accessors: "
                << snode.get_ch_from_parent_func_name() << std::endl;
      for(auto i = 0; i < ch_offsets.size(); i++) {
    // std::cout << " ch " << i << " offset: " << ch_offsets[i] << std::endl;
    std::cout << " ch " << i << " " << ch_snode_ids[i] << " offset: " << ch_offsets[i] << std::endl;
  }

    auto bb2 =
     llvm::BasicBlock::Create(*llvm_ctx_, "entry", funcfoo);

    llvm::IRBuilder<> builder2(bb2, bb2->begin());
    std::vector<llvm::Value *> args2;

    for (auto &arg : funcfoo->args()) {
      args2.push_back(&arg);
    }

    size_t offset = 0;
    if(parent != nullptr && parent->parent == nullptr) {
      std::cout << "parent is root snode\n";
      for(int i = 0; i < ch_offsets.size(); i++) {
        if(ch_snode_ids[i] == snode.id) {
          offset = ch_offsets[i];
          break;
        }
      }
      // offset = ch_offsets[]
      // offset = parent->child_id(&snode);
    }
    std::cout << "offset: " << offset << std::endl;
    llvm::Value *snode_ptr = builder2.CreateGEP(
        llvm::Type::getInt8Ty(*llvm_ctx_),
        builder2.CreateBitCast(args2[0], llvm::Type::getInt8PtrTy(*llvm_ctx_)),
        tlctx_->get_constant(offset));

    // builder.CreateRet(args[0]);
    // builder2.CreateRet(
    //     builder2.CreateBitCast(args2[0], llvm::Type::getInt8PtrTy(*llvm_ctx_)));
    builder2.CreateRet(snode_ptr);
      //  builder2.CreateBitCast( snode_ptr, llvm::Type::getInt8PtrTy(*llvm_ctx_));

    //       auto inp_type =
    //     llvm::PointerType::get(get_llvm_element_type(module.get(), parent), 0);


    // auto func = create_function(ft, snode.get_ch_from_parent_func_name());

    // auto bb = llvm::BasicBlock::Create(*llvm_ctx_, "entry", func);

    // llvm::IRBuilder<> builder(bb, bb->begin());
    // std::vector<llvm::Value *> args;

    // for (auto &arg : func->args()) {
    //   args.push_back(&arg);
    // }
    // llvm::Value *ret;
    // ret = builder.CreateGEP(get_llvm_element_type(module.get(), parent),
    //                         builder.CreateBitCast(args[0], inp_type),
    //                         {tlctx_->get_constant(0),
    //                          tlctx_->get_constant(parent->child_id(&snode))},
    //                         "getch");

    // builder.CreateRet(
    //     builder.CreateBitCast(ret, llvm::Type::getInt8PtrTy(*llvm_ctx_)));
  }

  for (auto &ch : snode.ch) {
    if (!ch->is_bit_level)
      generate_child_accessors(*ch);
  }

  stack.pop_back();
}

std::string StructCompilerLLVM::type_stub_name(SNode *snode) {
  return snode->node_type_name + "_type_stubs";
}

void StructCompilerLLVM::run(SNode &root) {
  TI_AUTO_PROF;
  // bottom to top
  collect_snodes(root);

  auto snodes_rev = snodes;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  for (auto &n : snodes_rev) {
    generate_types(*n);
  }

  generate_child_accessors(root);

  if (config_.print_struct_llvm_ir) {
    static FileSequenceWriter writer("taichi_struct_llvm_ir_{:04d}.ll",
                                     "struct LLVM IR");
    writer.write(module.get());
  }

  const char *dump_ir_env = std::getenv("TAICHI_DUMP_IR");
  const std::string dumpOutDir = "/tmp/ir/";
  if (dump_ir_env != nullptr) {
    std::filesystem::create_directories(dumpOutDir);

    std::string filename =
        dumpOutDir + "/" + std::string(module->getName()) + "_{:04d}_llvm.ll";
        static FileSequenceWriter writer(filename,
                                     "struct LLVM IR");
        writer.write(module.get());
  }

  TI_ASSERT((int)snodes.size() <= taichi_max_num_snodes);

  auto node_type = get_llvm_node_type(module.get(), &root);
  root_size = tlctx_->get_type_size(node_type);

  tlctx_->add_struct_module(std::move(module), root.get_snode_tree_id());
}

llvm::Type *StructCompilerLLVM::get_stub(llvm::Module *module,
                                         SNode *snode,
                                         uint32 index) {
  TI_ASSERT(module);
  TI_ASSERT(snode);
  auto stub = llvm::StructType::getTypeByName(module->getContext(),
                                              type_stub_name(snode));
  TI_ASSERT(stub);
  TI_ASSERT(stub->getStructNumElements() == 4);
  TI_ASSERT(0 <= index && index < 4);
  auto type = stub->getContainedType(index);
  TI_ASSERT(type);
  return type;
}

llvm::Type *StructCompilerLLVM::get_llvm_node_type(llvm::Module *module,
                                                   SNode *snode) {
  return get_stub(module, snode, 0);
}

llvm::Type *StructCompilerLLVM::get_llvm_body_type(llvm::Module *module,
                                                   SNode *snode) {
  return get_stub(module, snode, 1);
}

llvm::Type *StructCompilerLLVM::get_llvm_aux_type(llvm::Module *module,
                                                  SNode *snode) {
  return get_stub(module, snode, 2);
}

llvm::Type *StructCompilerLLVM::get_llvm_element_type(llvm::Module *module,
                                                      SNode *snode) {
  return get_stub(module, snode, 3);
}

llvm::Function *StructCompilerLLVM::create_function(llvm::FunctionType *ft,
                                                    std::string func_name) {
  return llvm::Function::Create(ft, llvm::Function::ExternalLinkage, func_name,
                                *module);
}

}  // namespace taichi::lang
