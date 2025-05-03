#include <memory>

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

void init_kernel_func() {
  // Empty function - we'll set the IR separately
}

// std::unique_ptr<taichi::lang::IRNode> get_ir_unique_ptr(taichi::lang::Block*
// block) {
//   taichi::lang::IRNode* node = block->get_ir_root();
//   // Create a unique_ptr that doesn't delete the node when it goes out of
//   scope
//   // since the block still owns it
//   return std::unique_ptr<taichi::lang::IRNode>(node,
//   [](taichi::lang::IRNode*) {});
// }

// std::unique_ptr<taichi::lang::IRNode,
// std::function<void(taichi::lang::IRNode*)>>
// get_ir_unique_ptr(taichi::lang::Block* block) {
//   taichi::lang::IRNode* node = block->get_ir_root();
//   // Create a unique_ptr with custom deleter that doesn't delete
//   std::function<void(taichi::lang::IRNode*)> no_delete =
//   [](taichi::lang::IRNode*) {}; return std::unique_ptr<taichi::lang::IRNode,
//   std::function<void(taichi::lang::IRNode*)>>(node, no_delete);
// }

// Replace your current get_ir_unique_ptr function with:
// std::unique_ptr<taichi::lang::IRNode> get_ir_unique_ptr(taichi::lang::Block*
// block) {
//   // Use the clone() method of IRNode to create a deep copy
//   return block->clone();
// }

std::unique_ptr<taichi::lang::IRNode> get_ir_unique_ptr(
    taichi::lang::Block *block) {
  // Create a new Block
  auto new_block = std::make_unique<taichi::lang::Block>();

  // Clone all statements from the original block to the new one
  for (auto &stmt : block->statements) {
    // Create a deep copy of each statement
    auto stmt_clone = stmt->clone();
    new_block->insert(std::move(stmt_clone));
  }

  // Return the new block as an IRNode
  return new_block;
}

// std::unique_ptr<taichi::lang::IRNode> get_ir_unique_ptr(taichi::lang::Block*
// block) {
//   // Create a completely new Block with the same statements
//   auto new_block = std::make_unique<taichi::lang::Block>();

//   // Re-add the same statements but created fresh

//   auto const_123 = new_block->push_back<taichi::lang::ConstStmt>(
//       taichi::lang::TypedConstant(taichi::lang::PrimitiveType::i32, 123));
//   new_block->push_back<taichi::lang::ReturnStmt>(const_123);

//   // Return the new block as an IRNode
//   return new_block;
// }

void run_snode() {
  /*
  import taichi as ti, numpy as np
  ti.init()
  #ti.init(print_ir = True)

  n = 10
  place = ti.field(dtype = ti.i32)
  ti.root.pointer(ti.i, n).place(place)

  @ti.kernel
  def init():
      for index in range(n):
          place[index] = index

  @ti.kernel
  def ret() -> ti.i32:
      sum = 0
      for index in place:
          sum = sum + place[index]
      return sum

  @ti.kernel
  def ext(ext_arr: ti.ext_arr()):
      for index in place:
          ext_arr[index] = place[index]

  init()
  print(ret())
  ext_arr = np.zeros(n, np.int32)
  ext(ext_arr)
  #ext_arr = place.to_numpy()
  print(ext_arr)
  */

  using namespace taichi;
  using namespace lang;
  auto program = Program(Arch::arm64);
  program.get_program_impl()->config->opt_level = 0;
  program.get_program_impl()->config->advanced_optimization = false;
  program.get_program_impl()->config->print_ir = true;
  // CompileConfig config;
  // config.arch = Arch::arm64;
  // config.opt_level = 0;
  // program.config = config;
  // program.set_compile_config(config);
  const auto &config = program.compile_config();
  // config.opt_level = 0;
  /*CompileConfig config_print_ir;
  config_print_ir.print_ir = true;
  prog_.config = config_print_ir;*/  // print_ir = True

  int n = 10;
  program.materialize_runtime();
  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->pointer(Axis(0), n);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/false);

  std::unique_ptr<Kernel> kernel_init, kernel_ret, kernel_ext;

  // {
  //   /*
  //   @ti.kernel
  //   def init():
  //     for index in range(n):
  //       place[index] = index
  //   */
  // IRBuilder builder;
  //   auto *zero = builder.get_int32(0);
  //   auto *n_stmt = builder.get_int32(n);
  //   auto *loop = builder.create_range_for(zero, n_stmt, 0, 4);
  //   {
  //     auto _ = builder.get_loop_guard(loop);
  //     auto *index = builder.get_loop_index(loop);
  //     auto *ptr = builder.create_global_ptr(place, {index});
  //     builder.create_global_store(ptr, index);
  //   }

  // kernel_init =
  //     std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  // }

  // {
  //   auto block = std::make_unique<Block>();
  //   auto const_type = taichi::lang::PrimitiveType::i32;
  //   auto const_123 =
  //   block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  //   // auto ret_stmt = block->push_back<ReturnStmt>(const_123);
  //   // ret_stmt->ret_type = const_type;
  //   auto ret_stmt = static_cast<ReturnStmt
  //   *>(block->push_back<ReturnStmt>(const_123));

  //   // Make sure ret_stmt has the proper type
  //   ret_stmt->ret_type = PrimitiveType::i32;
  //   std::cout << "ret values " << ret_stmt->values.size() << std::endl;
  //   std::cout << "ret values " << ret_stmt->values.size() << " " <<
  //   ret_stmt->values[0]->name() << std::endl;
  //   // auto ret_stmt =
  //   taichi::lang::Stmt::make_typed<taichi::lang::ReturnStmt>(const_123);
  //   // ret_stmt->ret_type = PrimitiveType::i32;
  //   // auto ret_stmt_ptr = ret_stmt.get();
  //   // if (ret_stmt_ptr) {
  //   //   ret_stmt_ptr->ret_type = PrimitiveType::i32;
  //   //   // Ensure const_123 has a proper type
  //   //   if (const_123->ret_type.is_primitive(PrimitiveTypeID::unknown)) {
  //   //     const_123->ret_type = PrimitiveType::i32;
  //   //   }
  //   // }
  //   // block->insert(ret_stmt.move());

  //   // Block block;
  //   // auto const_123 =
  //   block.push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  //   // block.push_back<ReturnStmt>(const_123);

  //   // auto block_clone = get_ir_unique_ptr(block.get());

  //   // Move the clone into the kernel constructor
  //   // kernel_init =
  //   //     std::make_unique<Kernel>(program, std::move(block_clone), "init");
  //       kernel_init =
  //       std::make_unique<Kernel>(program, block.release(), "init");
  //       // kernel_init->ir.get()->get_ir_root()
  //   // kernel_init =
  //   //     std::make_unique<Kernel>(program, block, "init");
  //   // kernel_init =
  //   //     std::make_unique<Kernel>(program, [](Kernel* kernel) {
  //   //       auto block = std::make_unique<Block>();
  //   //       auto const_123 =
  //   block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  //   //       block->push_back<ReturnStmt>(const_123);

  //   //       // Set the block as the kernel's IR using whatever method the
  //   Kernel API provides
  //   //       // This might involve something like:
  //   //       kernel->set_ir(block.release());
  //   //       // or some other method to transfer the IR to the kernel
  //   //     }, "init");

  //   // kernel_init =
  //   //     std::make_unique<Kernel>(program, get_ir_unique_ptr(block.get()),
  //   "init");
  //   // kernel_init =
  //   //     std::make_unique<Kernel>(program, init_kernel_func, "init");
  //       // init_kernel_func
  //     // kernel_init->set_ir_root(block.release());
  // }

  {
    /*
    @ti.kernel
    def ret():
      sum = 0
      for index in place:
        sum = sum + place[index];
      return sum
    */
    IRBuilder builder;
    std::cout << "1" << std::endl;
    auto *sum = builder.create_local_var(PrimitiveType::i32);
    auto *loop = builder.create_struct_for(pointer, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *sum_old = builder.create_local_load(sum);
      auto *place_index =
          builder.create_global_load(builder.create_global_ptr(place, {index}));
      builder.create_local_store(sum, builder.create_add(sum_old, place_index));
    }
    std::cout << "2" << std::endl;
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");
  }

  // {
  //   /*
  //   @ti.kernel
  //   def ext(ext: ti.ext_arr()):
  //     for index in place:
  //       ext[index] = place[index];
  //   # ext = place.to_numpy()
  //   */
  //   IRBuilder builder;
  //   auto *loop = builder.create_struct_for(pointer, 0, 4);
  //   {
  //     auto _ = builder.get_loop_guard(loop);
  //     auto *index = builder.get_loop_index(loop);
  //     auto *ext = builder.create_external_ptr(
  //         builder.create_arg_load({0}, PrimitiveType::i32, true, 0),
  //         {index});
  //     auto *place_index =
  //         builder.create_global_load(builder.create_global_ptr(place,
  //         {index}));
  //     builder.create_global_store(ext, place_index);
  //   }

  //   kernel_ext = std::make_unique<Kernel>(program, builder.extract_ir(),
  //   "ext"); kernel_ext->insert_arr_param(get_data_type<int>(),
  //   /*total_dim=*/1, {n}); kernel_ext->finalize_params();
  // }

  // auto ctx_init = kernel_init->make_launch_context();
  auto ctx_ret = kernel_ret->make_launch_context();
  // auto ctx_ext = kernel_ext->make_launch_context();
  // std::vector<int> ext_arr(n);
  // ctx_ext.set_arg_external_array_with_shape({0},
  // taichi::uint64(ext_arr.data()),
  //                                           n, {n});

  // {
  //   const auto &compiled_kernel_data =
  //       program.compile_kernel(config, program.get_device_caps(),
  //       *kernel_init);
  //   program.launch_kernel(compiled_kernel_data, ctx_init);
  //   std::cout << "res " << program.fetch_result<int>(0) << std::endl;
  // }
  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
    program.launch_kernel(compiled_kernel_data, ctx_ret);
    std::cout << "res " << program.fetch_result<int>(0) << std::endl;
  }
  // {
  //   const auto &compiled_kernel_data =
  //       program.compile_kernel(config, program.get_device_caps(),
  //       *kernel_ext);
  //   std::cout << "3" << std::endl;
  //   program.launch_kernel(compiled_kernel_data, ctx_ext);
  //   // for (int i = 0; i < n; i++)
  //   //   std::cout << ext_arr[i] << " ";
  //   std::cout << std::endl;
  // }
}
