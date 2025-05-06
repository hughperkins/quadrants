#include <memory>

#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

int main() {
  /*
  import taichi as ti, numpy as np
  ti.init()

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
  auto program = Program(host_arch());
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
    /*
    @ti.kernel
    def ret():
      sum = 0
      for index in place:
        sum = sum + place[index];
      return sum
    */
    IRBuilder builder;
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
    builder.create_return(builder.create_local_load(sum));

    kernel_ret = std::make_unique<Kernel>(program, builder.extract_ir(), "ret");
  }
  auto ctx_ret = kernel_ret->make_launch_context();

  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
    program.launch_kernel(compiled_kernel_data, ctx_ret);
    std::cout << "res " << program.fetch_result<int>(0) << std::endl;
  }
}
