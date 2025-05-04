#include "taichi/ir/ir_builder.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"

int main() {
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
  auto program = Program(host_arch());
  const auto &config = program.compile_config();

  int n = 10;
  program.materialize_runtime();
  auto *root = new SNode(0, SNodeType::root);
  auto *pointer = &root->pointer(Axis(0), n);
  auto *place = &pointer->insert_children(SNodeType::place);
  place->dt = PrimitiveType::i32;
  program.add_snode_tree(std::unique_ptr<SNode>(root), /*compile_only=*/false);

  std::unique_ptr<Kernel> kernel_init;

  {
    /*
    @ti.kernel
    def init():
      for index in range(n):
        place[index] = index
    */
    IRBuilder builder;
    auto *zero = builder.get_int32(0);
    auto *n_stmt = builder.get_int32(n);
    auto *loop = builder.create_range_for(zero, n_stmt, 0, 4);
    {
      auto _ = builder.get_loop_guard(loop);
      auto *index = builder.get_loop_index(loop);
      auto *ptr = builder.create_global_ptr(place, {index});
      builder.create_global_store(ptr, index);
    }

    kernel_init =
        std::make_unique<Kernel>(program, builder.extract_ir(), "init");
  }

  auto ctx_init = kernel_init->make_launch_context();

  {
    const auto &compiled_kernel_data =
        program.compile_kernel(config, program.get_device_caps(), *kernel_init);
    program.launch_kernel(compiled_kernel_data, ctx_init);
    std::cout << "afte running kernel_init" << std::endl;
  }
}
