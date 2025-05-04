#include <memory>

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
  program.get_program_impl()->config->opt_level = 0;
  program.get_program_impl()->config->external_optimization_level = 0;
  program.get_program_impl()->config->advanced_optimization = false;
  program.get_program_impl()->config->print_ir = true;
  const auto &config = program.compile_config();

  std::unique_ptr<Kernel> kernel_ret;

  auto block = std::make_unique<Block>();
  auto const_type = PrimitiveType::i32;
  auto const_123 =
      block->push_back<ConstStmt>(TypedConstant(PrimitiveType::i32, 123));
  block->push_back<ReturnStmt>(const_123);

  kernel_ret = std::make_unique<Kernel>(program, block.release(), "ret");
  kernel_ret->insert_ret(PrimitiveType::i32);
  kernel_ret->finalize_rets();

  auto ctx_ret = kernel_ret->make_launch_context();
  const auto &compiled_kernel_data =
      program.compile_kernel(config, program.get_device_caps(), *kernel_ret);
  program.materialize_runtime();
  program.launch_kernel(compiled_kernel_data, ctx_ret);
  std::cout << "res " << program.fetch_result<int>(0) << std::endl;
}
