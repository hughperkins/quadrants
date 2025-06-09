#pragma once

// Specialized Attributes and functions
struct DenseMeta : public StructMeta {
  int morton_dim;
};

STRUCT_FIELD(DenseMeta, morton_dim)

i32 Dense_get_num_elements(Ptr meta, Ptr node) {
  return ((StructMeta *)meta)->max_num_elements;
}

void Dense_activate(Ptr meta, Ptr node, int i) {
  // Dense elements are always active
}

u1 Dense_is_active(Ptr meta, Ptr node, int i) {
  return true;
}

i32 Dense_get_stride(Ptr meta, int i) {
  auto runtime = ((StructMeta *)meta)->runtime;
  auto shapeInfo = runtime->snode_shapes[((StructMeta *)meta)->snode_id];
  return shapeInfo.strides[i];
}

Ptr Dense_lookup_element(Ptr meta, Ptr node, int i) {
  auto runtime = ((StructMeta *)meta)->runtime;
  auto shapeInfo = runtime->snode_shapes[((StructMeta *)meta)->snode_id];
  struct StructMeta *sm = (StructMeta *)meta;
  // taichi_printf(runtime, "dumped_diag %i\n", runtime->dumped_diag[sm->snode_id]);
  auto dumped_diag = runtime->dumped_diag[sm->snode_id];
  if(!dumped_diag) {
    runtime->dumped_diag[sm->snode_id] = 1;
    taichi_printf(runtime, "snode_id = %i\n", sm->snode_id);
    for(auto i = 0; i < 5; i++) {
      taichi_printf(runtime, "  node_dense.h shapeInfo.strides[%i] = %i\n", i, shapeInfo.strides[i]);
    }
  }
  taichi_printf(runtime, "Dense_lookup_element snode_id = %i i = %i\n", sm->snode_id, i);
  return node + ((StructMeta *)meta)->element_size * i;
}
