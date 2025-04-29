#pragma once

struct RootMeta : public StructMeta {
  int tag;
};

STRUCT_FIELD(RootMeta, tag);

void Root_activate(Ptr meta, Ptr node, int i) {
  std::cout << "Root_activate" << std::endl;
}

u1 Root_is_active(Ptr meta, Ptr node, int i) {
  std::cout << "Root_is_active" << std::endl;
  return true;
}

Ptr Root_lookup_element(Ptr meta, Ptr node, int i) {
  // only one element
  std::cout << "Root_lookup_element: node" << std::endl;
  return node;
}

i32 Root_get_num_elements(Ptr meta, Ptr node) {
  std::cout << "Root_get_num_elements" << std::endl;
  return 1;
}
