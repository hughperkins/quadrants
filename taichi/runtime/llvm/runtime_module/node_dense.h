#pragma once

// Specialized Attributes and functions
struct DenseMeta : public StructMeta {
  int morton_dim;
};

STRUCT_FIELD(DenseMeta, morton_dim)

i32 Dense_get_num_elements(Ptr meta, Ptr node) {
  std::cout << "Dense_get_num_elements" << std::endl;
  return ((StructMeta *)meta)->max_num_elements;
}

void Dense_activate(Ptr meta, Ptr node, int i) {
  std::cout << "Dense_activate" << std::endl;
  // Dense elements are always active
}

u1 Dense_is_active(Ptr meta, Ptr node, int i) {
  std::cout << "Dense_is_active" << std::endl;
  return true;
}

Ptr Dense_lookup_element(Ptr meta, Ptr node, int i) {
  std::cout << "Dense_lookup_element meta " << ((void *)meta) << " node "
            << ((void *)node) << " i " << i << std::endl;
  Ptr res = node + ((StructMeta *)meta)->element_size * i;
  std::cout << "Dense_lookup_element got res" << std::endl;
  std::cout << "Dense_lookup_element got res" << std::endl;
  std::cout << "Dense_lookup_element got res" << std::endl;
  std::cout << "Dense_lookup_element got res" << std::endl;
  std::cout << "Dense_lookup_element got res" << std::endl;
  std::cout << "Dense_lookup_element got res" << std::endl;
  std::cout << "Dense_lookup_element got res here it is:" << ((void *)res)
            << std::endl;
  std::cout << "Dense_lookup_element got res here it is:" << ((void *)res)
            << std::endl;
  std::cout << "Dense_lookup_element got res here it is:" << ((void *)res)
            << std::endl;
  std::cout << "Dense_lookup_element got res here it is:" << ((void *)res)
            << std::endl;
  // std::cout << "Dense_lookup_element got res here it is:" << res <<
  // std::endl; std::cout << "Dense_lookup_element got res here it is:" << res
  // << std::endl; std::cout << "Dense_lookup_element got res here it is:" <<
  // res << std::endl; std::cout << "Dense_lookup_element got res here it is:"
  // << res << std::endl;
  return res;
}
