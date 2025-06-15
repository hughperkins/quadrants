
namespace taichi::lang {
struct ShapeInfo {
  int32_t strides[taichi_max_num_indices];
  int32_t element_size;
};
} // namespace taichi::lang
