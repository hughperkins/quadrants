
namespace taichi::lang {
struct ShapeInfo {
  int32_t strides[taichi_max_num_indices];
  int32_t element_size;
  size_t offset_bytes_in_parent_cell;
};
} // namespace taichi::lang
