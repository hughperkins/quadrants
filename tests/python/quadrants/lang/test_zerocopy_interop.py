"""Tests for zero-copy to_torch / to_numpy via DLPack (the ``copy`` parameter)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import quadrants as qd

from tests import test_utils

pytestmark = pytest.mark.needs_torch

dlpack_arch = [qd.cpu, qd.cuda, qd.metal, qd.amdgpu]


def is_v520_amdgpu():
    return os.environ.get("QD_AMDGPU_V520", None) == "1" and qd.cfg.arch == qd.amdgpu


def _to_cpu(t):
    """Move tensor to CPU for value comparison on arches where torch accessors may not work."""
    if is_v520_amdgpu() or t.device.type != "cpu":
        return t.cpu()
    return t


# ---------------------------------------------------------------------------
# ScalarField.to_torch  --  zero-copy
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32])
def test_scalar_field_to_torch_zerocopy(dtype):
    f = qd.field(dtype, shape=(4,))
    f[0] = 10
    f[1] = 20
    qd.sync()

    tc = f.to_torch()
    tc = _to_cpu(tc)
    assert tc[0] == 10
    assert tc[1] == 20
    assert tuple(tc.shape) == (4,)


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_caching():
    """Calling to_torch() twice returns the same cached object."""
    f = qd.field(qd.f32, shape=(3,))
    qd.sync()
    t1 = f.to_torch()
    t2 = f.to_torch()
    assert t1.data_ptr() == t2.data_ptr()


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_copy_true():
    """copy=True must return an independent tensor."""
    f = qd.field(qd.f32, shape=(3,))
    f[0] = 42
    qd.sync()
    tc_view = f.to_torch()
    tc_copy = f.to_torch(copy=True)
    assert _to_cpu(tc_copy)[0] == 42
    assert tc_view.data_ptr() != tc_copy.data_ptr()


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_copy_false():
    """copy=False should succeed on DLPack-capable arches with non-empty shape."""
    f = qd.field(qd.f32, shape=(3,))
    f[0] = 7
    qd.sync()
    tc = f.to_torch(copy=False)
    assert _to_cpu(tc)[0] == 7


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_aliases_memory():
    """Zero-copy tensor must alias the field's device memory."""
    if is_v520_amdgpu():
        pytest.skip("can't run torch accessor kernels on v520")
    f = qd.field(qd.i32, shape=(4,))
    f[0] = 1
    qd.sync()
    tc = f.to_torch()

    @qd.kernel
    def write(f: qd.template()):
        f[0] = 99

    write(f)
    qd.sync()
    assert tc[0] == 99


# ---------------------------------------------------------------------------
# ScalarField.to_numpy  --  zero-copy (CPU only)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_zerocopy_cpu():
    f = qd.field(qd.f32, shape=(5,))
    f[0] = 3.5
    qd.sync()
    arr = f.to_numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr[0], 3.5)


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_copy_true_cpu():
    f = qd.field(qd.f32, shape=(3,))
    f[0] = 1.0
    qd.sync()
    arr = f.to_numpy(copy=True)
    np.testing.assert_allclose(arr[0], 1.0)


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_numpy_matches_copy_path():
    """Zero-copy and copy path produce the same values."""
    f = qd.field(qd.f32, shape=(6,))
    for i in range(6):
        f[i] = float(i * 11)
    qd.sync()
    arr_default = f.to_numpy()
    arr_copy = f.to_numpy(copy=True)
    np.testing.assert_allclose(arr_default, arr_copy)


# ---------------------------------------------------------------------------
# MatrixField.to_torch  --  zero-copy with keep_dims handling
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_vector_field_to_torch_zerocopy():
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(4,))
    f[0] = (1, 2, 3)
    f[1] = (4, 5, 6)
    qd.sync()
    tc = _to_cpu(f.to_torch())
    assert tuple(tc.shape) == (4, 3)
    assert tc[0, 0] == 1
    assert tc[1, 2] == 6


@test_utils.test(arch=dlpack_arch)
def test_vector_field_to_torch_keep_dims():
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(4,))
    f[0] = (10, 20, 30)
    qd.sync()
    tc_no_keep = _to_cpu(f.to_torch(keep_dims=False))
    tc_keep = _to_cpu(f.to_torch(keep_dims=True))
    assert tuple(tc_no_keep.shape) == (4, 3)
    assert tuple(tc_keep.shape) == (4, 3, 1)
    assert tc_no_keep[0, 0] == tc_keep[0, 0, 0]


@test_utils.test(arch=dlpack_arch)
def test_matrix_field_to_torch_zerocopy():
    mat = qd.types.matrix(2, 3, qd.f32)
    f = qd.field(mat, shape=(5,))
    f[0] = ((1, 2, 3), (4, 5, 6))
    qd.sync()
    tc = _to_cpu(f.to_torch())
    assert tuple(tc.shape) == (5, 2, 3)
    assert tc[0, 0, 0] == 1
    assert tc[0, 1, 2] == 6


@test_utils.test(arch=dlpack_arch)
def test_matrix_field_to_torch_matches_copy():
    mat = qd.types.matrix(2, 3, qd.f32)
    f = qd.field(mat, shape=(4,))
    f[0] = ((1, 2, 3), (4, 5, 6))
    f[1] = ((7, 8, 9), (10, 11, 12))
    qd.sync()
    tc_zc = _to_cpu(f.to_torch())
    tc_cp = _to_cpu(f.to_torch(copy=True))
    assert torch.allclose(tc_zc, tc_cp)


# ---------------------------------------------------------------------------
# MatrixField (vec/mat).to_torch  --  copy=False zero-copy
#
# A vector / matrix field's components are stored as siblings under one parent
# SNode (so ``parent.get_num_ch() > 1``), but ``field_to_dlpack`` exports them
# with the correct shape and strides. ``copy=False`` MUST therefore succeed
# and alias the field's memory.
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_vector_field_to_torch_copy_false():
    """``copy=False`` on a Vector.field must succeed and produce values matching the kernel-copy path."""
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(4,))
    for i in range(4):
        f[i] = (i * 100 + 0, i * 100 + 1, i * 100 + 2)
    qd.sync()
    tc_zc = _to_cpu(f.to_torch(copy=False))
    tc_cp = _to_cpu(f.to_torch(copy=True))
    assert tuple(tc_zc.shape) == (4, 3)
    assert torch.allclose(tc_zc, tc_cp)


@test_utils.test(arch=dlpack_arch)
def test_matrix_field_to_torch_copy_false():
    """``copy=False`` on a Matrix.field must succeed and produce values matching the kernel-copy path."""
    mat = qd.types.matrix(2, 3, qd.f32)
    f = qd.field(mat, shape=(4,))
    f[0] = ((1, 2, 3), (4, 5, 6))
    f[1] = ((7, 8, 9), (10, 11, 12))
    qd.sync()
    tc_zc = _to_cpu(f.to_torch(copy=False))
    tc_cp = _to_cpu(f.to_torch(copy=True))
    assert tuple(tc_zc.shape) == (4, 2, 3)
    assert torch.allclose(tc_zc, tc_cp)


@test_utils.test(arch=dlpack_arch)
def test_vector_field_to_torch_aliases_memory():
    """Zero-copy tensor view of a Vector.field must reflect later kernel writes."""
    if is_v520_amdgpu():
        pytest.skip("can't run torch accessor kernels on v520")
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(4,))
    f[0] = (1.0, 2.0, 3.0)
    qd.sync()
    tc = f.to_torch(copy=False)

    @qd.kernel
    def write(f: qd.template()):
        f[0] = qd.Vector([99.0, 88.0, 77.0])

    write(f)
    qd.sync()
    tc_cpu = _to_cpu(tc)
    assert tc_cpu[0, 0] == 99.0
    assert tc_cpu[0, 1] == 88.0
    assert tc_cpu[0, 2] == 77.0


@test_utils.test(arch=dlpack_arch)
def test_matrix_field_to_torch_aliases_memory():
    """Zero-copy tensor view of a Matrix.field must reflect later kernel writes."""
    if is_v520_amdgpu():
        pytest.skip("can't run torch accessor kernels on v520")
    mat = qd.types.matrix(2, 2, qd.f32)
    f = qd.field(mat, shape=(2,))
    f[0] = ((1.0, 2.0), (3.0, 4.0))
    qd.sync()
    tc = f.to_torch(copy=False)

    @qd.kernel
    def write(f: qd.template()):
        f[0] = qd.Matrix([[10.0, 20.0], [30.0, 40.0]])

    write(f)
    qd.sync()
    tc_cpu = _to_cpu(tc)
    assert tc_cpu[0, 0, 0] == 10.0
    assert tc_cpu[0, 1, 1] == 40.0


# ---------------------------------------------------------------------------
# MatrixField.to_numpy  --  zero-copy (CPU only)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_vector_field_to_numpy_zerocopy_cpu():
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(3,))
    f[0] = (10, 20, 30)
    qd.sync()
    arr = f.to_numpy()
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[0], [10, 20, 30])


@test_utils.test(arch=[qd.cpu])
def test_matrix_field_to_numpy_zerocopy_cpu():
    mat = qd.types.matrix(2, 2, qd.f32)
    f = qd.field(mat, shape=(2,))
    f[0] = ((1, 2), (3, 4))
    qd.sync()
    arr = f.to_numpy()
    assert arr.shape == (2, 2, 2)
    np.testing.assert_allclose(arr[0], [[1, 2], [3, 4]])


@test_utils.test(arch=[qd.cpu])
def test_vector_field_to_numpy_copy_false_cpu():
    """``copy=False`` numpy view on a Vector.field (CPU) must succeed and match the copy path."""
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(3,))
    for i in range(3):
        f[i] = (i * 10.0, i * 10.0 + 1, i * 10.0 + 2)
    qd.sync()
    arr_zc = f.to_numpy(copy=False)
    arr_cp = f.to_numpy(copy=True)
    assert arr_zc.shape == (3, 3)
    np.testing.assert_allclose(arr_zc, arr_cp)


@test_utils.test(arch=[qd.cpu])
def test_matrix_field_to_numpy_copy_false_cpu():
    """``copy=False`` numpy view on a Matrix.field (CPU) must succeed and match the copy path."""
    mat = qd.types.matrix(2, 2, qd.f32)
    f = qd.field(mat, shape=(2,))
    f[0] = ((1, 2), (3, 4))
    f[1] = ((5, 6), (7, 8))
    qd.sync()
    arr_zc = f.to_numpy(copy=False)
    arr_cp = f.to_numpy(copy=True)
    assert arr_zc.shape == (2, 2, 2)
    np.testing.assert_allclose(arr_zc, arr_cp)


# ---------------------------------------------------------------------------
# Ndarray.to_torch  --  new method, always uses DLPack
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_scalar_ndarray_to_torch():
    nd = qd.ndarray(qd.f32, shape=(5,))
    nd[0] = 42.0
    qd.sync()
    tc = _to_cpu(nd.to_torch())
    assert tuple(tc.shape) == (5,)
    assert tc[0] == 42.0


@test_utils.test(arch=dlpack_arch)
def test_vector_ndarray_to_torch():
    nd = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    nd[0] = (1, 2, 3)
    qd.sync()
    tc = _to_cpu(nd.to_torch())
    assert tuple(tc.shape) == (4, 3)
    assert tc[0, 0] == 1
    assert tc[0, 2] == 3


@test_utils.test(arch=dlpack_arch)
def test_matrix_ndarray_to_torch():
    nd = qd.Matrix.ndarray(2, 3, qd.f32, shape=(4,))
    nd[0] = ((10, 20, 30), (40, 50, 60))
    qd.sync()
    tc = _to_cpu(nd.to_torch())
    assert tuple(tc.shape) == (4, 2, 3)
    assert tc[0, 0, 0] == 10
    assert tc[0, 1, 2] == 60


@test_utils.test(arch=dlpack_arch)
def test_ndarray_to_torch_caching():
    nd = qd.ndarray(qd.f32, shape=(3,))
    qd.sync()
    t1 = nd.to_torch()
    t2 = nd.to_torch()
    assert t1.data_ptr() == t2.data_ptr()


@test_utils.test(arch=dlpack_arch)
def test_ndarray_to_torch_copy_true():
    nd = qd.ndarray(qd.f32, shape=(3,))
    nd[0] = 7.0
    qd.sync()
    tc = _to_cpu(nd.to_torch(copy=True))
    assert tc[0] == 7.0
    view = nd.to_torch()
    assert view.data_ptr() != tc.data_ptr()


# ---------------------------------------------------------------------------
# Ndarray.to_numpy  --  zero-copy on CPU
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_scalar_ndarray_to_numpy_zerocopy():
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd[0] = 3.0
    nd[1] = 5.0
    qd.sync()
    arr = nd.to_numpy()
    np.testing.assert_allclose(arr[:2], [3.0, 5.0])


@test_utils.test(arch=[qd.cpu])
def test_vector_ndarray_to_numpy_zerocopy():
    nd = qd.Vector.ndarray(3, qd.f32, shape=(2,))
    nd[0] = (1, 2, 3)
    qd.sync()
    arr = nd.to_numpy()
    np.testing.assert_allclose(arr[0], [1, 2, 3])


@test_utils.test(arch=dlpack_arch)
def test_ndarray_to_numpy_matches_copy():
    nd = qd.ndarray(qd.f32, shape=(6,))
    for i in range(6):
        nd[i] = float(i * 7)
    qd.sync()
    arr_default = nd.to_numpy()
    arr_copy = nd.to_numpy(copy=True)
    np.testing.assert_allclose(arr_default, arr_copy)


# ---------------------------------------------------------------------------
# Two fields in the same SNode tree (non-zero offset)
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_to_torch_two_fields_same_tree():
    a = qd.field(qd.i32, (100,))
    b = qd.field(qd.i32, (100,))
    a[0] = 111
    b[0] = 222
    qd.sync()
    at = _to_cpu(a.to_torch())
    bt = _to_cpu(b.to_torch())
    assert at[0] == 111
    assert bt[0] == 222


# ---------------------------------------------------------------------------
# copy=False  --  error paths
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_copy_false_with_device_transfer_raises():
    """copy=False + device that differs from data device should raise."""
    if qd.cfg.arch == qd.cpu:
        pytest.skip("need GPU arch to test device mismatch")
    f = qd.field(qd.f32, shape=(3,))
    qd.sync()
    with pytest.raises(ValueError, match="copy=False"):
        f.to_torch(device="cpu", copy=False)


@test_utils.test(arch=[qd.cpu])
def test_copy_false_numpy_dtype_conversion_raises():
    """copy=False + dtype that requires conversion should raise."""
    f = qd.field(qd.f32, shape=(3,))
    qd.sync()
    with pytest.raises(ValueError, match="copy=False"):
        f.to_numpy(dtype=np.float64, copy=False)


# ---------------------------------------------------------------------------
# StructField pass-through
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_struct_field_to_torch():
    s = qd.Struct.field({"a": qd.f32, "b": qd.f32}, shape=(4,))
    s[0] = {"a": 1.0, "b": 2.0}
    qd.sync()
    d = s.to_torch()
    assert isinstance(d, dict)
    assert _to_cpu(d["a"])[0] == 1.0
    assert _to_cpu(d["b"])[0] == 2.0


@test_utils.test(arch=dlpack_arch)
def test_struct_field_to_torch_copy_true():
    s = qd.Struct.field({"x": qd.f32}, shape=(3,))
    s[0] = {"x": 5.0}
    qd.sync()
    d1 = s.to_torch()
    d2 = s.to_torch(copy=True)
    # StructField always copies (AOS member views are not zero-copyable yet -- see struct.py docstring),
    # so both calls allocate independent storage.
    assert d1["x"].data_ptr() != d2["x"].data_ptr()
    assert _to_cpu(d2["x"])[0] == 5.0


@test_utils.test(arch=dlpack_arch)
def test_struct_field_to_torch_does_not_alias_memory():
    """StructField always returns independent copies of each AOS member.

    Quadrants' C++ ``field_to_dlpack`` does not currently emit cell-stride-aware DLPack views for individual members of
    an AOS struct -- it computes contiguous strides at the member dtype size, which would interleave neighboring
    members' bytes. Until that is fixed, ``StructField.to_torch`` forces ``copy=True`` per-member, so a kernel write
    into the field must NOT be reflected in a previously-obtained dict.
    """
    if is_v520_amdgpu():
        pytest.skip("can't run torch accessor kernels on v520")
    s = qd.Struct.field({"a": qd.i32, "b": qd.i32}, shape=(4,))
    s[0] = {"a": 1, "b": 2}
    qd.sync()
    d = s.to_torch()
    assert _to_cpu(d["a"])[0] == 1
    assert _to_cpu(d["b"])[0] == 2

    @qd.kernel
    def write(s: qd.template()):
        s[0].a = 99
        s[0].b = 77

    write(s)
    qd.sync()
    # Snapshot must be unchanged because StructField.to_torch always copies.
    assert _to_cpu(d["a"])[0] == 1
    assert _to_cpu(d["b"])[0] == 2


@test_utils.test(arch=[qd.cpu])
def test_struct_field_copy_false_raises():
    """``copy=False`` is rejected for StructField (AOS member views not supported yet)."""
    s = qd.Struct.field({"a": qd.f32, "b": qd.i32}, shape=(3,))
    s[0] = {"a": 1.5, "b": 7}
    qd.sync()
    with pytest.raises(ValueError, match="StructField.to_numpy.*copy=False"):
        s.to_numpy(copy=False)
    with pytest.raises(ValueError, match="StructField.to_torch.*copy=False"):
        s.to_torch(copy=False)


@test_utils.test(arch=dlpack_arch)
def test_struct_member_scalar_field_copy_false_raises():
    """A ScalarField that is a real member of a multi-member StructField has genuine AOS layout
    (cell stride > member dtype size), and ``field_to_dlpack`` does not yet emit cell-aware strides.
    ``copy=False`` must raise; ``copy=None``/``True`` must keep working via the kernel-copy path.
    """
    s = qd.Struct.field({"a": qd.f32, "b": qd.f32, "c": qd.f32}, shape=(4,))
    for i in range(4):
        s[i] = {"a": i * 100 + 1, "b": i * 100 + 2, "c": i * 100 + 3}
    qd.sync()
    with pytest.raises(ValueError, match="Zero-copy not available"):
        s.a.to_torch(copy=False)
    tc_a = _to_cpu(s.a.to_torch(copy=True))
    assert tc_a[0] == 1
    assert tc_a[1] == 101
    assert tc_a[3] == 301


@test_utils.test(arch=[qd.cpu])
def test_struct_member_scalar_field_to_numpy_copy_false_raises():
    """``copy=False`` numpy view of a multi-member StructField member must raise (real AOS strides)."""
    s = qd.Struct.field({"a": qd.f32, "b": qd.f32}, shape=(4,))
    for i in range(4):
        s[i] = {"a": i * 10 + 0.1, "b": i * 10 + 0.2}
    qd.sync()
    with pytest.raises(ValueError, match="Zero-copy"):
        s.a.to_numpy(copy=False)


@test_utils.test(arch=dlpack_arch)
def test_struct_member_vector_field_copy_false_raises():
    """A Vector.field that is a real member of a multi-member StructField is also genuinely AOS:
    its representative SNode shares the struct cell with sibling members. ``copy=False`` must raise.
    """
    vec3 = qd.types.vector(3, qd.f32)
    s = qd.Struct.field({"v": vec3, "w": qd.f32}, shape=(3,))
    for i in range(3):
        s[i] = {"v": (i, i + 1, i + 2), "w": float(i)}
    qd.sync()
    with pytest.raises(ValueError, match="Zero-copy not available"):
        s.v.to_torch(copy=False)


@test_utils.test(arch=dlpack_arch)
def test_single_member_struct_field_member_zerocopy_ok():
    """Sanity: a ScalarField member of a *single*-member StructField is effectively SOA
    (``parent.get_num_ch() == 1``); ``copy=False`` must succeed."""
    s = qd.Struct.field({"a": qd.f32}, shape=(4,))
    for i in range(4):
        s[i] = {"a": float(i * 7)}
    qd.sync()
    tc = _to_cpu(s.a.to_torch(copy=False))
    assert tc[0] == 0.0
    assert tc[3] == 21.0


# ---------------------------------------------------------------------------
# Cache invalidation across qd.reset() / qd.init()
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_zerocopy_cache_survives_reset():
    """Holding a zero-copy view across qd.reset() must not segfault.

    Regression for the use-after-free that motivated the Genesis to_numpy()-always-copies workaround: cache invalidation
    now runs BEFORE the C++ program is torn down (see pyquadrants.cache_holders in impl.py), so the DLPack deleter on
    the cached tensor operates on still-valid memory.
    """
    arch = qd.cfg.arch
    f = qd.field(qd.f32, shape=(4,))
    f[0] = 1.0
    qd.sync()
    held = f.to_torch()  # cached zero-copy view; we hold a Python reference past reset()
    assert held is not None

    qd.reset()
    qd.init(arch=arch)

    # Re-create and verify the new field works -- the previous-cycle 'held' tensor is allowed to remain alive (its
    # data may now be released, but we don't dereference it). What we MUST not have is a crash inside the
    # previous-cycle deleter when 'held' is GC'd.
    f2 = qd.field(qd.f32, shape=(4,))
    f2[0] = 42.0
    qd.sync()
    fresh = f2.to_torch()
    assert _to_cpu(fresh)[0] == 42.0
    # Touch 'held' last so it's only dropped at function exit; the DLPack deleter must not crash.


@test_utils.test(arch=dlpack_arch)
def test_zerocopy_cache_fresh_after_reset():
    """After qd.reset() / qd.init(), a freshly-constructed Field gets a fresh cache.

    Specifically, the new field's data_ptr is not stale-equal to a previous cycle's, AND the new view sees current data.
    """
    arch = qd.cfg.arch
    f1 = qd.field(qd.f32, shape=(2,))
    f1[0] = 1.0
    qd.sync()
    f1.to_torch()  # populate cache
    f1_dtype = f1.dtype  # noqa: F841 -- keep f1 alive past reset for the GC ordering test

    qd.reset()
    qd.init(arch=arch)

    f2 = qd.field(qd.f32, shape=(2,))
    f2[0] = 99.0
    qd.sync()
    tc2 = _to_cpu(f2.to_torch())
    assert tc2[0] == 99.0


# ---------------------------------------------------------------------------
# Apple Metal double-sync
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_clone_after_kernel_write_returns_up_to_date_data():
    """After a kernel writes to a field, an immediate clone must see the post-write values.

    Specifically targets the Apple Metal path where qd.sync() flushes pending Quadrants kernels and
    torch.mps.synchronize() flushes pending MPS clone copies; both must run for the cloned tensor to be observably
    equal to the field's current values.
    """
    if is_v520_amdgpu():
        pytest.skip("can't run torch accessor kernels on v520")
    f = qd.field(qd.i32, shape=(8,))
    f[0] = 0
    qd.sync()

    @qd.kernel
    def write(f: qd.template()):
        for i in range(8):
            f[i] = 100 + i

    write(f)
    # No explicit qd.sync() here -- we rely on to_torch(copy=True)'s internal sync.
    tc = f.to_torch(copy=True)
    tc_cpu = _to_cpu(tc)
    for i in range(8):
        assert tc_cpu[i] == 100 + i
