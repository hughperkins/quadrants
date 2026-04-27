"""Zero-copy tensor interop between Quadrants and PyTorch / NumPy via DLPack.

Provides cached, zero-copy conversion of Fields and Ndarrays to ``torch.Tensor`` and ``numpy.ndarray`` using DLPack.
Returned tensors / arrays alias Quadrants device memory directly -- modifications on either side are visible on the
other (after the appropriate sync; see :class:`_ZerocopyCache`).

Public API used by :mod:`quadrants.lang._ndarray`, :mod:`quadrants.lang.field`, :mod:`quadrants.lang.matrix`, and
:mod:`quadrants.lang.struct`:

* :func:`can_zerocopy` -- predicate (call once, cache result on the source instance).
* :class:`_ZerocopyCache` -- per-instance container for the torch + numpy DLPack views.
* :func:`make_zerocopy_cache_if_supported` -- constructor that registers the owner with the runtime so the cache is
  invalidated on ``qd.reset()`` / ``qd.init()`` BEFORE C++ teardown.
* :func:`get_zerocopy_torch`, :func:`get_zerocopy_numpy` -- thin entry points used by every per-class ``to_torch`` /
  ``to_numpy``. Implement the always-zerocopy-then-clone semantic and the Apple Metal double-sync (``qd.sync()`` on
  read, ``torch.mps.synchronize()`` after ``.clone()`` / ``.to()``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.types import primitive_types

if TYPE_CHECKING:
    import torch as _torch_mod  # for type hints only

# Optional torch import: numpy zero-copy works without torch (np.from_dlpack), so importing this module must not fail
# on torch-less environments (e.g. the Vulkan CI runner).
try:
    import torch as _torch
    from torch.utils.dlpack import from_dlpack as _torch_from_dlpack

    _HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _torch_from_dlpack = None  # type: ignore[assignment]
    _HAS_TORCH = False


_ARCH_METAL = _qd_core.Arch.metal
_ARCH_VULKAN = _qd_core.Arch.vulkan
_ARCH_CPU = frozenset({_qd_core.Arch.x64, _qd_core.Arch.arm64})

_DLPACK_SUPPORTED_DTYPES = frozenset(
    {primitive_types.i32, primitive_types.i64, primitive_types.f32, primitive_types.f64, primitive_types.u1}
)


def _compute_torch_mps_supports_dlpack_bytes_offset() -> bool:
    """Required for zero-copy export of Fields on Apple Metal.

    Available since torch > 2.9.1 (see pytorch/pytorch#168193).
    """
    if not _HAS_TORCH:
        return False
    parts = _torch.__version__.replace("+", ".").split(".")[:3]
    try:
        return tuple(map(int, parts)) > (2, 9, 1)
    except ValueError:
        return False


# Evaluate once at import. Reviewer feedback on PR #450: lru_cache(maxsize=1) on a zero-arg helper adds per-call
# overhead with no benefit; a module constant is cheaper and clearer.
_TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET = _compute_torch_mps_supports_dlpack_bytes_offset()


def current_arch_is_cpu() -> bool:
    """``True`` when the active Quadrants backend is a CPU backend (x64 or arm64)."""
    return impl.current_cfg().arch in _ARCH_CPU


def can_zerocopy(
    is_field: bool,
    dtype=None,
    is_scalar_field: bool = False,
    shape: tuple[int, ...] = (),
    is_aos_struct_member: bool = False,
) -> bool:
    """Check whether zero-copy DLPack export is available for the current backend and data type.

    This is intended to be called **once** at the source instance's construction (or first access) and cached on the
    instance, not re-evaluated on every ``to_torch`` / ``to_numpy`` call.

    Args:
        is_field: ``True`` for SNode-backed Fields, ``False`` for Ndarrays.
        dtype: The Quadrants dtype (e.g. ``qd.f32``). Types not in the C++ DLPack whitelist (f16, u8, ...) return False.
        is_scalar_field: ``True`` when the source is a ``ScalarField`` (0-dim DLPack edge-case).
        shape: Batch shape of the field/ndarray.
        is_aos_struct_member: ``True`` when the field is a member of a multi-member ``StructField``. Quadrants' C++
            ``field_to_dlpack`` does not currently emit cell-stride-aware DLPack views for AOS struct members (it
            computes contiguous strides at the member dtype size, but the actual stride between consecutive elements of
            the same member is ``sizeof(cell)``), so the zerocopy view would interleave neighboring members' bytes.
            Force kernel-copy until the C++ export is fixed.

    Returns:
        ``True`` if zero-copy via DLPack is supported.
    """
    if dtype is not None and dtype not in _DLPACK_SUPPORTED_DTYPES:
        return False
    arch = impl.current_cfg().arch
    if arch == _ARCH_VULKAN:
        return False
    if is_field:
        if arch == _ARCH_METAL and not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
            return False
        # 0-dim ScalarFields lack DLPack bytes_offset support in current PyTorch.
        if is_scalar_field and not shape:
            return False
        if is_aos_struct_member:
            return False
    return True


class _DLPackV1Adapter:
    """Wraps a DLPack v0 PyCapsule into a v1-compatible object.

    Quadrants' C++ ``field_to_dlpack`` / ``ndarray_to_dlpack`` return raw PyCapsules (the v0 DLPack protocol). Modern
    NumPy (>= 1.23 with strict checks; mandatory in NumPy 2.x) requires the v1 protocol -- an object exposing
    ``__dlpack__`` and ``__dlpack_device__``. This adapter is a thin Python-side bridge so we don't have to touch the
    C++ DLPack export.

    CPU-only by construction; numpy zero-copy is gated on ``current_arch_is_cpu()`` upstream.
    """

    __slots__ = ("_capsule",)
    _KDLCPU = 1  # DLDeviceType::kDLCPU

    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None):
        return self._capsule

    def __dlpack_device__(self):
        return (self._KDLCPU, 0)


class _ZerocopyCache:
    """Per-instance cache of DLPack-backed views into a Quadrants Field / Ndarray.

    Holds two independent natural-layout slots:

    * ``_tc``: ``torch.Tensor`` filled via ``torch.utils.dlpack.from_dlpack`` on first access.
    * ``_np``: ``numpy.ndarray`` filled via ``numpy.from_dlpack`` on first access (CPU only). Independent of torch:
      numpy-only workloads never trigger a torch import.

    Plus per-layout caches, keyed on ``(layout_perm_tuple, target_shape)``:

    * ``_layout_tc`` / ``_layout_np``: dicts of permuted views. The build function does
      ``natural.reshape(target_shape).permute(*layout)`` once on miss and caches the result; subsequent calls return
      the cached tensor directly.
    * ``_last_layout_*_key`` / ``_last_layout_*_view``: a single-slot fast path that bypasses the dict on the common
      case where a callsite repeatedly asks for the same ``(layout, target_shape)``. One tuple-equality check + a
      cached return on the hit path (~30-50 ns), no dict hashing.

    All slots are filled lazily and may be ``None`` if not yet requested.

    Lifetime: the cache is invalidated by :func:`PyQuadrants.reset` (via ``cache_holders``) BEFORE the C++ program is
    torn down, so the DLPack deleters run while the underlying memory is still valid. Layout views share storage with
    ``_tc`` / ``_np`` so they're invalidated together.
    """

    __slots__ = (
        "_tc",
        "_np",
        "_layout_tc",
        "_layout_np",
        "_last_layout_tc_key",
        "_last_layout_tc_view",
        "_last_layout_np_key",
        "_last_layout_np_view",
    )

    def __init__(self):
        self._tc = None
        self._np = None
        self._layout_tc: dict = {}
        self._layout_np: dict = {}
        self._last_layout_tc_key: tuple | None = None
        self._last_layout_tc_view = None
        self._last_layout_np_key: tuple | None = None
        self._last_layout_np_view = None

    def _ensure_torch(self, owner) -> "_torch_mod.Tensor":
        if not _HAS_TORCH:
            raise RuntimeError("torch is not installed; zero-copy to_torch is unavailable")
        if self._tc is None:
            self._tc = _torch_from_dlpack(owner.to_dlpack())
        return self._tc

    def _ensure_numpy(self, owner) -> np.ndarray:
        if self._np is None:
            self._np = np.from_dlpack(_DLPackV1Adapter(owner.to_dlpack()))
        return self._np

    def _ensure_layout_torch(
        self,
        owner,
        layout: tuple[int, ...],
        target_shape: tuple[int, ...] | None,
    ) -> "_torch_mod.Tensor":
        """Return a layout-permuted (and optionally reshaped) torch view, cached.

        ``layout`` follows ``tensor.permute`` semantics: the i-th element is the input axis that ends up at output
        position i. ``target_shape``, when not ``None``, is applied as a ``.reshape(...)`` of the natural ``_tc`` view
        BEFORE the permute (this is how :class:`MatrixField` flattens / unflattens matrix dims into the batch shape).
        """
        key = (layout, target_shape)
        if key == self._last_layout_tc_key:
            return self._last_layout_tc_view
        view = self._layout_tc.get(key)
        if view is None:
            natural = self._ensure_torch(owner)
            base = natural.reshape(target_shape) if target_shape is not None and natural.shape != target_shape else natural
            # Pad layout with identity for trailing axes so callers can pass a permutation over the leading
            # batch dims and let matrix / vector dims pass through (matches ``tensor.movedim(layout, range(...))``).
            full_layout = layout if len(layout) == base.ndim else layout + tuple(range(len(layout), base.ndim))
            view = base.permute(*full_layout)
            self._layout_tc[key] = view
        self._last_layout_tc_key = key
        self._last_layout_tc_view = view
        return view

    def _ensure_layout_numpy(
        self,
        owner,
        layout: tuple[int, ...],
        target_shape: tuple[int, ...] | None,
    ) -> np.ndarray:
        """Return a layout-permuted (and optionally reshaped) numpy view, cached. See ``_ensure_layout_torch``."""
        key = (layout, target_shape)
        if key == self._last_layout_np_key:
            return self._last_layout_np_view
        view = self._layout_np.get(key)
        if view is None:
            natural = self._ensure_numpy(owner)
            base = natural.reshape(target_shape) if target_shape is not None and natural.shape != target_shape else natural
            full_layout = layout if len(layout) == base.ndim else layout + tuple(range(len(layout), base.ndim))
            view = base.transpose(full_layout)
            self._layout_np[key] = view
        self._last_layout_np_key = key
        self._last_layout_np_view = view
        return view

    def invalidate(self) -> None:
        """Drop all cached views. Idempotent."""
        self._tc = None
        self._np = None
        self._layout_tc.clear()
        self._layout_np.clear()
        self._last_layout_tc_key = None
        self._last_layout_tc_view = None
        self._last_layout_np_key = None
        self._last_layout_np_view = None


def make_zerocopy_cache_if_supported(
    owner,
    *,
    is_field: bool,
    dtype,
    is_scalar_field: bool = False,
    shape: tuple[int, ...] = (),
    is_aos_struct_member: bool = False,
) -> _ZerocopyCache | None:
    """Construct a ``_ZerocopyCache`` for ``owner`` if zero-copy is supported, else return ``None``.

    Also registers ``owner`` with ``pyquadrants.cache_holders`` so its cache is invalidated on ``qd.reset()`` /
    ``qd.init()`` BEFORE C++ teardown. The owner must define a method ``_invalidate_zerocopy_cache(self)`` that calls
    :meth:`_ZerocopyCache.invalidate`.

    Intended to be called at instance construction (or once via ``cached_property``).
    """
    if not can_zerocopy(
        is_field=is_field,
        dtype=dtype,
        is_scalar_field=is_scalar_field,
        shape=shape,
        is_aos_struct_member=is_aos_struct_member,
    ):
        return None
    cache = _ZerocopyCache()
    impl.get_runtime().cache_holders.add(owner)
    return cache


# Per-runtime cached result of ``impl.current_cfg().arch == _ARCH_METAL``. Looking this up via pybind11 (~1us per call)
# on every ``to_torch`` / ``to_numpy`` invocation dominates per-call overhead on CPU-backed hot loops where the metal
# sync is a no-op anyway. We cache the bool keyed on ``id(impl.pyquadrants)`` so the value is automatically refreshed
# when ``impl.reset()`` swaps the runtime singleton (the only legitimate way the arch changes mid-process).
_METAL_ARCH_CACHE: tuple[int, bool] | None = None


def _is_metal_arch() -> bool:
    global _METAL_ARCH_CACHE
    rt_id = id(impl.pyquadrants)
    cache = _METAL_ARCH_CACHE
    if cache is not None and cache[0] == rt_id:
        return cache[1]
    is_metal = impl.current_cfg().arch == _ARCH_METAL
    _METAL_ARCH_CACHE = (rt_id, is_metal)
    return is_metal


def _metal_sync_runtime() -> None:
    """Quadrants -> MPS sync. Required so the DLPack-backed MPS tensor sees pending kernel writes."""
    if _is_metal_arch():
        impl.get_runtime().sync()


def _metal_sync_torch() -> None:
    """MPS -> next operation sync. Required after ``.clone()`` / ``.to()`` so the resulting torch tensor
    has actually finished copying before the next user op (or the next Quadrants kernel) sees it."""
    if _HAS_TORCH and _is_metal_arch():
        _torch.mps.synchronize()


def get_zerocopy_torch(
    owner,
    *,
    copy: bool | None,
    device=None,
    layout: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
) -> "_torch_mod.Tensor | None":
    """Zero-copy entry point for ``to_torch``.

    Returns the cached zero-copy view, optionally reshaped, permuted, cloned, and / or moved to ``device``. The
    ``copy`` argument selects between view (``False`` / ``None``) and independent buffer (``True``); crucially, the
    zerocopy export is taken **even when ``copy=True``** and the result is then cloned -- DLPack export is cheaper
    than the kernel-copy fallback path even when followed by a full clone.

    Args:
        owner: A Field or Ndarray with a ``_zerocopy_cache: _ZerocopyCache | None`` attribute and a ``to_dlpack()``
            method.
        copy: ``None`` -> view; ``False`` -> view (raises if zerocopy unsupported); ``True`` -> clone.
        device: Optional torch device. If different from the view's device, performs a device transfer (incompatible
            with ``copy=False``).
        layout: Optional axis permutation tuple in ``tensor.permute`` semantics (i-th element is the input axis at
            output position i). When set, the returned view is the natural-layout view permuted by ``layout`` and
            cached separately for repeated calls.
        target_shape: Optional reshape applied to the natural-layout view before permuting / returning. Used by
            :class:`MatrixField` to flatten matrix dims into the batch shape. The cache keys layout views on
            ``(layout, target_shape)``, so different shapes get different cache slots.

    Returns:
        The torch tensor, or ``None`` when ``owner._zerocopy_cache is None`` (i.e. zero-copy is not supported for this
        instance) and ``copy is not False`` -- the caller should fall back to its kernel-copy path.

    Raises:
        ValueError: when ``copy=False`` but zerocopy is unsupported for this instance, or when a device transfer is
            required but ``copy=False``.
    """
    cache: _ZerocopyCache | None = owner._zerocopy_cache
    if cache is None:
        if copy is False:
            raise ValueError(
                f"Zero-copy not available for arch={impl.current_cfg().arch.name}, "
                f"dtype={getattr(owner, 'dtype', '?')}"
            )
        return None

    # Inlined ``_metal_sync_runtime()`` body. The wrapper-call frame shows up as ~0.3 us / call in cProfile on
    # franka_accessors-class workloads (43 to_torch calls / step); inlining elides it on non-metal where the body is a
    # no-op. Correctness across mid-process ``qd.init(arch=...)`` switches is preserved by ``_is_metal_arch`` keying its
    # cache on ``id(impl.pyquadrants)``, which rotates on ``impl.reset()``.
    if _is_metal_arch():
        impl.get_runtime().sync()

    if layout is not None:
        tc = cache._ensure_layout_torch(owner, layout, target_shape)
    else:
        tc = cache._ensure_torch(owner)
        if target_shape is not None and tc.shape != target_shape:
            tc = tc.reshape(target_shape)

    needs_device_transfer = device is not None and tc.device != _torch.device(device)
    if needs_device_transfer:
        if copy is False:
            raise ValueError(
                f"copy=False is incompatible with device transfer (data on {tc.device}, requested {device})"
            )
        out = tc.to(device)
        # Inlined ``_metal_sync_torch()`` body (``_HAS_TORCH`` is implied true: ``_ensure_*_torch`` above would have
        # raised otherwise).
        if _is_metal_arch():
            _torch.mps.synchronize()
        return out

    if copy is True:
        out = tc.clone()
        if _is_metal_arch():
            _torch.mps.synchronize()
        return out
    return tc


def get_zerocopy_numpy(
    owner,
    *,
    copy: bool | None,
    dtype_target=None,
    layout: tuple[int, ...] | None = None,
    target_shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    """Zero-copy entry point for ``to_numpy``.

    Numpy zero-copy is available only on CPU backends (numpy arrays cannot reference GPU memory).
    Uses ``numpy.from_dlpack`` directly -- no torch import required.

    Args:
        owner: A Field or Ndarray with a ``_zerocopy_cache: _ZerocopyCache | None`` attribute and a ``to_dlpack()``
            method.
        copy: ``None`` -> view; ``False`` -> view (raises if zerocopy unsupported); ``True`` -> copy.
        dtype_target: Optional numpy dtype. If different from the view's dtype, performs ``.astype()`` (incompatible
            with ``copy=False``).
        layout: Optional axis permutation tuple in ``np.transpose`` semantics (i-th element is the input axis at
            output position i). Cached per ``(layout, target_shape)``.
        target_shape: Optional reshape applied to the natural-layout array before permuting / returning. See
            :func:`get_zerocopy_torch`.

    Returns:
        The numpy array, or ``None`` when zerocopy is unsupported for this instance and ``copy is not False`` -- the
        caller should fall back to its kernel-copy path.

    Raises:
        ValueError: when ``copy=False`` but zerocopy is unsupported, or when a dtype conversion is required but
            ``copy=False``.
    """
    cache: _ZerocopyCache | None = owner._zerocopy_cache
    if cache is None or not current_arch_is_cpu():
        if copy is False:
            raise ValueError(
                f"Zero-copy numpy unavailable for arch={impl.current_cfg().arch.name}, "
                f"dtype={getattr(owner, 'dtype', '?')} (numpy zero-copy requires a CPU backend)"
            )
        return None

    if layout is not None:
        arr = cache._ensure_layout_numpy(owner, layout, target_shape)
    else:
        arr = cache._ensure_numpy(owner)
        if target_shape is not None and arr.shape != target_shape:
            arr = arr.reshape(target_shape)

    if dtype_target is not None and arr.dtype != dtype_target:
        if copy is False:
            raise ValueError("copy=False is incompatible with dtype conversion")
        return arr.astype(dtype_target)

    if copy is True:
        return arr.copy()
    return arr
