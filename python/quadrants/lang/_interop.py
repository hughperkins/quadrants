"""Zero-copy tensor interop between Quadrants and PyTorch / NumPy via DLPack.

Provides cached, zero-copy conversion of Fields and Ndarrays to ``torch.Tensor`` and ``numpy.ndarray`` using
DLPack. Returned tensors / arrays alias Quadrants device memory directly -- modifications on either side are
visible on the other (after the appropriate sync; see :class:`_ZerocopyCache`).

Public API used by :mod:`quadrants.lang._ndarray`, :mod:`quadrants.lang.field`, :mod:`quadrants.lang.matrix`,
and :mod:`quadrants.lang.struct`:

* :func:`can_zerocopy` -- predicate (call once, cache result on the source instance).
* :class:`_ZerocopyCache` -- per-instance container for the torch + numpy DLPack views.
* :func:`make_zerocopy_cache_if_supported` -- constructor that registers the owner with the runtime
  so the cache is invalidated on ``qd.reset()`` / ``qd.init()`` BEFORE C++ teardown.
* :func:`get_zerocopy_torch`, :func:`get_zerocopy_numpy` -- thin entry points used by every per-class
  ``to_torch`` / ``to_numpy``. Implement the always-zerocopy-then-clone semantic and the Apple Metal
  double-sync (``qd.sync()`` on read, ``torch.mps.synchronize()`` after ``.clone()`` / ``.to()``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.types import primitive_types

if TYPE_CHECKING:
    import torch as _torch_mod  # for type hints only

# Optional torch import: numpy zero-copy works without torch (np.from_dlpack), so importing this
# module must not fail on torch-less environments (e.g. the Vulkan CI runner).
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


# Evaluate once at import. Reviewer feedback on PR #450: lru_cache(maxsize=1) on a zero-arg helper
# adds per-call overhead with no benefit; a module constant is cheaper and clearer.
_TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET = _compute_torch_mps_supports_dlpack_bytes_offset()


def current_arch_is_cpu() -> bool:
    """``True`` when the active Quadrants backend is a CPU backend (x64 or arm64)."""
    return impl.current_cfg().arch in _ARCH_CPU


def can_zerocopy(is_field: bool, dtype=None, is_scalar_field: bool = False, shape: tuple[int, ...] = ()) -> bool:
    """Check whether zero-copy DLPack export is available for the current backend and data type.

    This is intended to be called **once** at the source instance's construction (or first access)
    and cached on the instance, not re-evaluated on every ``to_torch`` / ``to_numpy`` call.

    Args:
        is_field: ``True`` for SNode-backed Fields, ``False`` for Ndarrays.
        dtype: The Quadrants dtype (e.g. ``qd.f32``). Types not in the C++ DLPack whitelist (f16, u8, ...)
            return False.
        is_scalar_field: ``True`` when the source is a ``ScalarField`` (0-dim DLPack edge-case).
        shape: Batch shape of the field/ndarray.

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
    return True


class _DLPackV1Adapter:
    """Wraps a DLPack v0 PyCapsule into a v1-compatible object.

    Quadrants' C++ ``field_to_dlpack`` / ``ndarray_to_dlpack`` return raw PyCapsules (the v0 DLPack
    protocol). Modern NumPy (>= 1.23 with strict checks; mandatory in NumPy 2.x) requires the v1
    protocol -- an object exposing ``__dlpack__`` and ``__dlpack_device__``. This adapter is a
    thin Python-side bridge so we don't have to touch the C++ DLPack export.

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

    Holds two independent slots:

    * ``_tc``: ``torch.Tensor`` filled via ``torch.utils.dlpack.from_dlpack`` on first access.
    * ``_np``: ``numpy.ndarray`` filled via ``numpy.from_dlpack`` on first access (CPU only).
      Independent of torch: numpy-only workloads never trigger a torch import.

    Each slot is filled lazily and may be ``None`` if not yet requested.

    Lifetime: the cache is invalidated by :func:`PyQuadrants.reset` (via ``cache_holders``) BEFORE the
    C++ program is torn down, so the DLPack deleters run while the underlying memory is still valid.
    """

    __slots__ = ("_tc", "_np")

    def __init__(self):
        self._tc = None
        self._np = None

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

    def invalidate(self) -> None:
        """Drop both cached views. Idempotent."""
        self._tc = None
        self._np = None


def make_zerocopy_cache_if_supported(
    owner,
    *,
    is_field: bool,
    dtype,
    is_scalar_field: bool = False,
    shape: tuple[int, ...] = (),
) -> _ZerocopyCache | None:
    """Construct a ``_ZerocopyCache`` for ``owner`` if zero-copy is supported, else return ``None``.

    Also registers ``owner`` with ``pyquadrants.cache_holders`` so its cache is invalidated on
    ``qd.reset()`` / ``qd.init()`` BEFORE C++ teardown. The owner must define a method
    ``_invalidate_zerocopy_cache(self)`` that calls :meth:`_ZerocopyCache.invalidate`.

    Intended to be called at instance construction (or once via ``cached_property``).
    """
    if not can_zerocopy(is_field=is_field, dtype=dtype, is_scalar_field=is_scalar_field, shape=shape):
        return None
    cache = _ZerocopyCache()
    impl.get_runtime().cache_holders.add(owner)
    return cache


def _metal_sync_runtime() -> None:
    """Quadrants -> MPS sync. Required so the DLPack-backed MPS tensor sees pending kernel writes."""
    if impl.current_cfg().arch == _ARCH_METAL:
        impl.get_runtime().sync()


def _metal_sync_torch() -> None:
    """MPS -> next operation sync. Required after ``.clone()`` / ``.to()`` so the resulting torch tensor
    has actually finished copying before the next user op (or the next Quadrants kernel) sees it."""
    if _HAS_TORCH and impl.current_cfg().arch == _ARCH_METAL:
        _torch.mps.synchronize()


def get_zerocopy_torch(
    owner,
    *,
    copy: bool | None,
    device=None,
) -> "_torch_mod.Tensor | None":
    """Zero-copy entry point for ``to_torch``.

    Returns the cached zero-copy view, optionally cloned and / or moved to ``device``. The
    ``copy`` argument selects between view (``False`` / ``None``) and independent buffer (``True``);
    crucially, the zerocopy export is taken **even when ``copy=True``** and the result is then
    cloned -- DLPack export is cheaper than the kernel-copy fallback path even when followed by a
    full clone.

    Args:
        owner: A Field or Ndarray with a ``_zerocopy_cache: _ZerocopyCache | None`` attribute and a
            ``to_dlpack()`` method.
        copy: ``None`` -> view; ``False`` -> view (raises if zerocopy unsupported); ``True`` -> clone.
        device: Optional torch device. If different from the view's device, performs a device transfer
            (incompatible with ``copy=False``).

    Returns:
        The torch tensor, or ``None`` when ``owner._zerocopy_cache is None`` (i.e. zero-copy is not
        supported for this instance) and ``copy is not False`` -- the caller should fall back to its
        kernel-copy path.

    Raises:
        ValueError: when ``copy=False`` but zerocopy is unsupported for this instance, or when a
            device transfer is required but ``copy=False``.
    """
    cache: _ZerocopyCache | None = owner._zerocopy_cache
    if cache is None:
        if copy is False:
            raise ValueError("Zero-copy not available for this backend / dtype combination")
        return None

    _metal_sync_runtime()
    tc = cache._ensure_torch(owner)

    needs_device_transfer = device is not None and tc.device != _torch.device(device)
    if needs_device_transfer:
        if copy is False:
            raise ValueError(
                f"copy=False is incompatible with device transfer (data on {tc.device}, requested {device})"
            )
        out = tc.to(device)
        _metal_sync_torch()
        return out

    if copy is True:
        out = tc.clone()
        _metal_sync_torch()
        return out
    return tc


def get_zerocopy_numpy(
    owner,
    *,
    copy: bool | None,
    dtype_target=None,
) -> np.ndarray | None:
    """Zero-copy entry point for ``to_numpy``.

    Numpy zero-copy is available only on CPU backends (numpy arrays cannot reference GPU memory).
    Uses ``numpy.from_dlpack`` directly -- no torch import required.

    Args:
        owner: A Field or Ndarray with a ``_zerocopy_cache: _ZerocopyCache | None`` attribute and a
            ``to_dlpack()`` method.
        copy: ``None`` -> view; ``False`` -> view (raises if zerocopy unsupported); ``True`` -> copy.
        dtype_target: Optional numpy dtype. If different from the view's dtype, performs ``.astype()``
            (incompatible with ``copy=False``).

    Returns:
        The numpy array, or ``None`` when zerocopy is unsupported for this instance and
        ``copy is not False`` -- the caller should fall back to its kernel-copy path.

    Raises:
        ValueError: when ``copy=False`` but zerocopy is unsupported, or when a dtype conversion is
            required but ``copy=False``.
    """
    cache: _ZerocopyCache | None = owner._zerocopy_cache
    if cache is None or not current_arch_is_cpu():
        if copy is False:
            raise ValueError("Zero-copy numpy unavailable (requires a CPU backend and a supported dtype)")
        return None

    arr = cache._ensure_numpy(owner)

    if dtype_target is not None and arr.dtype != dtype_target:
        if copy is False:
            raise ValueError("copy=False is incompatible with dtype conversion")
        return arr.astype(dtype_target)

    if copy is True:
        return arr.copy()
    return arr
