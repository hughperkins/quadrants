# pyright: reportPrivateImportUsage=false
# Reason: torch.zeros is public torch API, but pyright 1.1.409+ flags it as
# private because torch's stubs don't re-export it via __all__.
from functools import cached_property
from typing import TYPE_CHECKING, cast

import quadrants.lang
from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._logging import warn
from quadrants.lang import _interop, impl
from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.lang.util import (
    in_python_scope,
    python_scope,
    to_numpy_type,
    to_pytorch_type,
)

if TYPE_CHECKING:
    from quadrants.lang.expr import Expr


class Field:
    """Quadrants field class.

    A field is constructed by a list of field members.
    For example, a scalar field has 1 field member, while a 3x3 matrix field has 9 field members.
    A field member is a Python Expr wrapping a C++ FieldExpression.

    Args:
        vars (List[Expr]): Field members.
    """

    def __init__(self, _vars):
        assert all(_vars)
        self.vars = _vars
        self.host_accessors = None
        self.grad = None
        self.dual = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: DataTypeCxx | None = None
        self.__name: str | None = None

    # TODO: why do we have snode and _snode, that return the same thing?
    @property
    def snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return self._snode

    @property
    def _snode(self) -> "quadrants.lang.snode.SNode":  # type: ignore
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return quadrants.lang.snode.SNode(self.vars[0].ptr.snode())  # type: ignore

    @property
    def shape(self) -> tuple[int, ...]:
        if not self._shape:
            self._shape = cast(tuple[int, ...], self._snode.shape)
        return self._shape

    @property
    def dtype(self) -> DataTypeCxx:
        if not self._dtype:
            self._dtype = cast(DataTypeCxx, self._snode._dtype)
        return self._dtype

    @property
    def _name(self) -> str:
        if not self.__name:
            self.__name = cast(str, self._snode._name)
        return self.__name

    def parent(self, n=1):
        """
        n (int): the number of levels going up from the representative SNode.
        """
        return self.snode.parent(n)

    def _get_field_members(self) -> list["Expr"]:
        return self.vars

    def _loop_range(self):
        """Gets SNode of representative field member for loop range info.

        Returns:
            quadrants_python.SNode: SNode of representative (first) field member.
        """
        return self.vars[0].ptr.snode()

    def _set_grad(self, grad: "Field") -> None:
        """Sets corresponding grad field (reverse mode)."""
        self.grad = grad

    def _set_dual(self, dual: "Field") -> None:
        """Sets corresponding dual field (forward mode)."""
        self.dual = dual

    def _invalidate_zerocopy_cache(self) -> None:
        """Hook called by ``impl.reset()`` (via ``pyquadrants.cache_holders``) before C++ teardown.

        Subclasses that support zero-copy define ``_zerocopy_cache`` as a ``cached_property``; we read it via
        ``__dict__.get`` so invalidation never triggers the lazy init.
        """
        cache = self.__dict__.get("_zerocopy_cache")
        if cache is not None:
            cache.invalidate()

    @python_scope
    def fill(self, val: int | float) -> None:
        raise NotImplementedError()

    @python_scope
    def to_numpy(self, dtype: DataTypeCxx | None = None, *, copy: bool | None = None):
        """Converts `self` to a numpy array.

        Args:
            copy: Controls copying behaviour:

                - ``None`` (default) -- zero-copy when possible, copy otherwise.
                - ``True`` -- always return an independent copy.
                - ``False`` -- require zero-copy; raises if not possible.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        raise NotImplementedError()

    @python_scope
    def to_torch(self, device=None, *, copy: bool | None = None):
        """Converts `self` to a torch tensor.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
            copy: Controls copying behaviour:

                - ``None`` (default) -- zero-copy when possible, copy otherwise.
                - ``True`` -- always return an independent copy.
                - ``False`` -- require zero-copy; raises if not possible.

        Returns:
            torch.tensor: The result torch tensor.
        """
        raise NotImplementedError()

    @python_scope
    def from_numpy(self, arr):
        """Loads all elements from a numpy array.

        The shape of the numpy array needs to be the same as `self`.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        raise NotImplementedError()

    @python_scope
    def _from_external_arr(self, arr):
        raise NotImplementedError()

    @python_scope
    def from_torch(self, arr):
        """Loads all elements from a torch tensor.

        The shape of the torch tensor needs to be the same as `self`.

        Args:
            arr (torch.tensor): The source torch tensor.
        """
        self._from_external_arr(arr.contiguous())

    @python_scope
    def copy_from(self, other: "Field") -> None:
        """Copies all elements from another field.

        The shape of the other field needs to be the same as `self`.
        """
        if not isinstance(other, Field):
            raise TypeError("Cannot copy from a non-field object")
        if self.shape != other.shape:
            raise ValueError(f"qd.field shape {self.shape} does not match" f" the source field shape {other.shape}")
        from quadrants._kernels import tensor_to_tensor  # pylint: disable=C0415

        tensor_to_tensor(self, other)

    @python_scope
    def __setitem__(self, key: list[int] | int | None, value: int | float) -> None:
        raise NotImplementedError()

    @python_scope
    def __getitem__(self, key: list[int] | int | None) -> int | float:
        raise NotImplementedError()

    def __str__(self) -> str:
        if quadrants.lang.impl.inside_kernel():
            return self.__repr__()  # make pybind11 happy, see Matrix.__str__
        if self._snode.ptr is None:
            return "<Field: Definition of this field is incomplete>"
        return str(self.to_numpy())

    def _pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key,)

        if len(key) != len(self.shape):
            raise AssertionError("Slicing is not supported on qd.field")

        return key + ((0,) * (_qd_core.get_max_num_indices() - len(key)))  # type: ignore

    def _initialize_host_accessors(self):
        if self.host_accessors:
            return
        quadrants.lang.impl.get_runtime().materialize()
        self.host_accessors = [SNodeHostAccessor(e.ptr.snode()) for e in self.vars]

    def _host_access(self, key):
        return [SNodeHostAccess(e, key) for e in self.host_accessors]  # type: ignore

    def __iter__(self):
        raise NotImplementedError("Struct for is only available in Quadrants scope.")


class ScalarField(Field):
    """Quadrants scalar field with SNode implementation.

    Args:
        var (Expr): Field member.
    """

    def __init__(self, var):
        super().__init__([var])

    def to_dlpack(self):
        """
        Note: caller is responsible for calling qd.sync() between modifying the field, and
        reading it.
        """
        impl.get_runtime().materialize()
        return impl.get_runtime().prog.field_to_dlpack(self._snode.ptr, 0, 0, 0)

    @cached_property
    def _zerocopy_cache(self) -> _interop._ZerocopyCache | None:
        """Lazily-constructed DLPack cache. ``None`` when zero-copy is unsupported.

        Computed once per instance (see review feedback #17 on PR #450). Registers ``self`` in
        ``pyquadrants.cache_holders`` so the cache is invalidated on ``qd.reset()`` / ``qd.init()`` BEFORE C++ teardown.
        """
        # A ScalarField that is a member of a multi-member StructField has AOS layout: its parent SNode (the struct
        # cell) holds multiple `place` children (one for each sibling struct member), and consecutive elements of
        # the same member are sizeof(cell) bytes apart. The C++ field_to_dlpack does not emit those strides yet,
        # so zerocopy would produce an interleaved (broken) view. Skip it.
        # A standalone ScalarField has exactly one component, so ``parent.get_num_ch() == 1``; ``> 1`` therefore
        # uniquely identifies the multi-member-struct case. (The MatrixField version generalizes to ``> n*m``,
        # since a vec/mat field's own components are also siblings of its parent SNode.)
        parent_snode = self.parent()._snode.ptr
        is_aos_struct_member = parent_snode.get_num_ch() > 1
        return _interop.make_zerocopy_cache_if_supported(
            self,
            is_field=True,
            dtype=self.dtype,
            is_scalar_field=True,
            shape=self.shape,
            is_aos_struct_member=is_aos_struct_member,
        )

    def fill(self, val):
        """Fills this scalar field with a specified value."""
        if in_python_scope():
            from quadrants._kernels import fill_field  # pylint: disable=C0415

            fill_field(self, val)
        else:
            from quadrants._funcs import (  # pylint: disable=C0415
                field_fill_quadrants_scope,  # pylint: disable=C0415
            )

            field_fill_quadrants_scope(self, val)

    def to_numpy(self, dtype=None, *, copy=None, layout=None):
        """Converts this field to a `numpy.ndarray`.

        Args:
            dtype: Optional target numpy dtype. Incompatible with ``copy=False`` if it differs from the field's native
                dtype.
            copy: ``None`` (default) and ``True`` return an independent copy. ``False`` returns a zero-copy DLPack view
                (requires CPU backend and a supported dtype) or raises ``ValueError``. Note: zero-copy numpy arrays
                alias the field's underlying C++ runtime memory; callers opting into ``copy=False`` are responsible for
                the buffer lifetime.
            layout: Optional axis permutation tuple in ``np.transpose`` semantics (i-th element is the input axis at
                output position i). The returned array is the natural-layout view permuted by ``layout`` and cached
                per perm-key. ``None`` (default) returns the natural layout.
        """
        # Slot-hit fast path for explicit zero-copy (``copy=False``) -- placed BEFORE the ``@python_scope``
        # assertion (which lives on ``_to_numpy_slow`` below) to drop ~540 ns of decorator overhead per call.
        # Only fires for ``copy=False`` because ``copy=None`` is documented to return an independent copy
        # (asymmetric with ``to_torch`` semantics; see docstring above).
        if copy is False and dtype is None:
            cache = self._zerocopy_cache
            if cache is not None:
                if layout is None:
                    _np = cache._np
                    if _np is not None:
                        return _np
                else:
                    if (
                        cache._last_layout_np_view is not None
                        and (layout is cache._last_layout_np_layout or layout == cache._last_layout_np_layout)
                        and cache._last_layout_np_target_shape is None
                    ):
                        return cache._last_layout_np_view
        return self._to_numpy_slow(dtype=dtype, copy=copy, layout=layout)

    @python_scope
    def _to_numpy_slow(self, dtype=None, *, copy=None, layout=None):
        """Slow-path body of :meth:`to_numpy`."""
        if self.parent()._snode.ptr.type == _qd_core.SNodeType.dynamic:
            warn(
                "You are trying to convert a dynamic snode to a numpy array, be aware that inactive items in the snode will be converted to zeros in the resulting array."
            )
        np_dtype_target = None
        if dtype is not None:
            np_dtype_target = to_numpy_type(dtype) if isinstance(dtype, _qd_core.DataTypeCxx) else dtype

        if copy is False:
            return _interop.get_zerocopy_numpy(self, copy=False, dtype_target=np_dtype_target, layout=layout)
        # copy is None or True: try fast zerocopy+clone path, else kernel fallback.
        arr = _interop.get_zerocopy_numpy(self, copy=True, dtype_target=np_dtype_target, layout=layout)
        if arr is not None:
            return arr

        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        import numpy as np  # pylint: disable=C0415

        arr = np.zeros(shape=self.shape, dtype=dtype)  # type: ignore
        from quadrants._kernels import tensor_to_ext_arr  # pylint: disable=C0415

        tensor_to_ext_arr(self, arr)
        # TODO: can we remove .runtime_ops here?
        quadrants.lang.runtime_ops.sync()  # type: ignore
        if layout is not None:
            full = layout if len(layout) == arr.ndim else layout + tuple(range(len(layout), arr.ndim))
            arr = arr.transpose(full)
        return arr

    def to_torch(self, device=None, *, copy=None, layout=None):
        """Converts this field to a `torch.tensor`.

        Args:
            device: Optional torch device. Incompatible with ``copy=False`` if it differs from the field's native
                device.
            copy: ``None`` (default) prefers zero-copy, ``True`` forces an independent copy, ``False`` requires
                zero-copy or raises.
            layout: Optional axis permutation tuple in ``tensor.permute`` semantics (i-th element is the input axis at
                output position i). The returned tensor is the natural-layout view permuted by ``layout`` and cached
                per perm-key for hot-loop reuse. ``None`` (default) returns the natural layout.
        """
        # Slot-hit fast path -- intentionally placed BEFORE the ``@python_scope`` assertion (which lives on
        # ``_to_torch_slow`` below), to drop ~540 ns of decorator overhead per call on the cache-hit path.
        # The Quadrants-scope assertion still fires on cache miss / first call. Misuse from inside a
        # ``@quadrants.kernel`` body is caught the first time, after which the IR-generation context would be
        # signalled by the next op anyway. Rationale: hot-loop callers (franka_accessors-class workloads) hit
        # this path 30-50 times per simulation step; the assertion is redundant on every call.
        if (copy is None or copy is False) and device is None:
            cache = self._zerocopy_cache
            if cache is not None:
                if layout is None:
                    _tc = cache._tc
                    if _tc is not None:
                        return _tc
                else:
                    # Identity comparison on split slot key. See ``_ZerocopyCache._ensure_layout_torch``.
                    if (
                        cache._last_layout_tc_view is not None
                        and (layout is cache._last_layout_tc_layout or layout == cache._last_layout_tc_layout)
                        and cache._last_layout_tc_target_shape is None
                    ):
                        return cache._last_layout_tc_view
        return self._to_torch_slow(device=device, copy=copy, layout=layout)

    @python_scope
    def _to_torch_slow(self, device=None, *, copy=None, layout=None):
        """Slow-path body of :meth:`to_torch` -- runs the python_scope assertion + DLPack chain + kernel fallback."""
        tc = _interop.get_zerocopy_torch(self, copy=copy, device=device, layout=layout)
        if tc is not None:
            return tc

        import torch  # pylint: disable=C0415

        # pylint: disable=E1101
        arr = torch.zeros(size=self.shape, dtype=to_pytorch_type(self.dtype), device=device)
        from quadrants._kernels import tensor_to_ext_arr  # pylint: disable=C0415

        tensor_to_ext_arr(self, arr)
        # TODO: can we remove .runtime_ops here?
        quadrants.lang.runtime_ops.sync()  # type: ignore
        if layout is not None:
            full = layout if len(layout) == arr.ndim else layout + tuple(range(len(layout), arr.ndim))
            arr = arr.permute(*full)
        return arr

    @python_scope
    def _from_external_arr(self, arr):
        if len(self.shape) != len(arr.shape):
            raise ValueError(f"qd.field shape {self.shape} does not match" f" the numpy array shape {arr.shape}")
        for i, _ in enumerate(self.shape):
            if self.shape[i] != arr.shape[i]:
                raise ValueError(f"qd.field shape {self.shape} does not match" f" the numpy array shape {arr.shape}")
        from quadrants._kernels import ext_arr_to_tensor  # pylint: disable=C0415

        ext_arr_to_tensor(arr, self)
        # TODO: can we remove .runtime_ops here?
        quadrants.lang.runtime_ops.sync()  # type: ignore

    @python_scope
    def from_numpy(self, arr):
        """Copies the data from a `numpy.ndarray` into this field."""
        if not arr.flags.c_contiguous:
            import numpy as np  # pylint: disable=C0415

            arr = np.ascontiguousarray(arr)
        self._from_external_arr(arr)

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self.host_accessors[0].setter(value, *self._pad_key(key))  # type: ignore

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        # Check for potential slicing behaviour
        # for instance: x[0, :]
        padded_key = self._pad_key(key)
        import numpy as np  # pylint: disable=C0415

        for key in padded_key:
            if not isinstance(key, (int, np.integer)):
                raise TypeError(
                    f"Detected illegal element of type: {type(key)}. "
                    f"Please be aware that slicing a qd.field is not supported so far."
                )
        return self.host_accessors[0].getter(*padded_key)  # type: ignore

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return "<qd.field>"


class SNodeHostAccessor:
    def __init__(self, snode):
        if _qd_core.is_real(snode.data_type()):
            write_func = snode.write_float
            read_func = snode.read_float
        else:

            def write_func(key, value):
                if value >= 0:
                    snode.write_uint(key, value)
                else:
                    snode.write_int(key, value)

            if _qd_core.is_signed(snode.data_type()):
                read_func = snode.read_int
            else:
                read_func = snode.read_uint

        def getter(*key):
            assert len(key) == _qd_core.get_max_num_indices()
            return read_func(key)

        def setter(value, *key):
            assert len(key) == _qd_core.get_max_num_indices()
            write_func(key, value)
            # same as above
            if (
                impl.get_runtime().target_tape
                and impl.get_runtime().target_tape.grad_checker  # type: ignore
                and not impl.get_runtime().grad_replaced
            ):
                for x in impl.get_runtime().target_tape.grad_checker.to_check:  # type: ignore
                    assert snode != x.snode.ptr, "Overwritten is prohibitive when doing grad check."
                impl.get_runtime().target_tape.insert(write_func, (key, value))  # type: ignore

        self.getter = getter
        self.setter = setter


class SNodeHostAccess:
    def __init__(self, accessor, key):
        self.accessor = accessor
        self.key = key


class BitpackedFields:
    """Quadrants bitpacked fields, where fields with quantized types are packed together.

    Args:
        max_num_bits (int): Maximum number of bits all fields inside can occupy in total. Only 32 or 64 is allowed.
    """

    def __init__(self, max_num_bits):
        self.fields = []
        self.bit_struct_type_builder = _qd_core.BitStructTypeBuilder(max_num_bits)

    def place(self, *args, shared_exponent=False):
        """Places a list of fields with quantized types inside.

        Args:
            *args (List[Field]): A list of fields with quantized types to place.
            shared_exponent (bool): Whether the fields have a shared exponent.
        """
        if shared_exponent:
            self.bit_struct_type_builder.begin_placing_shared_exponent()
        count = 0
        for arg in args:
            assert isinstance(arg, Field)
            for var in arg._get_field_members():
                self.fields.append((var.ptr, self.bit_struct_type_builder.add_member(var.ptr.get_dt())))
                count += 1
        if shared_exponent:
            self.bit_struct_type_builder.end_placing_shared_exponent()
            if count <= 1:
                raise QuadrantsSyntaxError("At least 2 fields need to be placed when shared_exponent=True")


__all__ = ["BitpackedFields", "Field", "ScalarField"]
