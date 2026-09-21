from __future__ import annotations

from collections.abc import Iterator
from multiprocessing import cpu_count
from pathlib import Path
from re import findall
from types import EllipsisType
from typing import TypeAlias
from warnings import warn

import h5py
import numpy as np
import numpy.typing as npt
from prefetch_generator import prefetch
from tifffile import PHOTOMETRIC, TiffFile, TiffFileError, TiffPage, imread

from ._dcimg import DCIMGFile

CPU_COUNT = cpu_count()

Items: TypeAlias = (
    int | slice | list | tuple | np.ndarray | np.integer | EllipsisType
)
PathTypes: TypeAlias = (
    str | Path | list[str] | list[Path] | npt.NDArray[str] | npt.NDArray[Path]
)

__all__ = ["Stack", "iter_chunks", "lazystack"]


def _format_info(obj, where: str) -> str:
    """Return a one-line summary of a stack or view."""
    image_nbytes_mb = 1e-6 * obj.image_nbytes
    nbytes_mb = 1e-6 * obj.nbytes
    return (
        f"{type(obj).__name__} object referencing {obj.shape[0]} "
        f"{obj.dtype} images of shape {obj.shape[1:]}. Each image "
        f"occupies {image_nbytes_mb:.2f} MB {where}, totalling "
        f"{nbytes_mb:.2f} MB."
    )


class Stack:
    """
    Base class for lazy image stack readers.

    Indexing with an integer returns a materialised 2D array. Indexing with a
    slice, list, or tuple returns a lazy ``View``. Use within a ``with`` block
    or close explicitly with ``close()``.

    Subclasses support a new file format by setting the required attributes
    and overriding the two read methods below. Everything else -- lazy
    slicing and indexing, ``asarray()``, and the context manager -- is
    inherited and format-agnostic, and the module-level ``iter_chunks()``
    works on any stack or view. See "Adding a format" in the README for a
    worked example.

    Required attributes:
        shape (tuple): Dimensions as (num_images, height, width).
        dtype (npt.DTypeLike): Data type of each image.
        image_nbytes (int): Bytes of a single image.
        nbytes (int): Total bytes of the stack.

    Required methods:
        _get_image(index): Return one image as a materialised 2D NumPy array.
        _get_images(indices): Return the requested images as a materialised
            3D NumPy array, ordered as given.

    Optional:
        _file: An open handle. If set, ``close()`` closes it and clears the
            attribute; the context manager calls ``close()`` on exit.

    Attributes:
        info (str): One-line summary of the stack (same as ``str(stack)``).
        ndim (int): Number of dimensions.
        size (int): Number of elements in the stack.
        itemsize (int): Length of one element in bytes.
    """

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, items: Items) -> npt.NDArray | View:
        return _init_view(self, items)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def __str__(self):
        """
        Return information about the current instance, including shape and disk
        usage of each image and the image stack.
        """
        return _format_info(self, "on disk")

    def asarray(self, dtype=None) -> npt.NDArray:
        """Return the stack's data as a materialised NumPy array."""
        return np.asarray(
            self._get_images(np.arange(self.shape[0])),
            dtype=dtype,
        )

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        if copy is False:
            raise ValueError("copy=False is not supported.")
        return self.asarray(dtype)

    def close(self):
        file_open = getattr(self, "_file", None)
        if file_open:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()

    @property
    def info(self):
        return str(self)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def itemsize(self):
        return self.dtype.itemsize


def iter_chunks(
    obj: npt.NDArray | View,
    chunk_size_gb: float = 5,
    num_prefetch: int = 1,
    axis: int = 0,
    step: int = 1,
) -> Iterator[tuple[npt.NDArray, int, int]]:
    """
    Yield successive chunks along an axis of a 3D NumPy-like array
    (or Stack/View) with prefetching.

    Args:
        obj: Array or view to be chunked.
        chunk_size_gb: Approximate chunk size allowance in GB.
        num_prefetch: Number of chunks to prefetch. Chunk size will be
            scaled accordingly so as not to exceed `chunk_size_gb`.
        axis: Axis to chunk over.
        step: Spacing along the chunk axis.

    Yields:
        tuple: (chunk, start, stop) where ``chunk`` is the chunked image data
        as a 3D array, ``start`` is the inclusive start index of the chunk,
        and ``stop`` is the exclusive stop index.
    """
    if step < 1:
        raise ValueError(f"`step` must be at least one, but got {step=}.")
    if obj.ndim != 3:
        raise ValueError(f"Only 3D arrays are supported, but got {obj.shape=}")
    if axis not in (0, 1, 2):
        raise ValueError(f"`axis` must be 0, 1, or 2, but got {axis}.")
    if num_prefetch < 1:
        raise ValueError(
            f"`num_prefetch` must be at least 1, but got {num_prefetch}."
        )

    # Calculate how much memory one item occupies along the batch axis.
    image_nbytes = obj.nbytes // obj.shape[axis]

    # Calculate how many items can fit into chunk allowance, ensuring this is
    #   never zero.
    chunk_size = (
        np.maximum(
            int(chunk_size_gb // (1e-9 * image_nbytes * (num_prefetch + 1))),
            1,
        )
        * step
    )

    slices = [slice(None)] * 3

    @prefetch(max_prefetch=num_prefetch)
    def _generator() -> Iterator[tuple[npt.NDArray, int, int]]:
        for start in range(0, obj.shape[axis], chunk_size):
            stop = np.minimum(obj.shape[axis], start + chunk_size)
            slices[axis] = slice(start, stop, step)

            indices = slices[axis] if axis == 0 else tuple(slices)

            # View is materialised by `np.asarray`.
            yield (
                np.asarray(obj[indices]),
                start,
                stop,
            )

    return _generator()


def _is_multichannel(page: TiffPage) -> bool:
    return (
        page.photometric
        not in (PHOTOMETRIC.MINISBLACK, PHOTOMETRIC.MINISWHITE)
        and page.photometric is not None
    ) or page.samplesperpixel != 1


def _to_indices(
    items: list | npt.NDArray, shape: tuple
) -> npt.NDArray[np.integer]:
    """
    Converts boolean mask to indices if necessary and returns a NumPy array.
    """
    items = np.asarray(items)

    if items.dtype == bool and len(items) != shape[0]:
        raise ValueError(
            "Boolean mask must have the same length as base object."
        )

    items = np.nonzero(items)[0] if items.dtype == bool else items

    if not np.issubdtype(items.dtype, np.integer):
        raise TypeError("Only integer or boolean indexing is supported.")

    return items


def _reject_bool(items: Items):
    if isinstance(items, bool | np.bool_):
        raise TypeError(
            "Single boolean indexing is invalid. Use a boolean mask with the "
            "same length as the base object."
        )


def _reject_newaxis(items: Items):
    if items is None:
        raise NotImplementedError(
            "None/newaxis indexing is not supported on lazystacks. "
            "Materialise the stack first, e.g., stack.asarray()[None]."
        )


def _reject_shape_modifiers(items: Items):
    _reject_bool(items)
    _reject_newaxis(items)


def _init_view(base: Stack, items: Items) -> npt.NDArray | View:
    """
    Handles view creation and dispatch of ``Stack`` indexing. Integer indexing
    of the first axis returns a materialised 2D NumPy array. Otherwise, a lazy
    ``View`` is returned.
    """
    _reject_shape_modifiers(items)

    if isinstance(items, int | np.integer):
        return base._get_image(items)

    if isinstance(items, slice):
        indices = np.arange(base.shape[0])[items]
        return View(base, indices)

    if isinstance(items, list | np.ndarray):
        indices = _to_indices(items, base.shape)
        return View(base, indices)

    if isinstance(items, tuple):
        for item in items:
            _reject_shape_modifiers(item)

        z_items = items[0]
        yx_indices = items[1:]

        if isinstance(z_items, int | np.integer):
            return base._get_image(z_items)[yx_indices]

        elif isinstance(z_items, slice | EllipsisType):
            z_indices = np.arange(base.shape[0])[z_items]

        elif isinstance(z_items, list | np.ndarray):
            z_indices = _to_indices(z_items, base.shape)

        else:
            raise TypeError(f"Unsupported z-axis item type: {z_items}.")

        return View(base, z_indices, yx_indices)

    else:
        raise TypeError(f"Unsupported item type: {items}.")


class View:
    """
    Provides a view into image data stored on disk. Slices of this class are
    also a view, but nested spatial slicing is not currently supported.
    Supports integer, slice, and NumPy fancy/array indexing: integer indexing
    results in a 2D array, otherwise another View is created.

    Materialise the view with ``asarray()`` or ``np.asarray(view)``.

    Attributes:
        shape (tuple): Dimensions of the view (num_images, height, width).
        info (str): One-line summary of the view (same as ``str(view)``).
        ndim (int): Number of dimensions.
        dtype (npt.DTypeLike): Data type of the underlying image data on disk.
        image_nbytes (int): Number of bytes of each image after spatial
            slicing.
        nbytes (int): Total number of bytes when materialised (``num_images``
            * ``image_nbytes``).
        size (int): Number of elements in the stack.
        itemsize (int): Length of one element in bytes.
    """

    def __init__(self, base: Stack, z_indices: Items, yx_indices: tuple = ()):
        self._base, self._z_indices = base, z_indices
        self._yx_indices = (
            ()
            if all(
                isinstance(item, slice) and item == slice(None)
                for item in yx_indices
            )
            else yx_indices
        )

        num_images = len(self._z_indices)
        if num_images == 0:
            raise ValueError("View is empty!")

        self.dtype = self._base.dtype

        dummy = np.broadcast_to(
            np.empty(1, dtype=self.dtype), self._base.shape[1:]
        )[self._yx_indices]

        self.shape = (num_images, *dummy.shape)
        self.image_nbytes = int(np.prod(self.shape[1:]) * self.dtype.itemsize)
        self.nbytes = self.image_nbytes * self.shape[0]

    def __getitem__(self, items: Items) -> npt.NDArray | View:
        _reject_shape_modifiers(items)

        if isinstance(items, int | np.integer):
            image = self._base[self._z_indices[items]]
            return image[self._yx_indices]

        elif isinstance(items, slice | list | np.ndarray):
            if isinstance(items, list | np.ndarray):
                items = _to_indices(items, self.shape)

            return View(self._base, self._z_indices[items], self._yx_indices)

        elif isinstance(items, tuple):
            for item in items:
                _reject_shape_modifiers(item)

            if self._yx_indices:
                raise NotImplementedError(
                    "Nested spatial indexing is not currently supported. "
                    "Please index the original stack with the full crop, or "
                    "materialise the view first with np.asarray(view)."
                )

            z_items = items[0]
            yx_indices = items[1:]

            if isinstance(z_items, list | np.ndarray):
                z_items = _to_indices(z_items, self.shape)

            elif not isinstance(
                z_items, int | np.integer | slice | EllipsisType
            ):
                raise TypeError(f"Unsupported z-axis item type: {z_items}.")

            return self._base[(self._z_indices[z_items],) + yx_indices]

        else:
            raise TypeError(f"Unsupported item type: {items}.")

    def asarray(self, dtype=None) -> npt.NDArray:
        """Return the view's data as a materialised NumPy array."""
        return np.asarray(
            self._base._get_images(self._z_indices)[
                (slice(None), *self._yx_indices)
            ],
            dtype=dtype,
        )

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        if copy is False:
            raise ValueError("copy=False is not supported.")
        return self.asarray(dtype)

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self):
        """
        Return information about the current view, including shape and
        materialised size of each image and the whole view.
        """
        return _format_info(self, "when materialised")

    @property
    def info(self):
        return str(self)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def itemsize(self):
        return self.dtype.itemsize


class HISStack(Stack):
    """
    Lazy reader for Hamamatsu HIS files.

    Maintains an array of byte offsets for each image, allowing lazy reading
    of single images and single seek/read of contiguous image chunks. HIS
    files do not support lazy spatial slicing.

    Inherits the shared ``Stack`` surface: ``shape``, ``dtype``,
    ``image_nbytes``, ``nbytes``, ``ndim``, ``size``, ``itemsize``, ``info``,
    and the lazy indexing methods.

    Attributes:
        metadata (dict[str, dict[str, str]]): Metadata from the file header.
    """

    def __init__(self, path: Path):
        self._file = path.open("rb")

        header = self._file.read(64)
        self._metadata_nbytes = int.from_bytes(header[2:4], byteorder="little")
        self._width = int.from_bytes(header[4:6], byteorder="little")
        self._height = int.from_bytes(header[6:8], byteorder="little")
        self._file_type = int.from_bytes(header[12:14], byteorder="little")
        num_images_from_header = int.from_bytes(
            header[14:18], byteorder="little"
        )

        self.image_nbytes = self._width * self._height * self._file_type
        self.nbytes = self.image_nbytes * num_images_from_header

        self.shape = (num_images_from_header, self._height, self._width)

        if self._file_type not in (1, 2):
            raise ValueError(
                f"Unrecognised data type: {self._file_type}. Must be 1 (uint8) "
                f"or 2 (uint16)."
            )
        self.dtype = (
            np.dtype(np.uint16) if self._file_type == 2 else np.dtype(np.uint8)
        )

        self.metadata = self._parse_metadata()
        self._image_offsets = self._calc_image_offsets()
        num_images_from_offsets = len(self._image_offsets)
        if num_images_from_offsets != num_images_from_header:
            warn(
                f"HIS header indicated {num_images_from_header} images, but "
                f"found {num_images_from_offsets}. Resizing to "
                f"{num_images_from_offsets}.",
                stacklevel=2,
            )
            self.shape = (num_images_from_offsets, self._height, self._width)
            self.nbytes = self.image_nbytes * num_images_from_offsets

    def _parse_metadata(self) -> dict[str, dict[str, str]]:
        self._file.seek(64, 0)

        metadata = self._file.read(self._metadata_nbytes)
        metadata = metadata.decode("utf-8", errors="ignore").rstrip("\x00")
        metadata = metadata.replace("[", "\n[")

        metadata_dict = {}
        for line in metadata.split("\n"):
            line = line.strip()

            if not line:
                continue

            values = line.split(",")
            category = values[0].lstrip("[").rstrip("]")
            metadata_dict[category] = {}

            for parameter in values[1:]:
                try:
                    key, value = findall(
                        r'([a-zA-Z0-9_]+)=("[^"]*"|[^,]*)', parameter
                    )[0]
                    metadata_dict[category][key] = value
                except IndexError:
                    continue

        return metadata_dict

    def _calc_image_offsets(self) -> npt.NDArray[np.integer]:
        self._file.seek(64 + self._metadata_nbytes, 0)

        image_offsets = [self._file.tell()]

        self._file.seek(self.image_nbytes, 1)

        for _ in range(1, self.shape[0]):
            header = self._file.read(64)

            if len(header) != 64:
                break

            gap = int.from_bytes(header[2:4], byteorder="little")

            image_offsets.append(self._file.tell() + gap)

            self._file.seek(self.image_nbytes + gap, 1)

        return np.asarray(image_offsets, dtype=np.int64)

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        self._file.seek(self._image_offsets[index], 0)
        image_bytes = self._file.read(self.image_nbytes)
        return np.frombuffer(image_bytes, dtype=self.dtype).reshape(
            (self._height, self._width)
        )

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        indices = np.asarray(indices)

        # Only contiguous, non-negative indices are guaranteed to map to
        #   increasing file offsets.
        if np.all(indices >= 0) and np.all(np.diff(indices) == 1):
            offsets = self._image_offsets[indices]
            start = offsets[0]
            stop = offsets[-1]
            self._file.seek(start, 0)
            buffer = self._file.read(stop - start + self.image_nbytes)

            out = np.empty(
                (len(indices), self._height, self._width), dtype=self.dtype
            )
            for i, offset in enumerate(offsets):
                image_bytes = buffer[
                    offset - start : offset - start + self.image_nbytes
                ]
                out[i] = np.frombuffer(image_bytes, dtype=self.dtype).reshape(
                    self._height, self._width
                )

        else:
            out = np.stack([self._get_image(index) for index in indices])

        return out


class DCIMGStack(Stack):
    """
    Lazy reader for Hamamatsu DCIMG files, wrapping ``DCIMGFile``.

    Inherits the shared ``Stack`` surface: ``shape``, ``dtype``,
    ``image_nbytes``, ``nbytes``, ``ndim``, ``size``, ``itemsize``, ``info``,
    and the lazy indexing methods.
    """

    def __init__(self, input_path: Path):
        self._file = DCIMGFile(input_path)
        self.shape = self._file.shape
        self.dtype = self._file.dtype
        self.image_nbytes = int(np.prod(self.shape[1:]) * self.dtype.itemsize)
        self.nbytes = self.image_nbytes * self.shape[0]

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        return np.array(self._file[index])

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        return np.asarray(self._file[indices])


class HDFStack(Stack):
    """
    Lazy reader for HDF files, wrapping an ``h5py.Dataset``. Datasets are
    expected to be grayscale and interpreted as (num_images, height, width).

    Inherits the shared ``Stack`` surface: ``shape``, ``dtype``,
    ``image_nbytes``, ``nbytes``, ``ndim``, ``size``, ``itemsize``, ``info``,
    and the lazy indexing methods.
    """

    def __init__(self, input_path: Path, dset_name: str):
        self._file = h5py.File(input_path, "r")
        self._images = self._file[dset_name]

        if self._images.ndim not in (2, 3):
            raise ValueError(
                f"Only stacks of 2D arrays are supported. Got "
                f"{self._images.shape[0]} {self._images.ndim}D stacks."
            )

        self._2d = self._images.ndim == 2
        self.shape = (
            (1, *self._images.shape) if self._2d else self._images.shape
        )

        self.dtype = self._images.dtype
        self.image_nbytes = int(np.prod(self.shape[1:]) * self.dtype.itemsize)
        self.nbytes = self.image_nbytes * self.shape[0]

    def _as_3d(self) -> npt.NDArray | h5py.Dataset:
        if self._2d:
            return np.asarray(self._images)[np.newaxis, :, :]

        return self._images

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        images = self._as_3d()
        return np.asarray(images[index])

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        images = self._as_3d()
        indices = np.asarray(indices)

        # h5py requires fancy indices to be non-negative and strictly
        #   increasing.
        if np.all(indices >= 0) and np.all(np.diff(indices) > 0):
            return np.asarray(images[indices])

        return np.stack([np.asarray(images[index]) for index in indices])


class TIFFStack(Stack):
    """
    Lazy reader for TIFF-family files via ``tifffile``.

    Files are read as stacks of 2D grayscale images. Colour/multichannel
    images and hyperstacks (more than three dimensions) are rejected, and
    pyramidal files load the highest-resolution level only. Files that store
    the whole stack as a single 3D image (rather than a sequence of 2D
    images) require the optional ``zarr`` package.

    Inherits the shared ``Stack`` surface: ``shape``, ``dtype``,
    ``image_nbytes``, ``nbytes``, ``ndim``, ``size``, ``itemsize``, ``info``,
    and the lazy indexing methods.
    """

    def __init__(self, image_paths: Path | list[Path] | npt.NDArray[Path]):
        self._paths = np.atleast_1d(image_paths)

        self._file = TiffFile(self._paths[0])

        if not self._file.series:
            raise ValueError(f"No series found in {self._paths}.")

        series = self._file.series[0]
        page = series.pages[0]

        if len(self._paths) > 1 and series.is_multifile:
            warn(
                "Passing multiple linked files has no effect -- the whole "
                "series is opened from the first file regardless. Independent "
                "files are ignored.",
                stacklevel=2,
            )
            self._paths = self._paths[:1]

        if len(self._paths) == 1:
            if series.ndim not in (2, 3):
                raise ValueError("Hyperstacks are not supported.")

            if _is_multichannel(page):
                raise ValueError(
                    "Colour/multichannel stacks are not supported."
                )

            if series.is_pyramidal:
                warn(
                    "Pyramidal series are partially supported. Taking maximum "
                    "resolution only.",
                    stacklevel=2,
                )

            # Remove None data from missing pages.
            self._images = np.array(
                [page for page in series if page is not None]
            )

            if page.ndim == 3:
                try:
                    import zarr

                    self._mode = "zarr"

                    # Take the first series and first pyramidal level only.
                    zarr_store = self._file.aszarr(series=0, level=0)
                    self._images = zarr.open(zarr_store, mode="r")
                    self.shape = self._images.shape

                except ImportError as e:
                    raise ImportError(
                        "Please install Zarr to enable reading of 3D-page "
                        "TIFF files."
                    ) from e

            else:
                self._mode = "pages"

                # Single-TIFF case.
                if series.ndim == 2:
                    self.shape = (1, *series.shape)
                else:
                    self.shape = (len(self._images), *series.shape[1:])

            self.dtype = series.dtype
            self.image_nbytes = int(
                np.prod(self.shape[1:]) * self.dtype.itemsize
            )
            self.nbytes = self.image_nbytes * self.shape[0]

        else:
            self._mode = "paths"
            self._file.close()
            self._file = None

            self.shape = None
            self.dtype = None

            for path in self._paths:
                try:
                    with TiffFile(path) as path_file:
                        if not path_file.series:
                            raise ValueError(
                                f"No series found in {path}. Ensure all paths "
                                f"specify valid TIFF files."
                            )

                        path_series = path_file.series[0]
                        if path_series.ndim != 2:
                            raise ValueError(
                                f"All paths must specify 2D images, but "
                                f"{path} is {path_series.ndim}D."
                            )

                        path_page = path_series.pages[0]
                        if _is_multichannel(path_page):
                            raise ValueError(
                                "Colour/multichannel images are not supported."
                            )

                        # Assign shape and dtype from first file -- all others
                        #   must match.
                        if self.shape is None:
                            self.shape = (len(self._paths), *path_series.shape)
                            self.dtype = path_series.dtype

                        if path_series.shape != self.shape[1:]:
                            raise ValueError(
                                "All images must have the same shape."
                            )
                        if path_series.dtype != self.dtype:
                            raise ValueError(
                                "All images must have the same data type."
                            )

                except TiffFileError as exc:
                    raise ValueError(
                        f"Unsupported file type: '{path}'. List/array input "
                        f"must contain paths to 2D grayscale images contained "
                        f"in TIFF-family files readable by `tifffile`."
                    ) from exc

            self.image_nbytes = int(
                np.prod(self.shape[1:]) * self.dtype.itemsize
            )
            self.nbytes = self.image_nbytes * self.shape[0]
            self._images = self._paths

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        if self._mode == "zarr":
            return self._images[index]

        if self._mode == "pages":
            return self._images[index].asarray()

        if self._mode == "paths":
            return imread(str(self._images[index]))

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        if self._mode == "zarr":
            return self._images[indices]

        if self._mode == "pages":
            out = np.empty((len(indices), *self.shape[1:]), dtype=self.dtype)

            for i, index in enumerate(indices):
                out[i] = self._images[index].asarray()

            return out

        if self._mode == "paths":
            images = imread(
                [str(path) for path in self._images[indices]],
                ioworkers=CPU_COUNT // 2,
            )

            return images[np.newaxis, :, :] if images.ndim == 2 else images


def _detect_format(path: PathTypes) -> type[Stack]:
    """
    Return the ``Stack`` subclass that handles ``path``.

    Add new formats here. Detection is ordered, so register a new extension
    branch before the ``tifffile`` fallback at the end, otherwise the
    fallback will claim the path.
    """
    # Single-file input.
    if isinstance(path, str | Path):
        path = Path(path)
        name = path.name.lower()

        if h5py.is_hdf5(path):
            return HDFStack
        if name.endswith(".dcimg"):
            return DCIMGStack
        if name.endswith(".his"):
            return HISStack

        try:
            with TiffFile(path):
                pass
        except TiffFileError as exc:
            raise ValueError(f"Unsupported file type: '{name}'.") from exc

        return TIFFStack

    if len(path) == 0:
        raise ValueError("Received an empty list or array.")

    return TIFFStack


def lazystack(path: PathTypes, dset_name: str | None = None) -> Stack:
    """
    Function for creating lazy image stack readers from an input path or a
    list/array of input paths, with support for multiple file formats.
    Currently supported formats are:

    - HDF5 (.h5, .hdf5, ... ), read as (num_images, height, width) grayscale
      stacks.
    - TIFF-family files via tifffile (.tif, .tiff, .ome.tif, ... ), e.g.
      multi-page TIFF, BigTIFF, Micro-Manager MMStack, and NDTiff. Grayscale
      only; colour/multichannel images and hyperstacks are rejected. Files
      storing the whole stack as one 3D image require the optional ``zarr``
      package.
    - Hamamatsu DCIMG (.dcimg).
    - Hamamatsu HIS (.his).

    The resulting object provides a unified and familiar interface for
    accessing image data and metadata. Underlying image data is only
    materialised upon being cast to a NumPy array, e.g., via `np.asarray()`.

    This function can and should be used within a context manager or closed
    explicitly after use.

    To add support for a new format, subclass `Stack` and register the class
    in `_detect_format`. See "Adding a format" in the README for details.

    Args:
        path: Path or list/array of paths to supported image files
            to be lazily opened. If a list/array is given, each file is
            treated as one frame; linked multi-file series are
            auto-discovered from the first file.
        dset_name: For HDF input only. Path within HDF file to
            dataset to be lazily opened.

    Returns:
        Lazy image stack reader instance.

    Notes:
        - The 'laziness' of arbitrary slicing depends on the layout of the
          underlying data on disk. For example, HDF files support lazy spatial
          slicing, but whole TIFF images must be materialised before spatial
          slicing can be applied.
        - Laziness covers lazystack's own indexing and slicing. NumPy
          functions that accept array-likes (e.g. ``np.take``, ``np.sum``)
          read the whole stack into memory.

    Examples:
        .. code-block:: python
            from lazystack import lazystack
            # Initialise reader (no images in memory).
            with lazystack("path/to/file") as images:
                # Access familiar NDArray attributes, e.g., shape, dtype, size.
                shape = images.shape
                # Slicing the lazystack produces a view (no images in memory).
                substack = images[0:50]
                # Slicing a view or lazystack also produces a view.
                subsubstack = substack[0:10]
                # Spatial slicing also produces a view.
                subsubsubstack = subsubstack[:, 100:, 50:-50]
                # Integer indexing materialises an image.
                image = subsubsubstack[5]
                # `np.asarray()` materialises the whole view.
                subsubsubstack = np.asarray(subsubsubstack)
    """
    stack_class = _detect_format(path)
    if stack_class is HDFStack:
        if dset_name is None:
            raise ValueError(
                "`dset_name` cannot be `None` for HDF file input."
            )
        return stack_class(path, dset_name)
    return stack_class(path)
