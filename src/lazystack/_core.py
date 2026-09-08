from __future__ import annotations

from collections.abc import Iterator
from multiprocessing import cpu_count
from pathlib import Path
from re import findall
from typing import TypeAlias
from warnings import warn

import h5py
import numpy as np
import numpy.typing as npt
from dcimg import DCIMGFile
from prefetch_generator import prefetch
from tifffile import TiffFile, imread

CPU_COUNT = cpu_count()

Items: TypeAlias = int | slice | list | tuple | np.ndarray | np.integer
PathTypes: TypeAlias = (
    str | Path | list[str] | list[Path] | npt.NDArray[str] | npt.NDArray[Path]
)

__all__ = ["Stack", "iter_chunks", "lazystack"]


class Stack:
    """
    Base class for lazy image stack readers.

    Indexing with an integer returns a materialised 2D array. Indexing with a
    slice, list, or tuple returns a lazy ``View``. Use within a ``with`` block
    or close explicitly with ``close()``.

    Attributes:
        shape (tuple): Dimensions as (num_images, height, width).
        dtype (npt.DTypeLike): Data type of each image.
        image_nbytes (int): Bytes of a single image.
        nbytes (int): Total bytes of the stack.
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
        image_nbytes_mb = 1e-6 * self.image_nbytes
        nbytes_mb = 1e-6 * self.nbytes
        return (
            f"{type(self).__name__} object referencing {self.shape[0]} "
            f"{self.dtype} images of shape {self.shape[1:]}. Each image "
            f"occupies {image_nbytes_mb:.2f} MB on disk, totalling "
            f"{nbytes_mb:.2f} MB."
        )

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

    @property
    def info(self):
        return str(self)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def itemsize(self):
        return self.dtype.itemsize


def iter_chunks(
    array: npt.NDArray | View,
    chunk_size_gb: float = 5,
    num_prefetch: int = 1,
    axis: int = 0,
    step: int = 1,
) -> Iterator[tuple[npt.NDArray, int, int]]:
    """
    Yield successive chunks along an axis of a 3D array with prefetching.

    Args:
        array: Array or view to be chunked.
        chunk_size_gb: Approximate chunk size allowance in GB.
        num_prefetch: Number of chunks to prefetch. Chunk size will be
            scaled accordingly so as not to exceed `chunk_size_gb`. This
            feature is experimental and may result in underutilisation of
            the chunk size allowance.
        axis: Axis to chunk over.
        step: Spacing along the chunk axis.

    Yields:
        tuple: (chunk, start, stop) where ``chunk`` is the chunked image data
        as a 3D array, ``start`` is the inclusive start index of the chunk,
        and ``stop`` is the exclusive stop index.
    """
    if step < 1:
        raise ValueError(f"`step` must be at least one, but got {step=}.")
    if len(array.shape) != 3:
        raise ValueError(
            f"Only 3D arrays are supported, but got {array.shape=}"
        )
    if axis not in (0, 1, 2):
        raise ValueError(f"`axis` must be 0, 1, or 2, but got {axis}.")
    if num_prefetch < 0:
        raise ValueError(
            f"`num_prefetch` must be positive, but got {num_prefetch}. Set "
            f"`num_prefetch=0` to remove prefetch limit."
        )

    # Calculate how much memory one item occupies along the batch axis.
    image_nbytes = array.nbytes // array.shape[axis]

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
        for start in range(0, array.shape[axis], chunk_size):
            stop = np.minimum(array.shape[axis], start + chunk_size)
            slices[axis] = slice(start, stop, step)

            # View is materialised by `np.asarray`.
            yield (
                np.asarray(array[tuple(slices)]),
                start,
                stop,
            )

    return _generator()


def _to_indices(items: list | npt.NDArray) -> npt.NDArray[np.integer]:
    """
    Converts boolean mask to indices if necessary and returns a NumPy array.
    """
    items = np.asarray(items)
    return np.nonzero(items)[0] if items.dtype == bool else items


def _init_view(base: Stack, items: Items) -> npt.NDArray | View:
    """
    Handles view creation and dispatch of ``Stack`` indexing. Integer indexing
    of the first axis returns a materialised 2D NumPy array. Otherwise, a lazy
    ``View`` is returned.
    """

    if isinstance(items, int | np.integer):
        return base._get_image(items)

    if isinstance(items, slice):
        indices = np.arange(base.shape[0])[items]
        return View(base, indices)

    if isinstance(items, list | np.ndarray):
        indices = _to_indices(items)
        return View(base, indices)

    if isinstance(items, tuple):
        z_items = items[0]
        yx_indices = items[1:]

        if isinstance(z_items, int | np.integer):
            return base._get_image(z_items)[yx_indices]

        elif isinstance(z_items, slice):
            z_indices = np.arange(base.shape[0])[z_items]

        elif isinstance(z_items, list | np.ndarray):
            z_indices = _to_indices(z_items)

        return View(base, z_indices, yx_indices)

    else:
        raise TypeError


class View:
    """
    Provides a view into image data stored on disk. Slices of this class are
    also a view, but nested spatial slicing is not currently supported.
    Supports integer, slice, and NumPy fancy/array indexing: integer indexing
    results in a 2D array, otherwise another View is created.

    Materialise the view with ``asarray()`` or ``np.asarray(view)``.

    Attributes:
        shape (tuple): Dimensions of the view (num_images, height, width).
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
        self._yx_indices = yx_indices

        num_images = len(self._z_indices)
        if num_images == 0:
            raise ValueError("View is empty!")

        test_image = self._base._get_image(self._z_indices[0])
        test_image = np.array(test_image[self._yx_indices])
        self.shape = (num_images, *test_image.shape)

        self.dtype = self._base.dtype
        self.image_nbytes = test_image.nbytes
        self.nbytes = self.image_nbytes * self.shape[0]

    def __getitem__(self, items: Items) -> npt.NDArray | View:
        if isinstance(items, int | np.integer):
            image = self._base[self._z_indices[items]]
            return image[self._yx_indices]

        elif isinstance(items, slice | list | np.ndarray):
            return View(self._base, self._z_indices[items], self._yx_indices)

        elif isinstance(items, tuple):
            if self._yx_indices:
                raise NotImplementedError(
                    "Nested spatial indexing is not currently supported. "
                    "Please index the original stack with the full crop, or "
                    "materialise the view first with np.asarray(view)."
                )
            return self._base[(self._z_indices[items[0]],) + items[1:]]

        else:
            raise TypeError

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

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def itemsize(self):
        return self.dtype.itemsize


class HISStack(Stack):
    """
    Lazy reader for Hamamatsu HIS files.

    Maintains an array of byte offsets for each image, allowing lazy reading
    of single images and single seek/read of contiguous image chunks. HIS
    files do not support lazy spatial slicing.

    Attributes:
        file (BinaryIO): Open binary file handle.
        metadata (dict[str, dict[str, str]]): Metadata from the file header.
    """

    def __init__(self, path: Path):
        self._file = path.open("rb")

        header = self._file.read(64)
        self.metadata_nbytes = int.from_bytes(header[2:4], byteorder="little")
        self.width = int.from_bytes(header[4:6], byteorder="little")
        self.height = int.from_bytes(header[6:8], byteorder="little")
        self.file_type = int.from_bytes(header[12:14], byteorder="little")
        num_images = int.from_bytes(header[14:18], byteorder="little")

        self.image_nbytes = self.width * self.height * self.file_type
        self.nbytes = self.image_nbytes * num_images

        self.shape = (num_images, self.height, self.width)
        self.dtype = (
            np.dtype(np.uint16) if self.file_type == 2 else np.dtype(np.uint8)
        )

        self.metadata = self._parse_metadata()
        self._image_offsets = self._calc_image_offsets()

    def __del__(self):
        self.close()

    def _parse_metadata(self) -> dict[str, dict[str, str]]:
        self._file.seek(64, 0)

        metadata = self._file.read(self.metadata_nbytes)
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
        self._file.seek(64 + self.metadata_nbytes, 0)

        image_offsets = np.empty(self.shape[0], dtype=np.int64)
        image_offsets[0] = self._file.tell()

        self._file.seek(self.image_nbytes, 1)

        for i in range(1, self.shape[0]):
            header = self._file.read(64)

            if not header:
                break

            gap = int.from_bytes(header[2:4], byteorder="little")

            image_offset = self._file.tell() + gap
            image_offsets[i] = image_offset

            self._file.seek(self.image_nbytes + gap, 1)

        return image_offsets

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        self._file.seek(self._image_offsets[index], 0)
        image_bytes = self._file.read(self.image_nbytes)
        return np.frombuffer(image_bytes, dtype=self.dtype).reshape(
            (self.height, self.width)
        )

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        if np.all(np.diff(indices) == 1):
            offsets = self._image_offsets[indices]
            start = offsets[0]
            stop = offsets[-1]
            self._file.seek(start, 0)
            buffer = self._file.read(stop - start + self.image_nbytes)

            out = np.empty(
                (len(indices), self.height, self.width), dtype=self.dtype
            )
            for i, offset in enumerate(offsets):
                image_bytes = buffer[
                    offset - start : offset - start + self.image_nbytes
                ]
                out[i] = np.frombuffer(image_bytes, dtype=self.dtype).reshape(
                    self.height, self.width
                )

        else:
            out = np.stack([self._get_image(index) for index in indices])

        return out

    def close(self):
        file_open = getattr(self, "_file", None)
        if file_open:
            self._file.close()
            self._file = None


class DCIMGStack(Stack):
    """
    Lazy reader for Hamamatsu DCIMG files, wrapping ``DCIMGFile``.

    Attributes:
        dcimg_file (DCIMGFile): Underlying DCIMG reader.
    """

    def __init__(self, input_path: Path):
        self._file = DCIMGFile(input_path)
        self.shape = self._file.shape
        self.dtype = self._file.dtype
        self.image_nbytes = np.array(self._file[0]).nbytes
        self.nbytes = self.image_nbytes * self.shape[0]

    def __del__(self):
        self.close()

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        return np.asarray(self._file[index])

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        return np.asarray(self._file[indices])

    def close(self):
        file_open = getattr(self, "_file", None)
        if file_open:
            self._file.close()
            self._file = None


class HDFStack(Stack):
    """
    Lazy reader for HDF files, wrapping an ``h5py.Dataset``.

    Attributes:
        images (h5py.Dataset): Underlying dataset.
    """

    def __init__(self, input_path: Path, dset_name: str):
        self._in_hdf = h5py.File(input_path, "r")
        self.images = self._in_hdf[dset_name]
        self.shape = self.images.shape
        self.dtype = self.images.dtype
        self.image_nbytes = np.array(self.images[0]).nbytes
        self.nbytes = self.image_nbytes * self.shape[0]

    def __del__(self):
        self.close()

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        return np.asarray(self.images[index])

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        return np.asarray(self.images[indices])

    def close(self):
        in_hdf = getattr(self, "_in_hdf", None)
        if in_hdf is not None:
            self._in_hdf.close()
            self._in_hdf = None


class MMStack(Stack):
    """
    Lazy reader for Micro-Manager (MMStack) OME-TIFF files.

    Attributes:
        images (npt.NDArray): Array of ``TiffPage`` objects from the MMStack
            series.
    """

    def __init__(self, image_paths: Path | list[Path] | npt.NDArray[Path]):
        image_paths = np.atleast_1d(image_paths)
        self._file = TiffFile(image_paths[0])
        if not self._file.series:
            raise ValueError(f"No series found in {image_paths}.")
        if len(image_paths) > 1:
            warn(
                "Passing multiple OME-TIFF has no effect -- the whole linked "
                "series is opened regardless."
            )

        # 3D data containts one series.
        series = self._file.series[0]

        # Remove None data from missing pages.
        self.images = np.array([page for page in series if page is not None])
        tmp = self.images[0].asarray()
        self.shape = (len(self.images), *tmp.shape)
        self.dtype = tmp.dtype
        self.image_nbytes = tmp.nbytes
        self.nbytes = self.image_nbytes * self.shape[0]

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        return self.images[index].asarray()

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        return np.stack([self.images[index].asarray() for index in indices])

    def close(self):
        file_open = getattr(self, "_file", None)
        if file_open:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


class TIFFStack(Stack):
    """
    Lazy reader for a sequence of TIFF files.

    Attributes:
        images (npt.NDArray[Path]): TIFF file paths (in caller-provided
            order).
    """

    def __init__(self, image_paths: Path | list[Path] | npt.NDArray[Path]):
        self.images = np.atleast_1d(image_paths)
        tmp = imread(self.images[0])

        if tmp.ndim > 2:
            raise ValueError(
                f"Only stacks of 2D arrays are supported. Got {tmp.ndim}."
            )

        self.shape = (len(self.images), tmp.shape[0], tmp.shape[1])
        self.dtype = tmp.dtype
        self.image_nbytes = tmp.nbytes
        self.nbytes = self.image_nbytes * self.shape[0]

    def _get_image(self, index: int | np.integer) -> npt.NDArray:
        return imread(str(self.images[index]))

    def _get_images(
        self, indices: list[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray:
        images = imread(
            [str(path) for path in self.images[indices]],
            ioworkers=CPU_COUNT // 2,
        )
        return images[np.newaxis, :, :] if images.ndim == 2 else images

    def close(self):
        pass


def _detect_format(path: PathTypes) -> type[Stack]:
    # Single-file input.
    if isinstance(path, str | Path):
        path = Path(path)
        name = path.name.lower()

        if h5py.is_hdf5(path):
            return HDFStack
        if ".dcimg" in name:
            return DCIMGStack
        if ".his" in name:
            return HISStack
        if ".ome." in name:
            return MMStack
        if ".tif" in name:
            return TIFFStack

        raise ValueError(f"Unsupported file type: '{name}'.")

    # List/array input. Only TIFF and MMStack supported (for now).
    paths = [Path(p) for p in path]
    names = [p.name.lower() for p in paths]
    if all(".ome." in name for name in names):
        return MMStack

    if all(".tif" in name and ".ome." not in name for name in names):
        return TIFFStack

    raise ValueError(f"Unsupported file type(s): '{names}'.")


def lazystack(path: PathTypes, dset_name: str | None = None) -> Stack:
    """
    Function for creating lazy image stack readers from an input path or a
    list/array of input paths, with support for multiple file formats.
    Currently supported formats are:

    - HDF5 (.h5, .hdf5, ... ).
    - TIFF (.tif, .tiff).
    - Micro-Manager TIFF (.ome.tif).
    - Hamamatsu DCIMG (.dcimg).
    - Hamamatsu HIS (.his).

    The resulting object provides a unified and familiar interface for
    accessing image data and metadata. Underlying image data is only
    materialised upon being cast to a NumPy array, e.g., via `np.asarray()`.

    This function can and should be used within a context manager or closed
    explicitly after use.

    Args:
        path: Path or list/array of paths to supported image files
            to be lazily opened. If a list/array is given, all files must
            be of the same type.
        dset_name: For HDF input only. Path within HDF file to
            dataset to be lazily opened.

    Returns:
        Lazy image stack reader instance.

    Notes:
        - The 'laziness' of arbitrary slicing depends on the layout of the
          underlying data on disk. For example, HDF files support lazy spatial
          slicing, but whole TIFF images must be materialised before spatial
          slicing can be applied.

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
