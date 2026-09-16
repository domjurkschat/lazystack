# lazystack

[![License](https://img.shields.io/github/license/domjurkschat/lazystack.svg?color=green)](https://github.com/domjurkschat/lazystack/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lazystack.svg?color=green)](https://pypi.org/project/lazystack)
[![Python Version](https://img.shields.io/pypi/pyversions/lazystack.svg?color=green&v=1)](https://python.org/downloads)
[![Tests](https://github.com/domjurkschat/lazystack/actions/workflows/ci.yml/badge.svg)](https://github.com/domjurkschat/lazystack/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/domjurkschat/lazystack/branch/main/graph/badge.svg)](https://codecov.io/gh/domjurkschat/lazystack)

This package provides a familiar interface for lazily reading scientific image 
stacks, particularly stacks of x-ray images. It's cross-platform, lightweight, 
and easy to use. Stacks opened via lazystack are indexable (including NumPy-style 
fancy indexing), iterable, sliceable, and only load the underlying data into 
memory when absolutely necessary. 

**Supported formats**

* HDF (via h5py) — datasets must be (num_images, height, width) grayscale images.
* TIFF and other TIFF-family files (via tifffile) — grayscale stacks only; colour/multichannel images and hyperstacks are not supported.
* Hamamatsu DCIMG (.dcimg).
* Hamamatsu HIS (.his).

Adding a format is designed to be straightforward. See [Adding a format](#adding-a-format).

TIFF-family files that store the whole stack as a single 3D image (rather than a sequence
of 2D images) require the optional `zarr` package (`pip install zarr`).

## Installation

```console
$ pip install lazystack
```

Requires Python 3.10 or later.

## Usage

Lazy stacks can be created by passing a path or a list/array of paths to the 
`lazystack` function. The underlying data is only materialised via integer 
indexing or upon being cast to a NumPy array, e.g., via `np.asarray()` or 
`.asarray()`. For example:

```python
from lazystack import lazystack

with lazystack("path/to/somehis.his") as images:
    # Access familiar NDArray attributes, e.g., shape, dtype, size.
    shape = images.shape
    # Slicing the lazystack produces a view (no images in memory).
    substack = images[0:50]
    # Slicing a view or lazystack also produces a view.
    subsubstack = substack[0:10]
    # Spatial slicing also produces a view.
    subsubsubstack = subsubstack[:, 100:, 50:-50]
    # Integer indexing materialises an image from the view.
    image = subsubsubstack[5]
    # `np.asarray()` materialises the whole view.
    subsubsubstack = np.asarray(subsubsubstack)
```

A lazystack exposes NumPy-like attributes: ``shape``, ``dtype``, ``ndim``, ``size`` (total
elements), and ``itemsize`` (bytes per element). Disk usage is available via
``image_nbytes`` (bytes per frame) and ``nbytes`` (bytes for the whole stack).
A one-line summary is available via ``str(stack)`` or ``stack.info``.

It's best to open lazystacks within a context manager, but you can also open 
and close them manually, e.g.:

```python
images = lazystack("path/to/somehis.his")
# Do some stuff.
# ...
# Don't forget to close!
images.close()
```

If you're opening HDF files, the desired dataset path (within the HDF file) 
must also be specified, e.g.:

```python
images = lazystack("path/to/some/hdf5.h5", "path/to/some/dset")
```

Directories of files can be opened by supplying a list or array of filenames, e.g.:

```python
filenames = sorted(input_path.glob("*.tif"))
with lazystack(filenames) as images:
    # Do some stuff.
```

Currently, this only supports TIFF files.

lazystack also provides `iter_chunks` for prefetching and yielding 
successive chunks along any axis of a lazystack (or regular 3D NumPy-like array), e.g.:

```python
from lazystack import lazystack, iter_chunks

# How many chunks will be prefetched.
num_prefetch = 1
# Memory allowance for each chunk (in GB).
chunk_size_gb = 1.0
# Axis to chunk over (must be 0, 1, or 2).
axis = 0
# Step along the chunk axis (must be at least 1).
step = 1

with lazystack("path/to/somedcimg.dcimg") as images:
    # `start_idx` and `stop_idx` specify the index bounds of each chunk, useful 
    #   for output.
    for chunk, start_idx, stop_idx in iter_chunks(
        images, 
        chunk_size_gb=chunk_size_gb,
        axis=axis,
        step=step,
        num_prefetch=num_prefetch
    ):
        # Do some stuff.
```

## Adding a format

Every supported format is a `Stack` subclass that sets four attributes and
overrides two methods. Lazy slicing, indexing, materialisation, and resource
management are handled generically by `View` and the base class, so a new
reader only needs to describe how images are read from disk.

**The contract**

| Member | Kind | Purpose |
| --- | --- | --- |
| `shape` | attribute | Dimensions as `(num_images, height, width)`. |
| `dtype` | attribute | NumPy dtype of each image. |
| `image_nbytes` | attribute | Bytes of a single image on disk. |
| `nbytes` | attribute | Total bytes of the stack. |
| `_get_image(index)` | method | Return one image as a materialised 2D NumPy array. |
| `_get_images(indices)` | method | Return the requested images as a materialised 3D NumPy array of shape `(len(indices), height, width)`. |
| `_file` | optional attribute | Open handle; closed by `close()` and the context manager. |

Both read methods return materialised NumPy arrays; a lazy `View` is not acceptable, since these are the hooks that actually read from disk. `indices` may be unordered, duplicated, or negative, and the returned images must follow the given order. If your backend can't do that directly — e.g. h5py requires non-negative, strictly increasing indices — fall back to per-image reads as `HDFStack` does.

**Minimal example**

```python
import numpy as np
from lazystack import Stack


class MyStack(Stack):
    """Lazy reader for the fictional .myformat container."""

    def __init__(self, path):
        # Kept as `_file` so `close()`/the context manager releases it.
        self._file = open(path, "rb")
        # shape is (num_images, height, width).
        self.shape = self._read_header()
        self.dtype = np.dtype(np.uint16)
        self.image_nbytes = np.prod(self.shape[1:]) * self.dtype.itemsize
        self.nbytes = self.image_nbytes * self.shape[0]

    def _get_image(self, index):
        frame = self._read_frame(index)
        return np.frombuffer(frame, self.dtype).reshape(self.shape[1:])

    def _get_images(self, indices):
        # Optimised multi-image retrieval preferred, otherwise something like:
        return np.stack([self._get_image(i) for i in indices])


# Opens the underlying file; the context manager closes `_file` on exit.
with MyStack("path/to/file.myformat") as stack:
    image = stack[0]
```

**What works out of the box**

* `View`-backed slicing and spatial cropping (`stack[0:50]`,
  `stack[:, 100:, 50:-50]`), so no image data is read until materialisation.
* Integer, slice, list, array, and boolean-mask indexing.
* `asarray()` / `np.asarray()` materialisation.
* `iter_chunks()` (a module-level helper) for prefetched chunking along any
  axis.
* Context-manager support, and `close()` which closes `_file`.
* `shape`, `dtype`, `ndim`, `size`, `itemsize`, `image_nbytes`, `nbytes`, and
  `info`.

**Register it**

`_detect_format()` in `src/lazystack/_core.py` maps a path to a reader class.
Add a branch that returns your class, placing it before the `tifffile`
fallback at the end of the function (otherwise the fallback claims the path):

```python
if Path(path).name.lower().endswith(".myformat"):
    return MyStack
```

If your reader needs constructor arguments (as `HDFStack` does for
`dset_name`), extend `lazystack()` to pass them through. Readers can also be
instantiated directly without touching `_detect_format`.

**Test it**

Add a writer and an `_open_stack` branch to `tests/conftest.py`, then include
your format name in `FORMATS`. The parametrised suites in `tests/test_core.py`
then exercise your reader against the shared indexing, attribute, and chunking
tests automatically. This usually satisfies the CI coverage floor on its own.

## Contributing

Contributions are very welcome! Don't hesitate to reach out if you have any
questions, and feel free to open an issue if you have any feedback or encounter any bugs.

If you're adding a new file format, see [Adding a format](#adding-a-format) for the reader contract and a minimal example.

To contribute, clone the repository and set up the development environment with
`uv`, which installs the project along with its development dependencies
(`pytest`, `ruff`, `zarr`):

```console
$ git clone https://github.com/domjurkschat/lazystack.git
$ cd lazystack
$ uv sync
```

Alternatively, install an editable copy with `pip`, then add the development
tools separately:

```console
$ pip install -e .
$ pip install pytest ruff zarr
```

Run the tests with:

```console
$ uv run pytest
```

CI enforces a minimum of 75% overall test coverage on pull requests, and
Codecov checks that new or changed lines are covered. Check coverage locally
with:

```console
$ uv run pytest --cov=lazystack --cov-report=term-missing
```

Linting and formatting are enforced by CI and can be run locally with:

```console
$ uv run ruff check
$ uv run ruff format
```

## Roadmap

* Other file formats -- see [Adding a format](#adding-a-format) to contribute a reader.
* Expand test suite.
* Nested spatial indexing.
* Stacks of stacks.
* Colour/multichannel and hyperstack support.


