# lazystack

[![License](https://img.shields.io/github/license/domjurkschat/lazystack.svg?color=green)](https://github.com/domjurkschat/lazystack/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/lazystack.svg?color=green)](https://pypi.org/project/lazystack)
[![Python Version](https://img.shields.io/pypi/pyversions/lazystack.svg?color=green&v=1)](https://python.org/downloads)
[![Tests](https://github.com/domjurkschat/lazystack/actions/workflows/ci.yml/badge.svg)](https://github.com/domjurkschat/lazystack/actions/workflows/ci.yml)

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
successive chunks along any axis of a lazystack (or regular 3D NumPy array), e.g.:

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

## Contributing

Contributions are very welcome! Don't hesitate to reach out if you have any 
questions, and feel free to open an issue if you have any feedback or encounter any bugs.

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

Linting and formatting are enforced by CI and can be run locally with:

```console
$ uv run ruff check
$ uv run ruff format
```

## Roadmap

* Expand test suite.
* Nested spatial indexing.
* Stacks of stacks.
* Other file formats.
* Colour/multichannel and hyperstack support.


