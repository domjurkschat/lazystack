import sys

import h5py
import numpy as np
import pytest
from tifffile import TiffWriter, imwrite

from lazystack._core import (
    DCIMGStack,
    HDFStack,
    HISStack,
    TIFFStack,
    View,
    _detect_format,
    iter_chunks,
    lazystack,
)


@pytest.fixture
def example_3d_data():
    return np.arange(120).reshape(10, 3, 4)


def test_iter_chunks_large_budget(example_3d_data):
    out = list(iter_chunks(example_3d_data, chunk_size_gb=1e9))
    chunk, start, stop = out[0]
    # Should only be one chunk.
    assert len(out) == 1
    # Shape should match original.
    assert chunk.shape == example_3d_data.shape
    # Start and stop should match original.
    assert (start, stop) == (0, example_3d_data.shape[0])
    # Data should match.
    assert np.array_equal(chunk, example_3d_data)


def test_iter_chunks_small_budget(example_3d_data):
    out = list(iter_chunks(example_3d_data, chunk_size_gb=1e-18))
    # Should be one image per chunk.
    assert len(out) == example_3d_data.shape[0]
    for i, (chunk, start, stop) in enumerate(out):
        # Shape, bounds, and data should match.
        assert chunk.shape == (1, *example_3d_data.shape[1:])
        assert (start, stop) == (i, i + 1)
        assert np.array_equal(chunk, example_3d_data[i : i + 1, :, :])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_iter_chunks_reconstruct(example_3d_data, axis):
    out = list(iter_chunks(example_3d_data, chunk_size_gb=16e-9, axis=axis))
    recon = np.concatenate([chunk for chunk, _, _ in out], axis=axis)
    # Chunks should stitch back together.
    assert np.array_equal(recon, example_3d_data)


def test_iter_chunks_step_decimates(example_3d_data):
    out = list(iter_chunks(example_3d_data, chunk_size_gb=16e-9, step=2))
    recon = np.concatenate([chunk for chunk, _, _ in out], axis=0)
    # Chunks should stitch back together.
    assert np.array_equal(recon, example_3d_data[::2])


def test_iter_chunks_reject_nonpositive_step(example_3d_data):
    with pytest.raises(ValueError):
        iter_chunks(example_3d_data, step=0)


def test_iter_chunks_reject_non_3d():
    with pytest.raises(ValueError):
        iter_chunks(np.zeros((5, 5)))


def test_iter_chunks_reject_bad_axis(example_3d_data):
    with pytest.raises(ValueError):
        iter_chunks(example_3d_data, axis=3)


def test_iter_chunks_reject_negative_prefetch(example_3d_data):
    with pytest.raises(ValueError):
        iter_chunks(example_3d_data, num_prefetch=-1)


@pytest.fixture
def example_stack(tmp_path, example_3d_data):
    path = tmp_path / "tmp.h5"
    with h5py.File(path, "w") as tmp_hdf:
        tmp_hdf.create_dataset(name="images", data=example_3d_data)
    return lazystack(path, dset_name="images")


def test_view_index(example_stack, example_3d_data):
    result = example_stack[0]
    true_result = example_3d_data[0]
    assert result.shape == true_result.shape
    assert np.array_equal(result, true_result)


def test_view_index_slice_spatial(example_stack, example_3d_data):
    result = example_stack[0, 1:, 2:]
    true_result = example_3d_data[0, 1:, 2:]
    assert result.shape == true_result.shape
    assert np.array_equal(result, true_result)


def test_view_slice(example_stack, example_3d_data):
    view = example_stack[0:5]
    data = example_3d_data[0:5]
    assert isinstance(view, View)
    assert view.shape == data.shape
    assert np.array_equal(view, data)


def test_view_slice_spatial(example_stack, example_3d_data):
    view = example_stack[0:5, 1:, 2:]
    data = example_3d_data[0:5, 1:, 2:]
    assert isinstance(view, View)
    assert view.shape == data.shape
    assert np.array_equal(view, data)


def test_view_list(example_stack, example_3d_data):
    view = example_stack[[0, 1, 2, 3, 4]]
    data = example_3d_data[[0, 1, 2, 3, 4]]
    assert isinstance(view, View)
    assert view.shape == data.shape
    assert np.array_equal(view, data)


def test_view_array(example_stack, example_3d_data):
    view = example_stack[np.arange(5)]
    data = example_3d_data[np.arange(5)]
    assert isinstance(view, View)
    assert view.shape == data.shape
    assert np.array_equal(view, data)


def test_view_mask(example_stack, example_3d_data):
    mask = np.zeros(10)
    mask[0:5] = 1
    mask = mask.astype(np.bool_)
    view = example_stack[mask]
    data = example_3d_data[mask]
    assert isinstance(view, View)
    assert view.shape == data.shape
    assert np.array_equal(view, data)


def test_view_materialisation(example_stack, example_3d_data):
    view = example_stack[0:5]
    data = example_3d_data[0:5]
    assert np.array_equal(view.asarray(), data)
    assert np.array_equal(np.asarray(view), data)
    assert np.array_equal(np.array(view), data)


def test_view_attributes(example_stack, example_3d_data):
    view = example_stack[0:5, 1:, 2:]
    data = example_3d_data[0:5, 1:, 2:]
    assert view.shape == data.shape
    assert view.dtype == data.dtype
    assert view.nbytes == data.nbytes
    assert view.image_nbytes == data.nbytes / data.shape[0]
    assert view.shape[0] == len(view)
    assert view.size == data.size
    assert view.itemsize == data.itemsize


def test_view_reject_empty(example_stack):
    with pytest.raises(ValueError):
        example_stack[5:5]


def test_view_reject_nested_spatial_indexing(example_stack):
    view = example_stack[:, 1:, 2:]
    with pytest.raises(NotImplementedError):
        view[:, 1:, 2:]


@pytest.fixture
def example_his(tmp_path, example_3d_data):
    buffer = bytearray(64)
    # Construct header.
    # Number of metadata bytes.
    buffer[2:4] = (2).to_bytes(2, "little")
    # Width.
    buffer[4:6] = (4).to_bytes(2, "little")
    # Height.
    buffer[6:8] = (3).to_bytes(2, "little")
    # File type.
    buffer[12:14] = (2).to_bytes(2, "little")
    # Number of images.
    buffer[14:18] = (10).to_bytes(4, "little")

    # Add the metadata bytes.
    buffer += bytearray(2)

    # Add the images with variable gaps. Header -> gap -> image -> repeat,
    #   except the first image.
    gaps = np.array([0, 2, 4, 2, 4, 0, 2, 6, 4, 10], dtype=np.uint16)
    images = example_3d_data.astype(np.uint16)
    for i, image in enumerate(images):
        if i > 0:
            image_header = bytearray(64)
            image_header[2:4] = int(gaps[i]).to_bytes(2, "little")
            buffer += image_header
            buffer += bytes(int(gaps[i]))
        buffer += image.tobytes()

    path = tmp_path / "tmp.his"
    path.write_bytes(buffer)

    return path


def test_his_attributes(example_his, example_3d_data):
    his = HISStack(example_his)
    data = example_3d_data.astype(np.uint16)
    assert his.shape == data.shape
    assert his.file_type == 2
    assert his.image_nbytes == data[0].nbytes
    assert his.nbytes == data.nbytes
    assert his.dtype == np.uint16
    assert his.size == data.size
    assert his.itemsize == data.itemsize


def test_his_get_image(example_his, example_3d_data):
    his = HISStack(example_his)
    data = example_3d_data.astype(np.uint16)
    assert np.array_equal(his[0], data[0])
    assert np.array_equal(his[[0, 5, 7]], data[[0, 5, 7]])
    assert np.array_equal(his[0:5], data[0:5])
    assert np.array_equal(his, data)


@pytest.fixture
def example_tiff_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"
    imwrite(output_path, example_3d_data[0].astype(np.uint16))
    return output_path


@pytest.fixture
def example_good_tiff_bad_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.dat"
    imwrite(output_path, example_3d_data[0].astype(np.uint16))
    return output_path


def test_lazystack_dispatch(
    tmp_path, example_stack, example_tiff_path, example_good_tiff_bad_path
):
    assert isinstance(
        lazystack(tmp_path / "tmp.h5", dset_name="images"), HDFStack
    )
    assert _detect_format("tmp.dcimg") is DCIMGStack
    assert _detect_format("tmp.his") is HISStack
    assert _detect_format(example_tiff_path) is TIFFStack
    assert _detect_format(example_good_tiff_bad_path) is TIFFStack


@pytest.fixture
def example_unsupported_tiff_path(tmp_path):
    output_path = tmp_path / "tmp.blah"
    output_path.write_bytes(b"This is not a TIFF file.")
    return output_path


def test_dispatch_reject_unsupported_type(example_unsupported_tiff_path):
    with pytest.raises(ValueError):
        _detect_format(example_unsupported_tiff_path)


def test_dispatch_reject_empty_list():
    with pytest.raises(ValueError):
        _detect_format([])


@pytest.fixture
def example_tiff_paths(tmp_path, example_3d_data):
    tiff_paths = []
    for i, image in enumerate(example_3d_data.astype(np.uint16)):
        path = tmp_path / f"{i:04}.tif"
        imwrite(path, image)
        tiff_paths.append(path)
    return tiff_paths


def test_dispatch_reject_unsupported_mix(
    example_unsupported_tiff_path, example_tiff_paths
):
    paths = np.concatenate(
        (example_tiff_paths, [example_unsupported_tiff_path])
    )
    with pytest.raises(ValueError):
        _detect_format(paths)


def test_lazystack_reject_no_hdf_dset_name(tmp_path, example_stack):
    # `example_stack` creates `tmp.h5`.
    with pytest.raises(ValueError):
        lazystack(tmp_path / "tmp.h5")


@pytest.fixture
def example_mmstack_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.ome.tif"
    with TiffWriter(output_path, ome=True, bigtiff=False) as writer:
        writer.write(
            example_3d_data.astype(np.uint16),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
    return output_path


def test_mm_attributes(example_mmstack_path, example_3d_data):
    mm = TIFFStack(example_mmstack_path)
    data = example_3d_data.astype(np.uint16)
    assert mm.shape == data.shape
    assert mm.image_nbytes == data[0].nbytes
    assert mm.nbytes == data.nbytes
    assert mm.dtype == np.uint16
    assert mm.size == data.size
    assert mm.itemsize == data.itemsize


def test_mm_get_image(example_mmstack_path, example_3d_data):
    mm = TIFFStack(example_mmstack_path)
    data = example_3d_data.astype(np.uint16)
    assert np.array_equal(mm[0], data[0])
    assert np.array_equal(mm[[0, 5, 7]], data[[0, 5, 7]])
    assert np.array_equal(mm[0:5], data[0:5])
    assert np.array_equal(mm, data)


@pytest.fixture
def example_rgb_tiff_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"
    data = example_3d_data.astype(np.uint16)
    example_rgb_image = data[0][:, :, np.newaxis]
    example_rgb_image = np.repeat(example_rgb_image, 3, axis=2)
    with TiffWriter(output_path) as writer:
        writer.write(
            example_rgb_image,
            photometric="rgb",
            metadata={"axes": "YXS"},
        )
    return output_path


def test_tiff_reject_rgb(example_rgb_tiff_path):
    with pytest.raises(ValueError):
        TIFFStack(example_rgb_tiff_path)


@pytest.fixture
def example_hyperstack_tiff_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"
    data = example_3d_data.astype(np.uint16)
    example_rgb_image = data[np.newaxis, :, :, :]
    with TiffWriter(output_path) as writer:
        writer.write(
            example_rgb_image,
            metadata={"axes": "TZYX"},
        )
    return output_path


def test_tiff_reject_hyperstack(example_hyperstack_tiff_path):
    with pytest.raises(ValueError):
        TIFFStack(example_hyperstack_tiff_path)


@pytest.fixture
def example_tiff_volumetric_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"
    with TiffWriter(output_path) as writer:
        writer.write(
            example_3d_data.astype(np.uint16),
            photometric="minisblack",
            volumetric=True,
        )
    return output_path


def test_tiff_volumetric_attributes(
    example_tiff_volumetric_path, example_3d_data
):
    tiffs = TIFFStack(example_tiff_volumetric_path)
    data = example_3d_data.astype(np.uint16)
    assert tiffs.shape == data.shape
    assert tiffs.image_nbytes == data[0].nbytes
    assert tiffs.nbytes == data.nbytes
    assert tiffs.dtype == np.uint16
    assert tiffs.size == data.size
    assert tiffs.itemsize == data.itemsize


def test_tiff_volumetric_get_image(
    example_tiff_volumetric_path, example_3d_data
):
    tiffs = TIFFStack(example_tiff_volumetric_path)
    data = example_3d_data.astype(np.uint16)
    assert np.array_equal(tiffs[0], data[0])
    assert np.array_equal(tiffs[[0, 5, 7]], data[[0, 5, 7]])
    assert np.array_equal(tiffs[0:5], data[0:5])
    assert np.array_equal(tiffs, data)


def test_tiff_volumetric_reject_no_zarr(
    monkeypatch, example_tiff_volumetric_path
):
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(ImportError):
        TIFFStack(example_tiff_volumetric_path)


def test_tiff_attributes(example_tiff_paths, example_3d_data):
    tiffs = TIFFStack(example_tiff_paths)
    data = example_3d_data.astype(np.uint16)
    assert tiffs.shape == data.shape
    assert tiffs.image_nbytes == data[0].nbytes
    assert tiffs.nbytes == data.nbytes
    assert tiffs.dtype == np.uint16
    assert tiffs.size == data.size
    assert tiffs.itemsize == data.itemsize


def test_tiff_get_image(example_tiff_paths, example_3d_data):
    tiffs = TIFFStack(example_tiff_paths)
    data = example_3d_data.astype(np.uint16)
    assert np.array_equal(tiffs[0], data[0])
    assert np.array_equal(tiffs[[0, 5, 7]], data[[0, 5, 7]])
    assert np.array_equal(tiffs[0:5], data[0:5])
    assert np.array_equal(tiffs, data)


def test_tiff_single(example_tiff_paths, example_3d_data):
    tiff = TIFFStack(example_tiff_paths[0])
    data = example_3d_data.astype(np.uint16)[0]
    data = data[np.newaxis, :, :]
    assert tiff.shape == data.shape
    assert np.array_equal(tiff, data)


@pytest.fixture
def example_3d_tiff_paths(tmp_path, example_3d_data):
    tiff_paths = []
    for i, image in enumerate(example_3d_data.astype(np.uint16)):
        image = image[np.newaxis, :, :]
        path = tmp_path / f"{i:04}.tif"
        imwrite(path, image)
        tiff_paths.append(path)
    return tiff_paths


def test_tiff_reject_3d_paths(example_3d_tiff_paths):
    with pytest.raises(ValueError):
        TIFFStack(example_3d_tiff_paths)


def test_tiff_reject_multi_rgb(example_rgb_tiff_path):
    with pytest.raises(ValueError):
        TIFFStack(np.repeat(example_rgb_tiff_path, 2))


def test_tiff_reject_multi_mixed_shape(tmp_path):
    path1 = tmp_path / "tmp1.tif"
    path2 = tmp_path / "tmp2.tif"

    imwrite(path1, np.zeros((2, 2), dtype=np.uint16))
    imwrite(path2, np.zeros((3, 3), dtype=np.uint16))

    with pytest.raises(ValueError):
        TIFFStack([path1, path2])


def test_tiff_reject_multi_mixed_dtype(tmp_path):
    path1 = tmp_path / "tmp1.tif"
    path2 = tmp_path / "tmp2.tif"

    imwrite(path1, np.zeros((2, 2), dtype=np.uint16))
    imwrite(path2, np.zeros((2, 2), dtype=np.uint32))

    with pytest.raises(ValueError):
        TIFFStack([path1, path2])
