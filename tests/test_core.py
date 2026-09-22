import sys

import h5py
import numpy as np
import pytest
from tifffile import (
    PHOTOMETRIC,
    TiffPageSeries,
    TiffWriter,
    imwrite,
)

from lazystack._core import (
    DCIMGStack,
    HDFStack,
    HISStack,
    TIFFStack,
    _detect_format,
    _is_multichannel,
    iter_chunks,
    lazystack,
)

INDEX_EXPRS = [
    pytest.param(0, id="int"),
    pytest.param(3, id="int-mid"),
    pytest.param(slice(0, 5), id="slice"),
    pytest.param(slice(None, None, -1), id="slice-reversed"),
    pytest.param([0, 1, 2, 3, 4], id="list"),
    pytest.param([4, 0, 3], id="list-unordered"),
    pytest.param([-1, 0], id="list-negative"),
    pytest.param([1, 1, 2], id="list-duplicates"),
    pytest.param([0, 2, 4], id="list-sparse"),
    pytest.param(np.arange(5), id="array"),
    pytest.param((slice(None), np.array([0, 1])), id="array-no-op-norm"),
    pytest.param((0, slice(1, None), slice(2, None)), id="int-spatial"),
    pytest.param(
        (slice(0, 5), slice(1, None), slice(2, None)), id="slice-spatial"
    ),
    pytest.param((Ellipsis, slice(None), slice(None)), id="ellipsis"),
    pytest.param(
        (np.array([0, 1]), slice(1, None), slice(2, None)),
        id="array-z-spatial",
    ),
]

NEWAXIS_EXPRS = [
    None,
    np.newaxis,
    (None,),
    (None, slice(None), slice(None)),
    (Ellipsis, None),
    (slice(None), None, slice(None)),
    (0, None),
]


def test_iter_chunks_large_budget(example_3d_data):
    out = list(iter_chunks(example_3d_data, chunk_size_gb=1e9))
    chunk, start, stop = out[0]

    assert len(out) == 1
    assert chunk.shape == example_3d_data.shape
    assert (start, stop) == (0, example_3d_data.shape[0])
    assert np.array_equal(chunk, example_3d_data)


def test_iter_chunks_small_budget(example_3d_data):
    out = list(iter_chunks(example_3d_data, chunk_size_gb=1e-18))

    assert len(out) == example_3d_data.shape[0]

    for i, (chunk, start, stop) in enumerate(out):
        expected = example_3d_data[i : i + 1, :, :]
        assert chunk.shape == (1, *example_3d_data.shape[1:])
        assert (start, stop) == (i, i + 1)
        assert np.array_equal(chunk, expected)


@pytest.mark.parametrize("num_prefetch", [1, 2, 3])
@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_iter_chunks_reconstruct(example_3d_data, axis, step, num_prefetch):
    slices = [slice(None)] * example_3d_data.ndim
    slices[axis] = slice(None, None, step)

    out = list(
        iter_chunks(
            example_3d_data,
            chunk_size_gb=16e-9,
            axis=axis,
            step=step,
            num_prefetch=num_prefetch,
        )
    )
    recon = np.concatenate([chunk for chunk, _, _ in out], axis=axis)

    expected = example_3d_data[tuple(slices)]

    assert np.array_equal(recon, expected)


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_iter_chunks_with_stack(example_hdf_path, axis, step):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        slices = [slice(None)] * stack.ndim
        slices[axis] = slice(None, None, step)

        out = list(
            iter_chunks(stack, chunk_size_gb=16e-9, axis=axis, step=step)
        )
        recon = np.concatenate([chunk for chunk, _, _ in out], axis=axis)

        expected = stack.asarray()[tuple(slices)]

        assert np.array_equal(recon, expected)


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_iter_chunks_with_view(example_hdf_path, axis, step):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        view = stack[0:5, :, :]

        slices = [slice(None)] * view.ndim
        slices[axis] = slice(None, None, step)

        out = list(
            iter_chunks(view, chunk_size_gb=16e-9, axis=axis, step=step)
        )
        recon = np.concatenate([chunk for chunk, _, _ in out], axis=axis)

        expected = view.asarray()[tuple(slices)]

        assert np.array_equal(recon, expected)


def test_iter_chunks_z_axis_with_spatially_cropped_view(example_hdf_path):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        view = stack[0:5, 1:, 2:]
        out = list(iter_chunks(view, chunk_size_gb=16e-9, axis=0))
        recon = np.concatenate([chunk for chunk, _, _ in out], axis=0)

        expected = view.asarray()

        assert np.array_equal(recon, expected)


@pytest.mark.parametrize("axis", [1, 2])
def test_iter_chunks_reject_view_nested_spatial_indexing(
    example_hdf_path, axis
):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        view = stack[:, 1:, 2:]
        with pytest.raises(NotImplementedError):
            list(iter_chunks(view, chunk_size_gb=16e-9, axis=axis))


def test_iter_chunks_reject_nonpositive_step(example_3d_data):
    with pytest.raises(ValueError):
        iter_chunks(example_3d_data, step=0)


def test_iter_chunks_reject_non_3d():
    with pytest.raises(ValueError):
        iter_chunks(np.zeros((5, 5)))


def test_iter_chunks_reject_bad_axis(example_3d_data):
    with pytest.raises(ValueError):
        iter_chunks(example_3d_data, axis=3)


@pytest.mark.parametrize("num_prefetch", [-1, 0])
def test_iter_chunks_reject_bad_prefetch(example_3d_data, num_prefetch):
    with pytest.raises(ValueError):
        iter_chunks(example_3d_data, num_prefetch=num_prefetch)


def test_attributes(indexable):
    object, expected = indexable

    assert object.shape == expected.shape
    assert object.ndim == expected.ndim
    assert object.dtype == expected.dtype
    assert object.image_nbytes == expected[0].nbytes
    assert object.nbytes == expected.nbytes
    assert object.size == expected.size
    assert object.itemsize == expected.itemsize
    assert len(object) == expected.shape[0]


@pytest.mark.parametrize("expr", INDEX_EXPRS)
def test_indexing(indexable, expr):
    object, expected = indexable

    assert np.array_equal(np.asarray(object[expr]), np.asarray(expected[expr]))


def test_indexing_mask(indexable):
    object, expected = indexable
    mask = np.zeros(len(object), dtype=bool)
    mask[::2] = True

    assert np.array_equal(np.asarray(object[mask]), np.asarray(expected[mask]))
    assert np.array_equal(
        np.asarray(object[mask, :, :]), np.asarray(expected[mask, :, :])
    )


def test_materialisation(indexable):
    object, expected = indexable

    assert np.array_equal(object.asarray(), expected)
    assert np.array_equal(np.asarray(object), expected)
    assert np.array_equal(np.array(object), expected)


def test_close_releases_handle(closable_stack):
    closable_stack.close()

    assert closable_stack._file is None


def test_close_is_idempotent(closable_stack):
    closable_stack.close()
    closable_stack.close()

    assert closable_stack._file is None


def test_exit_propagates_and_releases(closable_stack):
    with pytest.raises(RuntimeError), closable_stack:
        raise RuntimeError

    assert closable_stack._file is None


def test_2d_hdf(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.h5"
    data = example_3d_data[0].astype(np.uint16)
    with h5py.File(output_path, "w") as tmp_hdf:
        tmp_hdf.create_dataset(name="images", data=data, dtype="uint16")

    expected = data[np.newaxis, :, :]
    with lazystack(output_path, dset_name="images") as stack:
        assert stack.shape == expected.shape
        assert stack.image_nbytes == expected[0].nbytes
        assert np.array_equal(stack[0], expected[0])
        assert np.array_equal(stack.asarray(), expected)
        assert np.array_equal(
            stack[[0, 0]].asarray(), np.stack([expected[0], expected[0]])
        )


def test_dcimg_close_forwards(dcimg_stack):
    stack, fake = dcimg_stack
    stack.close()

    assert fake.closed
    assert stack._file is None


def test_dcimg_get_image_is_materialised(dcimg_stack):
    stack, fake = dcimg_stack
    image = stack[0]
    assert not np.shares_memory(image, fake.data)


def test_stack_reject_empty(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(ValueError),
    ):
        stack[5:5]


def test_view_reject_nested_spatial_indexing(example_hdf_path):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        view = stack[:, 1:, 2:]
        with pytest.raises(NotImplementedError):
            view[:, 1:, 2:]


def test_stack_reject_bad_mask_length(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(ValueError),
    ):
        stack[np.array([True, False])]


def test_stack_reject_single_bool(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(TypeError),
    ):
        stack[True]


def test_stack_reject_multi_bool(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(TypeError),
    ):
        stack[True, :, :]


@pytest.fixture
def example_good_tiff_bad_path(tmp_path):
    output_path = tmp_path / "tmp.dat"
    imwrite(output_path, np.zeros((2, 2), dtype=np.uint16))
    return output_path


def test_lazystack_dispatch(
    example_hdf_path, _example_paths, example_good_tiff_bad_path
):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        assert isinstance(stack, HDFStack)

    assert _detect_format("tmp.dcimg") is DCIMGStack
    assert _detect_format("tmp.his") is HISStack
    assert _detect_format(_example_paths["paged"]) is TIFFStack
    assert _detect_format(_example_paths["multi_file"]) is TIFFStack
    assert _detect_format(example_good_tiff_bad_path) is TIFFStack


@pytest.fixture
def example_unsupported_tiff_path(tmp_path):
    output_path = tmp_path / "tmp.blah"
    output_path.write_bytes(b"This is not a TIFF file.")
    return output_path


def test_dispatch_reject_unsupported_type(example_unsupported_tiff_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        _detect_format(example_unsupported_tiff_path)


def test_dispatch_reject_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        _detect_format([])


def test_dispatch_reject_unsupported_mix(
    example_unsupported_tiff_path, _example_paths
):
    paths = np.concatenate(
        (_example_paths["multi_file"], [example_unsupported_tiff_path])
    )

    assert _detect_format(paths) is TIFFStack

    with pytest.raises(ValueError, match="Unsupported file type"):
        lazystack(paths)


def test_lazystack_reject_no_hdf_dset_name(example_hdf_path):
    with pytest.raises(ValueError, match="dset_name"):
        lazystack(example_hdf_path)


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
    with pytest.raises(ValueError, match="Colour/multichannel"):
        TIFFStack(example_rgb_tiff_path)


@pytest.fixture
def example_tiff_miniswhite_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"
    data = example_3d_data[0].astype(np.uint16)

    with TiffWriter(output_path) as writer:
        writer.write(
            data,
            photometric="miniswhite",
            metadata={"axes": "YX"},
        )

    return output_path


def test_tiff_miniswhite(example_tiff_miniswhite_path, example_3d_data):
    data = example_3d_data[0].astype(np.uint16)

    with TIFFStack(example_tiff_miniswhite_path) as stack:
        assert np.array_equal(stack.asarray(), data[np.newaxis, :, :])


@pytest.fixture
def example_hyperstack_tiff_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"
    data = np.stack([example_3d_data, example_3d_data]).astype(np.uint16)

    with TiffWriter(output_path) as writer:
        writer.write(data, photometric="minisblack", metadata={"axes": "TZYX"})

    return output_path


def test_tiff_reject_hyperstack(example_hyperstack_tiff_path):
    with pytest.raises(ValueError, match="Hyperstacks"):
        TIFFStack(example_hyperstack_tiff_path)


def test_tiff_reject_volumetric_no_zarr(monkeypatch, _example_paths):
    monkeypatch.setitem(sys.modules, "zarr", None)

    with pytest.raises(ImportError, match="install Zarr"):
        TIFFStack(_example_paths["volumetric"])


def test_tiff_single(_example_paths, example_3d_data):
    data = example_3d_data.astype(np.uint16)[0][np.newaxis, :, :]

    with TIFFStack(_example_paths["multi_file"][0]) as tiff:
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
    with pytest.raises(ValueError, match="2D images"):
        TIFFStack(example_3d_tiff_paths)


def test_tiff_reject_multi_rgb(example_rgb_tiff_path):
    with pytest.raises(ValueError, match="2D images"):
        TIFFStack(np.repeat(example_rgb_tiff_path, 2))


def test_tiff_reject_multi_mixed_shape(tmp_path):
    path1 = tmp_path / "tmp1.tif"
    path2 = tmp_path / "tmp2.tif"

    imwrite(path1, np.zeros((2, 2), dtype=np.uint16))
    imwrite(path2, np.zeros((3, 3), dtype=np.uint16))

    with pytest.raises(ValueError, match="same shape"):
        TIFFStack([path1, path2])


def test_tiff_reject_multi_mixed_dtype(tmp_path):
    path1 = tmp_path / "tmp1.tif"
    path2 = tmp_path / "tmp2.tif"

    imwrite(path1, np.zeros((2, 2), dtype=np.uint16))
    imwrite(path2, np.zeros((2, 2), dtype=np.uint32))

    with pytest.raises(ValueError, match="same data type"):
        TIFFStack([path1, path2])


@pytest.fixture
def example_tiff_no_series(tmp_path):
    output_path = tmp_path / "tmp.tif"
    output_path.write_bytes(b"II*\x00\x00\x00\x00\x00")
    return output_path


def test_tiff_reject_no_series(example_tiff_no_series, _example_paths):
    with pytest.raises(ValueError, match="No series found"):
        lazystack(example_tiff_no_series)

    paths = [_example_paths["multi_file"][0], example_tiff_no_series]

    with pytest.raises(ValueError, match="No series found"):
        lazystack(paths)


def test_tiff_warns_multi_series(_example_paths, monkeypatch):
    original_init = TiffPageSeries.__init__

    def multifile_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.is_multifile = True

    monkeypatch.setattr(TiffPageSeries, "__init__", multifile_init)

    paths = [_example_paths["paged"], _example_paths["paged"]]
    with (
        pytest.warns(UserWarning, match="linked files"),
        TIFFStack(paths) as stack,
        TIFFStack(paths[0]) as expected_stack,
    ):
        assert np.array_equal(stack.asarray(), expected_stack.asarray())


@pytest.fixture
def example_tiff_palette_path(tmp_path, example_3d_data):
    output_path = tmp_path / "tmp.tif"

    data = example_3d_data[0].astype(np.uint8)
    cmap = np.zeros((3, 256), dtype=np.uint16)

    with TiffWriter(output_path) as writer:
        writer.write(data, photometric="palette", colormap=cmap)

    return output_path


def test_tiff_reject_palette(example_tiff_palette_path):
    with pytest.raises(ValueError, match="Colour/multichannel stacks"):
        lazystack(example_tiff_palette_path)

    with pytest.raises(ValueError, match="Colour/multichannel images"):
        lazystack([example_tiff_palette_path, example_tiff_palette_path])


def test_tiff_warn_pyramidal(tmp_path):
    output_path = tmp_path / "tmp.ome.tif"

    level0 = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    level1 = level0[:, ::2, ::2]

    with TiffWriter(output_path, ome=True) as writer:
        writer.write(
            level0,
            subifds=1,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        writer.write(
            level1,
            subfiletype=1,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )

    with (
        pytest.warns(UserWarning, match="Pyramidal"),
        lazystack(output_path) as stack,
    ):
        assert np.array_equal(stack.asarray(), level0)


def test_his_single(tmp_path, his_bytes, example_3d_data):
    data = example_3d_data[0].astype(np.uint16)[np.newaxis, :, :]

    path = tmp_path / "single.his"
    path.write_bytes(his_bytes(data))

    with HISStack(path) as stack:
        assert stack.shape == data.shape
        assert stack.image_nbytes == data[0].nbytes
        assert np.array_equal(stack[0], data[0])
        assert np.array_equal(stack.asarray(), data)


def test_his_parse_metadata(tmp_path, example_3d_data, his_bytes):
    data = example_3d_data.astype(np.uint16)
    metadata = (
        '[Camera,exposure=10.5,binning="2x2",gain=1\n'
        '[Stage,x=1.5,y=2.5,label="wide field"\n'
        "[Acquisition,frame_count=10,broken,gain=2\x00\x00"
    )

    path = tmp_path / "metadata.his"
    path.write_bytes(his_bytes(data, metadata=metadata))

    with HISStack(path) as stack:
        assert stack.metadata == {
            "Camera": {"exposure": "10.5", "binning": '"2x2"', "gain": "1"},
            "Stage": {"x": "1.5", "y": "2.5", "label": '"wide field"'},
            "Acquisition": {"frame_count": "10", "gain": "2"},
        }


def test_his_uint8(tmp_path, example_3d_data, his_bytes):
    data = example_3d_data.astype(np.uint8)

    path = tmp_path / "uint8.his"
    path.write_bytes(his_bytes(data, file_type=1))

    with HISStack(path) as stack:
        assert stack.shape == data.shape
        assert stack.dtype == np.uint8
        assert stack.image_nbytes == data[0].nbytes
        assert np.array_equal(stack.asarray(), data)


def test_his_reject_unrecognised_dtype(tmp_path, example_3d_data, his_bytes):
    data = example_3d_data.astype(np.uint16)

    path = tmp_path / "bad.his"
    path.write_bytes(his_bytes(data, file_type=3))

    with pytest.raises(ValueError):
        HISStack(path)


def test_his_reject_truncated(
    tmp_path, example_3d_data, his_bytes, his_header_fields
):
    data = example_3d_data.astype(np.uint16)

    buffer = bytearray(his_bytes(data))
    # HIS file only contains 10 images, but header reports 100.
    buffer[his_header_fields["frame_count"]] = (100).to_bytes(4, "little")

    path = tmp_path / "trunc.his"
    path.write_bytes(buffer)

    with pytest.warns(UserWarning):
        stack = HISStack(path)

    assert stack.shape == data.shape
    assert stack.nbytes == data.nbytes

    stack.close()


def test_is_multichannel(fake_page):
    assert _is_multichannel(fake_page(None, 1)) is False
    assert _is_multichannel(fake_page(PHOTOMETRIC.MINISBLACK, 1)) is False
    assert _is_multichannel(fake_page(PHOTOMETRIC.MINISWHITE, 1)) is False
    assert _is_multichannel(fake_page(PHOTOMETRIC.RGB, 3)) is True
    assert _is_multichannel(fake_page(PHOTOMETRIC.MINISBLACK, 3)) is True


def test_stack_copy_false(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(ValueError, match="copy=False"),
    ):
        np.array(stack, copy=False)


def test_stack_asarray_dtype(example_hdf_path):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        array = stack.asarray(dtype=np.uint8)
        assert array.dtype == np.uint8


def test_view_copy_false(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(ValueError, match="copy=False"),
    ):
        np.array(stack[:, 1:, 2:], copy=False)


def test_view_asarray_dtype(example_hdf_path):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        array = stack[:, 1:, 2:].asarray(dtype=np.uint8)
        assert array.dtype == np.uint8


@pytest.mark.parametrize("bad", ["x", 1.5])
def test_view_reject_unsupported_index_type(example_hdf_path, bad):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        view = stack[0:2]
        with pytest.raises(TypeError):
            view[bad, :, :]


def test_stack_str(example_hdf_path, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    expected_image_nbytes_mb = 1e-6 * data[0].nbytes
    expected_nbytes_mb = 1e-6 * data.nbytes

    with lazystack(example_hdf_path, dset_name="images") as stack:
        expected_str = (
            f"{type(stack).__name__} object referencing {data.shape[0]} "
            f"{data.dtype} images of shape {data.shape[1:]}. Each image "
            f"occupies {expected_image_nbytes_mb:.2f} MB on disk, totalling "
            f"{expected_nbytes_mb:.2f} MB."
        )

        assert str(stack) == expected_str
        assert stack.info == expected_str


def test_view_str(example_hdf_path):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        view = stack[2:8, 1:, 2:]

        image_nbytes_mb = 1e-6 * view.image_nbytes
        nbytes_mb = 1e-6 * view.nbytes
        expected_str = (
            f"View object referencing {view.shape[0]} {view.dtype} images of "
            f"shape {view.shape[1:]}. Each image occupies "
            f"{image_nbytes_mb:.2f} MB when materialised, totalling "
            f"{nbytes_mb:.2f} MB."
        )

        assert str(view) == expected_str
        assert view.info == expected_str


@pytest.mark.parametrize("expr", NEWAXIS_EXPRS)
def test_stack_reject_newaxis(example_hdf_path, expr):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(NotImplementedError),
    ):
        stack[expr]


@pytest.mark.parametrize("expr", NEWAXIS_EXPRS)
def test_view_reject_newaxis(example_hdf_path, expr):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(NotImplementedError),
    ):
        stack[:][expr]


def test_stack_reject_bool_in_spatial_position(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(TypeError),
    ):
        stack[:, True, :]


@pytest.mark.parametrize("bad", [[None], [0.5], ["x"]])
def test_stack_reject_non_integer_index_array(example_hdf_path, bad):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        with pytest.raises(TypeError):
            stack[bad]
        with pytest.raises(TypeError):
            stack[:][bad]


def test_view_reject_bad_mask_length(example_hdf_path):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises(ValueError),
    ):
        stack[:][np.array([True, False])]


@pytest.mark.parametrize("bad", ["x", 1.5])
def test_stack_reject_unsupported_index_type(example_hdf_path, bad):
    with lazystack(example_hdf_path, dset_name="images") as stack:
        with pytest.raises(TypeError):
            stack[bad]
        with pytest.raises(TypeError):
            stack[bad, :, :]
        with pytest.raises(TypeError):
            stack[:][bad]


@pytest.mark.parametrize("bad", [1.5, "x", [0.5]])
def test_stack_reject_invalid_spatial_index(example_hdf_path, bad):
    with (
        lazystack(example_hdf_path, dset_name="images") as stack,
        pytest.raises((IndexError, TypeError)),
    ):
        stack[:, bad, :]


def test_ellipsis_stack_view_parity(example_hdf_path, example_3d_data):
    data = example_3d_data.astype(np.uint16)

    with lazystack(example_hdf_path, dset_name="images") as stack:
        assert np.array_equal(np.asarray(stack[..., :, :]), data)
        assert np.array_equal(np.asarray(stack[2:8][..., :, :]), data[2:8])
        assert np.array_equal(np.asarray(stack[:, ..., :]), data)


def test_hdf_reject_bad_ndim(tmp_path, example_3d_data):
    bad_data = example_3d_data[np.newaxis, ...]

    output_path = tmp_path / "tmp.h5"
    with h5py.File(output_path, "w") as tmp_hdf:
        tmp_hdf.create_dataset(name="images", data=bad_data, dtype="uint16")

    with pytest.raises(ValueError, match="Only stacks of 2D arrays"):
        lazystack(output_path, "images")
