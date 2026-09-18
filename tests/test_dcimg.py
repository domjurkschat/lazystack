# MIT License
#
# Copyright (c) 2017 Giacomo Mazzamuto
# Copyright (c) 2026 Dominic Jurkschat
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Vendored for lazystack from https://github.com/domjurkschat/dcimg (v0.7.0),
# a fork of https://github.com/kushalbakshi/dcimg,
# itself a fork of https://github.com/lens-biophotonics/dcimg.

import numpy as np
import pytest

from lazystack._dcimg import (
    FILE_HEADER_DTYPE,
    FMT_NEW,
    FMT_OLD,
    NEW_CROP_INFO_DTYPE,
    NEW_SESSION_HEADER_DTYPE,
    SESSION_FOOTER2_DTYPE,
    SESSION_FOOTER_DTYPE,
    SESSION_HEADER_DTYPE,
    DCIMGFile,
)

TEST_VECTORS = [
    np.index_exp[...],
    np.index_exp[..., -1],
    np.index_exp[:, :, :],
    np.index_exp[..., ::-1],
    np.index_exp[..., ::-1, :],
    np.index_exp[::-1, ...],
    np.index_exp[::-1, ::-1, ::-1],
    np.index_exp[..., -10:-21:-1],
    np.index_exp[2, 0, -10:-21:-1],
    np.index_exp[2, 0:4, -10:-21:-1],
    np.index_exp[-4:-7:-1, 0, 0],
    np.index_exp[-2:-7:-2, 0, 0],
    np.index_exp[-2:-7:-3, 0, 0],
    np.index_exp[-2:-7:-4, 0, 0],
    np.index_exp[-1::-4, 0, 0],
    np.index_exp[-6:-4, 0, 0],
    np.index_exp[2:4, 0:4, 0:4],
    np.index_exp[..., 2:],
    np.index_exp[..., -18:-16],
    np.index_exp[..., -19:-17],
    np.index_exp[..., -17:-19],
    np.index_exp[..., -19::2],
    np.index_exp[2:4, 0:10, 0:10],
    np.index_exp[2:4, 7:10, 7:10],
    np.index_exp[2, 0, -2],
    np.index_exp[2:5, :6, :],
    np.index_exp[2:5, 0:4, -10:-21:-1],
    np.index_exp[2:5, :, -10:-21:-1],
    np.index_exp[2:5, 0],
    np.index_exp[0, 0, 0],
    np.index_exp[0, 0, 2],
    np.index_exp[5, 5, 5],
    np.index_exp[5, -11:-9, 0:4],
    np.index_exp[5, -9:-11:-1, 0:4],
    np.index_exp[5, -2:-5:-1, 0:4],
    np.index_exp[5, -10, 0:4],
    np.index_exp[-5, -10, 0:4],
    np.index_exp[-5, -10, -10:-21:-1],
    np.index_exp[-5, 0, -10:-21:-1],
    np.index_exp[:, :, -10:-21:-2],
    np.index_exp[:, :, 0:10:2],
    np.index_exp[:, 0:10:2, 1:10:2],
    np.index_exp[:, 0, 1:10:2],
    np.index_exp[5],
    np.index_exp[:5],
    np.index_exp[:50],
    np.index_exp[:50, :50, :50],
    np.index_exp[:3000, :3000, :3000],
    np.index_exp[:, 0:0:1, :],
    np.index_exp[..., 0:0:1],
    np.index_exp[[0, 2, 5]],
    np.index_exp[np.array([0, 2, 5])],
    np.index_exp[np.array([9, 8, 7])],
    np.index_exp[[0, 2, 5], 0:4, 0:4],
    np.index_exp[
        np.array(
            [True, False, True, False, True, False, True, False, True, False]
        )
    ],
    np.index_exp[np.uint16(1), np.uint32(10), np.int64(5) :],
    np.index_exp[0, 0, -2046:8],
    np.index_exp[:, :, :-1],
    np.index_exp[Ellipsis, slice(None)],
    [True, False, True, False, True, False, True, False, True, False],
]


_HEADER_SIZE = 256
_OLD_OFFSET_TO_DATA = 256
_NEW_OFFSET_TO_DATA = 1024
_CROP_INFO_OFFSET = 712


class DCIMGFileOverride4px(DCIMGFile):
    @property
    def _has_4px_data(self) -> bool:
        return True

    def _compute_target_line(self) -> None:
        self._target_line = 0


def _make_config():
    num_frames = 10
    first_4px = (65000 - np.arange(num_frames * 4)).reshape((num_frames, 4))

    # FMT_OLD reader.
    f_old = DCIMGFileOverride4px()
    header = np.zeros(1, dtype=SESSION_HEADER_DTYPE)
    header["nfrms"][0] = num_frames
    header["ysize"][0] = 2048
    header["xsize"][0] = 2048
    header["byte_depth"][0] = 2
    f_old._session_header = header
    f_old._images = np.arange(np.prod(f_old.shape), dtype=np.uint16).reshape(
        f_old.shape
    )
    f_old._images.flags.writeable = False
    f_old.first_4px_correction_enabled = True
    f_old._format_version = FMT_OLD
    f_old._first_4px = np.copy(first_4px)
    f_old._first_4px.flags.writeable = False
    f_old._compute_target_line()

    # FMT_NEW reader.
    f_new = DCIMGFile()
    header = np.zeros(1, dtype=NEW_SESSION_HEADER_DTYPE)
    header["nfrms"][0] = num_frames
    header["ysize"][0] = 2048
    header["xsize"][0] = 2048
    header["byte_depth"][0] = 2
    header["frame_footer_size"][0] = 32
    f_new._session_header = header
    file_header = np.zeros(1, dtype=FILE_HEADER_DTYPE)
    file_header["format_version"] = 0x1000000
    f_new._file_header = file_header
    f_new._images = np.arange(np.prod(f_new.shape), dtype=np.uint16).reshape(
        f_new.shape
    )
    f_new._images.flags.writeable = False
    f_new.first_4px_correction_enabled = True
    f_new._format_version = FMT_NEW
    f_new._first_4px = np.copy(first_4px)
    f_new._first_4px.flags.writeable = False
    f_new._compute_target_line()

    expected_old = np.copy(f_old._images)
    expected_old[:, 0, 0:4] = f_old._first_4px
    expected_old = np.copy(expected_old)
    expected_old.flags.writeable = False

    expected_new = np.copy(f_new._images)
    expected_new[:, 1023, 0:4] = f_new._first_4px
    expected_new = np.copy(expected_new)
    expected_new.flags.writeable = False

    return [(f_old, expected_old), (f_new, expected_new)]


@pytest.fixture(scope="module")
def config():
    return _make_config()


@pytest.mark.parametrize("value", TEST_VECTORS)
def test_getitem(config, value):
    for reader, expected in config:
        actual = reader[value]
        if actual.size == 0:
            assert actual.size == expected[value].size
        else:
            assert np.array_equal(expected[value], actual)


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param((0, [0, 1], slice(None)), id="fancy-not-on-z"),
        pytest.param(
            (slice(None), np.array([0, 1]), slice(None)), id="fancy-on-y"
        ),
        pytest.param(np.array([[0, 1], [2, 3]]), id="fancy-2d"),
    ],
)
def test_getitem_rejects(config, expr):
    for reader, _ in config:
        with pytest.raises(TypeError):
            reader[expr]


def _make_disabled_reader(fmt):
    num_frames, height, width = 2, 3, 4
    first_4px = (100 + np.arange(num_frames * 4)).reshape(num_frames, 4)

    reader = DCIMGFileOverride4px()
    header = np.zeros(1, dtype=SESSION_HEADER_DTYPE)
    header["nfrms"][0] = num_frames
    header["ysize"][0] = height
    header["xsize"][0] = width
    header["byte_depth"][0] = 2
    reader._session_header = header

    if fmt == FMT_NEW:
        file_header = np.zeros(1, dtype=FILE_HEADER_DTYPE)
        file_header["format_version"] = 0x1000000
        reader._file_header = file_header

    reader._images = np.arange(
        num_frames * height * width, dtype=np.uint16
    ).reshape(num_frames, height, width)
    reader._images.flags.writeable = False
    reader._format_version = fmt
    reader._first_4px = np.copy(first_4px)
    reader.first_4px_correction_enabled = False
    reader._compute_target_line()
    return reader


@pytest.mark.parametrize("fmt", [FMT_OLD, FMT_NEW])
def test_getitem_correction_disabled(fmt):
    reader = _make_disabled_reader(fmt)

    expected = np.copy(reader._images)
    expected[:, 0, 0:4] = 0

    assert np.array_equal(reader[:, :, 0:4], expected[:, :, 0:4])
    assert reader[1, 0, 2] == 0
    assert reader[1, 2, 2] == expected[1, 2, 2]


def test_dtype_reject_bad_depth():
    reader = DCIMGFile()
    header = np.zeros(1, dtype=SESSION_HEADER_DTYPE)
    header["byte_depth"] = 3
    reader._session_header = header

    with pytest.raises(ValueError, match="Unsupported byte depth"):
        _ = reader.dtype


def _dcimg_bytes(
    images,
    *,
    old=True,
    version=0x1000000,
    frame_footer_size=0,
    first_4px=None,
    y0=0,
    crop=(0, 0),
    bytes_per_row=None,
):
    """Build a synthetic DCIMG file as a mutable bytearray."""
    images = np.asarray(images)
    num_frames, height, width = images.shape
    byte_depth = images.dtype.itemsize
    bytes_per_row = bytes_per_row or width * byte_depth
    bytes_per_img = bytes_per_row * height

    if old:
        offset_to_data = _OLD_OFFSET_TO_DATA
        data_offset = _HEADER_SIZE + offset_to_data
        session_data_size = offset_to_data + bytes_per_img * num_frames
        footer_offset = _HEADER_SIZE + session_data_size
        total = footer_offset + 272 + 12 * num_frames + 64
    else:
        offset_to_data = _NEW_OFFSET_TO_DATA
        data_offset = _HEADER_SIZE + offset_to_data
        total = (
            data_offset + num_frames * (bytes_per_img + frame_footer_size) + 64
        )

    buffer = bytearray(total)

    file_header = np.frombuffer(buffer, dtype=FILE_HEADER_DTYPE, count=1)
    file_header["file_format"] = b"DCIMG"
    file_header["format_version"] = 0x7 if old else version
    file_header["nsess"] = 1
    file_header["nfrms"] = num_frames
    file_header["header_size"] = _HEADER_SIZE
    file_header["file_size"] = total
    file_header["file_size2"] = total

    frame_stride = bytes_per_img if old else bytes_per_img + frame_footer_size

    for frame in range(num_frames):
        frame_offset = data_offset + frame * frame_stride
        for row in range(height):
            row_offset = frame_offset + row * bytes_per_row
            buffer[row_offset : row_offset + width * byte_depth] = images[
                frame, row
            ].tobytes()

    if old:
        session_header = np.frombuffer(
            buffer,
            dtype=SESSION_HEADER_DTYPE,
            count=1,
            offset=_HEADER_SIZE,
        )
        session_header["nfrms"] = num_frames
        session_header["byte_depth"] = byte_depth
        session_header["xsize"] = width
        session_header["ysize"] = height
        session_header["bytes_per_row"] = bytes_per_row
        session_header["bytes_per_img"] = bytes_per_img
        session_header["offset_to_data"] = offset_to_data
        session_header["session_data_size"] = session_data_size

        footer_size = 128
        offset_to_4px = 0
        if first_4px is not None:
            offset_to_4px = footer_size - 8 * num_frames

        session_footer = np.frombuffer(
            buffer,
            dtype=SESSION_FOOTER_DTYPE,
            count=1,
            offset=footer_offset,
        )
        session_footer["format_version"] = 0x7
        session_footer["offset_to_2nd_struct"] = footer_size
        session_footer["footer_size"] = footer_size

        session_footer2 = np.frombuffer(
            buffer,
            dtype=SESSION_FOOTER2_DTYPE,
            count=1,
            offset=footer_offset + footer_size,
        )
        session_footer2["offset_to_4px"] = offset_to_4px
        session_footer2["4px_offset_in_frame"] = 0
        session_footer2["4px_size"] = 8

        if first_4px is not None:
            patch = np.frombuffer(
                buffer,
                dtype=images.dtype,
                count=num_frames * 4,
                offset=footer_offset + offset_to_4px,
            )
            patch[:] = np.asarray(first_4px, dtype=images.dtype).ravel()

        framestamps = np.frombuffer(
            buffer, dtype="<u4", count=num_frames, offset=footer_offset + 272
        )
        framestamps[:] = 1000 + np.arange(num_frames)

        timestamps = np.frombuffer(
            buffer,
            dtype="<u4",
            count=num_frames * 2,
            offset=footer_offset + 272 + 4 * num_frames,
        ).reshape(num_frames, 2)
        timestamps[:, 0] = 1000 + np.arange(num_frames)
        timestamps[:, 1] = 10 + np.arange(num_frames)

    else:
        session_header = np.frombuffer(
            buffer,
            dtype=NEW_SESSION_HEADER_DTYPE,
            count=1,
            offset=_HEADER_SIZE,
        )
        session_header["nfrms"] = num_frames
        session_header["byte_depth"] = byte_depth
        session_header["xsize"] = width
        session_header["ysize"] = height
        session_header["bytes_per_row"] = bytes_per_row
        session_header["bytes_per_img"] = bytes_per_img
        session_header["offset_to_data"] = offset_to_data
        session_header["frame_footer_size"] = frame_footer_size

        crop_info = np.frombuffer(
            buffer,
            dtype=NEW_CROP_INFO_DTYPE,
            count=1,
            offset=_HEADER_SIZE + _CROP_INFO_OFFSET,
        )
        crop_info["x0"] = 0
        crop_info["xsize"] = crop[0]
        crop_info["y0"] = y0
        crop_info["ysize"] = crop[1]

        for frame in range(num_frames):
            footer = data_offset + frame * frame_stride + bytes_per_img

            if frame_footer_size >= 12:
                framestamp = np.frombuffer(
                    buffer, dtype="<u4", count=1, offset=footer
                )
                framestamp[0] = 1000 + frame

                timestamp = np.frombuffer(
                    buffer, dtype="<u4", count=2, offset=footer + 4
                )
                timestamp[0] = 1000 + frame
                timestamp[1] = 10 + frame

            if first_4px is not None and frame_footer_size >= 20:
                patch = np.asarray(first_4px, dtype=images.dtype)[frame]
                buffer[footer + 12 : footer + 20] = patch.tobytes()

    return buffer


@pytest.fixture()
def dcimg_bytes():
    return _dcimg_bytes


def _set_field(buffer, dtype, offset, field, value):
    np.frombuffer(buffer, dtype=dtype, count=1, offset=offset)[field] = value


def _assert_metadata(reader, num_frames):
    expected_stamps = 1000 + np.arange(num_frames)
    assert np.array_equal(reader.framestamps, expected_stamps)

    expected_times = np.array(
        [
            np.datetime64((1000 + i) * 10**6 + 10 + i, "us")
            for i in range(num_frames)
        ]
    )
    assert np.array_equal(reader.timestamps, expected_times)
    assert reader.timestamp(0) == expected_times[0]


def test_dcimg_old_basic(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames, height, width = data.shape
    bytes_per_img = width * 2 * height
    path = tmp_path / "old.dcimg"
    path.write_bytes(dcimg_bytes(data))

    reader = DCIMGFile(path)

    try:
        assert reader.shape == data.shape
        assert reader.dtype == data.dtype
        assert reader.byte_depth == 2
        assert reader.width == width
        assert reader.height == height
        assert reader.bytes_per_row == width * 2
        assert reader.bytes_per_img == bytes_per_img
        assert reader.num_frames == num_frames
        assert reader.file_size == path.stat().st_size
        assert reader.file_path == path
        assert repr(reader) == (
            f'<DCIMGFile file_path="{path}" shape={data.shape} '
            f"dtype={data.dtype}>"
        )
        assert reader._session_footer_offset == (
            _HEADER_SIZE + _OLD_OFFSET_TO_DATA + bytes_per_img * num_frames
        )
        assert reader._has_4px_data is False
        assert reader._target_line == -1
        assert np.array_equal(reader[...], data)
        _assert_metadata(reader, num_frames)

    finally:
        reader.close()


def test_dcimg_old_4px(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames = data.shape[0]
    first_4px = (
        (65000 - np.arange(num_frames * 4))
        .reshape(num_frames, 4)
        .astype(np.uint16)
    )
    path = tmp_path / "old4px.dcimg"
    path.write_bytes(dcimg_bytes(data, first_4px=first_4px))

    expected = np.copy(data)
    expected[:, 0, 0:4] = first_4px

    reader = DCIMGFile(path)

    try:
        assert reader._has_4px_data is True
        assert reader._target_line == 0
        assert np.array_equal(reader[...], expected)
        assert np.array_equal(reader[3], expected[3])
        assert np.array_equal(reader[:, 0, 0:4], expected[:, 0, 0:4])

    finally:
        reader.close()


def test_dcimg_old_uint8(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint8)
    path = tmp_path / "old8.dcimg"
    path.write_bytes(dcimg_bytes(data))

    reader = DCIMGFile(path)

    try:
        assert reader.dtype == np.dtype(np.uint8)
        assert reader._has_4px_data is False
        assert np.array_equal(reader[...], data)

    finally:
        reader.close()


def test_dcimg_new_camlink(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames = data.shape[0]
    path = tmp_path / "new.dcimg"
    path.write_bytes(dcimg_bytes(data, old=False, frame_footer_size=32))

    reader = DCIMGFile(path)

    try:
        assert reader.shape == data.shape
        assert reader._has_4px_data is True
        assert reader._target_line == 1023
        assert np.array_equal(reader[...], data)
        _assert_metadata(reader, num_frames)

    finally:
        reader.close()


def test_dcimg_new_4px(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames = data.shape[0]
    first_4px = (
        (65000 - np.arange(num_frames * 4))
        .reshape(num_frames, 4)
        .astype(np.uint16)
    )
    path = tmp_path / "new4px.dcimg"
    path.write_bytes(
        dcimg_bytes(
            data,
            old=False,
            version=0x2000000,
            frame_footer_size=32,
            first_4px=first_4px,
        )
    )

    expected = np.copy(data)
    expected[:, 0, 0:4] = first_4px

    reader = DCIMGFile(path)

    try:
        assert reader._has_4px_data is True
        assert reader._target_line == 0
        assert np.array_equal(reader[...], expected)
        _assert_metadata(reader, num_frames)

    finally:
        reader.close()


def test_dcimg_new_big_footer(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames = data.shape[0]
    first_4px = (
        (60000 - np.arange(num_frames * 4))
        .reshape(num_frames, 4)
        .astype(np.uint16)
    )
    path = tmp_path / "new512.dcimg"
    path.write_bytes(
        dcimg_bytes(
            data,
            old=False,
            version=0x2000000,
            frame_footer_size=512,
            first_4px=first_4px,
        )
    )

    expected = np.copy(data)
    expected[:, 0, 0:4] = first_4px

    reader = DCIMGFile(path)

    try:
        assert reader._has_4px_data is True
        assert np.array_equal(reader[...], expected)

    finally:
        reader.close()


def test_dcimg_new_no_footer(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    path = tmp_path / "new0.dcimg"
    path.write_bytes(dcimg_bytes(data, old=False))

    reader = DCIMGFile(path)

    try:
        assert reader._has_4px_data is False
        assert np.array_equal(reader[...], data)

    finally:
        reader.close()


def test_dcimg_new_padded(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    _, height, width = data.shape
    bytes_per_row = width * 2 + 4
    path = tmp_path / "padded.dcimg"
    path.write_bytes(dcimg_bytes(data, old=False, bytes_per_row=bytes_per_row))

    reader = DCIMGFile(path)

    try:
        assert reader.bytes_per_row == bytes_per_row
        assert reader.bytes_per_img == bytes_per_row * height
        assert np.array_equal(reader[...], data)

    finally:
        reader.close()


def test_dcimg_new_binning(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames, height, width = data.shape
    first_4px = (
        (60000 - np.arange(num_frames * 4))
        .reshape(num_frames, 4)
        .astype(np.uint16)
    )
    path = tmp_path / "binned.dcimg"
    path.write_bytes(
        dcimg_bytes(
            data,
            old=False,
            frame_footer_size=32,
            first_4px=first_4px,
            y0=1019,
            crop=(2 * width, 2 * height),
        )
    )

    expected = np.copy(data)
    expected[:, 2, 0:4] = first_4px

    reader = DCIMGFile(path)

    try:
        assert reader.binning == 2
        assert reader._target_line == 2
        assert np.array_equal(reader[...], expected)

    finally:
        reader.close()


@pytest.mark.parametrize("version", [0x1000000, 0x1050000, 0x2000000])
def test_dcimg_new_versions(tmp_path, dcimg_bytes, example_3d_data, version):
    data = example_3d_data.astype(np.uint16)
    path = tmp_path / f"v{version:x}.dcimg"
    path.write_bytes(dcimg_bytes(data, old=False, version=version))

    reader = DCIMGFile(path)

    try:
        assert np.array_equal(reader[...], data)

    finally:
        reader.close()


def test_dcimg_reject_bad_magic(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)

    buffer = dcimg_bytes(data)
    _set_field(buffer, FILE_HEADER_DTYPE, 0, "file_format", b"XXXX")

    path = tmp_path / "bad.dcimg"
    path.write_bytes(buffer)

    with pytest.raises(ValueError, match="Invalid DCIMG file"):
        DCIMGFile(path)


@pytest.mark.parametrize("version", [0x0, 0x8, 0x2000001])
def test_dcimg_reject_bad_version(
    tmp_path, dcimg_bytes, example_3d_data, version
):
    data = example_3d_data.astype(np.uint16)

    buffer = dcimg_bytes(data)
    _set_field(buffer, FILE_HEADER_DTYPE, 0, "format_version", version)

    path = tmp_path / "bad.dcimg"
    path.write_bytes(buffer)

    with pytest.raises(ValueError, match="Invalid DCIMG format version"):
        DCIMGFile(path)


def test_dcimg_reject_bad_byte_depth(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)

    buffer = dcimg_bytes(data)
    _set_field(buffer, SESSION_HEADER_DTYPE, _HEADER_SIZE, "byte_depth", 3)

    path = tmp_path / "bad.dcimg"
    path.write_bytes(buffer)

    with pytest.raises(ValueError, match="Invalid byte-depth"):
        DCIMGFile(path)


def test_dcimg_reject_bad_bytes_per_img(
    tmp_path, dcimg_bytes, example_3d_data
):
    data = example_3d_data.astype(np.uint16)
    _, height, width = data.shape
    bytes_per_img = width * 2 * height

    buffer = dcimg_bytes(data)
    _set_field(
        buffer,
        SESSION_HEADER_DTYPE,
        _HEADER_SIZE,
        "bytes_per_img",
        bytes_per_img + 2,
    )

    path = tmp_path / "bad.dcimg"
    path.write_bytes(buffer)

    with pytest.raises(ValueError, match="invalid value for bytes_per_img"):
        DCIMGFile(path)


def test_dcimg_reject_binning_mismatch(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    _, height, width = data.shape

    buffer = dcimg_bytes(data, old=False)
    crop_info = np.frombuffer(
        buffer,
        dtype=NEW_CROP_INFO_DTYPE,
        count=1,
        offset=_HEADER_SIZE + _CROP_INFO_OFFSET,
    )
    crop_info["xsize"] = 2 * width
    crop_info["ysize"] = 3 * height

    path = tmp_path / "bad.dcimg"
    path.write_bytes(buffer)

    with pytest.raises(ValueError, match="different binning"):
        DCIMGFile(path)


def test_dcimg_reject_empty_file(tmp_path):
    path = tmp_path / "empty.dcimg"
    path.write_bytes(b"")

    with pytest.raises(ValueError):
        DCIMGFile(path)


def test_dcimg_reject_truncated_header(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    path = tmp_path / "trunc.dcimg"
    path.write_bytes(dcimg_bytes(data)[:40])

    with pytest.raises((ValueError, TypeError)):
        DCIMGFile(path)


def test_dcimg_reject_truncated_footer(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames, height, width = data.shape
    bytes_per_img = width * 2 * height
    footer_offset = (
        _HEADER_SIZE + _OLD_OFFSET_TO_DATA + bytes_per_img * num_frames
    )

    path = tmp_path / "trunc.dcimg"
    path.write_bytes(dcimg_bytes(data)[: footer_offset + 10])

    with pytest.raises((ValueError, TypeError)):
        DCIMGFile(path)


def test_dcimg_failed_open_releases_memmap(
    tmp_path, dcimg_bytes, example_3d_data
):
    data = example_3d_data.astype(np.uint16)

    path = tmp_path / "trunc.dcimg"
    path.write_bytes(dcimg_bytes(data)[:40])

    reader = DCIMGFile()

    with pytest.raises((ValueError, TypeError)):
        reader.open(path)

    assert reader._memmap is None


def test_dcimg_close_releases(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)

    path = tmp_path / "close.dcimg"
    path.write_bytes(dcimg_bytes(data))

    reader = DCIMGFile(path)
    reader.close()

    assert reader._memmap is None
    assert reader._images is None
    assert reader._first_4px is None

    reader.close()


def test_dcimg_reopen(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)
    num_frames = data.shape[0]
    first_4px = (
        (65000 - np.arange(num_frames * 4))
        .reshape(num_frames, 4)
        .astype(np.uint16)
    )
    path_4px = tmp_path / "4px.dcimg"
    path_4px.write_bytes(dcimg_bytes(data, first_4px=first_4px))

    path_plain = tmp_path / "plain.dcimg"
    path_plain.write_bytes(dcimg_bytes(data))

    reader = DCIMGFile(path_4px)
    assert reader._has_4px_data is True
    reader.open(path_plain)

    try:
        assert reader._has_4px_data is False
        assert reader.file_path == path_plain
        assert np.array_equal(reader[...], data)

    finally:
        reader.close()


def test_dcimg_open_path_arg(tmp_path, dcimg_bytes, example_3d_data):
    data = example_3d_data.astype(np.uint16)

    path = tmp_path / "arg.dcimg"
    path.write_bytes(dcimg_bytes(data))

    reader = DCIMGFile()
    reader.open(path)

    try:
        assert reader.file_path == path
        assert np.array_equal(reader[...], data)

    finally:
        reader.close()
