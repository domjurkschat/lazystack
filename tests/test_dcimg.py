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
    NEW_SESSION_HEADER_DTYPE,
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
]


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
    f_old._images = np.arange(
        np.prod(f_old.shape), dtype=np.uint16
    ).reshape(f_old.shape)
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
    f_new._images = np.arange(
        np.prod(f_new.shape), dtype=np.uint16
    ).reshape(f_new.shape)
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
