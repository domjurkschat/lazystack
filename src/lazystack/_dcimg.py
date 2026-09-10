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

# Based on information gathered from:
# https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py
# hamamatsuOrcaTools: https://github.com/orlandi/hamamatsuOrcaTools
# Python Microscopy: http://www.python-microscopy.org
#                    https://bitbucket.org/david_baddeley/python-microscopy

# Authors:
#   - Giacomo Mazzamuto <mazzamuto@lens.unifi.it> (original package)
#   - Dominic Jurkschat <dom.jurkschat@monash.edu> (lazystack)

# Vendored for lazystack from https://github.com/domjurkschat/dcimg (v0.7.0),
# a fork of https://github.com/kushalbakshi/dcimg,
# itself a fork of https://github.com/lens-biophotonics/dcimg.

"""
Vendored reader for Hamamatsu DCIMG files, adapted for lazystack.

This module vendors the single-file ``dcimg`` package. It's adapted from the
``domjurkschat/dcimg`` fork (v0.7.0), which itself adds NumPy 2.x support and
1-D fancy indexing along the frame (Z) axis on top of the upstream
``lens-biophotonics/dcimg`` reader (via ``kushalbakshi/dcimg``). ``DCIMGFile``
is memory-mapped and provides NumPy-style indexing plus frame/timestamp
metadata accessors.

The lazystack integration differs from the forks in that only the core read
path is kept: the ``zslice``/``zslice_idx``/``frame``/``whole`` convenience
methods, the ``as_dask_array`` helper, and the context-manager copy semantics
were removed, while the 4px-correction knob
(``first_4px_correction_enabled``) is retained as a plain bool. The module is
fully type-annotated to match the rest of lazystack.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

FILE_HEADER_DTYPE = np.dtype(
    [
        ("file_format", "S8"),
        ("format_version", "<u4"),  # 0x08
        ("skip", "5<u4"),  # 0x0c
        ("nsess", "<u4"),  # 0x20 ?
        ("nfrms", "<u4"),  # 0x24
        ("header_size", "<u4"),  # 0x28 ?
        ("skip2", "<u4"),  # 0x2c
        ("file_size", "<u8"),  # 0x30
        ("skip3", "2<u4"),  # 0x38
        ("file_size2", "<u8"),  # 0x40, repeated
    ]
)

SESSION_HEADER_DTYPE = np.dtype(
    [
        ("session_size", "<u8"),  # including footer
        ("skip1", "6<u4"),
        ("nfrms", "<u4"),
        ("byte_depth", "<u4"),
        ("skip2", "<u4"),
        ("xsize", "<u4"),
        ("bytes_per_row", "<u4"),
        ("ysize", "<u4"),
        ("bytes_per_img", "<u4"),
        ("skip3", "2<u4"),
        ("offset_to_data", "<u4"),
        ("session_data_size", "<u8"),  # header_size + x*y*byte_depth*nfrms
    ]
)

SESSION_FOOTER_DTYPE = np.dtype(
    [
        ("format_version", "<u4"),
        ("skip0", "<u4"),
        ("offset_to_2nd_struct", "<u8"),
        ("skip1", "2<u4"),
        ("offset_to_offset_to_end_of_data", "<u8"),
        ("skip2", "2<u4"),
        ("footer_size", "<u4"),
        ("skip3", "<u4"),
        # an almost empty part after the footer
        # contains 'offset_to_end_of_data', 0x00000000 0x00000000
        # repeated 2 * nfrms times
        ("2nd_footer_size", "<u4"),  # = 2 * nfrms * 16
        ("skip4", "19<u4"),
        ("offset_to_end_of_data", "<u8"),  # sum of the two offsets above
        ("skip5", "<u8"),
        ("offset_to_end_of_data_again", "<u8"),  # repeated
        ("skip6", "<u8"),  # repeated
    ]
)

SESSION_FOOTER2_DTYPE = np.dtype(
    [
        ("offset_to_offset_to_timestamps", "<u8"),
        ("skip0", "2<u4"),
        ("offset_to_offset_to_frame_counts", "<u8"),
        ("skip1", "2<u4"),
        ("offset_to_offset_to_4px", "<u8"),
        ("skip2", "2<u4"),
        ("offset_to_frame_counts", "<u8"),
        ("skip3", "2<u4"),
        ("offset_to_timestamps", "<u8"),
        ("skip4", "4<u4"),
        ("offset_to_4px", "<u8"),
        ("skip5", "<u4"),
        ("4px_offset_in_frame", "<u4"),
        # zero if there is no 4px correction info (e.g. cropped so the first
        # line is not included), 8 if 4px correction info is stored. Probably
        # the size in bytes of the 4px correction per frame (8 = 4 * 2).
        ("4px_size", "<u8"),
    ]
)

# newer versions of the dcimg format have a different header
NEW_SESSION_HEADER_DTYPE = np.dtype(
    [
        ("session_size", "<u8"),
        ("skip1", "13<u4"),
        ("nfrms", "<u4"),
        ("byte_depth", "<u4"),
        ("skip2", "<u4"),
        ("xsize", "<u4"),
        ("ysize", "<u4"),
        ("bytes_per_row", "<u4"),
        ("bytes_per_img", "<u4"),
        ("skip3", "2<u4"),
        ("offset_to_data", "<u8"),
        ("skip4", "5<u4"),
        ("frame_footer_size", "<u4"),
    ]
)

NEW_FRAME_FOOTER_CAMLINK_DTYPE = np.dtype(
    [
        ("progressive_number", "<u4"),
        ("timestamp", "<u4"),
        ("timestamp_frac", "<u4"),
        ("4px", "<u8"),
        ("zeros", "3<u4"),
    ]
)

NEW_CROP_INFO_DTYPE = np.dtype(
    [
        ("x0", "<u2"),
        ("xsize", "<u2"),
        ("y0", "<u2"),
        ("ysize", "<u2"),
    ]
)

FMT_OLD = 1
FMT_NEW = 2


class DCIMGFile:
    """Hamamatsu DCIMG file, memory-mapped.

    Image data is accessed via NumPy indexing: integer, slice, and 1-D fancy
    indexing along the frame (z) axis. The first 4 (or 8, for ``uint8``) pixels
    of certain lines are stored separately in the file and are patched in when
    ``first_4px_correction_enabled`` is ``True`` (default).

    Attributes:
        shape (tuple[int, int, int]): Dimensions as
            (num_images, height, width).
        dtype (np.dtype): Data type of each image.
        first_4px_correction_enabled (bool): Patch in the separately-stored
            first pixels when ``True``; zero them when ``False``.
        framestamps (npt.NDArray[np.uint32]): Framestamps of all frames.
        timestamps (npt.NDArray[np.datetime64]): Timestamps of all frames.
    """

    def __init__(self, file_path: str | Path | None = None) -> None:
        self._memmap = None
        self._images = None

        self._file_header = None
        self._session_header = None
        self._session_footer = None
        self._session_footer2 = None
        self._time_stamps = None
        self._frame_stamps = None

        self.file_path = file_path
        if file_path:
            self.file_path = Path(file_path)

        self._format_version = None
        self.x0 = 0
        self.y0 = 0
        self.binning = 1
        self._target_line = -1

        self.first_4px_correction_enabled = True
        self._first_4px = None

        if file_path is not None:
            self.open()

    def __repr__(self) -> str:
        return (
            f'<DCIMGFile file_path="{self.file_path}" '
            f"shape={self.shape} dtype={self.dtype}>"
        )

    def __del__(self) -> None:
        self.close()

    @property
    def file_size(self) -> int:
        """File size in bytes."""
        return self._file_header["file_size"].item()

    @property
    def num_frames(self) -> int:
        """Number of frames (z planes)."""
        return self._session_header["nfrms"].item()

    @property
    def byte_depth(self) -> int:
        """Number of bytes per pixel."""
        return self._session_header["byte_depth"].item()

    @property
    def dtype(self) -> np.dtype:
        """NumPy numerical dtype."""
        if self.byte_depth == 1:
            return np.dtype(np.uint8)
        if self.byte_depth == 2:
            return np.dtype(np.uint16)
        raise ValueError(f"Unsupported byte depth: {self.byte_depth}")

    @property
    def width(self) -> int:
        return self._session_header["xsize"].item()

    @property
    def height(self) -> int:
        return self._session_header["ysize"].item()

    @property
    def bytes_per_row(self) -> int:
        return self._session_header["bytes_per_row"].item()

    @property
    def bytes_per_img(self) -> int:
        return self._session_header["bytes_per_img"].item()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the whole image stack, as (num_images, height, width)."""
        return self.num_frames, self.height, self.width

    @property
    def _header_size(self) -> int:
        return self._file_header["header_size"].item()

    @property
    def _session_footer_offset(self) -> int:
        header = self._session_header
        if self._format_version == FMT_OLD:
            session_data_size = header["session_data_size"].item()
        else:  # FMT_NEW
            session_data_size = (
                header["offset_to_data"].item()
                + (header["bytes_per_img"].item() + 8) * self.num_frames
            )
        return self._header_size + session_data_size

    def open(self, file_path: str | Path | None = None) -> None:
        self.close()
        if file_path is not None:
            self.file_path = Path(file_path)

        self._memmap = np.memmap(self.file_path, mode="r")

        try:
            self._parse_header()
            self._parse_footer()
        except ValueError:
            self.close()
            raise

        bytes_per_pixel = self.byte_depth
        data_offset = (
            self._file_header["header_size"].item()
            + self._session_header["offset_to_data"].item()
        )
        frame_footer_size = None
        if self._format_version == FMT_OLD:
            if self._has_4px_data:
                offset = (
                    self._session_footer_offset
                    + self._session_footer2["offset_to_4px"].item()
                )
                self._first_4px = np.ndarray(
                    (self.num_frames, 4), self.dtype, self._memmap, offset
                )
            data_strides = (
                self.bytes_per_img,
                self.bytes_per_row,
                bytes_per_pixel,
            )
        elif self._format_version == FMT_NEW:
            frame_footer_size = self._session_header[
                "frame_footer_size"
            ].item()
            strides = (self.bytes_per_img + frame_footer_size, bytes_per_pixel)
            if self._has_4px_data:
                self._first_4px = np.ndarray(
                    (self.num_frames, 8 // self.byte_depth),
                    self.dtype,
                    self._memmap,
                    data_offset + self.bytes_per_img + 12,
                    strides,
                )
            padding = (
                self.bytes_per_img - self.width * self.height * bytes_per_pixel
            )
            padding //= self.height
            data_strides = (
                self.bytes_per_img + frame_footer_size,
                self.width * bytes_per_pixel + padding,
                bytes_per_pixel,
            )

        self._images = np.ndarray(
            self.shape, self.dtype, self._memmap, data_offset, data_strides
        )

        if self._format_version == FMT_OLD:
            # Framestamp offset.
            offset = self._session_footer_offset + 272
            self._frame_stamps = np.ndarray(
                self.num_frames, np.uint32, self._memmap, offset
            )

            # Timestamp offset.
            offset += 4 * self.num_frames
            self._time_stamps = np.ndarray(
                (self.num_frames, 2), np.uint32, self._memmap, offset
            )
        elif self._format_version == FMT_NEW:
            # Framestamps.
            offset = (
                self._file_header["header_size"].item()
                + self._session_header["offset_to_data"].item()
                + self.bytes_per_img
            )
            strides = self.bytes_per_img + frame_footer_size
            self._frame_stamps = np.ndarray(
                self.num_frames, np.uint32, self._memmap, offset, strides
            )

            # Timestamps.
            offset += 4
            strides = (self.bytes_per_img + frame_footer_size, 4)
            self._time_stamps = np.ndarray(
                (self.num_frames, 2), np.uint32, self._memmap, offset, strides
            )

        self._compute_target_line()

    def _compute_target_line(self) -> None:
        if self._format_version == FMT_OLD:
            if self._has_4px_data:
                self._target_line = (
                    self._session_footer2["4px_offset_in_frame"].item()
                    // self.bytes_per_row
                )
            else:
                self._target_line = -1
        elif self._file_header["format_version"].item() == 0x2000000:
            self._target_line = 0
        else:
            self._target_line = (1023 - self.y0) // self.binning

    def close(self) -> None:
        self._memmap = None

    def _parse_header(self) -> None:
        self._file_header = np.ndarray((1,), FILE_HEADER_DTYPE, self._memmap)

        if self._file_header["file_format"].item() != b"DCIMG":
            raise ValueError("Invalid DCIMG file")

        format_version = self._file_header["format_version"].item()
        if format_version == 0x7:
            session_dtype = SESSION_HEADER_DTYPE
            self._format_version = FMT_OLD
        elif format_version in (0x1000000, 0x2000000, 0x1050000):
            self._format_version = FMT_NEW
            session_dtype = NEW_SESSION_HEADER_DTYPE
        else:
            raise ValueError(
                f"Invalid DCIMG format version: {format_version:#x}"
            )

        self._session_header = np.ndarray(
            (1,), session_dtype, self._memmap, offset=self._header_size
        )

        if self._format_version == FMT_NEW:
            crop_offset = self._header_size + 712
            crop_info = np.ndarray(
                (1,), NEW_CROP_INFO_DTYPE, self._memmap, crop_offset
            )

            self.x0 = crop_info["x0"].item()
            self.y0 = crop_info["y0"].item()
            binning_x = crop_info["xsize"].item() // self.width
            binning_y = crop_info["ysize"].item() // self.height

            if binning_x != binning_y:
                raise ValueError("different binning in X and Y")

            if binning_x > 0:
                self.binning = binning_x

        if self.byte_depth not in (1, 2):
            raise ValueError(f"Invalid byte-depth: {self.byte_depth}")

        if self.bytes_per_img != self.bytes_per_row * self.height:
            raise ValueError("invalid value for bytes_per_img")

    def _parse_footer(self) -> None:
        if self._format_version != FMT_OLD:
            return

        self._session_footer = np.ndarray(
            (1,),
            SESSION_FOOTER_DTYPE,
            self._memmap,
            self._session_footer_offset,
        )

        offset = (
            self._session_footer_offset
            + self._session_footer["offset_to_2nd_struct"].item()
        )

        self._session_footer2 = np.ndarray(
            (1,), SESSION_FOOTER2_DTYPE, self._memmap, offset
        )

    @property
    def _has_4px_data(self) -> bool:
        """Whether the footer contains 4px correction data."""
        if self._format_version == FMT_NEW:
            if self._session_header["frame_footer_size"].item() >= 512:
                return True
            return (
                NEW_FRAME_FOOTER_CAMLINK_DTYPE.itemsize
                == self._session_header["frame_footer_size"].item()
            )

        footer_size = self._session_footer["footer_size"].item()
        offset_to_4px = self._session_footer2["offset_to_4px"].item()
        return footer_size == offset_to_4px + 8 * self.num_frames

    def __getitem__(
        self, items: int | slice | list | npt.NDArray[np.integer] | tuple
    ) -> npt.NDArray | np.generic | int:
        """Allow access to image data using NumPy indexing."""
        data = self._images[items]

        if data.size == 0:
            return data

        # A bare list/array is a z-axis fancy index; normalize to a tuple.
        if isinstance(items, (list, np.ndarray)):
            items = (items,)
        elif not isinstance(items, tuple):
            items = np.index_exp[items]

        # Normalize each axis; only the frame (z) axis may be fancy-indexed.
        indices = []
        for item in items:
            if item is Ellipsis:
                for _ in range(3 - len(items) + 1):
                    indices.append(slice(0, self.shape[len(indices)], 1))
                continue
            if isinstance(item, (list, np.ndarray)):
                if indices:
                    raise TypeError(f"Invalid type: {type(item)}")
                if np.ndim(item) != 1:
                    raise TypeError("Only 1-D fancy indexing is supported")
                indices.append(item)
                continue
            indices.append(
                self._index_to_slice(item, self.shape[len(indices)])
            )

        for _ in range(3 - len(indices)):
            indices.append(slice(0, self.shape[len(indices)], 1))

        z_index = (
            indices[0]
            if isinstance(indices[0], (list, np.ndarray))
            else indices[0].start
        )

        startx = indices[2].start
        stopx = indices[2].stop
        stepx = indices[2].step

        starty = indices[1].start
        stopy = indices[1].stop
        stepy = indices[1].step

        target_line = self._target_line
        condition_y = False
        if self._format_version == FMT_OLD and self._has_4px_data:
            condition_y = starty == 0 or stopy == 0
        elif self._format_version == FMT_NEW and self._has_4px_data:
            if stepy > 0:
                condition_y = starty <= target_line <= stopy
            elif stepy < 0:
                condition_y = stopy <= target_line <= starty

        num_first_pixels = 8 // self.byte_depth

        if condition_y and (
            (0 <= startx < num_first_pixels) or stopx < num_first_pixels
        ):
            if data.size == 1:
                if self.first_4px_correction_enabled:
                    data = self._first_4px[z_index, startx]
                else:
                    data = 0
                return data

            if startx < stopx:
                newstartx = 0
                if stopx > num_first_pixels:
                    newstopx = math.ceil(
                        (num_first_pixels - startx) / abs(stepx)
                    )
                else:
                    newstopx = (stopx - startx) // abs(stepx)
            else:
                newstopx = data.shape[-1]
                if data.shape[-1] < num_first_pixels:
                    newstartx = 0
                else:
                    newstartx = data.shape[-1] - num_first_pixels // abs(stepx)

            if newstartx == newstopx:
                return np.empty([0])

            newshape = [self._len_index(index) for index in indices]

            old_shape = data.shape

            data = np.reshape(data, newshape)

            newy = math.floor((target_line - starty) / stepy)
            if stepy < 0:
                newy -= 1

            index_exp = np.index_exp[..., newy, newstartx:newstopx]

            if not data.flags.writeable:
                data = np.copy(data)

            if self.first_4px_correction_enabled:
                bounds = sorted((startx, stopx))
                first_start = max(0, bounds[0])
                first_stop = min(8 // self.byte_depth, bounds[1])
                first_4px = self._first_4px[
                    items[0], first_start : first_stop : abs(stepx)
                ]

                if stepx < 0:
                    first_4px = first_4px[..., ::-1]
                data[index_exp] = first_4px
            else:
                data[index_exp] = 0

            data = np.reshape(data, old_shape)

        return data

    @property
    def framestamps(self) -> npt.NDArray[np.uint32]:
        """Framestamps of all frames."""
        return self._frame_stamps

    @property
    def timestamps(self) -> npt.NDArray[np.datetime64]:
        """Timestamps of all frames."""
        return np.asarray([self.timestamp(i) for i in range(self.num_frames)])

    def timestamp(self, frame: int) -> np.datetime64:
        """Timestamp of a single frame."""
        whole = int.from_bytes(self._time_stamps[frame, 0], "little")
        fraction = int.from_bytes(self._time_stamps[frame, 1], "little")
        return np.datetime64(whole * 10**6 + fraction, "us")

    @staticmethod
    def _index_to_slice(index: int | slice | np.integer, size: int) -> slice:
        if isinstance(index, int | np.integer):
            index = int(index)
            start, stop, step = index, index + 1, 1
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            step = index.step if index.step is not None else 1
        else:
            raise TypeError(f"Invalid type: {type(index)}")

        if start is None:
            start = 0 if step > 0 else size
        elif start < 0:
            start += size
            if stop is not None:
                stop += size
        elif start > size:
            start = size

        if stop is None:
            stop = size if step > 0 else 0
        elif stop < 0:
            stop += size
        elif stop > size:
            stop = size

        return slice(start, stop, step)

    @staticmethod
    def _len_index(index: slice | list | npt.NDArray[np.integer]) -> int:
        if isinstance(index, (list, np.ndarray)):
            if isinstance(index, np.ndarray) and index.dtype == np.bool_:
                return int(np.count_nonzero(index))
            return len(index)
        return math.ceil((index.stop - index.start) / index.step)
