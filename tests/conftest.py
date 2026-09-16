from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import pytest
from tifffile import TiffWriter, imwrite

from lazystack import lazystack

FORMATS = ["hdf", "his", "paged", "multi_file", "volumetric", "dcimg"]
# `multi_file` is excluded: paths-mode TIFF opens per read and already has no
#   persistent handle, so a close check would pass vacuously.
CLOSABLE = ["hdf", "his", "dcimg", "paged", "volumetric"]
HIS_FRAME_COUNT = slice(14, 18)


class FakeDCIMG:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.closed = False

    def __getitem__(self, index):
        return self.data[index]

    def close(self):
        self.closed = True


class FakePage:
    def __init__(self, photometric, samplesperpixel):
        self.photometric, self.samplesperpixel = photometric, samplesperpixel


@pytest.fixture()
def fake_page():
    return FakePage


@pytest.fixture(scope="module")
def example_3d_data():
    return np.arange(120).reshape(10, 3, 4)


def _write_hdf(directory: Path, data: npt.NDArray) -> Path:
    path = directory / "example.h5"
    with h5py.File(path, "w") as tmp_hdf:
        tmp_hdf.create_dataset(name="images", data=data, dtype="uint16")
    return path


def _his_bytes(data: npt.NDArray, file_type: int = 2) -> bytes:
    buffer = bytearray(64)
    # Number of metadata bytes.
    buffer[2:4] = (2).to_bytes(2, "little")
    # Width.
    buffer[4:6] = data.shape[2].to_bytes(2, "little")
    # Height.
    buffer[6:8] = data.shape[1].to_bytes(2, "little")
    # File type (uint16 by default).
    buffer[12:14] = (file_type).to_bytes(2, "little")
    # Number of images.
    buffer[HIS_FRAME_COUNT] = data.shape[0].to_bytes(4, "little")

    # Metadata placeholder.
    buffer += bytearray(2)

    # Add the images with variable gaps. Header -> gap -> image -> repeat,
    #   except the first image.
    gaps = np.resize(
        np.array([0, 2, 4, 2, 4, 0, 2, 6, 4, 10], dtype=np.uint16), len(data)
    )
    for i, image in enumerate(data):
        if i > 0:
            image_header = bytearray(64)
            image_header[2:4] = int(gaps[i]).to_bytes(2, "little")
            buffer += image_header
            buffer += bytes(int(gaps[i]))
        buffer += image.tobytes()

    return bytes(buffer)


@pytest.fixture()
def his_bytes():
    return _his_bytes


@pytest.fixture()
def his_header_fields():
    return {"frame_count": HIS_FRAME_COUNT}


def _write_his(directory: Path, data: npt.NDArray) -> Path:
    path = directory / "example.his"
    path.write_bytes(_his_bytes(data))
    return path


def _write_paged(directory: Path, data: npt.NDArray) -> Path:
    path = directory / "example_paged.ome.tif"
    with TiffWriter(path, ome=True, bigtiff=False) as writer:
        writer.write(
            data,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
    return path


def _write_multi_file(directory: Path, data: npt.NDArray) -> list[Path]:
    subdir = directory / "multi_file"
    subdir.mkdir(exist_ok=True)

    paths = []
    for i, image in enumerate(data):
        path = subdir / f"{i:04}.tif"
        imwrite(path, image)
        paths.append(path)
    return paths


def _write_volumetric(directory: Path, data: npt.NDArray) -> Path:
    path = directory / "example_volumetric.tif"
    with TiffWriter(path) as writer:
        writer.write(
            data,
            photometric="minisblack",
            volumetric=True,
        )
    return path


@pytest.fixture(scope="module")
def _example_paths(tmp_path_factory, example_3d_data):
    directory = tmp_path_factory.mktemp("example_paths")
    data = example_3d_data.astype(np.uint16)
    return {
        "hdf": _write_hdf(directory, data),
        "his": _write_his(directory, data),
        "paged": _write_paged(directory, data),
        "multi_file": _write_multi_file(directory, data),
        "volumetric": _write_volumetric(directory, data),
    }


@pytest.fixture
def example_hdf_path(_example_paths):
    return _example_paths["hdf"]


def _open_stack(fmt, paths, data, monkeypatch):
    if fmt == "dcimg":
        monkeypatch.setattr(
            "lazystack._core.DCIMGFile", lambda path: FakeDCIMG(data)
        )
        return lazystack("fake.dcimg")
    if fmt == "hdf":
        return lazystack(paths["hdf"], dset_name="images")
    return lazystack(paths[fmt])


@pytest.fixture(params=FORMATS)
def stack_and_data(request, _example_paths, example_3d_data, monkeypatch):
    data = example_3d_data.astype(np.uint16)
    stack = _open_stack(request.param, _example_paths, data, monkeypatch)
    yield stack, data
    stack.close()


@pytest.fixture(params=CLOSABLE)
def closable_stack(request, _example_paths, example_3d_data, monkeypatch):
    data = example_3d_data.astype(np.uint16)
    stack = _open_stack(request.param, _example_paths, data, monkeypatch)
    yield stack
    stack.close()


@pytest.fixture
def dcimg_stack(example_3d_data, monkeypatch):
    fake = FakeDCIMG(example_3d_data.astype(np.uint16))
    monkeypatch.setattr("lazystack._core.DCIMGFile", lambda path: fake)
    return lazystack("fake.dcimg"), fake


@pytest.fixture(params=["stack", "view"], ids=["stack", "view"])
def indexable(request, stack_and_data):
    stack, data = stack_and_data
    if request.param == "view":
        return stack[2:8], data[2:8]
    return stack, data
