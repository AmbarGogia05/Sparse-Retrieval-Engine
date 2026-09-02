import os
import shutil

from submission.retrieve import (
    _compress_index_sidecars,
    _materialize_index_sidecars,
)


def test_sidecar_compression_round_trip(tmp_path):
    terms = (b"repeated-term-prefix-" * 200) + b"tail"
    postings = bytes(range(256))
    (tmp_path / "terms.txt").write_bytes(terms)
    (tmp_path / "postings.bin").write_bytes(postings)

    _compress_index_sidecars(str(tmp_path))

    assert not (tmp_path / "terms.txt").exists()
    assert (tmp_path / "terms.txt.z").exists()
    assert (tmp_path / "postings.bin").read_bytes() == postings

    reader_dir, cleanup_dir = _materialize_index_sidecars(str(tmp_path))
    try:
        assert open(os.path.join(reader_dir, "terms.txt"), "rb").read() == terms
        assert open(os.path.join(reader_dir, "postings.bin"), "rb").read() == postings
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir)


def test_incompressible_sidecar_keeps_original_name(tmp_path):
    data = bytes(range(32))
    (tmp_path / "tf1.bin").write_bytes(data)

    _compress_index_sidecars(str(tmp_path))

    assert (tmp_path / "tf1.bin").read_bytes() == data
    assert not (tmp_path / "tf1.bin.z").exists()
