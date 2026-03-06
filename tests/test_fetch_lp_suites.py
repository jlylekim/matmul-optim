from __future__ import annotations

import gzip
from pathlib import Path

from benchmarks.datasets.fetch_lp_suites import extract_dataset_links, expand_raw_file, sha256_file


def test_extract_dataset_links_filters_supported_files() -> None:
    html = """
    <html><body>
      <a href="afiro.mps.gz">afiro</a>
      <a href="blend.sif">blend</a>
      <a href="readme.txt">readme</a>
      <a href="http://example.com/foo.MPS">foo</a>
      <a href="#anchor">anchor</a>
    </body></html>
    """
    urls = extract_dataset_links(html, base_url="https://netlib.org/lp/data/")
    assert "https://netlib.org/lp/data/afiro.mps.gz" in urls
    assert "https://netlib.org/lp/data/blend.sif" in urls
    assert "http://example.com/foo.MPS" in urls
    assert not any(u.endswith("readme.txt") for u in urls)


def test_expand_raw_file_gzip_roundtrip(tmp_path: Path) -> None:
    raw = tmp_path / "sample.mps.gz"
    payload = b"NAME TEST\nROWS\n N COST\nENDATA\n"
    with gzip.open(raw, "wb") as gz:
        gz.write(payload)
    expanded = tmp_path / "sample.mps"
    expand_raw_file(raw, expanded)
    assert expanded.read_bytes() == payload
    assert len(sha256_file(raw)) == 64

