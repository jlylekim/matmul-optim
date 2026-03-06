from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

SUITE_URLS: dict[str, str] = {
    "netlib": "https://netlib.org/lp/data/",
    "stochlp": "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/stochlp/",
    "misc": "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/",
}

SUPPORTED_EXTENSIONS = (".mps", ".mps.gz", ".sif", ".sif.gz")


@dataclass
class DatasetFileRecord:
    suite: str
    instance_name: str
    source_url: str
    raw_path: str
    expanded_path: str
    sha256: str
    size_bytes: int


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.links.append(v)


def _is_supported_file(name_or_url: str) -> bool:
    lowered = name_or_url.lower()
    return lowered.endswith(SUPPORTED_EXTENSIONS)


def extract_dataset_links(index_html: str, base_url: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(index_html)
    urls: list[str] = []
    for href in parser.links:
        if href.startswith("#"):
            continue
        full = urllib.parse.urljoin(base_url, href)
        if _is_supported_file(full):
            urls.append(full)
    return sorted(set(urls))


def _download_text(url: str, timeout_s: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "gemm-kkt-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _download_binary(url: str, dest: Path, timeout_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "gemm-kkt-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expanded_name(filename: str) -> str:
    if filename.lower().endswith(".gz"):
        return filename[:-3]
    return filename


def _instance_name_from_expanded(expanded_filename: str) -> str:
    name = expanded_filename
    lowered = name.lower()
    for ext in (".mps", ".sif"):
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def expand_raw_file(raw_path: Path, expanded_path: Path) -> None:
    expanded_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.suffix.lower() == ".gz":
        with gzip.open(raw_path, "rb") as gz, expanded_path.open("wb") as out:
            shutil.copyfileobj(gz, out)
        return
    if raw_path.resolve() == expanded_path.resolve():
        return
    shutil.copy2(raw_path, expanded_path)


def _collect_suite_records(
    *,
    suite: str,
    base_url: str,
    raw_dir: Path,
    expanded_dir: Path,
    timeout_s: float,
    refresh: bool,
    max_files: int | None,
) -> list[DatasetFileRecord]:
    html = _download_text(base_url, timeout_s=timeout_s)
    urls = extract_dataset_links(html, base_url=base_url)
    if max_files is not None and max_files > 0:
        urls = urls[:max_files]

    suite_raw = raw_dir / suite
    suite_expanded = expanded_dir / suite
    suite_raw.mkdir(parents=True, exist_ok=True)
    suite_expanded.mkdir(parents=True, exist_ok=True)

    records: list[DatasetFileRecord] = []
    seen_names: dict[str, int] = {}
    for url in urls:
        filename = Path(urllib.parse.urlparse(url).path).name
        if not _is_supported_file(filename):
            continue

        raw_path = suite_raw / filename
        expanded_filename = _expanded_name(filename)
        expanded_path = suite_expanded / expanded_filename

        if refresh or not raw_path.exists():
            _download_binary(url, raw_path, timeout_s=timeout_s)
        if refresh or not expanded_path.exists():
            expand_raw_file(raw_path=raw_path, expanded_path=expanded_path)

        instance = _instance_name_from_expanded(expanded_filename)
        count = seen_names.get(instance, 0)
        seen_names[instance] = count + 1
        if count > 0:
            instance = f"{instance}_{count}"

        records.append(
            DatasetFileRecord(
                suite=suite,
                instance_name=instance,
                source_url=url,
                raw_path=str(raw_path),
                expanded_path=str(expanded_path),
                sha256=sha256_file(raw_path),
                size_bytes=int(raw_path.stat().st_size),
            )
        )
    return records


def write_manifest(path: Path, rows: list[DatasetFileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "instance_name", "source_url", "raw_path", "expanded_path", "sha256", "size_bytes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.suite, r.instance_name)):
            writer.writerow(
                {
                    "suite": row.suite,
                    "instance_name": row.instance_name,
                    "source_url": row.source_url,
                    "raw_path": row.raw_path,
                    "expanded_path": row.expanded_path,
                    "sha256": row.sha256,
                    "size_bytes": row.size_bytes,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and index real LP benchmark suites")
    parser.add_argument(
        "--suites",
        nargs="*",
        default=["netlib", "stochlp", "misc"],
        choices=sorted(SUITE_URLS.keys()),
        help="Suites to download",
    )
    parser.add_argument("--raw-dir", type=str, default="benchmarks/datasets/raw")
    parser.add_argument("--expanded-dir", type=str, default="benchmarks/datasets/expanded")
    parser.add_argument("--manifest", type=str, default="benchmarks/datasets/real_lp_manifest.csv")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout per request")
    parser.add_argument("--refresh", action="store_true", help="Force re-download and re-expand files")
    parser.add_argument("--max-files-per-suite", type=int, default=0, help="Limit files per suite (0 means no limit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    expanded_dir = Path(args.expanded_dir)
    manifest_path = Path(args.manifest)
    max_files = None if int(args.max_files_per_suite) <= 0 else int(args.max_files_per_suite)

    all_rows: list[DatasetFileRecord] = []
    for suite in args.suites:
        print(f"[fetch] suite={suite} url={SUITE_URLS[suite]}")
        rows = _collect_suite_records(
            suite=suite,
            base_url=SUITE_URLS[suite],
            raw_dir=raw_dir,
            expanded_dir=expanded_dir,
            timeout_s=float(args.timeout_s),
            refresh=bool(args.refresh),
            max_files=max_files,
        )
        print(f"[fetch] suite={suite} files={len(rows)}")
        all_rows.extend(rows)

    write_manifest(manifest_path, all_rows)
    print(f"[fetch] manifest={manifest_path} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
