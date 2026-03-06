from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

SUITE_URLS: dict[str, list[str]] = {
    "netlib": [
        "https://netlib.org/lp/data/",
        "https://www.netlib.org/lp/data/",
    ],
    "stochlp": [
        "https://old.sztaki.hu/~meszaros/public_ftp/lptestset/stochlp/",
        "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/stochlp/",
        "http://www.sztaki.hu/~meszaros/public_ftp/lptestset/stochlp/",
    ],
    "misc": [
        "https://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/",
        "http://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/",
        "http://www.sztaki.hu/~meszaros/public_ftp/lptestset/misc/",
    ],
}

SUPPORTED_EXTENSIONS = (".mps", ".mps.gz", ".sif", ".sif.gz")
NON_INSTANCE_BASENAMES = {
    "ascii",
    "changes",
    "emps.c",
    "emps.f",
    "emps.exe.gz",
    "mpc.src",
    "minos",
    "readme",
    "nams.ps.gz",
    "stocfor3",
    "stocfor3.old",
    "truss",
}


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


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        s = data.strip()
        if s:
            self.parts.append(s)


def _is_supported_file(name_or_url: str) -> bool:
    lowered = name_or_url.lower()
    return lowered.endswith(SUPPORTED_EXTENSIONS)


def _looks_like_netlib_instance_href(href: str) -> bool:
    parsed = urllib.parse.urlparse(href)
    path = parsed.path
    if not path or path.endswith("/"):
        return False
    base = Path(path).name
    if not base:
        return False
    low = base.lower()
    if low in NON_INSTANCE_BASENAMES:
        return False
    if _is_supported_file(low):
        return True
    # NETLIB commonly uses extensionless EMPS-compressed names (e.g., "afiro").
    return "." not in low


def _extract_netlib_compressed_names(index_html: str) -> list[str]:
    txt_parser = _TextParser()
    txt_parser.feed(index_html)
    tokens = [t.strip() for t in txt_parser.parts if t.strip()]

    names: list[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low.startswith("* file:"):
            nm = tok.split(":", 1)[1].strip().split()[0]
            nxt = tokens[i + 1].lower() if i + 1 < len(tokens) else ""
            if "lang:" in nxt and "compressed mps" in nxt:
                names.append(nm)

    if names:
        return sorted(set(names))

    # Fallback regex path for odd HTML formatting.
    matches = re.finditer(
        r"file:\s*([A-Za-z0-9_.\-]+)\s*.*?lang:\s*compressed\s+MPS",
        index_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return sorted(set(m.group(1).strip() for m in matches))


def extract_dataset_links(index_html: str, base_url: str, *, suite: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(index_html)
    urls: list[str] = []
    for href in parser.links:
        if href.startswith("#"):
            continue
        full = urllib.parse.urljoin(base_url, href)
        if _is_supported_file(full):
            urls.append(full)
            continue
        if suite == "netlib" and _looks_like_netlib_instance_href(href):
            urls.append(full)
    if suite == "netlib":
        for name in _extract_netlib_compressed_names(index_html):
            if name in NON_INSTANCE_BASENAMES:
                continue
            urls.append(urllib.parse.urljoin(base_url, name))
    return sorted(set(urls))


def _download_text(url: str, timeout_s: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "gemm-kkt-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _download_text_with_fallback(urls: Iterable[str], timeout_s: float) -> tuple[str, str]:
    last_exc: Exception | None = None
    for url in urls:
        try:
            return _download_text(url, timeout_s=timeout_s), url
        except Exception as exc:  # pragma: no cover - exercised in integration
            last_exc = exc
    assert last_exc is not None
    raise last_exc


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


def _ensure_emps_decoder(tools_dir: Path, timeout_s: float, refresh: bool) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    emps_c = tools_dir / "emps.c"
    emps_bin = tools_dir / "emps"

    if refresh or not emps_c.exists():
        _download_binary("https://www.netlib.org/lp/data/emps.c", emps_c, timeout_s=timeout_s)
    if refresh or not emps_bin.exists() or emps_bin.stat().st_mtime < emps_c.stat().st_mtime:
        cmd = ["cc", "-O2", str(emps_c), "-o", str(emps_bin)]
        subprocess.run(cmd, check=True)
    return emps_bin


def _is_native_mps_or_sif_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(".mps") or lowered.endswith(".sif") or lowered.endswith(".mps.gz") or lowered.endswith(".sif.gz")


def _expanded_name_for(filename: str) -> str:
    lowered = filename.lower()
    if lowered.endswith(".mps.gz") or lowered.endswith(".sif.gz"):
        return filename[:-3]
    if lowered.endswith(".mps") or lowered.endswith(".sif"):
        return filename
    # extensionless / non-native files are expected to be EMPS-coded; decode to .mps
    return f"{filename}.mps"


def expand_raw_file(
    raw_path: Path,
    expanded_path: Path,
    *,
    emps_bin: Path | None = None,
) -> None:
    expanded_path.parent.mkdir(parents=True, exist_ok=True)
    filename = raw_path.name
    lower = filename.lower()
    is_native = _is_native_mps_or_sif_name(filename)

    if is_native:
        if lower.endswith(".gz"):
            with gzip.open(raw_path, "rb") as gz, expanded_path.open("wb") as out:
                shutil.copyfileobj(gz, out)
            return
        if raw_path.resolve() == expanded_path.resolve():
            return
        shutil.copy2(raw_path, expanded_path)
        return

    if emps_bin is None:
        raise RuntimeError(f"Cannot decode EMPS file `{raw_path}` without emps binary")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        emps_input = td_path / raw_path.name
        if raw_path.suffix.lower() == ".gz":
            with gzip.open(raw_path, "rb") as gz, emps_input.open("wb") as out:
                shutil.copyfileobj(gz, out)
        else:
            shutil.copy2(raw_path, emps_input)
        with expanded_path.open("wb") as out:
            subprocess.run([str(emps_bin), str(emps_input)], check=True, stdout=out)


def _collect_suite_records(
    *,
    suite: str,
    base_url: str,
    raw_dir: Path,
    expanded_dir: Path,
    timeout_s: float,
    refresh: bool,
    max_files: int | None,
    tools_dir: Path,
) -> list[DatasetFileRecord]:
    html, resolved_base = _download_text_with_fallback([base_url], timeout_s=timeout_s)
    urls = extract_dataset_links(html, base_url=resolved_base, suite=suite)
    if max_files is not None and max_files > 0:
        urls = urls[:max_files]

    suite_raw = raw_dir / suite
    suite_expanded = expanded_dir / suite
    suite_raw.mkdir(parents=True, exist_ok=True)
    suite_expanded.mkdir(parents=True, exist_ok=True)

    records: list[DatasetFileRecord] = []
    seen_names: dict[str, int] = {}
    emps_bin: Path | None = None
    if suite == "netlib" or suite in {"stochlp", "misc"}:
        emps_bin = _ensure_emps_decoder(tools_dir=tools_dir, timeout_s=timeout_s, refresh=refresh)
    for url in urls:
        filename = Path(urllib.parse.urlparse(url).path).name
        if not _is_supported_file(filename) and suite != "netlib":
            # STOCHLP/MISC often use extensionless *.gz names that are still valid LP data.
            if not filename.lower().endswith(".gz"):
                continue
        if suite == "netlib" and filename in NON_INSTANCE_BASENAMES:
            continue

        raw_path = suite_raw / filename
        expanded_filename = _expanded_name_for(filename)
        expanded_path = suite_expanded / expanded_filename

        if refresh or not raw_path.exists():
            _download_binary(url, raw_path, timeout_s=timeout_s)
        if refresh or not expanded_path.exists():
            expand_raw_file(raw_path=raw_path, expanded_path=expanded_path, emps_bin=emps_bin)

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
    parser.add_argument("--tools-dir", type=str, default="benchmarks/datasets/tools")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout per request")
    parser.add_argument("--refresh", action="store_true", help="Force re-download and re-expand files")
    parser.add_argument("--max-files-per-suite", type=int, default=0, help="Limit files per suite (0 means no limit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    expanded_dir = Path(args.expanded_dir)
    manifest_path = Path(args.manifest)
    tools_dir = Path(args.tools_dir)
    max_files = None if int(args.max_files_per_suite) <= 0 else int(args.max_files_per_suite)

    all_rows: list[DatasetFileRecord] = []
    for suite in args.suites:
        urls = SUITE_URLS.get(suite)
        if not urls:
            print(f"[fetch] skip suite={suite}: no configured URL")
            continue
        print(f"[fetch] suite={suite} urls={urls}")
        last_exc: Exception | None = None
        rows: list[DatasetFileRecord] = []
        for base_url in urls:
            try:
                rows = _collect_suite_records(
                    suite=suite,
                    base_url=base_url,
                    raw_dir=raw_dir,
                    expanded_dir=expanded_dir,
                    timeout_s=float(args.timeout_s),
                    refresh=bool(args.refresh),
                    max_files=max_files,
                    tools_dir=tools_dir,
                )
                break
            except Exception as exc:  # pragma: no cover - integration/network branch
                last_exc = exc
                print(f"[fetch] suite={suite} failed base={base_url}: {type(exc).__name__}: {exc}")
        if not rows and last_exc is not None:
            print(f"[fetch] suite={suite} unresolved error after URL fallbacks")
            raise last_exc
        print(f"[fetch] suite={suite} files={len(rows)}")
        all_rows.extend(rows)

    write_manifest(manifest_path, all_rows)
    print(f"[fetch] manifest={manifest_path} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
