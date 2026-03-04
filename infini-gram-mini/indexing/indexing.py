"""Build FM-index shards from raw text files (parquet, jsonl, json.gz, zst/zstd)."""

from __future__ import annotations

import argparse
import gc
import glob as _glob
import gzip
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

_SA_MIN_CHUNK: int = 1_000
_SDSL_HEADER: int = 8
_FEW_FILES_LIMIT: int = 50
_DEFAULT_DOC_SEP: bytes = b"\xff"
_SUPPORTED_EXTENSIONS: tuple[str, ...] = (".parquet", ".jsonl", ".json.gz", ".zst", ".zstd")
_INDEXING_DIR: Path = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_file(path: str) -> list[str]:
    """Load a source file and return its records as JSON-encoded strings."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.readlines()
    if path.endswith((".zst", ".zstd")):
        import io, zstandard as zstd
        def _gen():
            with open(path, "rb") as fh:
                reader = zstd.ZstdDecompressor().stream_reader(fh)
                yield from io.TextIOWrapper(reader, encoding="utf-8")
        return _gen()
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            return fh.readlines()
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return [json.dumps(row) + "\n" for row in pq.read_table(path).to_pylist()]
    raise ValueError(f"Unsupported file format: {path!r}")


def _count_lines(path: str) -> int:
    """Count the number of records in a source file without fully loading it."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    if path.endswith((".zst", ".zstd")):
        import io, zstandard as zstd
        with open(path, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            return sum(1 for _ in io.TextIOWrapper(reader, encoding="utf-8"))
    if path.endswith(".jsonl"):
        count = 0
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                count += chunk.count(b"\n")
        return count
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    raise ValueError(f"Unsupported file format: {path!r}")


def _load_file_slice(path: str, start_line: int, end_line: int) -> Iterator[str]:
    """Yield lines [start_line, end_line) from a source file.

    end_line=-1 means read to the end of the file.
    For uncompressed .jsonl this is fully streaming (no full load into memory).
    """
    stop = end_line if end_line != -1 else None
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from itertools.islice(fh, start_line, stop)
    elif path.endswith((".zst", ".zstd")):
        import io, zstandard as zstd
        with open(path, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            yield from itertools.islice(io.TextIOWrapper(reader, encoding="utf-8"), start_line, stop)
    elif path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            yield from itertools.islice(fh, start_line, stop)
    elif path.endswith(".parquet"):
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
        end = stop if stop is not None else len(rows)
        yield from (json.dumps(row) + "\n" for row in rows[start_line:end])
    else:
        raise ValueError(f"Unsupported file format: {path!r}")


def _parse_record(
    line: str, doc_sep: bytes, text_key: str, rel_path: str, linenum: int
) -> tuple[bytes, bytes]:
    """Parse one JSON record into a (data, meta) binary pair."""
    record = json.loads(line.strip("\n"))
    text   = record.pop(text_key)
    data   = doc_sep + text.encode("utf-8")
    meta   = (json.dumps({"path": rel_path, "linenum": linenum, "metadata": record}) + "\n").encode("utf-8")
    return data, meta


def _sdsl_finalize(fout: BinaryIO) -> None:
    """Append the null-byte sentinel, pad to 8-byte alignment, and write the SDSL header."""
    content_bytes = fout.tell() - _SDSL_HEADER
    fout.write(b"\x00")
    pad = 7 - content_bytes % 8
    if pad:
        fout.write(b"\x00" * pad)
    fout.seek(0)
    fout.write(int((content_bytes + 1) * 8).to_bytes(8, "little"))


def _prepare_few(
    files: list[tuple[str, int, int]], common_dir: str, save_dir: str,
    doc_sep: bytes, text_key: str, batch_size: int, cpus: int,
) -> None:
    """Sequential prepare path for small numbers of source files."""
    save = Path(save_dir)
    t0   = time.time()

    with (
        open(save / "text_data.sdsl", "wb") as ds_fout,
        open(save / "data_offset",    "wb") as od_fout,
        open(save / "text_meta.sdsl", "wb") as mt_fout,
        open(save / "meta_offset",    "wb") as om_fout,
        mp.get_context("fork").Pool(cpus) as pool,
    ):
        ds_fout.write(b"\x00" * _SDSL_HEADER)
        mt_fout.write(b"\x00" * _SDSL_HEADER)
        od = om = 0

        for path, start_line, end_line in files:
            rel = os.path.relpath(path, common_dir)
            batch, line_idx = [], start_line
            for line in _load_file_slice(path, start_line, end_line):
                batch.append((line, doc_sep, text_key, rel, line_idx))
                line_idx += 1
                if len(batch) >= batch_size:
                    parsed = pool.starmap(_parse_record, batch)
                    for data, meta in parsed:
                        ds_fout.write(data);  od_fout.write(od.to_bytes(8, "little")); od += len(data)
                        mt_fout.write(meta);  om_fout.write(om.to_bytes(8, "little")); om += len(meta)
                    del parsed, batch
                    batch = []
                    gc.collect()
            if batch:
                parsed = pool.starmap(_parse_record, batch)
                for data, meta in parsed:
                    ds_fout.write(data);  od_fout.write(od.to_bytes(8, "little")); od += len(data)
                    mt_fout.write(meta);  om_fout.write(om.to_bytes(8, "little")); om += len(meta)
                del parsed
            gc.collect()

        _sdsl_finalize(ds_fout)
        _sdsl_finalize(mt_fout)

    log.info("  few-files prepare done in %.1f s", time.time() - t0)


def _map_worker(
    filenum: int, path: str, start_line: int, end_line: int, tmp_dir: str,
    doc_sep: bytes, text_key: str, rel_path: str,
) -> None:
    """Parse one source file slice into temporary binary shard files."""
    tmp = Path(tmp_dir)
    with (
        open(tmp / f"text_data.{filenum:04d}",   "wb") as ds_fout,
        open(tmp / f"data_offset.{filenum:04d}", "wb") as od_fout,
        open(tmp / f"text_meta.{filenum:04d}",   "wb") as mt_fout,
        open(tmp / f"meta_offset.{filenum:04d}", "wb") as om_fout,
    ):
        od = om = 0
        for linenum, line in enumerate(_load_file_slice(path, start_line, end_line), start_line):
            data, meta = _parse_record(line, doc_sep, text_key, rel_path, linenum)
            ds_fout.write(data);  od_fout.write(od.to_bytes(8, "little")); od += len(data)
            mt_fout.write(meta);  om_fout.write(om.to_bytes(8, "little")); om += len(meta)


def _reduce_worker(
    filenum: int, data_off: int, meta_off: int,
    offset_off: int, tmp_dir: str, save_dir: str,
) -> None:
    """Splice one temporary shard into the pre-allocated output files."""
    tmp = Path(tmp_dir)
    for mode, text_off in (("data", data_off), ("meta", meta_off)):
        with (
            open(Path(save_dir) / f"text_{mode}.sdsl", "rb+") as tf,
            open(Path(save_dir) / f"{mode}_offset",    "rb+") as of_,
        ):
            tf.seek(_SDSL_HEADER + text_off)
            of_.seek(offset_off)
            tf.write((tmp / f"text_{mode}.{filenum:04d}").read_bytes())
            raw     = (tmp / f"{mode}_offset.{filenum:04d}").read_bytes()
            offsets = np.frombuffer(raw, dtype=np.uint8).view(np.uint64).copy()
            offsets += text_off
            of_.write(offsets.view(np.uint8).tobytes())


def _prepare_many(
    files: list[tuple[str, int, int]], common_dir: str, save_dir: str, temp_dir: str,
    doc_sep: bytes, text_key: str, cpus: int,
) -> None:
    """Parallel map-reduce prepare path for large numbers of source files."""
    save = Path(save_dir)
    tmp  = Path(temp_dir) / "files"
    t0   = time.time()

    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    with mp.get_context("fork").Pool(cpus) as pool:
        pool.starmap(_map_worker, [
            (i, path, start, end, str(tmp), doc_sep, text_key, os.path.relpath(path, common_dir))
            for i, (path, start, end) in enumerate(files)
        ])

        data_sizes = [os.path.getsize(tmp / f"text_data.{i:04d}")   for i in range(len(files))]
        meta_sizes = [os.path.getsize(tmp / f"text_meta.{i:04d}")   for i in range(len(files))]
        off_sizes  = [os.path.getsize(tmp / f"data_offset.{i:04d}") for i in range(len(files))]
        total_data, total_meta, total_off = sum(data_sizes), sum(meta_sizes), sum(off_sizes)

        # Sequential reduce: concatenate temp files in order — avoids concurrent random
        # writes to a single huge pre-allocated file (which causes filesystem contention).
        for mode, total, sizes in (
            ("data", total_data, data_sizes),
            ("meta", total_meta, meta_sizes),
        ):
            pad = 7 - total % 8
            with open(save / f"text_{mode}.sdsl", "wb") as fout:
                fout.write(int((total + 1) * 8).to_bytes(8, "little"))
                for i in range(len(files)):
                    fout.write((tmp / f"text_{mode}.{i:04d}").read_bytes())
                fout.write(b"\x00" * (1 + pad))

        running_data = running_meta = 0
        with (
            open(save / "data_offset", "wb") as od_fout,
            open(save / "meta_offset", "wb") as om_fout,
        ):
            for i in range(len(files)):
                for fout, mode, running in (
                    (od_fout, "data", running_data),
                    (om_fout, "meta", running_meta),
                ):
                    raw     = (tmp / f"{mode}_offset.{i:04d}").read_bytes()
                    offsets = np.frombuffer(raw, dtype=np.uint8).view(np.uint64).copy()
                    offsets += running
                    fout.write(offsets.view(np.uint8).tobytes())
                running_data += data_sizes[i]
                running_meta += meta_sizes[i]

    shutil.rmtree(tmp)
    log.info("  many-files prepare done in %.1f s", time.time() - t0)


def prepare(
    files: list[tuple[str, int, int]], common_dir: str, save_dir: str, temp_dir: str,
    doc_sep: bytes, text_key: str, batch_size: int, cpus: int,
) -> None:
    """Write text_{data,meta}.sdsl and {data,meta}_offset to save_dir."""
    save    = Path(save_dir)
    outputs = [save / n for n in ("text_data.sdsl", "data_offset", "text_meta.sdsl", "meta_offset")]
    if all(p.exists() for p in outputs):
        log.info("Step 1 (prepare): skipped -- outputs already exist.")
        return

    strategy = "few-files" if len(files) <= _FEW_FILES_LIMIT else "many-files"
    log.info("Step 1 (prepare): %d file(s), strategy=%s", len(files), strategy)

    if strategy == "few-files":
        _prepare_few(files, common_dir, save_dir, doc_sep, text_key, batch_size, cpus)
    else:
        _prepare_many(files, common_dir, save_dir, temp_dir, doc_sep, text_key, cpus)

    log.info("Step 1 (prepare): done.")


def _run(cmd: list[str], step: str) -> None:
    """Run a subprocess and raise RuntimeError with stderr on non-zero exit."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{step} failed (exit {result.returncode}):\n{result.stderr.strip()}")


def build_sa_bwt(save_dir: str, temp_dir: str, mem_gib: int, cpus: int, mode: str) -> None:
    """Build the suffix array and BWT for corpus mode ('data' or 'meta')."""
    save     = Path(save_dir)
    ds_path  = save / f"text_{mode}.sdsl"
    sa_path  = save / f"sa_{mode}.sdsl"
    bwt_path = save / f"bwt_{mode}.sdsl"

    if sa_path.exists() and bwt_path.exists():
        log.info("Step 2 (build_sa_bwt / %s): skipped -- outputs already exist.", mode)
        return

    rust = str(_INDEXING_DIR / "rust_indexing")
    t0   = time.time()
    log.info("Step 2 (build_sa_bwt / %s): starting ...", mode)

    with open(ds_path, "rb") as fh:
        ds_size = int.from_bytes(fh.read(_SDSL_HEADER), "little") // 8

    ratio = int(np.ceil(np.log2(ds_size) / 8))

    mem_bytes   = mem_gib * 1024 ** 3
    num_batches = 1
    while num_batches * (mem_bytes // 12) < ds_size:
        num_batches *= 2
    total_jobs = min(num_batches * cpus, ds_size // _SA_MIN_CHUNK)
    chunk      = ds_size // total_jobs
    log.info("  corpus=%d B, ratio=%d, %d batch(es) x %d job(s)", ds_size, ratio, num_batches, cpus)

    parts_dir = Path(temp_dir) / "parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir()

    log.info("  2.1 make-part ...")
    t_step = time.time()
    for batch_start in range(0, total_jobs, cpus):
        procs = []
        for i in range(batch_start, min(batch_start + cpus, total_jobs)):
            s = _SDSL_HEADER + i * chunk
            e = _SDSL_HEADER + min((i + 1) * chunk + _SA_MIN_CHUNK, ds_size)
            procs.append(subprocess.Popen([
                rust, "make-part",
                "--data-file",  str(ds_path), "--parts-dir", str(parts_dir),
                "--start-byte", str(s),       "--end-byte",  str(e),
                "--ratio",      str(ratio),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        for proc in procs:
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"rust_indexing make-part failed:\n{stderr.decode()}")
    log.info("  2.1 make-part: done in %.1f s", time.time() - t_step)

    merged_dir = Path(temp_dir) / "merged"
    bwt_dir    = Path(temp_dir) / "bwt"
    shutil.rmtree(merged_dir, ignore_errors=True); merged_dir.mkdir()
    shutil.rmtree(bwt_dir,    ignore_errors=True); bwt_dir.mkdir()

    log.info("  2.2 merge ...")
    t_step = time.time()
    _run([
        rust, "merge",
        "--data-file",   str(ds_path),  "--parts-dir",   str(parts_dir),
        "--merged-dir",  str(merged_dir), "--bwt-dir",   str(bwt_dir),
        "--num-threads", str(cpus),     "--hacksize",    str(_SA_MIN_CHUNK),
        "--ratio",       str(ratio),
    ], "rust_indexing merge")
    shutil.rmtree(parts_dir)
    log.info("  2.2 merge: done in %.1f s", time.time() - t_step)

    log.info("  2.3 concat ...")
    t_step = time.time()
    _run([
        rust, "concat",
        "--data-file",   str(ds_path),  "--merged-dir",  str(merged_dir),
        "--merged-file", str(sa_path),  "--bwt-dir",     str(bwt_dir),
        "--bwt-file",    str(bwt_path), "--num-threads", str(cpus),
        "--ratio",       str(ratio),
    ], "rust_indexing concat")
    shutil.rmtree(merged_dir)
    shutil.rmtree(bwt_dir)
    log.info("  2.3 concat: done in %.1f s", time.time() - t_step)

    log.info("Step 2 (build_sa_bwt / %s): done in %.1f s", mode, time.time() - t0)


def build_fm_index(save_dir: str) -> None:
    """Compress the suffix array and BWT into FM-index files via the C++ binary."""
    save = Path(save_dir)
    if (save / "data.fm9").exists() and (save / "meta.fm9").exists():
        log.info("Step 3 (build_fm_index): skipped -- outputs already exist.")
        return

    log.info("Step 3 (build_fm_index): starting ...")
    t0     = time.time()
    result = subprocess.run(
        [str(_INDEXING_DIR / "cpp" / "cpp_indexing"), save_dir],
        capture_output=True, text=True,
    )
    if result.stdout:
        log.info(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(f"cpp_indexing failed:\n{result.stderr.strip()}")
    log.info("Step 3 (build_fm_index): done in %.1f s", time.time() - t0)


def collect_input_files(data_dirs: list[str]) -> list[tuple[str, int]]:
    """Return a sorted (path, size_bytes) list for all supported files in data_dirs."""
    results: list[tuple[str, int]] = []
    for entry in data_dirs:
        entry = entry.rstrip("/")
        matches = sorted(_glob.glob(entry, recursive=True))
        if not matches:
            matches = [entry]
        for path in matches:
            if os.path.isfile(path):
                if path.endswith(_SUPPORTED_EXTENSIONS):
                    results.append((path, os.path.getsize(path)))
                else:
                    log.warning("Skipping unsupported file type: %s", path)
            elif os.path.isdir(path):
                for root, _, fnames in os.walk(path):
                    for fname in sorted(fnames):
                        if fname.endswith(_SUPPORTED_EXTENSIONS):
                            fpath = os.path.join(root, fname)
                            results.append((fpath, os.path.getsize(fpath)))
            else:
                raise FileNotFoundError(f"--data_dir entry not found: {path!r}")
    results.sort(key=lambda x: x[0])
    return results


def split_into_shards(
    files: list[tuple[str, int]], num_shards: int
) -> list[list[tuple[str, int, int]]]:
    """Greedily partition files into num_shards groups of roughly equal total size.

    Files larger than the per-shard target are split into line-range slices so
    that no single shard exceeds the target.  Each entry in the returned lists is
    a (path, start_line, end_line) tuple, where end_line=-1 means "to end of file".
    """
    if not files:
        return []
    total  = sum(s for _, s in files)
    target = total / num_shards

    # Expand files that exceed the target into multiple line-range slices.
    slices: list[tuple[str, int, int, int]] = []  # (path, start, end, approx_size)
    for path, size in files:
        if size <= target:
            slices.append((path, 0, -1, size))
        else:
            n_pieces   = math.ceil(size / target)
            log.info("  Splitting %s into %d pieces (%.1f GiB each)",
                     os.path.basename(path), n_pieces, size / n_pieces / 1024 ** 3)
            line_count = _count_lines(path)
            lines_per  = math.ceil(line_count / n_pieces)
            for i in range(n_pieces):
                start = i * lines_per
                end   = min((i + 1) * lines_per, line_count)
                slices.append((path, start, end, int(size * (end - start) / line_count)))

    shards: list[list[tuple[str, int, int]]] = []
    current: list[tuple[str, int, int]] = []
    current_size = 0
    for path, start, end, size in slices:
        current.append((path, start, end))
        current_size += size
        if current_size >= target and len(shards) < num_shards - 1:
            shards.append(current)
            current, current_size = [], 0
    if current:
        shards.append(current)
    return shards


def index_shard(
    shard_files: list[tuple[str, int, int]], common_dir: str, save_dir: str,
    temp_dir: str, args: argparse.Namespace,
) -> None:
    """Run the full three-step indexing pipeline for a single shard."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    prepare(shard_files, common_dir, save_dir, temp_dir,
            args.doc_sep, args.text_key, args.batch_size, args.cpus)
    build_sa_bwt(save_dir, temp_dir, args.mem, args.cpus, mode="data")
    build_sa_bwt(save_dir, temp_dir, args.mem, args.cpus, mode="meta")
    build_fm_index(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FM-index shards from raw text files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   required=True, nargs="+",
                        help="One or more source directories, files, or glob patterns "
                             "(e.g. /data/dir1 /data/dir2 or '/data/**/*.parquet').")
    parser.add_argument("--save_dir",   required=True,
                        help="Root output directory; each shard is written to save_dir/NN/.")
    parser.add_argument("--temp_dir",   default=None,
                        help="Scratch space for intermediate files (defaults to save_dir).")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Number of index shards to create.")
    parser.add_argument("--shard_id",   type=int, default=None,
                        help="Process only this shard (0-indexed). Useful for SLURM array jobs.")
    parser.add_argument("--text_key",   default="text",
                        help="JSON field name containing the document text.")
    parser.add_argument("--doc_sep",    default="ff",
                        help="Document separator byte as a two-character hex string (e.g. 'ff').")
    parser.add_argument("--batch_size", type=int, default=65_536,
                        help="Records per batch in the few-files prepare path.")
    parser.add_argument("--cpus",       type=int, default=mp.cpu_count(),
                        help="Number of CPU cores to use.")
    parser.add_argument("--mem",        type=int, required=True,
                        help="Available memory in GiB.")
    parser.add_argument("--ulimit",     type=int, default=1_048_576,
                        help="Target open-file-descriptor limit (capped at the system hard limit).")
    args = parser.parse_args()

    if sys.byteorder != "little":
        raise RuntimeError("Only little-endian systems are supported.")

    args.save_dir = args.save_dir.rstrip("/")
    args.temp_dir = (args.temp_dir or args.save_dir).rstrip("/")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive, got {args.batch_size}")
    if args.cpus <= 0:
        raise ValueError(f"--cpus must be positive, got {args.cpus}")

    try:
        args.doc_sep = bytes.fromhex(args.doc_sep)
    except ValueError:
        raise ValueError(f"--doc_sep must be a two-character hex string (e.g. 'ff'), got {args.doc_sep!r}")

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target  = min(args.ulimit, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    log.info("File descriptor limit: %d (hard: %d).", target, hard)

    all_files = collect_input_files(args.data_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No supported files found for --data_dir {args.data_dir!r}"
        )
    log.info("Found %d source file(s), total %.2f GiB.",
             len(all_files), sum(s for _, s in all_files) / 1024 ** 3)

    all_paths  = [p for p, _ in all_files]
    common_dir = os.path.commonpath(all_paths)
    if os.path.isfile(common_dir):
        common_dir = os.path.dirname(common_dir)
    log.info("Common source root: %s", common_dir)

    shards    = split_into_shards(all_files, args.num_shards)
    shard_ids = [args.shard_id] if args.shard_id is not None else range(len(shards))
    log.info("Split into %d shard(s).", len(shards))

    for sid in shard_ids:
        if sid >= len(shards):
            log.info("Shard %02d: no files assigned (only %d shard(s) total), skipping.", sid, len(shards))
            continue
        shard_save  = os.path.join(args.save_dir, f"{sid:02d}")
        shard_temp  = os.path.join(args.temp_dir,  f"tmp_{sid:02d}")
        shard_files = shards[sid]

        if (Path(shard_save) / "data.fm9").exists() and (Path(shard_save) / "meta.fm9").exists():
            log.info("Shard %02d: already complete, skipping.", sid)
            continue

        log.info("Shard %02d: %d slice(s) -> %s", sid, len(shard_files), shard_save)
        index_shard(shard_files, common_dir, shard_save, shard_temp, args)
        shutil.rmtree(shard_temp, ignore_errors=True)


if __name__ == "__main__":
    main()
