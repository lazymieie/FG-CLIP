#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from array import array
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Optional

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


IMAGE_TOKEN_PATTERN = re.compile(r"<image>")
SHELL_DEFAULT_ASSIGNMENT_RE = re.compile(r'^([A-Z0-9_]+)="\$\{[A-Z0-9_]+:-(.*)\}"$')
SHELL_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)(:-([^}]*))?\}|\$([A-Z0-9_]+)")
APPEND_MANIFEST_RE = re.compile(
    r'^append_manifest_line\s+"(?P<source>[^"]+)"\s+"(?P<limit>[^"]*)"\s+"(?P<seed>[^"]*)"\s+"(?P<index>[^"]*)"$'
)
WORKER_STORE = None
WORKER_TOKENIZER = None
WORKER_NORMALIZE_LOWERCASE = True
WORKER_STRIP_IMAGE_TOKEN = True
WORKER_MAX_SEQ_LENGTH = 196
WORKER_BATCH_SIZE = 512
FILE_HANDLE_CACHE = None
FILE_HANDLE_CACHE_LIMIT = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute caption length distributions for the same data sources used by FG-CLIP2 "
            "training manifests or stage1 shell scripts."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--data-path",
        help="Training data path. Supports a manifest .txt, a .json/.jsonl file, or a directory of .json/.jsonl files.",
    )
    source_group.add_argument(
        "--train-script",
        help="Training shell script such as scripts/train/stage1_fgclip2_longonly_multisource.sh.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Tokenizer model dir. If omitted, only raw character-length stats are computed.",
    )
    parser.add_argument(
        "--index-cache-root",
        default=None,
        help="Optional cache directory for auto-generated offset indexes.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=196,
        help="Training max sequence length used for truncation stats. Default matches stage1 long-caption training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for tokenizer calls.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes per source. Default: 1.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of samples per worker task chunk when --num-workers > 1.",
    )
    parser.add_argument(
        "--max-open-files-per-worker",
        type=int,
        default=32,
        help="Per-process LRU cap for open dataset files. Default: 32.",
    )
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=None,
        help="Optional cap for analyzed samples per source after manifest-level sampling is applied.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress to stderr every N analyzed samples per source. Set 0 to disable.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars and use periodic text logging only.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Default: print JSON to stdout only.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=10,
        help="How many error types to keep per source.",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable the training-style lowercase normalization before tokenization.",
    )
    parser.add_argument(
        "--keep-image-token",
        action="store_true",
        help="Keep literal <image> tokens instead of stripping them like training does.",
    )
    parser.add_argument(
        "--include-full-histogram",
        action="store_true",
        help="Include exact raw/token length histograms in the JSON output.",
    )
    return parser.parse_args()


@dataclass
class SourceSpec:
    label: str
    data_path: str
    max_records: Optional[int] = None
    sample_seed: Optional[int] = None
    index_file: Optional[str] = None
    origin: Optional[str] = None


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def build_progress(
    total: int,
    source_label: str,
    disable_progress_bar: bool,
):
    if disable_progress_bar or tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=source_label,
        unit="samples",
        dynamic_ncols=True,
        leave=True,
        smoothing=0.05,
    )


class FileHandleCache:
    def __init__(self, max_open_files: int):
        self.max_open_files = max_open_files
        self._handles: OrderedDict[str, object] = OrderedDict()

    def get(self, path: str):
        handle = self._handles.pop(path, None)
        if handle is not None and not handle.closed:
            self._handles[path] = handle
            return handle

        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

        while len(self._handles) >= self.max_open_files:
            _old_path, old_handle = self._handles.popitem(last=False)
            try:
                old_handle.close()
            except Exception:
                pass

        handle = open(path, "rb")
        self._handles[path] = handle
        return handle

    def close_all(self) -> None:
        while self._handles:
            _path, handle = self._handles.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass


def configure_file_handle_cache(max_open_files: int) -> None:
    global FILE_HANDLE_CACHE
    global FILE_HANDLE_CACHE_LIMIT

    if max_open_files <= 0:
        raise ValueError("max_open_files must be >= 1")

    FILE_HANDLE_CACHE_LIMIT = max_open_files
    if FILE_HANDLE_CACHE is not None:
        FILE_HANDLE_CACHE.close_all()
    FILE_HANDLE_CACHE = FileHandleCache(max_open_files)


def get_cached_file_handle(path: str):
    global FILE_HANDLE_CACHE
    if FILE_HANDLE_CACHE is None:
        FILE_HANDLE_CACHE = FileHandleCache(FILE_HANDLE_CACHE_LIMIT)
    return FILE_HANDLE_CACHE.get(path)


def index_meta_path(index_file: str) -> str:
    return f"{index_file}.meta.json"


def make_auto_index_path(
    data_file: str,
    index_cache_root: Optional[str],
    max_records: Optional[int] = None,
    sample_seed: Optional[int] = None,
) -> Optional[str]:
    if not index_cache_root:
        return None

    abs_path = os.path.abspath(data_file)
    digest = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:16]
    basename = os.path.basename(abs_path)
    suffixes = []
    if max_records is not None:
        suffixes.append(f"limit{max_records}")
    if sample_seed is not None:
        suffixes.append(f"seed{sample_seed}")
    suffix = "" if not suffixes else "." + ".".join(suffixes)
    return os.path.join(index_cache_root, f"{basename}.{digest}{suffix}.idx")


def read_index_meta(index_file: str) -> Optional[dict]:
    meta_path = index_meta_path(index_file)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_index_meta(index_file: str, meta: dict) -> None:
    meta_path = index_meta_path(index_file)
    meta_dir = os.path.dirname(meta_path)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)
    tmp_meta_path = f"{meta_path}.tmp.{os.getpid()}"
    with open(tmp_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_meta_path, meta_path)


def build_index_meta(
    data_file: str,
    index_kind: str,
    max_records: Optional[int],
    sample_seed: Optional[int],
    total_records: int,
) -> dict:
    return {
        "source_file": os.path.abspath(data_file),
        "source_size": os.path.getsize(data_file),
        "source_mtime": os.path.getmtime(data_file),
        "index_kind": index_kind,
        "max_records": max_records,
        "sample_seed": sample_seed,
        "total_records": total_records,
        "index_file": None,
    }


def index_matches_source(
    meta: Optional[dict],
    data_file: str,
    max_records: Optional[int],
    sample_seed: Optional[int],
) -> bool:
    if meta is None:
        return True

    abs_path = os.path.abspath(data_file)
    if os.path.abspath(meta.get("source_file", "")) != abs_path:
        return False
    if meta.get("source_size") != os.path.getsize(data_file):
        return False
    if meta.get("source_mtime") != os.path.getmtime(data_file):
        return False
    if meta.get("max_records") != max_records:
        return False
    if meta.get("sample_seed") != sample_seed:
        return False
    return True


def load_existing_offsets_if_valid(
    index_file: Optional[str],
    data_file: str,
    offset_array: array,
    max_records: Optional[int],
    sample_seed: Optional[int],
    items_per_record: int,
) -> bool:
    if index_file is None or not os.path.exists(index_file):
        return False

    meta = read_index_meta(index_file)
    if not index_matches_source(meta, data_file, max_records, sample_seed):
        return False

    del offset_array[:]
    index_size = os.path.getsize(index_file)
    with open(index_file, "rb") as f:
        offset_array.fromfile(f, index_size // offset_array.itemsize)

    if len(offset_array) % items_per_record != 0:
        raise ValueError(f"Index file {index_file} is corrupted.")

    expected_records = None if max_records is None else max_records * items_per_record
    if expected_records is not None and len(offset_array) != expected_records:
        raise ValueError(
            f"Index file {index_file} has {len(offset_array) // items_per_record} records, expected {max_records}."
        )
    return True


class JsonArrayOffsetStore:
    def __init__(
        self,
        data_file: str,
        max_records: Optional[int] = None,
        sample_seed: Optional[int] = None,
        index_file: Optional[str] = None,
    ):
        self.data_file = data_file
        self.offsets = array("Q")
        self._fp = None

        if load_existing_offsets_if_valid(index_file, data_file, self.offsets, max_records, sample_seed, 2):
            return

        rng = None
        if sample_seed is not None and max_records is not None:
            import random

            rng = random.Random(sample_seed)

        record_count = 0
        saw_array_start = False
        in_string = False
        escape = False
        brace_depth = 0
        object_start = None
        reached_array_end = False

        with open(data_file, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                chunk_start = f.tell() - len(chunk)

                for idx, byte in enumerate(chunk):
                    absolute_pos = chunk_start + idx
                    if not saw_array_start:
                        if byte in b" \t\r\n":
                            continue
                        if byte != ord("["):
                            raise ValueError(f"{data_file} must be a JSON array of objects.")
                        saw_array_start = True
                        continue

                    if brace_depth == 0:
                        if byte in b" \t\r\n,":
                            continue
                        if byte == ord("]"):
                            reached_array_end = True
                            break
                        if byte != ord("{"):
                            raise ValueError(f"{data_file} must contain JSON objects at the top level.")
                        object_start = absolute_pos
                        brace_depth = 1
                        in_string = False
                        escape = False
                        continue

                    if in_string:
                        if escape:
                            escape = False
                        elif byte == ord("\\"):
                            escape = True
                        elif byte == ord('"'):
                            in_string = False
                        continue

                    if byte == ord('"'):
                        in_string = True
                    elif byte == ord("{"):
                        brace_depth += 1
                    elif byte == ord("}"):
                        brace_depth -= 1
                        if brace_depth == 0:
                            object_end = absolute_pos + 1
                            record_count += 1
                            if rng is not None:
                                if len(self) < max_records:
                                    self.offsets.extend([object_start, object_end])
                                else:
                                    sample_idx = rng.randrange(record_count)
                                    if sample_idx < max_records:
                                        self.offsets[2 * sample_idx] = object_start
                                        self.offsets[2 * sample_idx + 1] = object_end
                            else:
                                self.offsets.extend([object_start, object_end])
                                if max_records is not None and len(self) >= max_records:
                                    break

                if max_records is not None and rng is None and len(self) >= max_records:
                    break
                if reached_array_end:
                    break

        if not saw_array_start:
            raise ValueError(f"{data_file} must be a JSON array of objects.")
        if brace_depth != 0 or in_string:
            raise ValueError(f"{data_file} is not a valid JSON array of objects.")
        if rng is not None and max_records > record_count:
            raise ValueError(
                f"Cannot sample {max_records} records from {data_file}; only {record_count} records found."
            )

        if index_file is not None:
            index_dir = os.path.dirname(index_file)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
            tmp_index_file = f"{index_file}.tmp.{os.getpid()}"
            with open(tmp_index_file, "wb") as f:
                self.offsets.tofile(f)
            os.replace(tmp_index_file, index_file)
            meta = build_index_meta(data_file, "json_array", max_records, sample_seed, len(self))
            meta["index_file"] = index_file
            write_index_meta(index_file, meta)

    def __len__(self) -> int:
        return len(self.offsets) // 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _file(self):
        return get_cached_file_handle(self.data_file)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = self.offsets[2 * index]
        end = self.offsets[2 * index + 1]
        f = self._file()
        f.seek(start)
        payload = f.read(end - start)
        return json.loads(payload.decode("utf-8"))


class JsonlOffsetStore:
    def __init__(
        self,
        data_file: str,
        max_records: Optional[int] = None,
        sample_seed: Optional[int] = None,
        index_file: Optional[str] = None,
    ):
        self.data_file = data_file
        self.offsets = array("Q")
        self._fp = None

        if load_existing_offsets_if_valid(index_file, data_file, self.offsets, max_records, sample_seed, 1):
            return

        rng = None
        if sample_seed is not None and max_records is not None:
            import random

            rng = random.Random(sample_seed)

        record_count = 0
        with open(data_file, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record_count += 1
                if rng is not None:
                    if len(self.offsets) < max_records:
                        self.offsets.append(offset)
                    else:
                        sample_idx = rng.randrange(record_count)
                        if sample_idx < max_records:
                            self.offsets[sample_idx] = offset
                else:
                    self.offsets.append(offset)
                    if max_records is not None and len(self.offsets) >= max_records:
                        break

        if rng is not None and max_records > record_count:
            raise ValueError(
                f"Cannot sample {max_records} records from {self.data_file}; only {record_count} records found."
            )

        if index_file is not None:
            index_dir = os.path.dirname(index_file)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
            tmp_index_file = f"{index_file}.tmp.{os.getpid()}"
            with open(tmp_index_file, "wb") as f:
                self.offsets.tofile(f)
            os.replace(tmp_index_file, index_file)
            meta = build_index_meta(data_file, "jsonl", max_records, sample_seed, len(self.offsets))
            meta["index_file"] = index_file
            write_index_meta(index_file, meta)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def _file(self):
        return get_cached_file_handle(self.data_file)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)

        f = self._file()
        f.seek(self.offsets[index])
        line = f.readline()
        return json.loads(line.decode("utf-8"))


class ConcatStore:
    def __init__(self, stores):
        self.stores = [store for store in stores if len(store) > 0]
        self.cumulative_sizes = []
        total = 0
        for store in self.stores:
            total += len(store)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        store_idx = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if store_idx == 0 else self.cumulative_sizes[store_idx - 1]
        return self.stores[store_idx][index - previous_size]


def parse_manifest_line(line: str, manifest_dir: str) -> tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    parts = line.strip().split()
    if not parts:
        return None, None, None, None

    data_path = parts[0]
    max_records = None
    sample_seed = None
    index_file = None
    if len(parts) > 1:
        max_records = int(parts[1])
    if len(parts) > 2:
        sample_seed = int(parts[2])
    if len(parts) > 3:
        index_file = parts[3]

    if not os.path.isabs(data_path):
        data_path = os.path.join(manifest_dir, data_path)
    if index_file is not None and not os.path.isabs(index_file):
        index_file = os.path.join(manifest_dir, index_file)

    return data_path, max_records, sample_seed, index_file


def build_store_for_spec(source: SourceSpec, index_cache_root: Optional[str]):
    data_path = os.path.abspath(source.data_path)
    if os.path.isdir(data_path):
        stores = []
        for json_file in sorted(glob.glob(os.path.join(data_path, "*.json"))):
            stores.append(
                JsonArrayOffsetStore(
                    json_file,
                    index_file=make_auto_index_path(json_file, index_cache_root),
                )
            )
        for jsonl_file in sorted(glob.glob(os.path.join(data_path, "*.jsonl"))):
            stores.append(
                JsonlOffsetStore(
                    jsonl_file,
                    index_file=make_auto_index_path(jsonl_file, index_cache_root),
                )
            )
        return ConcatStore(stores)

    resolved_index_file = source.index_file
    if resolved_index_file is None:
        resolved_index_file = make_auto_index_path(
            data_path,
            index_cache_root,
            max_records=source.max_records,
            sample_seed=source.sample_seed,
        )

    if data_path.endswith(".jsonl"):
        return JsonlOffsetStore(
            data_path,
            max_records=source.max_records,
            sample_seed=source.sample_seed,
            index_file=resolved_index_file,
        )
    if data_path.endswith(".json"):
        return JsonArrayOffsetStore(
            data_path,
            max_records=source.max_records,
            sample_seed=source.sample_seed,
            index_file=resolved_index_file,
        )
    raise ValueError(f"Unsupported data path: {data_path}")


def get_caption(item) -> str:
    if "caption" in item:
        caption = item["caption"]
        if not isinstance(caption, str):
            raise TypeError(f"caption must be a string, got {type(caption).__name__}")
        return caption

    messages = item.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
                content = message["content"]
                if not isinstance(content, str):
                    raise TypeError(f"message content must be a string, got {type(content).__name__}")
                return content
        for message in messages:
            if isinstance(message, dict) and message.get("content"):
                content = message["content"]
                if not isinstance(content, str):
                    raise TypeError(f"message content must be a string, got {type(content).__name__}")
                return content

    raise KeyError("caption is required, or messages must contain a content field")


def normalize_caption(caption: str, lowercase: bool, strip_image_token: bool) -> str:
    text = caption
    if strip_image_token:
        text = IMAGE_TOKEN_PATTERN.sub("", text)
    text = text.strip()
    if lowercase:
        text = text.lower()
    return text


def summarize_hist(hist: Counter[int], include_full_histogram: bool) -> dict:
    total = sum(hist.values())
    if total == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "percentiles": {},
            "buckets": {},
            **({"histogram": {}} if include_full_histogram else {}),
        }

    lengths = sorted(hist.items())
    total_value = sum(length * count for length, count in lengths)

    def percentile(percent: float) -> int:
        threshold = total * percent / 100.0
        running = 0
        for length, count in lengths:
            running += count
            if running >= threshold:
                return length
        return lengths[-1][0]

    values_only = [length for length, _count in lengths]
    max_length = values_only[-1]
    base_edges = sorted({16, 32, 64, 96, 128, 160, 196, 256, 384, 512, max_length})
    buckets: dict[str, int] = {"0": hist.get(0, 0)}

    start = 1
    for edge in base_edges:
        if edge < start:
            continue
        buckets[f"{start}-{edge}"] = sum(count for length, count in lengths if start <= length <= edge)
        start = edge + 1

    overflow_start = base_edges[-1] + 1
    overflow_count = sum(count for length, count in lengths if length >= overflow_start)
    if overflow_count:
        buckets[f"{overflow_start}+"] = overflow_count

    summary = {
        "count": total,
        "min": values_only[0],
        "max": values_only[-1],
        "mean": total_value / total,
        "percentiles": {
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        },
        "buckets": buckets,
    }
    if include_full_histogram:
        summary["histogram"] = {str(length): count for length, count in lengths}
    return summary


class LengthAccumulator:
    def __init__(self):
        self.sample_count = 0
        self.empty_count = 0
        self.skipped_count = 0
        self.raw_char_hist: Counter[int] = Counter()
        self.token_hist: Counter[int] = Counter()
        self.token_clipped_hist: Counter[int] = Counter()
        self.truncated_count = 0
        self.error_types: Counter[str] = Counter()

    def add_raw(self, raw_length: int) -> None:
        self.sample_count += 1
        self.raw_char_hist[raw_length] += 1
        if raw_length == 0:
            self.empty_count += 1

    def add_token_length(self, token_length: int, max_seq_length: int) -> None:
        self.token_hist[token_length] += 1
        clipped_length = min(token_length, max_seq_length)
        self.token_clipped_hist[clipped_length] += 1
        if token_length > max_seq_length:
            self.truncated_count += 1

    def add_error(self, exc: Exception) -> None:
        self.skipped_count += 1
        self.error_types[type(exc).__name__] += 1

    def merge(self, other: "LengthAccumulator") -> None:
        self.sample_count += other.sample_count
        self.empty_count += other.empty_count
        self.skipped_count += other.skipped_count
        self.raw_char_hist.update(other.raw_char_hist)
        self.token_hist.update(other.token_hist)
        self.token_clipped_hist.update(other.token_clipped_hist)
        self.truncated_count += other.truncated_count
        self.error_types.update(other.error_types)

    def finalize(self, max_seq_length: int, include_full_histogram: bool, top_errors: int) -> dict:
        result = {
            "num_samples": self.sample_count,
            "num_empty_captions": self.empty_count,
            "num_skipped_samples": self.skipped_count,
            "raw_char_length": summarize_hist(self.raw_char_hist, include_full_histogram),
            "top_error_types": [
                {"error_type": error_type, "count": count}
                for error_type, count in self.error_types.most_common(top_errors)
            ],
        }
        if self.token_hist:
            result["token_length_before_truncation"] = summarize_hist(self.token_hist, include_full_histogram)
            result["token_length_used_by_training"] = summarize_hist(self.token_clipped_hist, include_full_histogram)
            result["num_truncated"] = self.truncated_count
            result["truncation_ratio"] = self.truncated_count / self.sample_count if self.sample_count else 0.0
            result["max_seq_length"] = max_seq_length
        return result


def supports_fork_multiprocessing() -> bool:
    try:
        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


def flush_token_batch(
    captions: list[str],
    tokenizer,
    accumulator: LengthAccumulator,
    max_seq_length: int,
) -> None:
    if not captions:
        return
    encoded = tokenizer(
        captions,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    for input_ids in encoded["input_ids"]:
        accumulator.add_token_length(len(input_ids), max_seq_length)
    captions.clear()


def build_source_payload(
    source: SourceSpec,
    store_length: int,
    accumulator: LengthAccumulator,
    args: argparse.Namespace,
) -> dict:
    payload = accumulator.finalize(
        max_seq_length=args.max_seq_length,
        include_full_histogram=args.include_full_histogram,
        top_errors=args.top_errors,
    )
    payload["label"] = source.label
    payload["data_path"] = os.path.abspath(source.data_path)
    payload["num_records_available"] = store_length
    payload["max_records"] = source.max_records
    payload["sample_seed"] = source.sample_seed
    payload["index_file"] = source.index_file
    payload["origin"] = source.origin
    return payload


def init_worker(
    lowercase: bool,
    strip_image_token: bool,
    max_seq_length: int,
    batch_size: int,
    max_open_files_per_worker: int,
) -> None:
    global WORKER_NORMALIZE_LOWERCASE
    global WORKER_STRIP_IMAGE_TOKEN
    global WORKER_MAX_SEQ_LENGTH
    global WORKER_BATCH_SIZE
    WORKER_NORMALIZE_LOWERCASE = lowercase
    WORKER_STRIP_IMAGE_TOKEN = strip_image_token
    WORKER_MAX_SEQ_LENGTH = max_seq_length
    WORKER_BATCH_SIZE = batch_size
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_file_handle_cache(max_open_files_per_worker)


def process_range(task: tuple[int, int]) -> tuple[int, LengthAccumulator]:
    start, end = task
    accumulator = LengthAccumulator()
    token_batch: list[str] = []

    for idx in range(start, end):
        try:
            item = WORKER_STORE[idx]
            caption = get_caption(item)
            normalized = normalize_caption(
                caption,
                lowercase=WORKER_NORMALIZE_LOWERCASE,
                strip_image_token=WORKER_STRIP_IMAGE_TOKEN,
            )
            accumulator.add_raw(len(normalized))
            if WORKER_TOKENIZER is not None:
                token_batch.append(normalized)
                if len(token_batch) >= WORKER_BATCH_SIZE:
                    flush_token_batch(token_batch, WORKER_TOKENIZER, accumulator, WORKER_MAX_SEQ_LENGTH)
        except Exception as exc:
            accumulator.add_error(exc)

    if WORKER_TOKENIZER is not None:
        flush_token_batch(token_batch, WORKER_TOKENIZER, accumulator, WORKER_MAX_SEQ_LENGTH)

    return end - start, accumulator


def analyze_source_single_process(
    source: SourceSpec,
    store,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[dict, LengthAccumulator]:
    accumulator = LengthAccumulator()
    token_batch: list[str] = []

    num_to_process = len(store)
    if args.max_samples_per_source is not None:
        num_to_process = min(num_to_process, args.max_samples_per_source)

    progress = build_progress(num_to_process, source.label, args.no_progress_bar)
    start_time = time.perf_counter()
    try:
        for idx in range(num_to_process):
            try:
                item = store[idx]
                caption = get_caption(item)
                normalized = normalize_caption(
                    caption,
                    lowercase=not args.no_lowercase,
                    strip_image_token=not args.keep_image_token,
                )
                accumulator.add_raw(len(normalized))
                if tokenizer is not None:
                    token_batch.append(normalized)
                    if len(token_batch) >= args.batch_size:
                        flush_token_batch(token_batch, tokenizer, accumulator, args.max_seq_length)
            except Exception as exc:
                accumulator.add_error(exc)

            if progress is not None:
                progress.update(1)
            elif args.progress_every and (idx + 1) % args.progress_every == 0:
                elapsed = time.perf_counter() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0.0
                eprint(f"[{source.label}] analyzed {idx + 1}/{num_to_process} samples ({rate:.1f} samples/s)")

        if tokenizer is not None:
            flush_token_batch(token_batch, tokenizer, accumulator, args.max_seq_length)
    finally:
        if progress is not None:
            elapsed = time.perf_counter() - start_time
            rate = accumulator.sample_count / elapsed if elapsed > 0 else 0.0
            progress.set_postfix(
                samples=accumulator.sample_count,
                skipped=accumulator.skipped_count,
                rate=f"{rate:.1f}/s",
            )
            progress.close()

    payload = build_source_payload(source, len(store), accumulator, args)
    return payload, accumulator


def analyze_source_multi_process(
    source: SourceSpec,
    store,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[dict, LengthAccumulator]:
    global WORKER_STORE
    global WORKER_TOKENIZER

    num_to_process = len(store)
    if args.max_samples_per_source is not None:
        num_to_process = min(num_to_process, args.max_samples_per_source)

    if num_to_process == 0:
        empty = LengthAccumulator()
        return build_source_payload(source, len(store), empty, args), empty

    worker_count = min(args.num_workers, num_to_process)
    chunk_size = min(args.chunk_size, num_to_process)
    tasks = [
        (start, min(start + chunk_size, num_to_process))
        for start in range(0, num_to_process, chunk_size)
    ]

    ctx = mp.get_context("fork")
    WORKER_STORE = store
    WORKER_TOKENIZER = tokenizer
    accumulator = LengthAccumulator()
    progress = build_progress(num_to_process, source.label, args.no_progress_bar)
    processed_so_far = 0
    start_time = time.perf_counter()

    try:
        with ctx.Pool(
            processes=worker_count,
            initializer=init_worker,
            initargs=(
                not args.no_lowercase,
                not args.keep_image_token,
                args.max_seq_length,
                args.batch_size,
                args.max_open_files_per_worker,
            ),
        ) as pool:
            for processed_count, partial_accumulator in pool.imap_unordered(process_range, tasks, chunksize=1):
                accumulator.merge(partial_accumulator)
                processed_so_far += processed_count

                if progress is not None:
                    progress.update(processed_count)
                    elapsed = time.perf_counter() - start_time
                    rate = processed_so_far / elapsed if elapsed > 0 else 0.0
                    progress.set_postfix(
                        skipped=accumulator.skipped_count,
                        rate=f"{rate:.1f}/s",
                    )
                elif args.progress_every and processed_so_far % args.progress_every < processed_count:
                    elapsed = time.perf_counter() - start_time
                    rate = processed_so_far / elapsed if elapsed > 0 else 0.0
                    eprint(
                        f"[{source.label}] analyzed {processed_so_far}/{num_to_process} samples "
                        f"({rate:.1f} samples/s, workers={worker_count})"
                    )
    finally:
        WORKER_STORE = None
        WORKER_TOKENIZER = None
        if progress is not None:
            elapsed = time.perf_counter() - start_time
            rate = accumulator.sample_count / elapsed if elapsed > 0 else 0.0
            progress.set_postfix(
                samples=accumulator.sample_count,
                skipped=accumulator.skipped_count,
                rate=f"{rate:.1f}/s",
            )
            progress.close()

    payload = build_source_payload(source, len(store), accumulator, args)
    payload["num_workers"] = worker_count
    payload["chunk_size"] = chunk_size
    return payload, accumulator


def analyze_source(
    source: SourceSpec,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[dict, LengthAccumulator]:
    store = build_store_for_spec(source, args.index_cache_root)
    if args.num_workers <= 1:
        return analyze_source_single_process(source, store, tokenizer, args)
    if not supports_fork_multiprocessing():
        eprint(
            f"Fork-based multiprocessing is not available in this environment; "
            f"falling back to single-process mode for {source.label}."
        )
        return analyze_source_single_process(source, store, tokenizer, args)
    return analyze_source_multi_process(source, store, tokenizer, args)


def resolve_shell_expr(expr: str, values: dict[str, str]) -> str:
    resolved = expr
    for _ in range(20):
        previous = resolved

        def replacer(match: re.Match) -> str:
            braced_name = match.group(1)
            braced_default = match.group(3)
            simple_name = match.group(4)
            if braced_name is not None:
                value = values.get(braced_name, os.environ.get(braced_name, ""))
                if value:
                    return value
                if braced_default is not None:
                    return resolve_shell_expr(braced_default, values)
                return ""
            if simple_name is not None:
                return values.get(simple_name, os.environ.get(simple_name, ""))
            return ""

        resolved = SHELL_VAR_RE.sub(replacer, resolved)
        if resolved == previous:
            break
    return resolved


def resolve_shell_token(token: str, values: dict[str, str]) -> str:
    if token == "":
        return ""
    if token.startswith("$") and " " not in token:
        return resolve_shell_expr(token, values)
    return resolve_shell_expr(token, values)


def parse_sources_from_train_script(script_path: str) -> tuple[list[SourceSpec], dict[str, str]]:
    interested_vars = {
        "ROOT",
        "MODEL_DIR",
        "DATA_WORK_DIR",
        "INDEX_CACHE_ROOT",
        "COYO_SOURCE",
        "LLAVA_SOURCE",
        "DENSEFUSION_SOURCE",
        "WUKONG_SOURCE",
        "ZERO_SOURCE",
        "FINEHARD_SOURCE",
        "COYO_RECORD_LIMIT",
        "LLAVA_RECORD_LIMIT",
        "DENSEFUSION_RECORD_LIMIT",
        "COYO_SAMPLE_SEED",
        "LLAVA_SAMPLE_SEED",
        "DENSEFUSION_SAMPLE_SEED",
        "COYO_INDEX_PATH",
        "LLAVA_INDEX_PATH",
        "DENSEFUSION_INDEX_PATH",
    }
    values: dict[str, str] = {}
    manifest_entries: list[SourceSpec] = []

    with open(script_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        match = SHELL_DEFAULT_ASSIGNMENT_RE.match(stripped)
        if not match:
            continue
        var_name, default_expr = match.groups()
        if var_name not in interested_vars:
            continue
        if var_name in os.environ:
            values[var_name] = os.environ[var_name]
        else:
            values[var_name] = resolve_shell_expr(default_expr, values)

    script_dir = os.path.dirname(os.path.abspath(script_path))
    for line in lines:
        stripped = line.strip()
        match = APPEND_MANIFEST_RE.match(stripped)
        if not match:
            continue

        source_token = match.group("source")
        limit_token = match.group("limit")
        seed_token = match.group("seed")
        index_token = match.group("index")

        source_value = resolve_shell_token(source_token, values)
        limit_value = resolve_shell_token(limit_token, values)
        seed_value = resolve_shell_token(seed_token, values)
        index_value = resolve_shell_token(index_token, values)

        label = source_token.lstrip("$")
        data_path = source_value
        if data_path and not os.path.isabs(data_path):
            data_path = os.path.join(script_dir, data_path)

        index_file = index_value or None
        if index_file is not None and not os.path.isabs(index_file):
            index_file = os.path.join(script_dir, index_file)

        manifest_entries.append(
            SourceSpec(
                label=label,
                data_path=data_path,
                max_records=int(limit_value) if limit_value else None,
                sample_seed=int(seed_value) if seed_value else None,
                index_file=index_file,
                origin=os.path.abspath(script_path),
            )
        )

    if not manifest_entries:
        raise ValueError(f"No append_manifest_line entries found in {script_path}")
    return manifest_entries, values


def parse_sources_from_data_path(data_path: str) -> list[SourceSpec]:
    abs_data_path = os.path.abspath(data_path)
    if abs_data_path.endswith(".txt"):
        manifest_dir = os.path.dirname(abs_data_path)
        sources = []
        with open(abs_data_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                parsed_data_path, max_records, sample_seed, index_file = parse_manifest_line(line, manifest_dir)
                if not parsed_data_path:
                    continue
                sources.append(
                    SourceSpec(
                        label=f"manifest_line_{line_number}",
                        data_path=parsed_data_path,
                        max_records=max_records,
                        sample_seed=sample_seed,
                        index_file=index_file,
                        origin=abs_data_path,
                    )
                )
        return sources

    if os.path.isdir(abs_data_path):
        sources = []
        for json_file in sorted(glob.glob(os.path.join(abs_data_path, "*.json"))):
            sources.append(SourceSpec(label=os.path.basename(json_file), data_path=json_file, origin=abs_data_path))
        for jsonl_file in sorted(glob.glob(os.path.join(abs_data_path, "*.jsonl"))):
            sources.append(SourceSpec(label=os.path.basename(jsonl_file), data_path=jsonl_file, origin=abs_data_path))
        return sources

    return [SourceSpec(label=os.path.basename(abs_data_path), data_path=abs_data_path, origin=abs_data_path)]


def load_tokenizer(model_dir: Optional[str], explicit: bool):
    if model_dir is None:
        return None
    if AutoTokenizer is None:
        if explicit:
            raise RuntimeError("transformers is required for tokenizer-based stats but is not installed.")
        eprint("transformers is not installed in this environment; falling back to raw-length stats only.")
        return None
    return AutoTokenizer.from_pretrained(model_dir)


def main() -> None:
    args = parse_args()
    model_dir_explicit = args.model_dir is not None

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be >= 1")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be >= 1")
    if args.max_open_files_per_worker <= 0:
        raise ValueError("--max-open-files-per-worker must be >= 1")
    if args.max_seq_length <= 0:
        raise ValueError("--max-seq-length must be >= 1")

    if args.num_workers > 1:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    configure_file_handle_cache(args.max_open_files_per_worker)

    script_vars = {}
    if args.train_script:
        sources, script_vars = parse_sources_from_train_script(args.train_script)
        if args.model_dir is None and script_vars.get("MODEL_DIR"):
            args.model_dir = script_vars["MODEL_DIR"]
        if args.index_cache_root is None and script_vars.get("INDEX_CACHE_ROOT"):
            args.index_cache_root = script_vars["INDEX_CACHE_ROOT"]
    else:
        sources = parse_sources_from_data_path(args.data_path)

    tokenizer = load_tokenizer(args.model_dir, explicit=model_dir_explicit)

    overall = LengthAccumulator()
    source_summaries = []
    for source in sources:
        eprint(f"Analyzing {source.label}: {source.data_path}")
        summary, source_accumulator = analyze_source(source, tokenizer, args)
        source_summaries.append(summary)
        overall.merge(source_accumulator)

    output = {
        "input_mode": "train_script" if args.train_script else "data_path",
        "train_script": os.path.abspath(args.train_script) if args.train_script else None,
        "data_path": os.path.abspath(args.data_path) if args.data_path else None,
        "model_dir": args.model_dir,
        "index_cache_root": args.index_cache_root,
        "max_seq_length": args.max_seq_length,
        "normalization": {
            "lowercase": not args.no_lowercase,
            "strip_image_token": not args.keep_image_token,
        },
        "num_sources": len(source_summaries),
        "sources": source_summaries,
        "overall": overall.finalize(
            max_seq_length=args.max_seq_length,
            include_full_histogram=args.include_full_histogram,
            top_errors=args.top_errors,
        ),
    }

    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
