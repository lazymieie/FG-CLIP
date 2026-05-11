#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from array import array
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parents[2]
ANALYZE_SCRIPT_PATH = ROOT / "scripts" / "train" / "analyze_caption_length_distribution.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


analyze_mod = load_module("fgclip_caption_analyze", ANALYZE_SCRIPT_PATH)
sys.path.insert(0, str(ROOT))
from fgclip2.train import train_fgclip2 as train_mod  # noqa: E402


WORKER_SOURCES = None
WORKER_TOKENIZER = None
WORKER_STORE = None
WORKER_MAX_CAPTION_TOKENS = None
WORKER_LOWERCASE = True
WORKER_STRIP_IMAGE_TOKEN = True
WORKER_MAX_OPEN_FILES = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and cache valid caption indices for FG-CLIP training."
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
        help="Tokenizer model dir. If omitted with --train-script, resolve from script defaults.",
    )
    parser.add_argument(
        "--index-cache-root",
        default=None,
        help="Directory for auto-generated cache files. If omitted with --train-script, resolve from script defaults.",
    )
    parser.add_argument(
        "--max-caption-tokens",
        type=int,
        default=None,
        help="Keep samples whose main caption token length is <= this value. If omitted with --train-script, resolve from script defaults.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(os.cpu_count() or 1, 1),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of samples processed per task chunk.",
    )
    parser.add_argument(
        "--max-open-files-per-worker",
        type=int,
        default=16,
        help="Per-process LRU cap for open dataset files.",
    )
    parser.add_argument(
        "--cn-pair-root",
        default=None,
        help="Optional Chinese pair root used by training. Included in cache identity.",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Disable lowercase normalization before token length calculation.",
    )
    parser.add_argument(
        "--keep-image-token",
        action="store_true",
        help="Keep literal <image> tokens instead of stripping them like training does.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild cache even if a valid cache file already exists.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit cache output path. Default uses training-compatible auto path.",
    )
    return parser.parse_args()


def resolve_from_train_script(script_path: str) -> dict[str, Optional[str]]:
    sources, values = analyze_mod.parse_sources_from_train_script(script_path)
    script_vars = {
        "sources": sources,
        "model_dir": values.get("MODEL_DIR"),
        "index_cache_root": values.get("INDEX_CACHE_ROOT"),
    }

    script_dir = os.path.dirname(os.path.abspath(script_path))
    shell_vars = {}
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            match = analyze_mod.SHELL_DEFAULT_ASSIGNMENT_RE.match(stripped)
            if not match:
                continue
            var_name, default_expr = match.groups()
            shell_vars[var_name] = analyze_mod.resolve_shell_expr(default_expr, shell_vars)

    data_path = shell_vars.get("DATA_PATH")
    if data_path and not os.path.isabs(data_path):
        data_path = os.path.join(script_dir, data_path)
    script_vars["data_path"] = data_path

    max_seq_length = shell_vars.get("MAX_SEQ_LENGTH")
    script_vars["max_caption_tokens"] = int(max_seq_length) if max_seq_length else None
    return script_vars


def load_sources(args: argparse.Namespace):
    if args.train_script:
        resolved = resolve_from_train_script(args.train_script)
        sources = resolved["sources"]
        if args.model_dir is None:
            args.model_dir = resolved["model_dir"]
        if args.index_cache_root is None:
            args.index_cache_root = resolved["index_cache_root"]
        if args.data_path is None:
            args.data_path = resolved["data_path"]
        if args.max_caption_tokens is None:
            args.max_caption_tokens = resolved["max_caption_tokens"]
    else:
        sources = analyze_mod.parse_sources_from_data_path(args.data_path)

    if not args.data_path:
        raise ValueError("Failed to resolve data_path for cache identity.")
    if not args.model_dir:
        raise ValueError("model_dir is required.")
    if not args.index_cache_root:
        raise ValueError("index_cache_root is required.")
    if args.max_caption_tokens is None or args.max_caption_tokens <= 0:
        raise ValueError("max_caption_tokens must be a positive integer.")
    return sources


def build_store_from_sources(sources, index_cache_root: Optional[str]):
    stores = [analyze_mod.build_store_for_spec(source, index_cache_root) for source in sources]
    return analyze_mod.ConcatStore(stores)


def init_worker(
    sources,
    model_dir: str,
    index_cache_root: Optional[str],
    max_caption_tokens: int,
    lowercase: bool,
    strip_image_token: bool,
    max_open_files_per_worker: int,
) -> None:
    global WORKER_SOURCES, WORKER_TOKENIZER, WORKER_STORE, WORKER_MAX_CAPTION_TOKENS
    global WORKER_LOWERCASE, WORKER_STRIP_IMAGE_TOKEN, WORKER_MAX_OPEN_FILES

    WORKER_SOURCES = sources
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    WORKER_STORE = build_store_from_sources(sources, index_cache_root)
    WORKER_MAX_CAPTION_TOKENS = max_caption_tokens
    WORKER_LOWERCASE = lowercase
    WORKER_STRIP_IMAGE_TOKEN = strip_image_token
    WORKER_MAX_OPEN_FILES = max_open_files_per_worker
    analyze_mod.configure_file_handle_cache(max_open_files_per_worker)


def process_chunk(chunk: tuple[int, int]) -> tuple[int, list[int]]:
    start, end = chunk
    valid_indices = []
    tokenizer = WORKER_TOKENIZER
    store = WORKER_STORE

    for idx in range(start, end):
        try:
            item = store[idx]
            caption = analyze_mod.get_caption(item)
            normalized = analyze_mod.normalize_caption(
                caption,
                lowercase=WORKER_LOWERCASE,
                strip_image_token=WORKER_STRIP_IMAGE_TOKEN,
            )
            token_ids = tokenizer(
                [normalized],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"][0]
            if len(token_ids) <= WORKER_MAX_CAPTION_TOKENS:
                valid_indices.append(idx)
        except Exception:
            # Match training-side semantics: keep the sample if prefiltering cannot
            # confidently decide due to malformed records or transient parse issues.
            valid_indices.append(idx)

    return start, valid_indices


def chunk_ranges(total: int, chunk_size: int):
    for start in range(0, total, chunk_size):
        yield (start, min(start + chunk_size, total))


def acquire_lock(lock_file: str, cache_path: str, args: argparse.Namespace, total_source_records: int) -> bool:
    start_time = time.time()
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
            return True
        except FileExistsError:
            if not args.force:
                cached_indices = array("Q")
                if train_mod.load_existing_valid_indices_if_valid(
                    cache_path,
                    args.data_path,
                    args.cn_pair_root,
                    args.model_dir,
                    args.max_caption_tokens,
                    total_source_records,
                    cached_indices,
                ):
                    print(
                        f"Cache already built by another process: {cache_path} "
                        f"({len(cached_indices)}/{total_source_records} kept)",
                        flush=True,
                    )
                    return False
            if time.time() - start_time > train_mod.AUTO_INDEX_LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"Timed out waiting for valid index cache lock: {lock_file}")
            time.sleep(train_mod.AUTO_INDEX_LOCK_POLL_SECONDS)


def main() -> None:
    args = parse_args()
    sources = load_sources(args)

    store = build_store_from_sources(sources, args.index_cache_root)
    total_source_records = len(store)

    cache_path = args.output or train_mod.make_valid_indices_cache_path(
        args.data_path,
        args.index_cache_root,
        args.max_caption_tokens,
        args.model_dir,
        args.cn_pair_root,
    )
    if not cache_path:
        raise ValueError("Failed to resolve cache_path.")

    cached_indices = array("Q")
    if not args.force and train_mod.load_existing_valid_indices_if_valid(
        cache_path,
        args.data_path,
        args.cn_pair_root,
        args.model_dir,
        args.max_caption_tokens,
        total_source_records,
        cached_indices,
    ):
        print(
            f"Using existing cache: {cache_path} "
            f"({len(cached_indices)}/{total_source_records} kept)",
            flush=True,
        )
        return

    lock_file = f"{cache_path}.lock"
    if not acquire_lock(lock_file, cache_path, args, total_source_records):
        return

    try:
        if not args.force and train_mod.load_existing_valid_indices_if_valid(
            cache_path,
            args.data_path,
            args.cn_pair_root,
            args.model_dir,
            args.max_caption_tokens,
            total_source_records,
            cached_indices,
        ):
            print(
                f"Using existing cache: {cache_path} "
                f"({len(cached_indices)}/{total_source_records} kept)",
                flush=True,
            )
            return

        chunks = list(chunk_ranges(total_source_records, args.chunk_size))
        print(
            f"Building valid caption index cache: {cache_path}\n"
            f"Total samples={total_source_records} workers={args.num_workers} chunk_size={args.chunk_size} "
            f"max_caption_tokens={args.max_caption_tokens}",
            flush=True,
        )

        valid_indices = []
        pool_kwargs = {
            "processes": args.num_workers,
            "initializer": init_worker,
            "initargs": (
                sources,
                args.model_dir,
                args.index_cache_root,
                args.max_caption_tokens,
                not args.no_lowercase,
                not args.keep_image_token,
                args.max_open_files_per_worker,
            ),
        }

        with Pool(**pool_kwargs) as pool:
            iterator = pool.imap(process_chunk, chunks)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(chunks), desc="Building valid_indices", unit="chunk")
            for _chunk_start, chunk_valid_indices in iterator:
                valid_indices.extend(chunk_valid_indices)

        valid_index_array = array("Q", valid_indices)
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_cache_path, "wb") as f:
            valid_index_array.tofile(f)
        os.replace(tmp_cache_path, cache_path)

        meta = train_mod.build_valid_indices_meta(
            args.data_path,
            args.cn_pair_root,
            args.model_dir,
            args.max_caption_tokens,
            total_source_records,
            len(valid_indices),
        )
        meta["index_file"] = cache_path
        train_mod.write_index_meta(cache_path, meta)

        print(
            f"Saved valid index cache: {cache_path}\n"
            f"Kept {len(valid_indices)}/{total_source_records} samples "
            f"({len(valid_indices) / total_source_records:.4%})",
            flush=True,
        )
    finally:
        try:
            os.remove(lock_file)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
