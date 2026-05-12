from __future__ import annotations
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import hashlib
from typing import Dict, Optional, Sequence, List
from array import array
from bisect import bisect_right
import time
import signal
import threading

import torch
import torch.multiprocessing as torch_mp
import random


import glob
import transformers

from torch.utils.data import Dataset
from fgclip2.train.local_trainer import CLIPTrainer


import torch.distributed as dist

import copy
import os
import json
import torch
import re
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange
# import cv2
from random import choice
from PIL import Image

import gzip
from io import BytesIO
import base64
from torch.utils.data import  IterableDataset
import random
import numpy as np

from fgclip2.model.strcs.fgclip2 import FG_CLIP2_Model
from transformers import AutoProcessor,Siglip2ImageProcessor


from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


import gc



local_rank = None
IMAGE_TOKEN_PATTERN = re.compile(r"<image>")

try:
    torch_mp.set_sharing_strategy("file_system")
except (RuntimeError, ValueError):
    pass


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class SampleTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame):
    raise SampleTimeoutError("Sample processing timed out.")


def call_with_timeout(timeout_seconds: int, fn, *args, **kwargs):
    if not timeout_seconds or timeout_seconds <= 0:
        return fn(*args, **kwargs)
    if threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def append_jsonl_record(log_path: Optional[str], record: dict) -> None:
    if not log_path:
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        if exc.errno != 24:
            raise


def load_image_rgb(image_path: str):
    with Image.open(image_path) as image:
        width, height = image.size
        rgb_image = image.convert("RGB")
    return rgb_image, width, height


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="qihoo360/fg-clip2-base")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    download_root: Optional[str] = field(default=None)
    log_scale: float = 4.6052
    loss_type: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    extra_image_folders: Optional[str] = field(
        default=None,
        metadata={"help": "Additional image roots for resolving relative image paths. Use ':' or ',' to separate multiple paths."},
    )
    index_cache_root: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for auto-generated index cache files for json/jsonl datasets."},
    )
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    max_seq_length: int = 64*4-60
    base_seq_length: int = 64
    use_long_caption: bool = field(
        default=True,
        metadata={"help": "Whether to train with the long caption image-text loss."},
    )
    long_caption_field: str = field(
        default="caption",
        metadata={"help": "JSON field to use as the long-caption source. Use 'messages' to read assistant content from messages."},
    )
    use_short_caption: bool = field(
        default=True,
        metadata={"help": "Whether to train with the short caption image-text loss."},
    )
    short_caption_field: str = field(
        default="short_caption",
        metadata={"help": "JSON field to use as the short-caption source. Use 'messages' to read assistant content from messages."},
    )
    box_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_hard_neg: bool = field(default=False)
    cn_pair_root: Optional[str] = field(default=None)
    cn_image_root: Optional[str] = field(default=None)
    missing_image_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a jsonl log file for missing or unreadable training images."},
    )
    large_image_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a jsonl log file for skipped over-large training images."},
    )
    bad_sample_log_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a jsonl log file for samples skipped due to per-sample exceptions."},
    )
    sample_timeout_seconds: int = field(
        default=20,
        metadata={"help": "Timeout in seconds for per-sample image loading/conversion. Set 0 to disable."},
    )
    preprocess_timeout_seconds: int = field(
        default=20,
        metadata={"help": "Timeout in seconds for image preprocessing in the collator. Set 0 to disable."},
    )
    max_caption_tokens: int = field(
        default=0,
        metadata={"help": "Skip training samples whose main long caption token length exceeds this value. Set 0 to disable."},
    )
    max_image_pixels: int = field(
        default=50000000,
        metadata={"help": "Skip images whose width * height exceeds this value. Set 0 to disable."},
    )
    max_num_patches: int = 0

    def __post_init__(self):
        if not self.use_long_caption and not self.use_short_caption:
            raise ValueError("At least one of use_long_caption or use_short_caption must be True.")


    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    train_use_word_size: int = 8
    text_model_lr: Optional[float] = None
    from_siglip2: bool = field(default=False)
    cn_and_en_2_train: bool = field(default=False)
    naflex_train: bool = field(default=False)


from datetime import datetime
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

import ast

AUTO_INDEX_LOCK_POLL_SECONDS = 2
AUTO_INDEX_LOCK_TIMEOUT_SECONDS = 12 * 60 * 60


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
    source_file = meta.get("source_file", meta.get("jsonl"))
    if source_file is not None and os.path.abspath(source_file) != abs_path:
        return False

    source_size = meta.get("source_size", meta.get("jsonl_size"))
    if source_size is not None and source_size != os.path.getsize(data_file):
        return False

    source_mtime = meta.get("source_mtime", meta.get("jsonl_mtime"))
    if source_mtime is not None and source_mtime != os.path.getmtime(data_file):
        return False

    meta_records = meta.get("max_records", meta.get("sample_size"))
    if meta_records != max_records:
        return False

    meta_seed = meta.get("sample_seed", meta.get("seed"))
    if meta_seed != sample_seed:
        return False

    return True


def load_existing_offsets_if_valid(
    index_file: Optional[str],
    data_file: str,
    offset_array: array,
    max_records: Optional[int],
    sample_seed: Optional[int],
    items_per_record: int,
):
    if index_file is None or not os.path.exists(index_file):
        return False

    try:
        meta = read_index_meta(index_file)
        if not index_matches_source(meta, data_file, max_records, sample_seed):
            return False

        del offset_array[:]
        index_size = os.path.getsize(index_file)
        with open(index_file, "rb") as f:
            offset_array.fromfile(f, index_size // offset_array.itemsize)
    except FileNotFoundError:
        # Another rank may be atomically replacing the index or meta file.
        # Treat this as a cache miss and let the caller retry.
        return False

    if len(offset_array) % items_per_record != 0:
        raise ValueError(f"Index file {index_file} is corrupted.")

    expected_records = None if max_records is None else max_records * items_per_record
    if expected_records is not None and len(offset_array) != expected_records:
        raise ValueError(
            f"Index file {index_file} has {len(offset_array) // items_per_record} records, expected {max_records}."
        )

    return True


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def path_signature(path: Optional[str]) -> dict:
    if not path:
        return {
            "path": None,
            "exists": False,
            "size": None,
            "mtime": None,
            "content_sha1": None,
        }

    abs_path = os.path.abspath(path)
    exists = os.path.exists(abs_path)
    content_sha1 = None
    if exists and os.path.isfile(abs_path) and abs_path.endswith(".txt"):
        content_sha1 = file_sha1(abs_path)
    return {
        "path": abs_path,
        "exists": exists,
        "size": os.path.getsize(abs_path) if exists else None,
        "mtime": os.path.getmtime(abs_path) if exists else None,
        "content_sha1": content_sha1,
    }


def make_valid_indices_cache_path(
    data_path: str,
    index_cache_root: Optional[str],
    max_caption_tokens: int,
    tokenizer_name_or_path: Optional[str],
    cn_pair_root: Optional[str] = None,
) -> Optional[str]:
    if not index_cache_root or max_caption_tokens <= 0:
        return None

    payload = {
        "data_path": os.path.abspath(data_path),
        "cn_pair_root": os.path.abspath(cn_pair_root) if cn_pair_root else None,
        "max_caption_tokens": max_caption_tokens,
        "tokenizer_name_or_path": tokenizer_name_or_path,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    basename = os.path.basename(os.path.abspath(data_path))
    return os.path.join(
        index_cache_root,
        f"{basename}.{digest}.caption_le_{max_caption_tokens}.valid.idx",
    )


def build_valid_indices_meta(
    data_path: str,
    cn_pair_root: Optional[str],
    tokenizer_name_or_path: Optional[str],
    max_caption_tokens: int,
    total_source_records: int,
    total_valid_records: int,
) -> dict:
    return {
        "index_kind": "caption_valid_indices",
        "data_path": path_signature(data_path),
        "cn_pair_root": path_signature(cn_pair_root),
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "max_caption_tokens": max_caption_tokens,
        "total_source_records": total_source_records,
        "total_valid_records": total_valid_records,
        "index_file": None,
    }


def valid_indices_cache_matches(
    meta: Optional[dict],
    data_path: str,
    cn_pair_root: Optional[str],
    tokenizer_name_or_path: Optional[str],
    max_caption_tokens: int,
    total_source_records: int,
) -> bool:
    if meta is None:
        return False

    if meta.get("index_kind") != "caption_valid_indices":
        return False

    if meta.get("tokenizer_name_or_path") != tokenizer_name_or_path:
        return False

    if meta.get("max_caption_tokens") != max_caption_tokens:
        return False

    if meta.get("total_source_records") != total_source_records:
        return False

    for key, expected_path in (("data_path", data_path), ("cn_pair_root", cn_pair_root)):
        expected = path_signature(expected_path)
        actual = meta.get(key, {})
        for field_name in ("path", "exists", "size"):
            if actual.get(field_name) != expected.get(field_name):
                return False
        if expected.get("content_sha1") is not None:
            if actual.get("content_sha1") != expected.get("content_sha1"):
                return False
        elif actual.get("mtime") != expected.get("mtime"):
            return False

    return True


def load_existing_valid_indices_if_valid(
    index_file: Optional[str],
    data_path: str,
    cn_pair_root: Optional[str],
    tokenizer_name_or_path: Optional[str],
    max_caption_tokens: int,
    total_source_records: int,
    valid_index_array: array,
) -> bool:
    if index_file is None or not os.path.exists(index_file):
        return False

    try:
        meta = read_index_meta(index_file)
        if not valid_indices_cache_matches(
            meta,
            data_path,
            cn_pair_root,
            tokenizer_name_or_path,
            max_caption_tokens,
            total_source_records,
        ):
            return False

        del valid_index_array[:]
        index_size = os.path.getsize(index_file)
        with open(index_file, "rb") as f:
            valid_index_array.fromfile(f, index_size // valid_index_array.itemsize)
    except FileNotFoundError:
        return False

    if any(index >= total_source_records for index in valid_index_array):
        raise ValueError(f"Valid index cache {index_file} is corrupted.")

    expected_valid_records = meta.get("total_valid_records")
    if expected_valid_records is not None and len(valid_index_array) != expected_valid_records:
        raise ValueError(
            f"Valid index cache {index_file} has {len(valid_index_array)} records, expected {expected_valid_records}."
        )

    return True


class IndexBuildLock:
    def __init__(
        self,
        index_file: str,
        data_file: str,
        max_records: Optional[int],
        sample_seed: Optional[int],
        items_per_record: int,
    ):
        self.index_file = index_file
        self.data_file = data_file
        self.max_records = max_records
        self.sample_seed = sample_seed
        self.items_per_record = items_per_record
        self.lock_file = f"{index_file}.lock"
        self.acquired = False

    def acquire(self) -> None:
        start_time = time.time()
        while True:
            try:
                fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(f"{os.getpid()}\n")
                self.acquired = True
                return
            except FileExistsError:
                if time.time() - start_time > AUTO_INDEX_LOCK_TIMEOUT_SECONDS:
                    raise TimeoutError(f"Timed out waiting for index lock: {self.lock_file}")

                if os.path.exists(self.index_file):
                    offsets = array("Q")
                    if load_existing_offsets_if_valid(
                        self.index_file,
                        self.data_file,
                        offsets,
                        self.max_records,
                        self.sample_seed,
                        self.items_per_record,
                    ):
                        return

                time.sleep(AUTO_INDEX_LOCK_POLL_SECONDS)

    def release(self) -> None:
        if self.acquired:
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                pass
        self.acquired = False


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
        self.index_file = index_file

        if load_existing_offsets_if_valid(
            index_file,
            data_file,
            self.offsets,
            max_records,
            sample_seed,
            items_per_record=2,
        ):
            return

        rng = random.Random(sample_seed) if sample_seed is not None and max_records is not None else None
        record_count = 0
        saw_array_start = False
        in_string = False
        escape = False
        brace_depth = 0
        object_start = None

        reached_array_end = False

        build_lock = None if index_file is None else IndexBuildLock(index_file, data_file, max_records, sample_seed, 2)
        try:
            if build_lock is not None:
                build_lock.acquire()
                if load_existing_offsets_if_valid(
                    index_file,
                    data_file,
                    self.offsets,
                    max_records,
                    sample_seed,
                    items_per_record=2,
                ):
                    return

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
        finally:
            if build_lock is not None:
                build_lock.release()

    def __len__(self):
        return len(self.offsets) // 2

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = self.offsets[2 * index]
        end = self.offsets[2 * index + 1]
        with open(self.data_file, "rb") as f:
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
        self.index_file = index_file

        if load_existing_offsets_if_valid(
            index_file,
            data_file,
            self.offsets,
            max_records,
            sample_seed,
            items_per_record=1,
        ):
            return

        rng = random.Random(sample_seed) if sample_seed is not None else None
        record_count = 0

        build_lock = None if index_file is None else IndexBuildLock(index_file, data_file, max_records, sample_seed, 1)
        try:
            if build_lock is not None:
                build_lock.acquire()
                if load_existing_offsets_if_valid(
                    index_file,
                    data_file,
                    self.offsets,
                    max_records,
                    sample_seed,
                    items_per_record=1,
                ):
                    return

            with open(data_file, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        record_count += 1
                        if rng is not None and max_records is not None:
                            if len(self.offsets) < max_records:
                                self.offsets.append(offset)
                            else:
                                sample_idx = rng.randrange(record_count)
                                if sample_idx < max_records:
                                    self.offsets[sample_idx] = offset
                        else:
                            self.offsets.append(offset)
                        if rng is None and max_records is not None and len(self.offsets) >= max_records:
                            break
            if sample_seed is not None and max_records is not None:
                if max_records > record_count:
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
        finally:
            if build_lock is not None:
                build_lock.release()

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fp"] = None
        return state

    def __getitem__(self, index):
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)

        with open(self.data_file, "rb") as f:
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

    def __len__(self):
        if not self.cumulative_sizes:
            return 0
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        store_idx = bisect_right(self.cumulative_sizes, index)
        previous_size = 0 if store_idx == 0 else self.cumulative_sizes[store_idx - 1]
        return self.stores[store_idx][index - previous_size]


def parse_manifest_line(line: str, manifest_dir: str):
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


def build_data_store(
    data_path: str,
    max_records: Optional[int] = None,
    sample_seed: Optional[int] = None,
    index_file: Optional[str] = None,
    index_cache_root: Optional[str] = None,
):
    data_path = os.path.abspath(data_path)
    resolved_index_file = index_file
    if resolved_index_file is None:
        resolved_index_file = make_auto_index_path(
            data_path,
            index_cache_root,
            max_records=max_records,
            sample_seed=sample_seed,
        )

    if data_path.endswith(".jsonl"):
        return JsonlOffsetStore(
            data_path,
            max_records=max_records,
            sample_seed=sample_seed,
            index_file=resolved_index_file,
        )

    if data_path.endswith(".json"):
        return JsonArrayOffsetStore(
            data_path,
            max_records=max_records,
            sample_seed=sample_seed,
            index_file=resolved_index_file,
        )

    if data_path.endswith(".txt"):
        stores = []
        manifest_dir = os.path.dirname(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                json_file, json_limit, json_sample_seed, json_index_file = parse_manifest_line(line, manifest_dir)
                if not json_file:
                    continue
                stores.append(
                    build_data_store(
                        json_file,
                        max_records=json_limit,
                        sample_seed=json_sample_seed,
                        index_file=json_index_file,
                        index_cache_root=index_cache_root,
                    )
                )
        return ConcatStore(stores)

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



class LazySupervisedBboxDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None,tokenizer=None):
        super(LazySupervisedBboxDataset, self).__init__()

        data_store = build_data_store(data_path, index_cache_root=data_args.index_cache_root)
        self.en_data_length = len(data_store)

        if data_args.cn_pair_root is not None:
            cn_data_store = build_data_store(data_args.cn_pair_root, index_cache_root=data_args.index_cache_root)
            data_store = ConcatStore([data_store, cn_data_store])

        self.all_data_length = len(data_store)


        rank0_print("Formatting inputs...Skip in lazy mode")

        self.total_len = 1000
        self.tokenizer = tokenizer
        self.data_store = data_store
        self.max_anns = 4

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.data_path = data_path
        self.image_root = data_args.image_folder
        self.extra_image_roots = self.parse_image_roots(data_args.extra_image_folders)
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.use_long_caption = data_args.use_long_caption
        self.long_caption_field = data_args.long_caption_field
        self.use_short_caption = data_args.use_short_caption
        self.short_caption_field = data_args.short_caption_field
        self.box_image_size = data_args.box_image_size
        self.add_box_loss = data_args.add_box_loss
        self.use_hard_neg = data_args.use_hard_neg
        self.max_caption_tokens = data_args.max_caption_tokens
        self.cn_image_root = data_args.cn_image_root
        self.missing_image_log_path = data_args.missing_image_log_path
        self.large_image_log_path = data_args.large_image_log_path
        self.bad_sample_log_path = data_args.bad_sample_log_path
        self.sample_timeout_seconds = data_args.sample_timeout_seconds
        self.max_image_pixels = data_args.max_image_pixels
        self.logged_missing_images = set()
        self.logged_large_images = set()
        self.logged_bad_samples = set()
        self.valid_indices = self.build_valid_indices()

    def __len__(self):
        return len(self.valid_indices)

    def get_caption(self, item):
        if "caption" in item:
            return item["caption"]

        messages = item.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
                    return message["content"]
            for message in messages:
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]

        raise KeyError("caption is required, or messages must contain a content field")

    def get_message_caption(self, item):
        messages = item.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "assistant" and message.get("content"):
                    return message["content"]
            for message in messages:
                if isinstance(message, dict) and message.get("content"):
                    return message["content"]

        raise KeyError("messages must contain a content field")

    def get_configured_caption(self, item, field_name: str, default_field_name: str):
        if field_name == "messages":
            return self.get_message_caption(item)

        if field_name == "caption" and default_field_name == "caption":
            return self.get_caption(item)

        if field_name in item:
            return item[field_name]

        raise KeyError(f"{field_name} is required for configured caption field")

    def get_image_path(self, item):
        if "f_path" in item:
            return item["f_path"]

        images = item.get("images")
        if isinstance(images, list) and len(images) > 0:
            return images[0]

        raise KeyError("f_path is required, or images must contain at least one path")

    def parse_image_roots(self, value):
        if not value:
            return []
        image_roots = []
        for part in value.replace(",", os.pathsep).split(os.pathsep):
            part = part.strip()
            if part:
                image_roots.append(part)
        return image_roots

    def caption_exceeds_max_tokens(self, caption: str) -> bool:
        if self.max_caption_tokens <= 0 or self.tokenizer is None:
            return False

        input_ids = self.tokenizer(
            [caption.lower()],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        ).input_ids
        return len(input_ids[0]) > self.max_caption_tokens

    def build_valid_indices(self):
        if not self.use_long_caption:
            return list(range(len(self.data_store)))

        if self.max_caption_tokens <= 0 or self.tokenizer is None:
            return list(range(len(self.data_store)))

        tokenizer_name_or_path = getattr(self.tokenizer, "name_or_path", None)
        cache_path = make_valid_indices_cache_path(
            self.data_path,
            self.data_args.index_cache_root,
            self.max_caption_tokens,
            tokenizer_name_or_path,
            self.data_args.cn_pair_root,
        )
        cached_indices = array("Q")
        if load_existing_valid_indices_if_valid(
            cache_path,
            self.data_path,
            self.data_args.cn_pair_root,
            tokenizer_name_or_path,
            self.max_caption_tokens,
            len(self.data_store),
            cached_indices,
        ):
            rank0_print(
                f"Loaded caption-filtered index cache: {cache_path} "
                f"({len(cached_indices)}/{len(self.data_store)} samples kept)"
            )
            return list(cached_indices)

        lock_file = None if cache_path is None else f"{cache_path}.lock"
        lock_acquired = False
        rank0_print(
            f"Filtering samples by caption token length <= {self.max_caption_tokens} before training..."
        )
        try:
            if lock_file is not None:
                start_time = time.time()
                while True:
                    try:
                        fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        with os.fdopen(fd, "w", encoding="utf-8") as f:
                            f.write(f"{os.getpid()}\n")
                        lock_acquired = True
                        break
                    except FileExistsError:
                        if time.time() - start_time > AUTO_INDEX_LOCK_TIMEOUT_SECONDS:
                            raise TimeoutError(f"Timed out waiting for valid index cache lock: {lock_file}")

                        if load_existing_valid_indices_if_valid(
                            cache_path,
                            self.data_path,
                            self.data_args.cn_pair_root,
                            tokenizer_name_or_path,
                            self.max_caption_tokens,
                            len(self.data_store),
                            cached_indices,
                        ):
                            rank0_print(
                                f"Loaded caption-filtered index cache: {cache_path} "
                                f"({len(cached_indices)}/{len(self.data_store)} samples kept)"
                            )
                            return list(cached_indices)

                        time.sleep(AUTO_INDEX_LOCK_POLL_SECONDS)

                if load_existing_valid_indices_if_valid(
                    cache_path,
                    self.data_path,
                    self.data_args.cn_pair_root,
                    tokenizer_name_or_path,
                    self.max_caption_tokens,
                    len(self.data_store),
                    cached_indices,
                ):
                    rank0_print(
                        f"Loaded caption-filtered index cache: {cache_path} "
                        f"({len(cached_indices)}/{len(self.data_store)} samples kept)"
                    )
                    return list(cached_indices)

            valid_indices = []
            for cur_idx in range(len(self.data_store)):
                try:
                    item = self.data_store[cur_idx]
                    caption = IMAGE_TOKEN_PATTERN.sub(
                        "",
                        self.get_configured_caption(item, self.long_caption_field, "caption"),
                    ).strip()
                    if self.caption_exceeds_max_tokens(caption):
                        continue
                    valid_indices.append(cur_idx)
                except Exception:
                    # Keep the sample in the candidate set; existing runtime guards will handle
                    # unreadable/malformed records without collapsing the dataset at init time.
                    valid_indices.append(cur_idx)

            if cache_path is not None:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}"
                valid_index_array = array("Q", valid_indices)
                with open(tmp_cache_path, "wb") as f:
                    valid_index_array.tofile(f)
                os.replace(tmp_cache_path, cache_path)
                meta = build_valid_indices_meta(
                    self.data_path,
                    self.data_args.cn_pair_root,
                    tokenizer_name_or_path,
                    self.max_caption_tokens,
                    len(self.data_store),
                    len(valid_indices),
                )
                meta["index_file"] = cache_path
                write_index_meta(cache_path, meta)

            rank0_print(
                f"Caption length filtering kept {len(valid_indices)}/{len(self.data_store)} samples."
            )
            return valid_indices
        finally:
            if lock_acquired and lock_file is not None:
                try:
                    os.remove(lock_file)
                except FileNotFoundError:
                    pass

    def resolve_image_name(self, image_path, is_cn):
        if os.path.isabs(image_path):
            return image_path

        candidate_roots = []
        if is_cn and self.cn_image_root:
            candidate_roots.append(self.cn_image_root)
        if self.image_root:
            candidate_roots.append(self.image_root)
        candidate_roots.extend(self.extra_image_roots)

        first_candidate = None
        for root in candidate_roots:
            candidate = os.path.join(root, image_path)
            if first_candidate is None:
                first_candidate = candidate
            if os.path.exists(candidate):
                return candidate

        if first_candidate is not None:
            return first_candidate
        return image_path

    def log_missing_image(self, index, image_path, image_name, error):
        if self.missing_image_log_path is None:
            return

        if image_name in self.logged_missing_images:
            return

        self.logged_missing_images.add(image_name)
        log_dir = os.path.dirname(self.missing_image_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        record = {
            "index": index,
            "image_path": image_path,
            "resolved_path": image_name,
            "error": error,
        }
        append_jsonl_record(self.missing_image_log_path, record)

    def log_large_image(self, index, image_path, image_name, width, height):
        if self.large_image_log_path is None:
            return

        if image_name in self.logged_large_images:
            return

        self.logged_large_images.add(image_name)
        log_dir = os.path.dirname(self.large_image_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        record = {
            "index": index,
            "image_path": image_path,
            "resolved_path": image_name,
            "width": width,
            "height": height,
            "pixels": width * height,
            "max_image_pixels": self.max_image_pixels,
        }
        append_jsonl_record(self.large_image_log_path, record)

    def log_bad_sample(self, index, item, image_path, image_name, error):
        if self.bad_sample_log_path is None:
            return

        sample_key = (index, image_name, error)
        if sample_key in self.logged_bad_samples:
            return

        self.logged_bad_samples.add(sample_key)
        log_dir = os.path.dirname(self.bad_sample_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        caption_preview = None
        try:
            preview_source = None
            if self.use_long_caption:
                preview_source = self.get_configured_caption(item, self.long_caption_field, "caption")
            elif self.use_short_caption:
                preview_source = self.get_configured_caption(item, self.short_caption_field, "short_caption")
            if preview_source is not None:
                caption_preview = IMAGE_TOKEN_PATTERN.sub("", preview_source).strip()[:256]
        except Exception:
            pass

        record = {
            "index": index,
            "image_path": image_path,
            "resolved_path": image_name,
            "error": error,
            "item_id": item.get("id") if isinstance(item, dict) else None,
            "item_keys": sorted(item.keys()) if isinstance(item, dict) else None,
            "caption_preview": caption_preview,
        }
        append_jsonl_record(self.bad_sample_log_path, record)

    def load_valid_item(self, i):
        dataset_len = len(self.valid_indices)
        for offset in range(dataset_len):
            dataset_idx = (i + offset) % dataset_len
            cur_idx = self.valid_indices[dataset_idx]
            try:
                item = self.data_store[cur_idx]
                caption = None
                if self.use_long_caption:
                    caption = IMAGE_TOKEN_PATTERN.sub(
                        "",
                        self.get_configured_caption(item, self.long_caption_field, "caption"),
                    ).strip()
                image_path = self.get_image_path(item)
                caption_short = None

                if "is_cn" not in item.keys():
                    is_cn = False
                    if self.use_short_caption:
                        caption_short = self.get_configured_caption(item, self.short_caption_field, "short_caption")
                        caption_short = "a photo of " + caption_short
                else:
                    is_cn = True
                    if self.use_short_caption:
                        caption_short = self.get_configured_caption(item, self.short_caption_field, "short_caption")

                image_name = self.resolve_image_name(image_path, is_cn)
                image, width, height = call_with_timeout(
                    self.sample_timeout_seconds,
                    load_image_rgb,
                    image_name,
                )
                if self.max_image_pixels > 0 and width * height > self.max_image_pixels:
                    self.log_large_image(cur_idx, image_path, image_name, width, height)
                    image.close()
                    continue
            except (FileNotFoundError, OSError) as e:
                self.log_missing_image(
                    cur_idx,
                    image_path if "image_path" in locals() else None,
                    image_name if "image_name" in locals() else None,
                    repr(e),
                )
                continue
            except Exception as e:
                self.log_bad_sample(
                    cur_idx,
                    item if "item" in locals() else None,
                    image_path if "image_path" in locals() else None,
                    image_name if "image_name" in locals() else None,
                    repr(e),
                )
                continue

            return cur_idx, item, caption, caption_short, is_cn, image, image_path, image_name

        raise RuntimeError("No readable image was found in the dataset.")


    @property
    def modality_lengths(self):
        length_list = []
        for cur_idx in self.valid_indices:
            if cur_idx < self.en_data_length:
                length_list.append(1)
            else:
                length_list.append(0)
        return length_list
 

    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        cur_idx, item, caption, caption_short, is_cn, image, image_path, image_name = self.load_valid_item(i)
        

        prewidth, preheight = image.size

        if self.data_args.max_num_patches !=0:
            # NOTE The low unilateral resolution may cause a bug, forced to resize.
            if prewidth < 128 or preheight < 128:
                image = image.resize((self.box_image_size, self.box_image_size))

            width, height = image.size
            max_img_token = (width//16)*(height//16)
            image_tensor = image
            pixel_attention_mask = None
            spatial_shapes = None
        else:
            image = image.resize((self.box_image_size, self.box_image_size))
            width, height = image.size
            max_img_token = (width//16)*(height//16)
            pixel_attention_mask = None
            spatial_shapes = None
            image_tensor = self.preprocess(images=image, return_tensors='pt')['pixel_values'][0]

        
        max_img_token = torch.tensor([max_img_token])

        text = None
        if self.use_long_caption:
            text = torch.tensor(
                self.tokenizer([caption.lower()], max_length=self.max_length, padding="max_length", truncation=True).input_ids,
                dtype=torch.long,
            )
        short_text = None
        if self.use_short_caption:
            short_text = torch.tensor(
                self.tokenizer([caption_short.lower()], max_length=self.base_length, padding="max_length", truncation=True).input_ids,
                dtype=torch.long,
            )
        tensor_device = text.device if text is not None else short_text.device



        if self.add_box_loss:

            box_texts = []
            total_num = self.max_anns
            if "is_cn" not in item.keys():
                bbox_info = item["bbox_info"]
                valid_num = min(len(bbox_info), self.max_anns)
            else:
                valid_num = 0

            boxes_template = torch.zeros((total_num, 4), device=tensor_device)
            width, height = image.size

            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = random.choice([bbox_data["short_expr"], bbox_data["long_expr"]])
                else:
                    box = [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.000000000]
                    box_caption = ""


                box_tensor = torch.tensor(box[:4])
                boxes_template[i] = box_tensor

                if box[0] > box[2] or box[1] > box[3]:
                    raise ValueError("Box coordinates are invalid.")

                left = int(box[0] * width)
                top = int(box[1] * height)
                right = int(box[2] * width)
                bottom = int(box[3] * height)
                box_text = torch.tensor(self.tokenizer([box_caption.lower()], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=tensor_device)
                box_texts.append(box_text)

            box_texts = torch.cat(box_texts,dim=0)

            bbox_num = torch.tensor([valid_num], device=tensor_device)

        if self.use_hard_neg:
            hard_texts = []

            width, height = image.size
            total_num = self.max_anns
           
            if "is_cn" not in item.keys():
                bbox_info = item["bbox_info"]
                valid_num = min(len(bbox_info), self.max_anns)
            else:
                valid_num = 0

            hard_boxes = torch.zeros((total_num, 4), device=tensor_device)
            valid_hard = 0
            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = bbox_data["short_expr"]
                    
                    box_tensor = torch.tensor(box[:4])
                    if box[0] > box[2] or box[1] > box[3]:
                        raise ValueError("Box coordinates are invalid.")
    
                    if bbox_data["flag_short_neg"] == 1:
                        cur_texts = [box_caption]
                        hard_negs = bbox_data["short_expr_negs"]
                        for key in hard_negs.keys():
                            cur_texts.append(hard_negs[key].lower())
                        box_text = torch.tensor(self.tokenizer(cur_texts, max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=tensor_device)
                        hard_texts.append(box_text)

                        hard_boxes[valid_hard] = box_tensor
                        valid_hard = valid_hard+1
    
                        left = int(box[0] * width)
                        top = int(box[1] * height)
                        right = int(box[2] * width)
                        bottom = int(box[3] * height)
  

            valid_hard = torch.tensor([valid_hard], device=tensor_device)
   
            if len(hard_texts) > 0:
                hard_texts = torch.cat(hard_texts,dim=0)
            else:
                hard_texts = None

        data_dict = {}
        data_dict['image'] = image_tensor
        data_dict['sample_index'] = cur_idx
        data_dict['image_path'] = image_path
        data_dict['resolved_path'] = image_name
        data_dict['pixel_attention_mask'] = pixel_attention_mask
        data_dict['spatial_shapes'] = spatial_shapes
        
        
        data_dict['text'] = text
        data_dict['short_text'] = short_text

        data_dict['add_box_loss'] = self.add_box_loss
        data_dict['use_hard_neg'] = self.use_hard_neg
        data_dict['max_img_token'] = max_img_token
        data_dict['is_cn'] = is_cn

        if self.add_box_loss:
            # data_dict['box_images'] = box_images
            data_dict['box_texts'] = box_texts
            data_dict['box_infos'] = boxes_template
            data_dict['box_nums'] = bbox_num
        if self.use_hard_neg:

            data_dict['hard_texts'] = hard_texts
            data_dict['hard_infos'] = hard_boxes
            data_dict['hard_nums'] = valid_hard
            
        return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    preprocess: transformers.Siglip2ImageProcessor
    is_naflex: bool
    bad_sample_log_path: Optional[str] = None
    preprocess_timeout_seconds: int = 20

    def determine_max_value(self,values):

        max_val = torch.max(values).item()

        if max_val > 784:
            return 1024
        elif max_val > 576:
            return 784
        elif max_val > 256:
            return 576
        elif max_val > 128:
            return 256
        else:
            return 128

    def __call__(self, instances: Sequence[Dict]):
        
        batch = {}

        if self.is_naflex:
            batch_max_img_token = self.determine_max_value(torch.stack([instance['max_img_token'] for instance in instances]))

            pixel_values = []
            pixel_attention_masks = []
            spatial_shapes = []

            for instance in instances:

                try:
                    rgb_image = call_with_timeout(self.preprocess_timeout_seconds, instance['image'].convert, "RGB")
                    image_input = call_with_timeout(
                        self.preprocess_timeout_seconds,
                        self.preprocess,
                        images=rgb_image,
                        max_num_patches=batch_max_img_token,
                        return_tensors='pt',
                    )
                except Exception as e:
                    append_jsonl_record(
                        self.bad_sample_log_path,
                        {
                            "index": instance.get("sample_index"),
                            "image_path": instance.get("image_path"),
                            "resolved_path": instance.get("resolved_path"),
                            "error": f"collator_preprocess_failed: {repr(e)}",
                        },
                    )
                    width, height = 384, 384  # 
                    channels = 3  
                    black_image_array = np.zeros((height, width, channels), dtype=np.uint8)
                    black_image = Image.fromarray(black_image_array, mode="RGB")
                    image_input = self.preprocess(images=black_image, max_num_patches=batch_max_img_token, return_tensors='pt')

                pixel_values.append(image_input["pixel_values"])
                pixel_attention_masks.append(image_input["pixel_attention_mask"])
                spatial_shapes.append(image_input["spatial_shapes"])

            batch['pixel_values'] = torch.cat(pixel_values,dim=0)
            batch['pixel_attention_mask'] = torch.cat(pixel_attention_masks,dim=0)
            batch['spatial_shapes'] = torch.cat(spatial_shapes,dim=0)

        else:
            batch['pixel_attention_mask'] = None
            batch['spatial_shapes'] = None
            images = [instance['image'] for instance in instances]
            batch['pixel_values'] = torch.stack(images)

        texts = [instance['text'] for instance in instances]

        if any(text is None for text in texts):
            batch['text_long'] = None
            batch['text_long_flag'] = torch.tensor([0], device=batch['pixel_values'].device)
        else:
            batch['text_long_flag'] = torch.tensor([1], device=batch['pixel_values'].device)
            batch['text_long'] = torch.cat(texts,dim=0)

        short_texts = [instance['short_text'] for instance in instances]
        if any(short_text is None for short_text in short_texts):
            batch['text_short'] = None
        else:
            batch['text_short'] = torch.cat(short_texts,dim=0)
        
        batch["add_box_loss"] = instances[0]["add_box_loss"]
        batch["use_hard_neg"] = instances[0]["use_hard_neg"]
        batch["sample_indices"] = [instance["sample_index"] for instance in instances]
        batch["image_paths"] = [instance["image_path"] for instance in instances]
        batch["resolved_paths"] = [instance["resolved_path"] for instance in instances]
        
        if batch["add_box_loss"]:

            box_texts = [instance['box_texts'] for instance in instances]
            batch['box_texts'] = torch.cat(box_texts,dim=0)
            box_infos = [instance['box_infos'] for instance in instances]
            batch['box_infos'] = torch.cat(box_infos,dim=0)
            box_nums = [instance['box_nums'] for instance in instances]
            batch['box_nums'] = torch.cat(box_nums, dim=0)
            
        if batch["use_hard_neg"] :
            hard_texts = []
            for instance in instances:
                if instance['hard_texts'] != None:
                    hard_texts.append(instance['hard_texts'])
            if len(hard_texts)!=0:
                batch['hard_texts'] = torch.cat(hard_texts,dim=0)
            else:
                batch['hard_texts'] = None
            hard_infos = [instance['hard_infos'] for instance in instances]
            batch['hard_infos'] = torch.cat(hard_infos,dim=0)
            hard_nums = [instance['hard_nums'] for instance in instances]
            batch['hard_nums'] = torch.cat(hard_nums, dim=0)                

        return batch




def make_supervised_data_module(data_args,img_preprocess,tokenizer,is_naflex) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    

    train_dataset = LazySupervisedBboxDataset(
                                data_path=data_args.data_path,
                                data_args=data_args,
                                img_preprocess=img_preprocess,tokenizer=tokenizer,)
            
    data_collator = DataCollatorForSupervisedDataset(
        preprocess=img_preprocess,
        is_naflex=is_naflex,
        bad_sample_log_path=data_args.bad_sample_log_path,
        preprocess_timeout_seconds=data_args.preprocess_timeout_seconds,
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not data_args.use_long_caption and not data_args.use_short_caption:
        raise ValueError("At least one of --use_long_caption or --use_short_caption must be enabled.")
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    # compute_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

    assert training_args.naflex_train

    if training_args.naflex_train:
        assert data_args.max_num_patches in [128, 256, 576, 784, 1024, 4096]
        image_processor = Siglip2ImageProcessor.from_pretrained(model_args.base_model)
    else:
        pass

    model = FG_CLIP2_Model.from_pretrained(model_args.model_name_or_path)

    config = model.config
    import numpy as np

    model.logit_scale_finegraind = torch.nn.Parameter(torch.ones([]) * model_args.log_scale)
    model.logit_scale_hardneg = torch.nn.Parameter(torch.ones([]) * model_args.log_scale)
    
    if training_args.from_siglip2:
        print("copy and resize")
        model.resize_postion_embeding()
        model.copy_weight()
        print("copy_weight")
        model.copy_dense_feature_head()
        print("copy_dense_feature_head")
        print("fine")

    model.world_size = training_args.train_use_word_size
    model.loss_type = model_args.loss_type

    data_module = make_supervised_data_module(data_args=data_args,img_preprocess=image_processor,tokenizer=tokenizer,is_naflex=training_args.naflex_train)
    
    model.to(dtype=compute_dtype, device=training_args.device)

    # old: --gradient_checkpointing_kwargs {"use_reentrant":True} \
    training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}

    trainer = CLIPTrainer(model=model,
                        args=training_args,
                        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
