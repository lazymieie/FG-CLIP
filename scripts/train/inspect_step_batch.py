#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import Any

import torch

from fgclip2.train.train_fgclip2 import DataArguments, LazySupervisedBboxDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay a training step and dump the batch sample paths for each rank."
    )
    parser.add_argument("--data-path", required=True, help="Training data path or manifest.")
    parser.add_argument("--image-folder", default=None, help="Primary image root.")
    parser.add_argument("--extra-image-folders", default=None, help="Additional image roots.")
    parser.add_argument("--cn-image-root", default=None, help="Chinese image root, if used.")
    parser.add_argument("--index-cache-root", default=None, help="Index cache root.")
    parser.add_argument("--step", type=int, required=True, help="1-based optimizer step to inspect.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        required=True,
        help="Gradient accumulation steps used in training.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        required=True,
        help="Per-device train batch size used in training.",
    )
    parser.add_argument("--world-size", type=int, required=True, help="Total number of ranks.")
    parser.add_argument("--data-seed", type=int, required=True, help="Training data seed.")
    parser.add_argument(
        "--global-ranks",
        default=None,
        help="Comma-separated global ranks to inspect. Default: all ranks.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Default: print JSON to stdout only.",
    )
    parser.add_argument(
        "--caption-preview-len",
        type=int,
        default=120,
        help="Max caption preview length per sample in output.",
    )
    return parser.parse_args()


def build_dataset(args: argparse.Namespace) -> LazySupervisedBboxDataset:
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        extra_image_folders=args.extra_image_folders,
        cn_image_root=args.cn_image_root,
        index_cache_root=args.index_cache_root,
        use_short_caption=False,
        add_box_loss=False,
        use_hard_neg=False,
    )
    return LazySupervisedBboxDataset(
        data_path=args.data_path,
        data_args=data_args,
        img_preprocess=None,
        tokenizer=None,
    )


def get_rank_list(args: argparse.Namespace) -> list[int]:
    if not args.global_ranks:
        return list(range(args.world_size))
    ranks = []
    for part in args.global_ranks.split(","):
        part = part.strip()
        if part:
            ranks.append(int(part))
    return ranks


def get_sample_record(
    dataset: LazySupervisedBboxDataset,
    dataset_index: int,
    caption_preview_len: int,
) -> dict[str, Any]:
    item = dataset.data_store[dataset_index]
    caption = dataset.get_caption(item)
    image_path = dataset.get_image_path(item)
    is_cn = "is_cn" in item
    resolved_path = dataset.resolve_image_name(image_path, is_cn)
    return {
        "dataset_index": dataset_index,
        "item_id": item.get("id") if isinstance(item, dict) else None,
        "image_path": image_path,
        "resolved_path": resolved_path,
        "is_cn": is_cn,
        "caption_preview": caption[:caption_preview_len],
    }


def main() -> None:
    args = parse_args()
    if args.step <= 0:
        raise ValueError("--step must be >= 1")

    dataset = build_dataset(args)
    num_samples = len(dataset)

    generator = torch.Generator()
    generator.manual_seed(args.data_seed)
    permutation = torch.randperm(num_samples, generator=generator, dtype=torch.int64)

    start_microstep = (args.step - 1) * args.gradient_accumulation_steps
    ranks = get_rank_list(args)

    result: dict[str, Any] = {
        "step": args.step,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "world_size": args.world_size,
        "data_seed": args.data_seed,
        "dataset_length": num_samples,
        "microsteps": [],
    }

    for microstep_offset in range(args.gradient_accumulation_steps):
        local_microstep = start_microstep + microstep_offset
        microstep_record = {
            "microstep_index_zero_based": local_microstep,
            "microstep_index_one_based": local_microstep + 1,
            "ranks": [],
        }
        for global_rank in ranks:
            base_batch_index = local_microstep * args.world_size + global_rank
            start = base_batch_index * args.per_device_train_batch_size
            end = start + args.per_device_train_batch_size
            batch_indices = permutation[start:end].tolist()
            samples = [
                get_sample_record(dataset, dataset_index, args.caption_preview_len)
                for dataset_index in batch_indices
            ]
            microstep_record["ranks"].append(
                {
                    "global_rank": global_rank,
                    "node_rank": global_rank // 8,
                    "local_rank": global_rank % 8,
                    "base_batch_index": base_batch_index,
                    "start_offset": start,
                    "end_offset_exclusive": end,
                    "num_samples": len(samples),
                    "samples": samples,
                }
            )
        result["microsteps"].append(microstep_record)

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
