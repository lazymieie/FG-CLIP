#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from PIL import Image
from transformers import Siglip2ImageProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze offline-replayed step batches and rank suspicious samples.")
    parser.add_argument("--input", required=True, help="Path to step batch JSON produced by inspect_step_batch.py.")
    parser.add_argument("--model-dir", required=True, help="SigLIP2 model dir for Siglip2ImageProcessor.")
    parser.add_argument("--global-ranks", default=None, help="Comma-separated global ranks to inspect. Default: all in input.")
    parser.add_argument("--top-k", type=int, default=50, help="How many suspicious samples to print.")
    parser.add_argument("--slow-threshold-seconds", type=float, default=5.0, help="Mark samples slower than this as suspicious.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def determine_max_value(max_img_token: int) -> int:
    if max_img_token > 784:
        return 1024
    if max_img_token > 576:
        return 784
    if max_img_token > 256:
        return 576
    if max_img_token > 128:
        return 256
    return 128


def get_rank_filter(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def iter_samples(payload: dict[str, Any], rank_filter: set[int] | None):
    for microstep in payload.get("microsteps", []):
        microstep_id = microstep["microstep_index_one_based"]
        for rank_info in microstep.get("ranks", []):
            global_rank = rank_info["global_rank"]
            if rank_filter is not None and global_rank not in rank_filter:
                continue
            for sample in rank_info.get("samples", []):
                yield microstep_id, rank_info, sample


def analyze_sample(processor: Siglip2ImageProcessor, sample: dict[str, Any]) -> dict[str, Any]:
    path = sample["resolved_path"]
    record: dict[str, Any] = {
        "dataset_index": sample.get("dataset_index"),
        "item_id": sample.get("item_id"),
        "image_path": sample.get("image_path"),
        "resolved_path": path,
        "caption_preview": sample.get("caption_preview"),
    }
    t0 = time.perf_counter()
    image = Image.open(path)
    t1 = time.perf_counter()
    image.load()
    t2 = time.perf_counter()
    rgb = image.convert("RGB")
    t3 = time.perf_counter()
    width, height = rgb.size
    max_img_token = (width // 16) * (height // 16)
    patch_bucket = determine_max_value(max_img_token)
    processor(images=rgb, max_num_patches=patch_bucket, return_tensors="pt")
    t4 = time.perf_counter()

    record.update(
        {
            "width": width,
            "height": height,
            "pixels": width * height,
            "max_img_token": max_img_token,
            "patch_bucket": patch_bucket,
            "open_seconds": round(t1 - t0, 4),
            "load_seconds": round(t2 - t1, 4),
            "convert_seconds": round(t3 - t2, 4),
            "preprocess_seconds": round(t4 - t3, 4),
            "total_seconds": round(t4 - t0, 4),
        }
    )
    return record


def main() -> None:
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rank_filter = get_rank_filter(args.global_ranks)
    processor = Siglip2ImageProcessor.from_pretrained(args.model_dir)

    results = []
    errors = []
    seen = set()

    for microstep_id, rank_info, sample in iter_samples(payload, rank_filter):
        sample_key = (sample.get("dataset_index"), sample.get("resolved_path"))
        if sample_key in seen:
            continue
        seen.add(sample_key)

        base = {
            "microstep_index_one_based": microstep_id,
            "global_rank": rank_info["global_rank"],
            "node_rank": rank_info["node_rank"],
            "local_rank": rank_info["local_rank"],
        }
        try:
            analyzed = analyze_sample(processor, sample)
            analyzed.update(base)
            results.append(analyzed)
        except Exception as exc:
            error_record = dict(base)
            error_record.update(
                {
                    "dataset_index": sample.get("dataset_index"),
                    "item_id": sample.get("item_id"),
                    "image_path": sample.get("image_path"),
                    "resolved_path": sample.get("resolved_path"),
                    "caption_preview": sample.get("caption_preview"),
                    "error": repr(exc),
                }
            )
            errors.append(error_record)

    results.sort(key=lambda x: x["total_seconds"], reverse=True)
    suspicious = [r for r in results if r["total_seconds"] >= args.slow_threshold_seconds]

    summary = {
        "input": args.input,
        "top_k": args.top_k,
        "slow_threshold_seconds": args.slow_threshold_seconds,
        "num_analyzed": len(results),
        "num_errors": len(errors),
        "num_suspicious": len(suspicious),
        "errors": errors[: args.top_k],
        "slowest_samples": results[: args.top_k],
        "suspicious_samples": suspicious[: args.top_k],
    }

    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
