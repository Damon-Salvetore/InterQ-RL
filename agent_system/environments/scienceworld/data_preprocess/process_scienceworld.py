#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT_FILES = {
    "train": "train.parquet",
    "dev": "dev.parquet",
    "test": "test.parquet",
    "monitor": "dev_monitor.parquet",
}


def _build_row(
    *,
    split: str,
    index: int,
    task_name: str,
    variation_id: int,
    official_split: str,
) -> dict[str, Any]:
    return {
        "data_source": f"scienceworld_{split}/{task_name}",
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"ScienceWorld task {task_name}, variation {variation_id}."
                ),
            }
        ],
        "ability": "interactive_science",
        "reward_model": {
            "style": "rule",
            "ground_truth": {"final_score": None},
        },
        "extra_info": {
            "index": int(index),
            "split": split,
            "official_split": official_split,
            "task_name": task_name,
            "variation_id": int(variation_id),
            "simplification": "",
        },
    }


def build_records(
    catalog,
    *,
    monitor_per_task: int = 10,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if int(monitor_per_task) <= 0:
        raise ValueError("monitor_per_task must be positive")

    records = {split: [] for split in OUTPUT_FILES}
    task_manifest = {}
    for task_name in sorted(str(name) for name in catalog.get_task_names()):
        catalog.load(task_name)
        split_ids = {
            "train": sorted({int(value) for value in catalog.get_variations_train()}),
            "dev": sorted({int(value) for value in catalog.get_variations_dev()}),
            "test": sorted({int(value) for value in catalog.get_variations_test()}),
        }
        if not all(split_ids.values()):
            raise ValueError(f"ScienceWorld task {task_name!r} has an empty split")
        if not set(split_ids["train"]).isdisjoint(split_ids["dev"]):
            raise ValueError(f"ScienceWorld task {task_name!r} train/dev overlap")
        if not set(split_ids["train"]).isdisjoint(split_ids["test"]):
            raise ValueError(f"ScienceWorld task {task_name!r} train/test overlap")
        if not set(split_ids["dev"]).isdisjoint(split_ids["test"]):
            raise ValueError(f"ScienceWorld task {task_name!r} dev/test overlap")

        for split in ("train", "dev", "test"):
            for variation_id in split_ids[split]:
                records[split].append(
                    _build_row(
                        split=split,
                        index=len(records[split]),
                        task_name=task_name,
                        variation_id=variation_id,
                        official_split=split,
                    )
                )
        for variation_id in split_ids["dev"][: int(monitor_per_task)]:
            records["monitor"].append(
                _build_row(
                    split="monitor",
                    index=len(records["monitor"]),
                    task_name=task_name,
                    variation_id=variation_id,
                    official_split="dev",
                )
            )

        task_manifest[task_name] = {
            "train": split_ids["train"],
            "dev": split_ids["dev"],
            "test": split_ids["test"],
            "monitor": split_ids["dev"][: int(monitor_per_task)],
        }

    manifest = {
        "task_count": len(task_manifest),
        "simplification": "",
        "env_step_limit": 100,
        "reward": "final_official_score_div_100",
        "valid_action_aid": True,
        "monitor_policy": f"all_dev_up_to_{int(monitor_per_task)}_per_task",
        "split_counts": {
            split: len(rows)
            for split, rows in records.items()
        },
        "tasks": task_manifest,
    }
    return records, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_records(
    records: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    file_manifest = {}
    for split, filename in OUTPUT_FILES.items():
        path = output_dir / filename
        pd.DataFrame(records[split]).to_parquet(path, index=False)
        file_manifest[split] = {
            "path": str(path),
            "rows": len(records[split]),
            "sha256": _sha256(path),
        }
    complete_manifest = dict(manifest)
    complete_manifest["files"] = file_manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(complete_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return complete_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--monitor-per-task", type=int, default=10)
    args = parser.parse_args()

    from scienceworld import ScienceWorldEnv

    catalog = ScienceWorldEnv(envStepLimit=100)
    try:
        records, manifest = build_records(
            catalog,
            monitor_per_task=args.monitor_per_task,
        )
    finally:
        catalog.close()
    complete_manifest = write_records(records, manifest, args.output_dir)
    print(json.dumps(complete_manifest["split_counts"], sort_keys=True))
    print(f"Wrote ScienceWorld data manifest to {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
