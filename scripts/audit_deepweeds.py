"""Audit the fixed DeepWeeds fold-0 split and save reproducible EDA outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def audit(images_dir: Path, labels_dir: Path, out_dir: Path) -> dict:
    labels = pd.read_csv(labels_dir / "labels.csv")
    if not {"Filename", "Label", "Species"}.issubset(labels.columns):
        raise ValueError("labels.csv must contain Filename, Label, Species")
    if labels["Filename"].duplicated().any():
        raise ValueError("labels.csv contains duplicate filenames")

    class_names = (labels[["Label", "Species"]].drop_duplicates()
                   .sort_values("Label").set_index("Label")["Species"])
    splits = {
        name: pd.read_csv(labels_dir / f"{name}_subset0.csv")
        for name in ("train", "val", "test")
    }
    sets = {}
    counts = {}
    for name, frame in splits.items():
        if not {"Filename", "Label"}.issubset(frame.columns):
            raise ValueError(f"{name} split must contain Filename and Label")
        if frame["Filename"].duplicated().any():
            raise ValueError(f"{name} split contains duplicate filenames")
        sets[name] = set(frame["Filename"])
        counts[name] = frame["Label"].value_counts().reindex(class_names.index, fill_value=0)

    overlaps = {
        f"{left}_{right}": len(sets[left] & sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    union = set.union(*sets.values())
    missing_images = sorted(name for name in union if not (images_dir / name).is_file())
    split_counts_match_labels = union == set(labels["Filename"])
    if any(overlaps.values()) or missing_images or not split_counts_match_labels:
        raise ValueError(
            f"Invalid fold 0: overlaps={overlaps}, missing_images={len(missing_images)}, "
            f"split_union_matches_labels={split_counts_match_labels}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(counts, index=class_names.index)
    table.index = class_names.loc[table.index]
    ax = table.plot.bar(figsize=(12, 6), width=0.82)
    ax.set(title="DeepWeeds fold 0 class distribution", xlabel="Class", ylabel="Image count")
    ax.legend(title="Split")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=180)
    plt.close()

    report = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "total_images": len(labels),
        "split_sizes": {key: len(value) for key, value in splits.items()},
        "class_counts": {
            key: {str(class_names.loc[index]): int(value) for index, value in series.items()}
            for key, series in counts.items()
        },
        "overlap_counts": overlaps,
        "missing_images": missing_images,
        "split_union_matches_labels": split_counts_match_labels,
    }
    (out_dir / "split_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("images"))
    parser.add_argument("--labels", type=Path, default=Path("data/labels"))
    parser.add_argument("--out", type=Path, default=Path("eda"))
    args = parser.parse_args()
    print(json.dumps(audit(args.images, args.labels, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
