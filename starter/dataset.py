"""DeepWeeds fold-0 loading, validation, transforms, and DataLoaders."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import pandas as pd

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read the authors' untouched fold-0 CSVs. Other folds are bonus-only."""
    if fold != 0:
        raise ValueError("Bài lab chính dùng fold 0; fold khác chỉ dùng cho phần thưởng.")
    labels_dir = Path(labels_dir)
    paths = [labels_dir / f"{split}_subset{fold}.csv" for split in ("train", "val", "test")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu CSV fold 0: " + ", ".join(missing))
    frames = [pd.read_csv(path) for path in paths]
    for path, frame in zip(paths, frames):
        required = {"Filename", "Label", "Species"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} thiếu cột {sorted(required - set(frame.columns))}")
        if frame["Filename"].isna().any() or frame["Label"].isna().any():
            raise ValueError(f"{path} có Filename/Label rỗng")
        frame["Filename"] = frame["Filename"].astype(str)
        frame["Label"] = frame["Label"].astype(int)
    return tuple(frames)


def _image_lookup(images_dir: Path) -> dict[str, Path]:
    files = [p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    by_name: dict[str, Path] = {}
    duplicates = set()
    for path in files:
        if path.name in by_name:
            duplicates.add(path.name)
        else:
            by_name[path.name] = path
    if duplicates:
        sample = sorted(duplicates)[:5]
        raise ValueError(f"Tên ảnh trùng nhau trong thư mục dữ liệu: {sample}")
    return by_name


def discover_images_dir(data_dir: str | Path = "data") -> Path:
    """Find the directory holding the extracted images, including nested archives."""
    root = Path(data_dir)
    candidates = [root / "images", root]
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
                                       for p in candidate.rglob("*")):
            return candidate
    raise FileNotFoundError(f"Không tìm thấy ảnh JPG dưới {root}; hãy giải nén images.zip.")


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Check fold-0 integrity, class counts, overlaps, coverage, and image files."""
    frames = {"train": train_df, "val": val_df, "test": test_df}
    sets: dict[str, set[str]] = {}
    for name, frame in frames.items():
        if not {"Filename", "Label"}.issubset(frame.columns):
            raise ValueError(f"Tập {name} thiếu cột Filename hoặc Label")
        names = frame["Filename"].astype(str)
        if names.duplicated().any():
            examples = names[names.duplicated()].head().tolist()
            raise ValueError(f"Tập {name} có Filename trùng: {examples}")
        labels = pd.to_numeric(frame["Label"], errors="coerce")
        if labels.isna().any() or not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"Tập {name} có Label ngoài khoảng 0..8 hoặc không hợp lệ")
        sets[name] = set(names)

    overlap = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    bad_overlap = {key: value[:5] for key, value in overlap.items() if value}
    if bad_overlap:
        raise ValueError(f"Fold 0 bị giao nhau theo Filename: {bad_overlap}")
    union = set.union(*sets.values())
    if len(union) != 17_509:
        raise ValueError(f"Fold 0 phải có đúng 17.509 ảnh duy nhất; tìm thấy {len(union)}")

    lookup = _image_lookup(Path(images_dir))
    missing = sorted(union - lookup.keys())
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh dưới {images_dir}, ví dụ: {missing[:5]}")
    counts = {name: int(len(frame)) for name, frame in frames.items()}
    per_class = {
        name: {CLASS_NAMES[i]: int((frame["Label"].astype(int) == i).sum())
               for i in range(NUM_CLASSES)}
        for name, frame in frames.items()
    }
    result = {
        "n": counts,
        "per_class": per_class,
        "overlap": {key: len(value) for key, value in overlap.items()},
        "union": len(union),
        "images_dir": str(Path(images_dir).resolve()),
    }
    print(f"Fold 0: {counts}; hợp={len(union):,}; giao={result['overlap']}; ảnh tồn tại đủ.")
    for split, values in per_class.items():
        print(f"{split}: {values}")
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    from torchvision import transforms as T

    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        resize_size = img_size + 32 if img_size == 224 else img_size
        return T.Compose([T.Resize(resize_size), T.CenterCrop(img_size), T.ToTensor(), normalize])
    if aug not in {"basic", "color", "trivial", "randaug", "none"}:
        raise ValueError(f"Augmentation không hỗ trợ: {aug}")
    ops = [T.RandomResizedCrop(img_size, scale=(0.75, 1.0)), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=7))
    ops.extend([T.ToTensor(), normalize])
    if aug == "none":
        ops[0] = T.Resize((img_size, img_size))
        ops[1] = T.Lambda(lambda image: image)
    return T.Compose(ops)


class DeepWeedsDataset:
    """Lightweight map-style dataset; PyTorch's DataLoader accepts this protocol."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self._lookup = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        from PIL import Image
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = self.images_dir / filename
        if not path.is_file():
            if self._lookup is None:
                self._lookup = _image_lookup(self.images_dir)
            path = self._lookup.get(filename, path)
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform else image
        return tensor, int(row["Label"]), filename


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    dataset = DeepWeedsDataset(df, images_dir, transform)
    if sampler not in (None, "balanced"):
        raise ValueError("sampler phải là None hoặc 'balanced'")
    weights = None
    shuffle = bool(train and sampler is None)
    if train and sampler == "balanced":
        labels = df["Label"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        sample_weights = 1.0 / np.maximum(counts[labels], 1)
        weights = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double),
                                        num_samples=len(labels), replacement=True)

    def seed_worker(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
        random.seed(seed)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=weights,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=bool(train and len(dataset) >= batch_size),
                      worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
