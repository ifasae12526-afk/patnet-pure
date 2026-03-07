# build_splits.py
import os
import random
from pathlib import Path
import argparse

IMG_EXTS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
MSK_EXTS = [".tif", ".tiff", ".png"]

def find_existing_file(stem: str, folder: Path, exts):
    """Return first existing file for stem+ext in folder, else None."""
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def collect_pairs(images_dir: Path, labels_dir: Path, mask_mode: str):
    """
    mask_mode:
      - "suffix": mask stem = f"{id}_mask"
      - "same":   mask stem = f"{id}"
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    # ambil semua image yang punya ekstensi di IMG_EXTS
    image_files = []
    for ext in IMG_EXTS:
        image_files.extend(images_dir.glob(f"*{ext}"))

    pairs = []
    missing_labels = []
    for img_path in sorted(image_files):
        img_id = img_path.stem  # contoh: Lombok406
        if mask_mode == "suffix":
            mask_stem = f"{img_id}_mask"
        elif mask_mode == "same":
            mask_stem = img_id
        else:
            raise ValueError("mask_mode must be 'suffix' or 'same'")

        mask_path = find_existing_file(mask_stem, labels_dir, MSK_EXTS)
        if mask_path is None:
            missing_labels.append(img_path.name)
            continue

        pairs.append((img_id, img_path, mask_path))

    return pairs, missing_labels

def write_split(split_path: Path, ids):
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        for _id in ids:
            f.write(_id + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root dataset path containing images/ and labels/")
    ap.add_argument("--images", type=str, default="img")
    ap.add_argument("--labels", type=str, default="label")
    ap.add_argument("--out", type=str, default="splits",
                    help="Output folder name inside root (default: splits)")
    ap.add_argument("--mask_mode", type=str, default="suffix",
                    choices=["suffix", "same"],
                    help="suffix: Lombok406 -> Lombok406_mask.* ; same: Lombok406 -> Lombok406.*")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_items", type=int, default=3,
                    help="Minimal jumlah pasangan valid untuk bikin split (safety).")
    args = ap.parse_args()

    # validasi rasio
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+val+test must sum to 1.0, got {total}")

    root = Path(args.root)
    images_dir = root / args.images
    labels_dir = root / args.labels
    out_dir = root / args.out

    pairs, missing_labels = collect_pairs(images_dir, labels_dir, args.mask_mode)
    if len(pairs) < args.min_items:
        raise RuntimeError(
            f"Valid pairs found: {len(pairs)} (<{args.min_items}). "
            f"Check folder structure / mask naming."
        )

    ids = [p[0] for p in pairs]
    random.seed(args.seed)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    # sisanya masuk test
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    write_split(out_dir / "trn.txt", train_ids)
    write_split(out_dir / "val.txt", val_ids)
    write_split(out_dir / "test.txt", test_ids)

    print("=== Split summary ===")
    print(f"Total valid pairs : {n}")
    print(f"Train            : {len(train_ids)}")
    print(f"Val              : {len(val_ids)}")
    print(f"Test             : {len(test_ids)}")
    print(f"Output folder    : {out_dir}")

    if missing_labels:
        print("\n[Warning] Images without matching labels (ignored):")
        for name in missing_labels[:30]:
            print(" -", name)
        if len(missing_labels) > 30:
            print(f" ... and {len(missing_labels)-30} more")

if __name__ == "__main__":
    main()
