import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def compute_stats(files):
    returns = []
    successes = []

    for f in files:
        data = np.load(f)
        returns.append(float(data["rewards"].sum()))
        successes.append(float(data["success"]))

    if len(files) == 0:
        return {
            "num": 0,
            "success_rate": np.nan,
            "avg_return": np.nan,
            "min_return": np.nan,
            "max_return": np.nan,
        }

    return {
        "num": len(files),
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
    }


def copy_files(files, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    for old in out_dir.glob("*.npz"):
        old.unlink()

    for f in files:
        shutil.copy2(f, out_dir / f.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="dataset")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.npz"))

    if len(files) == 0:
        raise RuntimeError(f"No .npz files found in {src_dir}")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    n = len(files)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    test_files = [files[i] for i in test_idx]

    copy_files(train_files, out_root / "train")
    copy_files(val_files, out_root / "val")
    copy_files(test_files, out_root / "test")

    rows = []

    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        stats = compute_stats(split_files)
        stats["dataset"] = args.prefix
        stats["split"] = split_name
        rows.append(stats)

    df = pd.DataFrame(rows)
    csv_path = out_root / f"{args.prefix}_split_stats.csv"
    df.to_csv(csv_path, index=False)

    md_path = out_root / f"{args.prefix}_split_stats.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    print("=" * 80)
    print("Split finished")
    print("source:", src_dir)
    print("out_root:", out_root)
    print("num files:", n)
    print(df)
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
