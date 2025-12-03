# split_new_data.py
import argparse, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Data", help="Root data folder")
    ap.add_argument("--classes", nargs="+", default=["pm","approx","neq","equiv"])
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", nargs="+", default=[".png",".jpg",".jpeg",".bmp"])
    ap.add_argument("--mode", choices=["move","copy","link"], default="move",
                    help="How to place files into train/test")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    train_root = root / "train"
    test_root  = root / "test"
    train_root.mkdir(parents=True, exist_ok=True)
    test_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    total_moved = 0
    for cls in args.classes:
        src_dir = root / cls
        if not src_dir.exists():
            print(f"[skip] {src_dir} (not found)")
            continue

        files = [p for p in src_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in set(args.exts)]
        if not files:
            print(f"[info] {src_dir}: no eligible files")
            continue

        random.shuffle(files)
        n = len(files)
        n_test = max(1, int(round(args.test_ratio * n)))
        test_files = set(files[:n_test])
        train_files = files[n_test:]

        # Ensure class subfolders exist
        tr_dir = train_root / cls; tr_dir.mkdir(parents=True, exist_ok=True)
        te_dir = test_root  / cls; te_dir.mkdir(parents=True, exist_ok=True)

        def place(p: Path, dst_dir: Path):
            dst = dst_dir / p.name
            # If a same-named file already exists, skip to avoid clobber
            if dst.exists():
                print(f"[skip exists] {dst}")
                return 0
            if args.dry_run:
                print(f"[dry] {args.mode} {p} -> {dst}")
                return 1
            if args.mode == "move":
                shutil.move(str(p), str(dst))
            elif args.mode == "copy":
                shutil.copy2(str(p), str(dst))
            else:  # link
                dst.symlink_to(p.resolve())
            return 1

        moved_c = 0
        for p in test_files:
            moved_c += place(p, te_dir)
        for p in train_files:
            moved_c += place(p, tr_dir)

        total_moved += moved_c
        print(f"[done] class '{cls}': {len(test_files)} -> test, {len(train_files)} -> train (placed {moved_c}/{n})")

        # Optional: delete now-empty source dir
        if not args.dry_run and src_dir.exists() and not any(src_dir.iterdir()):
            try:
                src_dir.rmdir()
                print(f"[clean] removed empty {src_dir}")
            except OSError:
                pass

    print(f"[summary] placed {total_moved} files total.")

if __name__ == "__main__":
    main()