#!/usr/bin/env python3
"""
Download specific files from the Kaggle dataset `reubensuju/celeb-df-v2` into Test-Video/Real or Test-Video/Fake.

Prerequisites:
- Install the Kaggle CLI / Python package: `pip install kaggle`
- Place your Kaggle API token at `~/.kaggle/kaggle.json` (or set `KAGGLE_CONFIG_DIR`).

Usage examples:
# First list available files:
# kaggle datasets files -d reubensuju/celeb-df-v2

# Download specific files and mark them as fake:
# python3 scripts/download_kaggle_sample.py -l fake -f "Fake/video1.mp4" "Fake/video2.mp4"

# Download a single file and mark as real:
# python3 scripts/download_kaggle_sample.py -l real -f "Real/videoA.mp4"
"""

import argparse
import os
import tempfile
import shutil
import zipfile
import random
from kaggle.api.kaggle_api_extended import KaggleApi


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_file(api, dataset, file_name, dest_dir, force=False):
    tmp_dir = tempfile.mkdtemp()
    try:
        api.dataset_download_file(dataset, file_name, path=tmp_dir, force=force, quiet=True)
    except Exception as e:
        shutil.rmtree(tmp_dir)
        raise

    # Move or extract downloaded contents
    for entry in os.listdir(tmp_dir):
        full = os.path.join(tmp_dir, entry)
        if zipfile.is_zipfile(full):
            with zipfile.ZipFile(full) as z:
                z.extractall(dest_dir)
        else:
            # Move files into destination directory
            target_path = os.path.join(dest_dir, entry)
            shutil.move(full, target_path)

    shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(description="Download selected files from Kaggle dataset into Test-Video folders.")
    parser.add_argument("-f", "--files", nargs="+", help="One or more filenames (as shown by `kaggle datasets files`) to download.")
    parser.add_argument("-l", "--label", choices=["real", "fake"], help="Label to place files under (real|fake). If omitted, label is inferred per-file.")
    parser.add_argument("--auto", type=int, help="Automatically select N files from the dataset and download them.")
    parser.add_argument("--list-file", help="Path to a local list file containing dataset-relative paths or basenames, or the dataset file name to download and read.")
    parser.add_argument("--count", type=int, default=30, help="Number of files to download when using --auto or a list (default: 30)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle selection when using auto or list mode.")
    parser.add_argument("--force", action="store_true", help="Force re-download if file exists.")
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    dataset = "reubensuju/celeb-df-v2"
    repo_root = os.path.abspath(os.getcwd())
    target_base = os.path.join(repo_root, "Test-Video")
    ensure_dir(target_base)

    # Helper: list all files in the dataset
    def list_dataset_files():
        info = api.dataset_list_files(dataset)
        return [f.name for f in info.files]

    # Resolve a local or dataset list file into dataset-relative paths
    def resolve_list_file(list_file_name):
        # If local file exists, read it
        if os.path.isabs(list_file_name) or os.path.exists(list_file_name):
            with open(list_file_name, "r", encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            return lines

        # Otherwise, try to download that file from the dataset and read it
        tmpdir = tempfile.mkdtemp()
        try:
            api.dataset_download_file(dataset, list_file_name, path=tmpdir, force=args.force, quiet=True)
            # find any extracted files
            entries = []
            for entry in os.listdir(tmpdir):
                full = os.path.join(tmpdir, entry)
                if zipfile.is_zipfile(full):
                    with zipfile.ZipFile(full) as z:
                        z.extractall(tmpdir)
                elif os.path.isfile(full):
                    entries.append(full)

            # pick the first txt-like file
            txts = [p for p in entries if p.lower().endswith((".txt",))]
            if txts:
                with open(txts[0], "r", encoding="utf-8") as fh:
                    lines = [l.strip() for l in fh if l.strip()]
                return lines
        finally:
            shutil.rmtree(tmpdir)

        return []

    # If auto mode or list provided, build files_to_download
    files_to_download = []
    all_dataset_files = None
    if args.list_file:
        lines = resolve_list_file(args.list_file)
        if not lines:
            print(f"No entries found in list file {args.list_file}")
            return
        # lazy load dataset file list for matching
        all_dataset_files = list_dataset_files()
        resolved = []
        for entry in lines:
            # If entry already looks like a dataset-relative path and exists in dataset files, use it
            if entry in all_dataset_files:
                resolved.append(entry)
                continue
            # Otherwise try to match by basename
            matches = [f for f in all_dataset_files if os.path.basename(f) == os.path.basename(entry) or f.endswith(entry)]
            if matches:
                resolved.append(matches[0])
            else:
                print(f"Warning: couldn't find a dataset file matching '{entry}'")

        files_to_download = resolved
    elif args.auto:
        all_dataset_files = list_dataset_files()
        # filter to likely video/archive files
        candidates = [f for f in all_dataset_files if f.lower().endswith((".mp4", ".mov", ".avi", ".zip", ".tar", ".tar.gz"))]
        if not candidates:
            print("No video files found in dataset via API.")
            return

        # Try to split candidates into real/fake by folder name
        real_candidates = [f for f in candidates if f.lower().startswith("celeb-real") or f.lower().startswith("youtube-real") or "real" in f.lower()]
        fake_candidates = [f for f in candidates if f.lower().startswith("celeb-synthesis") or "synth" in f.lower() or "fake" in f.lower()]

        n = args.count if args.count else args.auto
        chosen = []
        # balanced selection
        want_each = n // 2
        if real_candidates:
            chosen += random.sample(real_candidates, min(want_each, len(real_candidates)))
        if fake_candidates:
            chosen += random.sample(fake_candidates, min(want_each, len(fake_candidates)))

        remaining = n - len(chosen)
        remaining_pool = [f for f in candidates if f not in chosen]
        if args.shuffle:
            random.shuffle(remaining_pool)
        if remaining > 0 and remaining_pool:
            chosen += remaining_pool[:remaining]

        files_to_download = chosen
        print(f"Auto-selected {len(files_to_download)} files (requested {n}).")
    elif args.files:
        files_to_download = args.files
    else:
        print("No files specified. Use -f/--files or --auto N to select files.")
        return

    # Limit to requested count if set
    if args.count and len(files_to_download) > args.count:
        if args.shuffle:
            random.shuffle(files_to_download)
        files_to_download = files_to_download[: args.count]

    for fname in files_to_download:
        # determine per-file label if not forced
        if args.label:
            per_label = args.label
        else:
            low = fname.lower()
            if "real" in low or fname.split("/")[0].lower() == "real":
                per_label = "real"
            else:
                per_label = "fake"

        per_label_dir = "Real" if per_label == "real" else "Fake"
        per_target_dir = os.path.join(target_base, per_label_dir)
        ensure_dir(per_target_dir)

        print(f"Downloading '{fname}' into {per_label_dir}...")
        try:
            download_file(api, dataset, fname, per_target_dir, force=args.force)
        except Exception as e:
            print(f"Failed to download {fname}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
