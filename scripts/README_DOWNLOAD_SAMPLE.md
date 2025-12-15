Download selected files from Kaggle dataset `reubensuju/celeb-df-v2`

Prerequisites
- Install Kaggle Python package: `pip install kaggle`
- Get your Kaggle API token: go to Kaggle > Account > Create API Token and save the resulting `kaggle.json` to `~/.kaggle/kaggle.json` (chmod 600 recommended).

List files in the dataset
```bash
kaggle datasets files -d reubensuju/celeb-df-v2
```

Download specific files and place them in `Test-Video` organized as `Real` or `Fake`:
```bash
# Example: download two fake videos by dataset-relative path
python3 scripts/download_kaggle_sample.py -l fake -f "Celeb-synthesis/video1.mp4" "Celeb-synthesis/video2.mp4"

# Example: download a single real video by basename (script will try to resolve)
python3 scripts/download_kaggle_sample.py -f "videoA.mp4" -l real
```

Download using a local list file (one entry per line). Entries may be dataset-relative paths or basenames. The script will try to resolve basenames to dataset files.
```bash
# Download up to 30 files listed in local_list.txt
python3 scripts/download_kaggle_sample.py --list-file local_list.txt --count 30 --shuffle
```

Download automatically-selected files (balanced real/fake) from the dataset:
```bash
# Auto-select ~30 videos and download them
python3 scripts/download_kaggle_sample.py --auto 30 --count 30 --shuffle
```

Notes
- Filenames must match the paths printed by `kaggle datasets files` for this dataset.
- The script downloads each file into a temporary directory, extracts if zipped, and moves content into `Test-Video/Real` or `Test-Video/Fake`.
- If you want me to run the download for you, provide the list of filenames you want and confirm you have placed `kaggle.json` in `~/.kaggle/` (or grant permission to run with your token).