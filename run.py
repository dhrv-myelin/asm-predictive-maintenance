import subprocess
from pathlib import Path

parent_dir = Path("/home/varenyapathak/Downloads/CED")
folders = sorted([f for f in parent_dir.iterdir() if f.is_dir()])
total = len(folders)

for i, folder in enumerate(folders, 1):
    print(f"[{i}/{total}] Processing: {folder}")

    subprocess.run(
        [
            "uv",
            "run",
            "src/main.py",
            "--mode",
            "record",
            "--input",
            str(folder),
        ],
        check=True,
        cwd="services/log_parser",   # ⭐ correct working directory
    )

print(f"Done! Processed {total} folders.")