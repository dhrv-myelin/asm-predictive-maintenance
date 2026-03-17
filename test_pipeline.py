import subprocess
from pathlib import Path
import tempfile
import shutil

parent_dir = Path("/home/varenyapathak/Downloads/CED")
folders = sorted([f for f in parent_dir.iterdir() if f.is_dir()])
total = len(folders)

# Copy the log_parser folder to temp
temp_parser = Path(tempfile.mkdtemp()) / "log_parser"
shutil.copytree("services/log_parser", temp_parser)

for i, folder in enumerate(folders, 1):
    print(f"[{i}/{total}] Testing: {folder}")

    # Create temp input folder with first 50 lines
    temp_input = Path(tempfile.mkdtemp())
    for file in folder.glob("*"):
        if file.is_file():
            with open(file, "r", errors="ignore") as f_in, open(temp_input / file.name, "w") as f_out:
                for j, line in enumerate(f_in):
                    if j >= 50: 
                        break
                    f_out.write(line)

    try:
        subprocess.run(
            [
                "uv",
                "run",
                "src/main.py",
                "--mode", "record",
                "--once",
                "--input", str(temp_input)
            ],
            check=True,
            cwd=temp_parser,
            timeout=60
        )
        print(f"✅ Success: {folder}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {folder}")
    except subprocess.TimeoutExpired:
        print(f"⏱ Timeout: {folder}")
    finally:
        shutil.rmtree(temp_input)

shutil.rmtree(temp_parser)
print("Test run completed!")