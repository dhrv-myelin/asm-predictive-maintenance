import subprocess

from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

import threading

import time
 
parent_dir = Path("/home/saibhavana/Desktop/ASM/CED")

folders = sorted([f for f in parent_dir.iterdir() if f.is_dir()])

total = len(folders)

print_lock = threading.Lock()

stats = []
 
def process_folder(i, folder):

    start = time.time()

    with print_lock:

        print(f"▶ [{i}/{total}] Starting: {folder.name}", flush=True)
 
    result = subprocess.run(

        ["uv", "run", "src/main.py", "--mode", "record", "--once", "--input", str(folder)],

        check=True,

        cwd="services/log_parser",

        capture_output=True,

        text=True,

    )
 
    elapsed = time.time() - start

    with print_lock:

        print(f"✓ [{i}/{total}] Done: {folder.name} ({elapsed:.1f}s)", flush=True)
 
    return i, folder.name, elapsed, "success"
 
start_all = time.time()

print(f"🚀 Launching {total} folders with 12 workers...\n", flush=True)
 
with ThreadPoolExecutor(max_workers=12) as executor:

    futures = {executor.submit(process_folder, i, f): f for i, f in enumerate(folders, 1)}
 
    done, failed = 0, 0

    for future in as_completed(futures):

        try:

            i, name, elapsed, status = future.result()

            done += 1

            stats.append({"folder": name, "elapsed": elapsed, "status": "✓ success"})

            eta = ((time.time() - start_all) / done) * (total - done)

            with print_lock:

                print(f"  Progress: {done}/{total} done | failed: {failed} | ETA: {eta/60:.1f} mins remaining", flush=True)

        except subprocess.CalledProcessError as e:

            failed += 1

            folder = futures[future]

            stats.append({"folder": folder.name, "elapsed": 0, "status": f"✗ failed"})

            with print_lock:

                print(f"✗ FAILED: {folder.name}\n{e.stderr}", flush=True)
 
total_time = time.time() - start_all
 
# Summary

print(f"\n{'='*60}")

print(f"{'FOLDER':<15} {'STATUS':<12} {'TIME':>10}")

print(f"{'-'*60}")

for s in sorted(stats, key=lambda x: x["folder"]):

    mins = s["elapsed"] / 60

    time_str = f"{mins:.1f} mins" if s["elapsed"] > 0 else "—"

    print(f"{s['folder']:<15} {s['status']:<12} {time_str:>10}")
 
print(f"{'-'*60}")

print(f"\nProcessed : {done}/{total} folders")

print(f"Failed    : {failed}")

print(f"Total time: {total_time/60:.1f} mins")

print(f"Avg/folder: {total_time/total:.1f}s")

if stats:

    successful = [s["elapsed"] for s in stats if s["elapsed"] > 0]

    if successful:

        print(f"Fastest   : {min(successful)/60:.1f} mins")

        print(f"Slowest   : {max(successful)/60:.1f} mins")

print(f"{'='*60}")

 