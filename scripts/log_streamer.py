import sys
import io
import time
import re
import os
import glob
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- SETTINGS ---
INPUT_PATH = "../data/raw_logs/machine_logs_156_cycles.txt"   # Path to source(s)
OUTPUT_FILE = "../data/App.log"    # File for Vector/LogParser to watch
MAX_BYTES = 10 * 1024 * 1024    # 10MB rotation trigger
BACKUP_COUNT = 3               # Keep 3 old files
TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
TS_REGEX = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')


# System Logger: For you to read (Console/Debug.log)
sys_logger = logging.getLogger("System")
logging.StreamHandler(sys.stdout).setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
sys_logger.addHandler(logging.StreamHandler(sys.stdout))
sys_logger.setLevel(logging.INFO)

# Data Logger: For Vector to read (tracked.log ONLY)
data_logger = logging.getLogger("LogStreamer")
data_logger.propagate = False  # <--- CRITICAL: Prevents data leaking to sys_logger
data_handler = RotatingFileHandler(OUTPUT_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8')
data_handler.terminator = '\n' # Force Unix-style ending; Windows handles this fine and it stops the \r doubling.
data_handler.setFormatter(logging.Formatter('%(message)s'))
data_logger.addHandler(data_handler)
data_logger.setLevel(logging.INFO)



def safe_log_line(text):
    """Encodes to UTF-8 and back to ASCII, ignoring characters 
    the Windows console can't handle (like the arrow)."""
    if sys.platform == "win32":
        # This strips out the arrow and replaces it with a '?' 
        # so the script doesn't crash.
        return text.encode('ascii', 'replace').decode('ascii')
    return text

def get_anchor_timestamp(files):
    """Peek at the first line of the first file to set the relative 'Start Time'."""
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:  # Just get the first valid line with a timestamp
                    match = re.search(TS_REGEX, line)
                    if match:
                        return datetime.strptime(match.group(), TIME_FORMAT)
        except Exception as e:
            sys_logger.error(f"Error peeking at {file_path}: {e}")
            continue
    return None


def stream():    
    # Resolve input files
    if "*" in INPUT_PATH or os.path.isdir(INPUT_PATH):
        files = sorted(glob.glob(INPUT_PATH))
    else:
        files = [INPUT_PATH]

    ref_start_dt = get_anchor_timestamp(files)
    if not ref_start_dt:
        sys_logger.error("Error: Could not find any valid timestamps in the reference files.")
        return

    sim_start_dt = datetime.now()
    sys_logger.info(f"log_streamer.py started.")
    sys_logger.info(f"Watching: {INPUT_PATH} | Writing to: {OUTPUT_FILE}")

    try:
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_line = line.rstrip('\r\n') # Remove original line endings immediately
                    if not clean_line:
                        continue


                    match = re.search(TS_REGEX, line)
                    if not match:
                        continue

                    # Calculate timing offset
                    current_line_ts = datetime.strptime(match.group(), TIME_FORMAT)
                    offset = current_line_ts - ref_start_dt
                    target_time = sim_start_dt + offset
                    
                    # Wait for real-time to match log-time
                    sleep_duration = (target_time - datetime.now()).total_seconds()
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

                    # Update timestamp to 'target_time' and emit
                    new_ts_str = target_time.strftime(TIME_FORMAT)[:-3]
                    final_line = line.replace(match.group(), new_ts_str).strip()
                    
                    data_logger.info(final_line)
                    # Simple stdout feedback
                    sys_logger.info(f"[{new_ts_str}] Emitted log line...")

    except KeyboardInterrupt:
        sys_logger.error("\nStreamer stopped by user.")
    finally:
        sys_logger.info("Done.")

if __name__ == "__main__":
    stream()
    