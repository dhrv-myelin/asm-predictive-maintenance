#!/bin/bash

# Export current working directory as PYTHONPATH
export PYTHONPATH="$(pwd)"

PARENT_DIR="/home/dhruvkumarjiguda/code/asm-predictive-maintenance/data/Weld_logs_utf8/"
WORKDIR="services/log_parser"

# Parameters
MAX_JOBS=${1:-4} # Number of parallel jobs (default: 4)
LIMIT=${2:-0}    # Number of files to process (0 = all)

echo "Max parallel jobs: $MAX_JOBS"
echo "Limit: $LIMIT"

# Get all .txt files recursively, sorted descending
mapfile -t files < <(find "$PARENT_DIR" -type f -name "*.txt" | sort -r)

TOTAL=${#files[@]}

# Apply LIMIT if provided
if [ "$LIMIT" -gt 0 ]; then
  files=("${files[@]:0:$LIMIT}")
fi

COUNT=${#files[@]}
echo "Processing $COUNT files (out of $TOTAL)"

# Export variables for subshell
export WORKDIR

# Function to process a file
process_file() {
  file="$1"
  echo "Processing: $file"

  (cd "$WORKDIR" && uv run src/main.py --mode record --input "$file" --once)
}

export -f process_file

# Run in parallel
printf "%s\n" "${files[@]}" | xargs -I {} -P "$MAX_JOBS" bash -c 'process_file "$@"' _ {}

echo "Done!"
