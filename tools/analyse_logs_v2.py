import pandas as pd
import re
from pathlib import Path

def get_signature(line):
    # 1. Remove Timestamps (HH:MM:SS)
    line = re.sub(r'\d{2}:\d{2}:\d{2}(\.\d+)?', '<TIME>', line)
    # 2. Mask Pallet IDs
    line = re.sub(r'P_\w+', '<PALLET_ID>', line)
    # 3. Mask Hex/Hardware tags
    line = re.sub(r'0x[0-9A-Fa-f]+', '<HEX>', line)
    # 4. Mask generic numbers
    line = re.sub(r'\b\d+\b', '<NUM>', line)
    return line.strip()

# --- NEW CONCATENATION LOGIC ---
base_path = Path('/home/vijayalagappanalagappan/Downloads/asm_logs_till_14jan') # Update this path
all_lines = []

# .rglob searches recursively for any file named App.log
for log_file in base_path.rglob('App.log'):
    print(f"Processing: {log_file}")
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines.extend(f.readlines())

if not all_lines:
    print("No App.log files found. Please check your base_path.")
else:
    # Process the aggregated data
    df = pd.DataFrame(all_lines, columns=['raw_log'])
    df['signature'] = df['raw_log'].apply(get_signature)

    # Group by signature and count frequency
    summary = df.groupby('signature').size().reset_index(name='frequency')
    summary = summary.sort_values(by='frequency', ascending=False)

    # Save to Excel (Note: use .xlsx extension if using to_excel, or .csv for to_csv)
    output_file = 'data/log_signature_audit_v1.csv'
    summary.to_csv(output_file, index=False)
    
    print("---")
    print(f"Audit complete.")
    print(f"Total log lines processed: {len(all_lines)}")
    print(f"Unique log structures found: {len(summary)}")