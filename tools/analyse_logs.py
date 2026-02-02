import pandas as pd
import re

def get_signature(line):
    # 1. Remove Timestamps (HH:MM:SS)
    line = re.sub(r'\d{2}:\d{2}:\d{2}(\.\d+)?', '<TIME>', line)
    # 2. Mask Pallet IDs (assuming P_ followed by digits/chars)
    line = re.sub(r'P_\w+', '<PALLET_ID>', line)
    # 3. Mask Hex/Hardware tags (0x...)
    line = re.sub(r'0x[0-9A-Fa-f]+', '<HEX>', line)
    # 4. Mask generic numbers (counts, retry attempts)
    line = re.sub(r'\b\d+\b', '<NUM>', line)
    return line.strip()

# Load logs
with open('/home/vijayalagappanalagappan/Downloads/tarun_prepped.log', 'r') as f:
    lines = f.readlines()

df = pd.DataFrame(lines, columns=['raw_log'])
df['signature'] = df['raw_log'].apply(get_signature)

# Group by signature and count frequency
summary = df.groupby('signature').size().reset_index(name='frequency')
summary = summary.sort_values(by='frequency', ascending=False)

# Save to Excel for easy manual mapping
summary.to_csv('data/log_signature_audit.xlsx', index=False)
print(f"Audit complete. Found {len(summary)} unique log structures.")