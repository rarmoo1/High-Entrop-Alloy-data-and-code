#!/usr/bin/env python3
"""
Batch 2 compositions — copy batch2_compositions.json to Turing
and run: python3 setup_batch2.py
Then: sbatch --array=5-14 04_submit_array.sh
"""
import json, os
from pathlib import Path

compositions = json.loads(Path("batch2_compositions.json").read_text())

for comp_data in compositions:
    idx  = comp_data["index"]
    comp = comp_data["composition"]
    out_dir = Path(f"alloy_results/alloy_{idx:04d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "results.json"
    if not meta.exists():
        meta.write_text(json.dumps({
            "index":       idx,
            "composition": comp,
            "n_atoms":     32,
            "status":      "pending"
        }, indent=2))
        print(f"Created: {meta}")
    else:
        print(f"Exists:  {meta} — skipping")

print("\nNow run:")
print("  sbatch --array=5-14 04_submit_array.sh")
