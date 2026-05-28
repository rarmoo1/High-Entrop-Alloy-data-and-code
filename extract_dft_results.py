#!/usr/bin/env python3
"""
extract_dft_results.py
─────────────────────────────────────────────────────────────────────────────
Reads all converged Quantum ESPRESSO SCF outputs in alloy_results/ and
computes ΔHf for each composition using the correct integer SQS atom counts
from the SLURM log files.

Usage (run from ~/HEA_discovery_SCF on Turing):
    python3 extract_dft_results.py

Output:
    - Console table of all results
    - dft_results_all.json  (one record per converged alloy)
    - dft_results_all.csv   (same, for spreadsheet import)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import csv
from pathlib import Path

# ── Reference energies (nspin=1, same PAW potentials as alloy calcs) ─────────
REF = {
    "Ni": -429.15353392,
    "Fe": -329.21297094,
    "Cr": -248.22183841,
    "Co": -298.47032558,
    "Al": -39.50329180,
    "Mn": -211.18325206,
}

# Conversion: Ry → kJ/mol
EV_TO_KJ = 1312.75

# ── Miedema binary interaction parameters (kJ/mol) ───────────────────────────
MIEDEMA = {
    ("Ni", "Fe"): -2,  ("Ni", "Cr"): -7,  ("Ni", "Co"):  0,
    ("Ni", "Al"): -22, ("Ni", "Mn"): -8,  ("Fe", "Cr"): -1,
    ("Fe", "Co"): -1,  ("Fe", "Al"): -11, ("Fe", "Mn"):  0,
    ("Cr", "Co"): -4,  ("Cr", "Al"): -10, ("Cr", "Mn"):  0,
    ("Co", "Al"): -19, ("Co", "Mn"): -5,  ("Al", "Mn"): -19,
}


def miedema_prediction(comp):
    """ΔH_mix from Miedema model (kJ/mol). comp = {el: fraction}."""
    els = [el for el, f in comp.items() if f > 0]
    H = 0.0
    for i, e1 in enumerate(els):
        for e2 in els[i + 1:]:
            key = (e1, e2) if (e1, e2) in MIEDEMA else (e2, e1)
            H += 4 * comp[e1] * comp[e2] * MIEDEMA.get(key, 0)
    return round(H, 3)


def integer_occupancy(comp, n=32):
    """
    Convert fractional composition to integer atom counts summing to n.
    Uses largest-remainder method to avoid rounding errors.
    """
    raw = {el: f * n for el, f in comp.items() if f > 0}
    floored = {el: int(v) for el, v in raw.items()}
    remainder = n - sum(floored.values())
    # Give extra atoms to elements with largest fractional parts
    order = sorted(raw, key=lambda e: raw[e] - floored[e], reverse=True)
    for i in range(remainder):
        floored[order[i]] += 1
    return floored


def read_slurm_occupancy(logs_dir, idx):
    """
    Read actual SQS integer atom counts from SLURM stdout log.
    Returns dict like {'Ni': 16, 'Fe': 10, 'Cr': 6} or None if not found.
    """
    for log in Path(logs_dir).glob(f"HEA_array.*_{idx}.out"):
        m = re.search(
            r"Site occupancies:\s+Counter\(\{(.*?)\}\)", log.read_text()
        )
        if m:
            occ = {}
            for p in re.finditer(r"'(\w+)':\s*(\d+)", m.group(1)):
                occ[p.group(1)] = int(p.group(2))
            return occ
    return None


def parse_scf_output(out_path):
    """
    Extract total energy and best SCF accuracy from QE output.
    Returns (energy_Ry, best_accuracy_Ry) or (None, None).
    """
    text = Path(out_path).read_text()
    # The "!" prefix marks the final converged energy
    converged = re.findall(r"!\s+total energy\s+=\s+([-\d.]+)\s+Ry", text)
    if not converged:
        return None, None
    energy = float(converged[-1])
    accuracies = re.findall(r"estimated scf accuracy\s+<\s+([\d.E+\-]+)", text)
    best_acc = min(float(a) for a in accuracies) if accuracies else None
    pressure = None
    m = re.search(r"P=\s+([\d.]+)", text)
    if m:
        pressure = float(m.group(1))
    return energy, best_acc, pressure


# ── Main extraction loop ──────────────────────────────────────────────────────
results = []
base = Path("alloy_results")
logs = base / "logs"

print(f"\n{'='*100}")
print(f"  DFT Results Extraction  —  NiCoCrFeMnAl HEA System")
print(f"{'='*100}")
print(f"\n  {'Idx':>4}  {'Composition':<28}  "
      f"{'SQS atoms':<30}  "
      f"{'ΔHf DFT':>10}  {'±':>6}  "
      f"{'Miedema':>8}  {'Ratio':>7}  "
      f"{'P(kbar)':>8}  Status")
print(f"  {'-'*97}")

for alloy_dir in sorted(base.glob("alloy_*")):
    meta_path = alloy_dir / "results.json"
    out_path  = alloy_dir / "step1_scf.out"
    if not meta_path.exists() or not out_path.exists():
        continue

    data = json.loads(meta_path.read_text())
    idx  = data["index"]
    comp = data["composition"]
    n    = data.get("n_atoms", 32)

    # ── Get integer atom counts ───────────────────────────────────────────────
    # Priority: SLURM log (actual SQS) > integer_occupancy (fallback)
    occ = read_slurm_occupancy(logs, idx)
    occ_source = "SQS log"
    if occ is None:
        occ = integer_occupancy({k: v for k, v in comp.items() if v > 0}, n)
        occ_source = "rounded"

    n_atoms = sum(occ.values())

    # ── Parse QE output ───────────────────────────────────────────────────────
    parsed = parse_scf_output(out_path)
    e_total, best_acc, pressure = parsed

    if e_total is None:
        label = " ".join(f"{el}{int(v*100)}" for el, v in comp.items() if v > 0)
        print(f"  {idx:>4}  {label:<28}  {'(not converged)':<30}  "
              f"{'—':>10}  {'—':>6}  {'—':>8}  {'—':>7}  {'—':>8}  NOT CONVERGED")
        continue

    # ── ΔHf from DFT ─────────────────────────────────────────────────────────
    e_ref = sum(n_atoms_el * REF[el] for el, n_atoms_el in occ.items())
    delta_e = e_total - e_ref
    dHf = delta_e * EV_TO_KJ / n_atoms
    unc = (best_acc * EV_TO_KJ / n_atoms) if best_acc else None

    # ── Miedema prediction ────────────────────────────────────────────────────
    H_mied = miedema_prediction(comp)
    ratio  = f"{dHf/H_mied:.2f}×" if H_mied != 0 else "—"

    # ── Sanity check ─────────────────────────────────────────────────────────
    physical = "✓" if -150 < dHf < 10 else "⚠ CHECK"

    # ── Format for display ────────────────────────────────────────────────────
    comp_label = " ".join(f"{el}{int(v*100)}" for el, v in comp.items() if v > 0)
    occ_label  = " ".join(f"{n_atoms_el}{el}" for el, n_atoms_el in occ.items())
    unc_str    = f"±{unc:.2f}" if unc else "?"
    p_str      = f"{pressure:.1f}" if pressure else "?"

    print(f"  {idx:>4}  {comp_label:<28}  {occ_label:<30}  "
          f"{dHf:>+10.2f}  {unc_str:>6}  "
          f"{H_mied:>+8.2f}  {ratio:>7}  "
          f"{p_str:>8}  {physical}")

    # ── Build result record ───────────────────────────────────────────────────
    record = {
        "index":               idx,
        "composition":         {k: v for k, v in comp.items() if v > 0},
        "sqs_occupancy":       occ,
        "occ_source":          occ_source,
        "n_atoms":             n_atoms,
        "e_total_Ry":          e_total,
        "e_ref_Ry":            round(e_ref, 6),
        "delta_e_Ry":          round(delta_e, 6),
        "delta_hf_kJ_mol":     round(dHf, 3),
        "uncertainty_kJ_mol":  round(unc, 3) if unc else None,
        "scf_accuracy_Ry":     best_acc,
        "miedema_kJ_mol":      H_mied,
        "ratio_dft_miedema":   round(dHf / H_mied, 2) if H_mied != 0 else None,
        "pressure_kbar":       pressure,
        "n_elements":          len([v for v in comp.values() if v > 0]),
    }
    results.append(record)

    # Update results.json in place
    data.update({
        "status":              "scf_converged",
        "total_energy_Ry":     e_total,
        "sqs_occupancy":       occ,
        "n_atoms_actual":      n_atoms,
        "e_ref_Ry":            round(e_ref, 6),
        "delta_hf_kJ_mol":     round(dHf, 3),
        "delta_hf_miedema_kJ": H_mied,
        "scf_accuracy_Ry":     best_acc,
        "uncertainty_kJ_mol":  round(unc, 3) if unc else None,
        "pressure_kbar":       pressure,
    })
    meta_path.write_text(json.dumps(data, indent=2))

# ── Summary statistics ────────────────────────────────────────────────────────
converged = [r for r in results if r["delta_hf_kJ_mol"] is not None]
if converged:
    print(f"\n  {'='*97}")
    print(f"  Converged calculations: {len(converged)}")
    dHfs   = [r["delta_hf_kJ_mol"] for r in converged]
    mieds  = [r["miedema_kJ_mol"] for r in converged]
    ratios = [r["ratio_dft_miedema"] for r in converged
              if r["ratio_dft_miedema"] is not None]
    print(f"  ΔHf DFT   range: {min(dHfs):+.2f} to {max(dHfs):+.2f} kJ/mol")
    print(f"  Miedema   range: {min(mieds):+.2f} to {max(mieds):+.2f} kJ/mol")
    if ratios:
        print(f"  DFT/Miedema ratio: {min(ratios):.2f}× to {max(ratios):.2f}×  "
              f"(mean {sum(ratios)/len(ratios):.2f}×)")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_out = Path("dft_results_all.json")
json_out.write_text(json.dumps(results, indent=2))
print(f"\n  Saved: {json_out}  ({len(results)} records)")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_out = Path("dft_results_all.csv")
if results:
    fields = ["index", "n_elements", "delta_hf_kJ_mol", "uncertainty_kJ_mol",
              "miedema_kJ_mol", "ratio_dft_miedema", "pressure_kbar",
              "scf_accuracy_Ry", "e_total_Ry", "n_atoms"]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_out}")

print()
