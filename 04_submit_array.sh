#!/bin/bash
# =============================================================================
# 04_submit_array.sh
# =============================================================================
# SLURM job array script for the NiCoCrFeMnAl HEA DFT screening.
# Each array task runs one alloy composition independently — this is the
# correct fix for the MPI collapse that occurred in the previous approach.
#
# How to use
# ----------
# 1. Run scripts 01 and 02 first:
#       python3 01_generate_compositions.py    # creates compositions.csv
#       sbatch 04_submit_references.sh         # creates reference_energies.json
#
# 2. Check how many compositions were generated:
#       N=$(( $(wc -l < compositions.csv) - 1 ))
#       echo "Total compositions: $N"
#
# 3. Submit the full array:
#       sbatch --array=0-$((N-1)) 04_submit_array.sh
#
#    Or a subset (e.g. first 10 for testing):
#       sbatch --array=0-9 04_submit_array.sh
#
#    Or specific indices:
#       sbatch --array=42,55,67 04_submit_array.sh
#
#    Or with a max concurrent limit (avoid swamping the queue):
#       sbatch --array=0-78%20 04_submit_array.sh   # max 20 running at once
#
# Output structure
# ----------------
# alloy_results/
#   alloy_0000/
#     vcrelax.in / vcrelax.out
#     scf.in / scf.out
#     strain_0.in / strain_0.out  (5 strain points)
#     ...
#     results.json      ← parsed results for this alloy
#   alloy_0001/
#     ...
#   HEA_array.JOBID.out   ← SLURM stdout for each task
#   HEA_array.JOBID.err   ← SLURM stderr for each task
#
# =============================================================================

#SBATCH --job-name=HEA_array
#SBATCH --output=alloy_results/logs/HEA_array.%A_%a.out
#SBATCH --error=alloy_results/logs/HEA_array.%A_%a.err
#SBATCH --partition=long
#SBATCH --time=2-00:00:00          # 2 days per alloy — vc-relax + 5 SCFs
#SBATCH --nodes=1
#SBATCH --ntasks=8                 # 8 MPI tasks per alloy
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G                  # 16 GB per task (32-atom cell, moderate k-mesh)
#SBATCH --hint=nomultithread

# NOTE: Do NOT set --array here — pass it on the sbatch command line
# so you can easily control the range without editing this file.

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
module purge  || true
module load python/3.9.12                   || true
module load quantum-espresso/7.1/llmvxn4   || true

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Threading: keep at 1 — parallelism comes from MPI, not threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Number of MPI tasks — matches --ntasks above
export QE_NP=${SLURM_NTASKS:-8}

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Job ID     : ${SLURM_JOB_ID}"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Node       : ${HOSTNAME}"
echo "  MPI tasks  : ${QE_NP}"
echo "  Start time : $(date)"
echo "============================================================"

# Confirm required files exist
BASE_DIR="/home/rarmoo/HEA_discovery_SCF"
cd "${BASE_DIR}"

[ -f compositions.csv ]         || { echo "ERROR: compositions.csv not found"; exit 2; }
[ -f 03_run_one_alloy.py ]      || { echo "ERROR: 03_run_one_alloy.py not found"; exit 2; }
[ -d pseudos ]                  || { echo "ERROR: pseudos/ directory not found"; exit 2; }
command -v pw.x >/dev/null 2>&1 || { echo "ERROR: pw.x not on PATH"; exit 2; }

# reference_energies.json is optional — ΔHf will be skipped if missing
if [ ! -f reference_energies.json ]; then
    echo "WARNING: reference_energies.json not found — ΔHf will not be computed."
    echo "         Run 02_run_references.py first to enable enthalpy calculations."
fi

# ── Per-alloy output directory ────────────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID}
ALLOY_DIR="alloy_results/alloy_$(printf '%04d' ${TASK_ID})"
mkdir -p "${ALLOY_DIR}"
mkdir -p "alloy_results/logs"

echo "  Output dir : ${ALLOY_DIR}"

# Check if this alloy is already complete (supports re-submission after failures)
if [ -f "${ALLOY_DIR}/results.json" ]; then
    STATUS=$(python3 -c "
import json, sys
try:
    d = json.load(open('${ALLOY_DIR}/results.json'))
    print(d.get('status', 'unknown'))
except:
    print('unreadable')
")
    if [ "${STATUS}" = "complete" ]; then
        echo "  Alloy ${TASK_ID} already complete — skipping."
        echo "  (Delete ${ALLOY_DIR}/results.json to force rerun)"
        exit 0
    else
        echo "  Previous run status: ${STATUS} — rerunning."
    fi
fi

# ── Run the DFT driver ────────────────────────────────────────────────────────
echo "  Running alloy index ${TASK_ID} ..."

python3 -u 03_run_one_alloy.py \
    --index        "${TASK_ID}" \
    --compositions "compositions.csv" \
    --pseudos      "pseudos" \
    --ref-energies "reference_energies.json" \
    --nmpi         "${QE_NP}" \
    --outdir       "${ALLOY_DIR}" \
    --lattice-a    3.54 \

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Alloy ${TASK_ID} finished at $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
