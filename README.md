# Neural Network Prediction of Elastic Modulus and Enthalpy of Formation in NiCoCrFeMnAl High Entropy Alloys

**Author:** Richard Armoo  
**Institution:** Department of Civil, Environmental and Architectural Engineering, Worcester Polytechnic Institute  

---

## Overview

This repository contains all code, data, and DFT scripts supporting an ongoing
research project investigating the empirical-to-physical domain gap in machine
learning prediction of HEA properties.

Six ML models are trained on a 53,124-composition empirical dataset and evaluated
against three external test sets of increasing physical rigour. A pilot DFT study
(11 ternary compositions, Quantum ESPRESSO 7.1) directly measures the gap between
Miedema mixing rule labels and first-principles formation enthalpies.

> **Note:** This repository accompanies a manuscript currently under preparation.
> Citation information will be added upon publication.
> Please contact the author before using this work in publications.

---

## Repository structure

```
NiCoCrFeMnAl-HEA-ML-DFT/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── training/
│   │   └── empirical_dataset.csv          53,124-composition training set
│   │                                       (Miedema enthalpy + VRT modulus labels)
│   ├── external_test/
│   │   ├── mp_enthalpy_test_clean.csv      175 Materials Project DFT enthalpy entries
│   │   ├── mp_modulus_test_clean.csv       62 Materials Project DFT modulus entries
│   │   └── literature_test_clean_with_sources.csv   21-composition literature set
│   │                                                  with DOIs and source keys
│   └── dft_results/
│       ├── dft_results_all.csv             11 converged DFT calculations
│       └── dft_results_all.json            same data, JSON format
│
├── notebooks/
│   ├── HEA_NN_Pipeline_Corrected.ipynb    Main pipeline: training, evaluation,
│   │                                       bootstrap CIs, calibration, optimiser
│   └── HEA_Benchmark_Models_Extended.ipynb Benchmark models (LR, RF, XGB, SVR, MTL)
│                                           + full MP DFT evaluation
│
├── scripts/
│   ├── dft/
│   │   ├── 03_run_one_alloy.py            Runs one QE SCF calculation (SQS + pw.x)
│   │   ├── 04_submit_array.sh             SLURM job array submission script
│   │   ├── setup_batch2.py                Creates alloy_results/ input directories
│   │   ├── extract_dft_results.py         Post-processing: reads QE outputs,
│   │   │                                   computes ΔHf, outputs CSV/JSON
│   │   └── README_DFT.md                  Full DFT setup and usage guide
│   └── data/
│       └── fetch_mp_data.py               Fetches data from Materials Project API
│
└── figures/
    ├── fig1_dataset_overview.png
    ├── fig3_formula_vs_ml.png
    └── fig4_modulus_inversion.png
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the NN pipeline (Google Colab recommended)

Upload `notebooks/HEA_NN_Pipeline_Corrected.ipynb` to Google Colab.
The notebook loads data from embedded Google Sheets links — no manual upload needed.

Run cells top to bottom. Key outputs:
- Trained model weights
- Bootstrap confidence intervals on all three external test sets
- Calibration slopes
- Figures 3 and 4
- Top-10 candidate alloy compositions

### 3. Run benchmark models

Upload `notebooks/HEA_Benchmark_Models_Extended.ipynb` to Colab.
Trains LR, RF, XGB, SVR, MTL and evaluates on all external test sets.

### 4. Reproduce DFT results

Requires a cluster running Quantum ESPRESSO 7.1 with SLURM.
See `scripts/dft/README_DFT.md` for full setup instructions.

```bash
# On HPC cluster:
python3 scripts/dft/03_run_one_alloy.py --index 0
sbatch scripts/dft/04_submit_array.sh

# After jobs complete:
python3 scripts/dft/extract_dft_results.py
```

Raw QE input/output files are available from the author on request.

---

## Data

### Training dataset

53,124 compositions at 5 at.% resolution across the full NiCoCrFeMnAl simplex.
Labels computed from Miedema regular solution model (enthalpy) and
Voigt–Reuss–Tamura averaging (elastic modulus).

### External test sets

| File | n | Source | Property |
|------|---|--------|----------|
| `mp_enthalpy_test_clean.csv` | 175 | Materials Project | PBE DFT formation energy |
| `mp_modulus_test_clean.csv` | 62 | Materials Project | VRH Young's modulus from elastic tensor |
| `literature_test_clean_with_sources.csv` | 21 | 6 published studies (see file for DOIs) | Experimental / DFT |

### DFT results

11 converged NiFe-X ternary calculations. Settings: QE 7.1, PBE-PAW, 80 Ry
cutoff, 4×4×4 k-mesh, nspin=1, fixed FCC geometry (a₀=3.54 Å), 32-atom SQS.

---

## Key results (preliminary — manuscript in preparation)

| Test set | ΔHf R² (NN) | E R² (NN) |
|----------|-------------|-----------|
| Generated hold-out (n=10,625) | 0.993 | 0.975 |
| Literature set (n=21) | −0.773 | −2.940 |
| Materials Project DFT (n=175/62) | 0.396 | −0.302 |

DFT pilot study: Miedema underestimates mixing enthalpy by 2.4–9.7× across
NiFeCr, NiFeCo, and NiFeAl ternary subsystems.

---

## Contact

Richard Armoo  
rarmoo@wpi.edu  
Department of Civil, Environmental and Architectural Engineering  
Worcester Polytechnic Institute, Worcester MA 01609

---

## Citation

Manuscript in preparation. Citation details will be added upon publication.  
Please contact the author before citing or building on this work.

---

## License

To be determined upon manuscript acceptance.  
Until then, all rights reserved. Contact the author for permissions.
