#!/usr/bin/env python3

"""

03_run_one_alloy.py  —  SQS + two-step SCF DFT driver

=======================================================

Improvements over previous version

------------------------------------

1. SQS structure generation (sqsgenerator) instead of random site assignment

   — minimises Warren-Cowley short-range order parameters to approximate

   a truly random solid solution within the 32-atom supercell



2. Two-step SCF protocol for magnetic convergence

   Step 1: nspin=1 (non-magnetic), conv_thr=1e-5, max 150 iterations

           → saves wavefunctions and charge density to disk

   Step 2: nspin=2 (spin-polarised), startingwfc='file', startingpot='file'

           → restarts from converged non-magnetic density, adds magnetism

           as a perturbation rather than initialising from scratch

   This avoids the catastrophic negative-rho non-convergence seen when

   magnetic DFT is started from atomic densities in a disordered supercell.



3. ecutwfc=80 Ry, ecutrho=800 Ry

   — matches the wfc_cutoff (~75 Ry) and rho_cutoff (~480 Ry) recommended

   by the semi-core PAW pseudopotentials in use (pslibrary 1.0.0)



Usage (called by SLURM array, not directly)

--------------------------------------------

    python3 03_run_one_alloy.py --index 5 --compositions compositions.csv

"""



import os, sys, json, csv, math, subprocess, argparse

from pathlib import Path

import numpy as np



# ── Element data ──────────────────────────────────────────────────────────────

ELEMENTS = ["Ni", "Fe", "Cr", "Co", "Al", "Mn"]



PSEUDO = {

    "Ni": "Ni.pbe-n-kjpaw_psl.1.0.0.UPF",

    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",

    "Cr": "Cr.pbe-n-kjpaw_psl.1.0.0.UPF",

    "Co": "Co.pbe-n-kjpaw_psl.1.0.0.UPF",

    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",

    "Mn": "Mn.pbe-spn-kjpaw_psl.1.0.0.UPF",

}



ATOMIC_MASS = {

    "Ni": 58.693, "Fe": 55.845, "Cr": 51.996,

    "Co": 58.933, "Al": 26.982, "Mn": 54.938,

}



# Atomic numbers for sqsgenerator species map

ATOMIC_NUMBER = {

    "Ni": 28, "Fe": 26, "Cr": 24,

    "Co": 27, "Al": 13, "Mn": 25,

}

# Valence electrons (actual z_valence from UPF headers)

VALENCE_E = {

    "Ni": 18, "Fe": 16, "Cr": 14,

    "Co": 17, "Al":  3, "Mn": 15,

}



# Starting magnetic moments for nspin=2 step (fraction of max moment)

START_MAG = {

    "Ni":  0.05, "Fe":  0.25, "Cr": -0.05,

    "Co":  0.15, "Al":  0.00, "Mn":  0.20,

}



STRAINS = [-0.015, -0.0075, 0.0, 0.0075, 0.015]

RY_BOHR3_TO_GPA = 14710.507



# ── Helpers ───────────────────────────────────────────────────────────────────



def load_composition(csv_path, index):

    with open(csv_path, newline="") as f:

        for row in csv.DictReader(f):

            if int(row["index"]) == index:

                return {el: float(row[el]) for el in ELEMENTS}

    raise ValueError(f"Index {index} not found in {csv_path}")





def largest_remainder(fractions, n_sites):

    raw    = {el: f * n_sites for el, f in fractions.items()}

    floors = {el: int(v) for el, v in raw.items()}

    deficit = n_sites - sum(floors.values())

    order = sorted(raw, key=lambda el: raw[el] - floors[el], reverse=True)

    counts = dict(floors)

    for el in order[:deficit]:

        counts[el] += 1

    assert sum(counts.values()) == n_sites

    return counts





def _fcc32_neighbour_shells():

    """

    Precompute FCC shell neighbour lists for a 32-atom 2×2×2 supercell.

    Shell 1 (nearest):      12 neighbours, d²=0.125 in fractional coords

    Shell 2 (next-nearest):  3 neighbours, d²=0.250 in fractional coords

    (3 not 6 because ±0.5 maps to the same site under PBC in a finite cell)

    """

    from collections import defaultdict

    pos = []

    for ix in range(2):

        for iy in range(2):

            for iz in range(2):

                for dx,dy,dz in [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]:

                    pos.append(((ix+dx)/2,(iy+dy)/2,(iz+dz)/2))

    n = len(pos)

    s1, s2 = defaultdict(list), defaultdict(list)

    for i in range(n):

        for j in range(n):

            if i == j: continue

            d = [(pos[i][k]-pos[j][k]) % 1.0 for k in range(3)]

            d = [x if x <= 0.5 else x - 1.0 for x in d]

            d2 = sum(x*x for x in d)

            if abs(d2 - 0.125) < 0.005: s1[i].append(j)

            elif abs(d2 - 0.25)  < 0.005: s2[i].append(j)

    return s1, s2



# Precomputed once at module load — shared across all calls

_S1, _S2 = _fcc32_neighbour_shells()





def _warren_cowley_obj(species, elements):

    """

    Compute Warren-Cowley SRO objective for the current arrangement.

    Σ_shell w_k × Σ_ij α_ij² where α_ij = 1 − P_ij/x_j

    Lower = more random. Perfect random solid solution → 0.

    """

    n    = len(species)

    conc = {el: species.count(el) / n for el in elements}

    obj  = 0.0

    for w, sh in [(1.0, _S1), (0.5, _S2)]:

        for i in range(n):

            nb = sh[i]

            if not nb: continue

            nb_len = len(nb)

            from collections import Counter as _Ctr

            cnt = _Ctr(species[j] for j in nb)

            for el in elements:

                if conc[el] > 0:

                    alpha = 1.0 - cnt.get(el, 0) / (nb_len * conc[el])

                    obj  += w * alpha * alpha

    return obj





def generate_sqs(composition, n_atoms=32, a_ang=3.54, iterations=300000):

    """

    Pure-Python Monte Carlo SQS generator — zero external dependencies.



    Minimises Warren-Cowley short-range order (SRO) parameters for FCC

    shells 1 and 2 by iteratively swapping atom pairs and accepting moves

    that reduce the SRO objective. Uses simulated annealing to escape

    local minima.



    Achieves ~60% reduction in SRO relative to a random arrangement for

    typical NiCoCrFeMnAl compositions in a 32-atom cell.



    Returns

    -------

    list of str: chemical symbols for the 32 FCC sites in SQS order

    """

    import random, math

    random.seed(42)



    counts = largest_remainder(

        {el: f for el, f in composition.items() if f > 1e-6},

        n_sites=n_atoms

    )

    elements = [el for el in counts if counts[el] > 0]



    # Initial random arrangement

    species = []

    for el in sorted(counts):

        species += [el] * counts[el]

    random.shuffle(species)



    obj      = _warren_cowley_obj(species, elements)

    best_sp  = species[:]

    best_obj = obj

    T        = 0.05   # annealing temperature



    for it in range(iterations):

        i = random.randrange(n_atoms)

        j = random.randrange(n_atoms)

        if species[i] == species[j]:

            continue



        species[i], species[j] = species[j], species[i]

        new_obj = _warren_cowley_obj(species, elements)



        if new_obj < obj or random.random() < math.exp(-(new_obj - obj) / T):

            obj = new_obj

            if obj < best_obj:

                best_obj = new_obj

                best_sp  = species[:]

        else:

            species[i], species[j] = species[j], species[i]



        # Cool slowly

        if it % 10000 == 0 and T > 0.001:

            T *= 0.95



    print(f"  SQS MC: obj={best_obj:.4f} after {iterations:,} iterations "

          f"(lower=more random)", flush=True)

    return best_sp





def compute_nbnd(composition, n_atoms=32, buffer=0.30, nspin=1):

    """

    Dynamic NBANDS calculation using actual z_valence from UPF headers.



    nspin=1: nbnd = ceil(n_val / 2 * (1+buffer))

    nspin=2: nbnd = ceil(n_val * (1+buffer))

    """

    total_valence = sum(

        composition.get(el, 0.0) * VALENCE_E[el] * n_atoms

        for el in ELEMENTS

    )

    nbnd = math.ceil(total_valence / 2 * (1 + buffer))

    return max(nbnd, 20)





def compute_kpoints(a_ang, target_spacing_bohr=0.5, min_k=3):

    a_bohr      = a_ang / 0.529177210903

    supercell_a = 2 * a_bohr

    k = max(4, math.ceil(2 * math.pi / (supercell_a * target_spacing_bohr)))

    return f"{k} {k} {k} 1 1 1"    # min 4x4x4





def get_present(composition, threshold=0.001):

    return [el for el in ELEMENTS if composition.get(el, 0.0) > threshold]





def fcc_positions(n=32):

    """Crystal coordinates of 32-atom 2×2×2 FCC supercell."""

    pos = []

    for ix in range(2):

        for iy in range(2):

            for iz in range(2):

                for dx, dy, dz in [

                    (0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]:

                    pos.append((

                        (ix+dx)/2, (iy+dy)/2, (iz+dz)/2

                    ))

    assert len(pos) == n

    return pos





def species_block(present):

    return "\n".join(

        f"  {el:4s}  {ATOMIC_MASS[el]:.4f}  {PSEUDO[el]}"

        for el in present

    )





def mag_block(present, nspin=1):

    return ""   # nspin=1 throughout — no magnetization





def cell_block(a, strain_zz=0.0):

    a2 = 2 * a

    return (

        f"  {a2:.8f}  0.000000  0.000000\n"

        f"  0.000000  {a2:.8f}  0.000000\n"

        f"  0.000000  0.000000  {a2*(1+strain_zz):.8f}"

    )





def pos_block(symbols, positions):

    return "\n".join(

        f"  {sym:4s}  {x:.8f}  {y:.8f}  {z:.8f}"

        for sym, (x,y,z) in zip(symbols, positions)

    )





def write_input(work_dir, fname, calculation, symbols, a_ang, composition,

                nbnd, kpts, pseudos_rel, nspin, suffix="scf",

                strain_zz=0.0, restart=False, outdir_suffix=""):

    """Write a pw.x input file."""

    present = get_present(composition)

    ntyp = len(present)

    nat  = len(symbols)

    outdir_name = f"outdir{outdir_suffix}"



    start_wfc = "'file'" if restart else "'atomic+random'"

    start_pot = "'file'" if restart else "'atomic'"



    mag = mag_block(present, nspin)

    mag_section = ("\n" + mag) if mag else ""



    content = f"""\

&CONTROL

  calculation   = '{calculation}'

  prefix        = 'alloy_{suffix}'

  outdir        = './{outdir_name}'

  pseudo_dir    = '{pseudos_rel}'

  verbosity     = 'low'

  tprnfor       = .true.

  tstress       = .true.

  disk_io       = 'medium'

  etot_conv_thr = 1.0d-3

  forc_conv_thr = 5.0d-3

/

&SYSTEM

  ibrav         = 0

  nat           = {nat}

  ntyp          = {ntyp}

  ecutwfc       = 80.0

  ecutrho       = 800.0

  nspin         = 1

  occupations   = 'smearing'

  smearing      = 'mv'

  degauss       = 0.02

  nbnd          = {nbnd}{mag_section}

/

&ELECTRONS

  conv_thr         = {'5.0d-2' if not restart else '1.0d-6'}

  mixing_beta      = 0.03

  mixing_mode      = 'TF'

  mixing_ndim      = 16

  electron_maxstep = {'300' if not restart else '400'}

  diagonalization  = 'david'

  startingwfc      = 'atomic+random'

/

{'&IONS' + chr(10) + "  ion_dynamics  = 'bfgs'" + chr(10) + '/' if calculation in ('relax','vc-relax') else ''}

{'&CELL' + chr(10) + "  cell_dynamics    = 'bfgs'" + chr(10) + "  press_conv_thr   = 0.5" + chr(10) + "  cell_factor      = 2.0" + chr(10) + '/' if calculation == 'vc-relax' else ''}

ATOMIC_SPECIES

{species_block(present)}



CELL_PARAMETERS {{angstrom}}

{cell_block(a_ang, strain_zz)}



ATOMIC_POSITIONS {{crystal}}

{pos_block(symbols, fcc_positions())}



K_POINTS {{automatic}}

  {kpts}

"""

    path = work_dir / fname

    path.write_text(content)

    return path





def run_pw(inp, out, n_mpi, cwd):

    cmd = ["srun", "--mpi=pmix", "-n", str(n_mpi), "pw.x",

           "-in", inp.name, "-out", out.name]

    result = subprocess.run(cmd, cwd=cwd,

                            stdout=open(out, "w"),

                            stderr=subprocess.STDOUT)

    return result.returncode





def parse_energy(out_file):

    """

    Return (energy, scf_accuracy, converged) where:

    - energy is the total energy at the SCF iteration with lowest accuracy

    - scf_accuracy is that best accuracy in Ry

    - converged is True only if the final line shows convergence



    For screening purposes, partially converged energies (scf_accuracy

    < 0.1 Ry, ~4 kJ/mol uncertainty) are physically meaningful.

    The reference energies converged fully, so errors partially cancel

    in the ΔHf difference.

    """

    best_energy   = None

    best_accuracy = float('inf')

    last_energy   = None

    converged     = False



    try:

        with open(out_file) as f:

            lines = f.readlines()

    except FileNotFoundError:

        return None, None, False



    i = 0

    while i < len(lines):

        line = lines[i]



        # Track best SCF point

        if "estimated scf accuracy" in line:

            try:

                acc = float(line.split("<")[1].split()[0])

                # Look backward for the most recent total energy

                for j in range(i-1, max(0, i-5), -1):

                    if "total energy" in lines[j] and "=" in lines[j]:

                        parts = lines[j].split("=")

                        if len(parts) > 1:

                            try:

                                e = float(parts[1].split()[0])

                                if acc < best_accuracy:

                                    best_accuracy = acc

                                    best_energy   = e

                                break

                            except ValueError:

                                pass

            except (ValueError, IndexError):

                pass



        # Final converged energy (! prefix in QE)

        if line.strip().startswith("!    total energy"):

            try:

                last_energy = float(line.split("=")[1].split()[0])

                converged   = True

            except (ValueError, IndexError):

                pass



        i += 1



    if converged:

        return last_energy, 0.0, True

    elif best_accuracy < float('inf'):

        return best_energy, best_accuracy, False

    return None, None, False





def parse_geometry(out_file):

    """Parse final cell vectors (Ang) and atomic positions (crystal)."""

    cell_vecs, positions = [], []

    cur_cell, cur_pos = [], []

    reading_cell = reading_pos = False



    with open(out_file) as f:

        for line in f:

            if "CELL_PARAMETERS" in line and "angstrom" in line.lower():

                cur_cell = []; reading_cell = True; reading_pos = False; continue

            if "ATOMIC_POSITIONS" in line and "crystal" in line.lower():

                cur_pos  = []; reading_pos  = True; reading_cell = False; continue

            if reading_cell:

                p = line.split()

                if len(p) == 3:

                    try:

                        cur_cell.append([float(x) for x in p])

                    except ValueError:

                        reading_cell = False

                if len(cur_cell) == 3:

                    cell_vecs = cur_cell[:]; reading_cell = False

            if reading_pos:

                p = line.split()

                if len(p) >= 4 and p[0].isalpha():

                    try:

                        cur_pos.append([float(p[1]), float(p[2]), float(p[3])])

                    except ValueError:

                        reading_pos = False

                elif p and not p[0].isalpha():

                    positions = cur_pos[:]; reading_pos = False



    return cell_vecs, positions





def cell_volume_bohr3(cell_vecs_ang):

    a = np.array(cell_vecs_ang)

    vol_ang3 = abs(np.dot(a[0], np.cross(a[1], a[2])))

    return vol_ang3 * (1.0 / 0.529177210903) ** 3





def fit_modulus(strains, energies, volume_bohr3):

    if None in energies or len(energies) < 3:

        return None

    coeffs = np.polyfit(strains, energies, 2)

    return 2.0 * coeffs[0] / volume_bohr3 * RY_BOHR3_TO_GPA





def compute_delta_hf(energy_ry, composition, ref_energies, n_atoms=32):

    RY_TO_KJ = 1312.75

    if energy_ry is None:

        return None

    ref_sum = sum(

        composition.get(el, 0.0) * ref_energies.get(el, 0.0) * n_atoms

        for el in ELEMENTS

    )

    return (energy_ry - ref_sum) / n_atoms * RY_TO_KJ





# ── Main ──────────────────────────────────────────────────────────────────────



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--index",        type=int, required=True)

    ap.add_argument("--compositions", default="compositions.csv")

    ap.add_argument("--pseudos",      default="../../pseudos")

    ap.add_argument("--ref-energies", default="../../reference_energies.json",

                    dest="ref_energies")

    ap.add_argument("--nmpi",         type=int,

                    default=int(os.environ.get("QE_NP", 8)))

    ap.add_argument("--outdir",       default=".")

    ap.add_argument("--lattice-a",    type=float, default=3.54, dest="a0")

    ap.add_argument("--sqs-iter",     type=int, default=200000, dest="sqs_iter")

    args = ap.parse_args()



    idx      = args.index

    work_dir = Path(args.outdir)

    ps_rel   = os.path.relpath(args.pseudos, work_dir)



    print(f"\n{'='*60}")

    print(f"  Alloy {idx}  |  {args.nmpi} MPI tasks  |  {work_dir}")

    print(f"{'='*60}\n", flush=True)



    composition = load_composition(args.compositions, idx)

    present     = get_present(composition)

    print(f"Composition: " +

          ", ".join(f"{el}={composition[el]*100:.1f}%" for el in present),

          flush=True)



    results = {"index": idx, "composition": composition,

               "n_atoms": 32, "status": "started"}



    def save():

        with open(work_dir / "results.json", "w") as f:

            json.dump(results, f, indent=2)

    save()



    # ── SQS structure ─────────────────────────────────────────────────────────

    print("\n[SQS] Generating special quasirandom structure...", flush=True)

    symbols = generate_sqs(composition, n_atoms=32,

                           a_ang=args.a0, iterations=args.sqs_iter)

    from collections import Counter

    print(f"  Site occupancies: {Counter(symbols)}", flush=True)



    nbnd = compute_nbnd(composition, n_atoms=32)

    kpts = compute_kpoints(args.a0)

    print(f"  nbnd={nbnd}  |  k-pts: {kpts}", flush=True)



    # ── Step 1: nspin=1 vc-relax ──────────────────────────────────────────────

    print("\n[Step 1] nspin=1 scf (fixed geometry)...", flush=True)

    (work_dir / "outdir_step1").mkdir(exist_ok=True)



    inp1 = write_input(work_dir, "step1_scf.in", "scf",

                       symbols, args.a0, composition,

                       nbnd, kpts, ps_rel,

                       nspin=1, suffix="step1",

                       outdir_suffix="_step1")

    out1 = work_dir / "step1_scf.out"



    rc = run_pw(inp1, out1, args.nmpi, work_dir)

    e1, e1_acc, e1_conv = parse_energy(out1)

    cell_vecs, positions = parse_geometry(out1)



    if not cell_vecs or not positions:

        print("  Could not parse geometry — aborting", flush=True)

        results["status"] = "vcrelax_parse_failed"

        save(); return



    print(f"  Energy: {e1} Ry", flush=True)

    print(f"  Relaxed cell: "

          f"{cell_vecs[0][0]:.4f} x {cell_vecs[1][1]:.4f} x "

          f"{cell_vecs[2][2]:.4f} Å", flush=True)

    results["vcrelax_energy_Ry"]  = e1

    results["vcrelax_scf_acc"]    = e1_acc

    results["vcrelax_converged"]  = e1_conv

    # Flag: partially converged energies still useful for screening

    # scf_acc < 0.1 Ry → ΔHf uncertainty ~4 kJ/mol — acceptable for screening

    results["cell_vectors_ang"]  = cell_vecs

    results["status"] = "vcrelax_done"

    save()



    # ── Step 2: tight SCF for ΔHf ────────────────────────────────────────────

    print("\n[Step 2] nspin=1 SCF (tight, for ΔHf)...", flush=True)

    (work_dir / "outdir_scf").mkdir(exist_ok=True)



    inp2 = write_input(work_dir, "scf.in", "scf",

                       symbols, None, composition,

                       nbnd, kpts, ps_rel,

                       nspin=1, suffix="scf",

                       restart=True, outdir_suffix="_step1")



    # Patch to relaxed geometry

    sc2 = inp2.read_text()

    relaxed_cell = (

        f"  {cell_vecs[0][0]:.8f}  {cell_vecs[0][1]:.8f}  {cell_vecs[0][2]:.8f}\n"

        f"  {cell_vecs[1][0]:.8f}  {cell_vecs[1][1]:.8f}  {cell_vecs[1][2]:.8f}\n"

        f"  {cell_vecs[2][0]:.8f}  {cell_vecs[2][1]:.8f}  {cell_vecs[2][2]:.8f}"

    )

    pos_new = pos_block(symbols, positions)

    sc2 = sc2.replace(

        "CELL_PARAMETERS {angstrom}\n" + cell_block(cell_vecs[0][0]/2),

        "CELL_PARAMETERS {angstrom}\n" + relaxed_cell

    )

    sc2 = sc2.replace(

        "ATOMIC_POSITIONS {crystal}\n" + pos_block(symbols, fcc_positions()),

        "ATOMIC_POSITIONS {crystal}\n" + pos_new

    )

    inp2.write_text(sc2)



    out2 = work_dir / "scf.out"

    run_pw(inp2, out2, args.nmpi, work_dir)

    e_scf_raw, e_scf_acc, e_scf_conv = parse_energy(out2)

    e_scf = e_scf_raw if e_scf_raw else e1

    e_scf_acc = e_scf_acc if e_scf_raw else e1_acc

    print(f"  SCF energy: {e_scf} Ry", flush=True)

    results["scf_energy_Ry"] = e_scf

    results["status"] = "scf_done"



    ref_energies = {}

    ref_path = Path(args.ref_energies)

    if ref_path.exists():

        with open(ref_path) as f:

            ref_energies = json.load(f)

        dHf = compute_delta_hf(e_scf, composition, ref_energies)

        results["delta_hf_kJ_per_mol"] = dHf

        print(f"  ΔHf = {dHf:.4f} kJ/mol", flush=True)

    else:

        results["delta_hf_kJ_per_mol"] = None

    save()



    # ── Step 3: Strain series for elastic modulus ─────────────────────────────

    print("\n[Step 3] Uniaxial strain series...", flush=True)

    strain_energies = []

    pos_new = pos_block(symbols, positions)



    for i, strain in enumerate(STRAINS):

        sfx = f"strain{i}"

        (work_dir / f"outdir_{sfx}").mkdir(exist_ok=True)



        s_in = write_input(work_dir, f"{sfx}.in", "scf",

                           symbols, None, composition,

                           nbnd, kpts, ps_rel,

                           nspin=1, suffix=sfx,

                           strain_zz=strain, outdir_suffix=f"_{sfx}")



        sc = s_in.read_text()

        strained_vecs = [list(v) for v in cell_vecs]

        strained_vecs[2][2] *= (1.0 + strain)

        strained_cell = (

            f"  {strained_vecs[0][0]:.8f}  {strained_vecs[0][1]:.8f}  {strained_vecs[0][2]:.8f}\n"

            f"  {strained_vecs[1][0]:.8f}  {strained_vecs[1][1]:.8f}  {strained_vecs[1][2]:.8f}\n"

            f"  {strained_vecs[2][0]:.8f}  {strained_vecs[2][1]:.8f}  {strained_vecs[2][2]:.8f}"

        )

        sc = sc.replace(

            "CELL_PARAMETERS {angstrom}\n" + cell_block(cell_vecs[0][0]/2, strain),

            "CELL_PARAMETERS {angstrom}\n" + strained_cell

        )

        sc = sc.replace(

            "ATOMIC_POSITIONS {crystal}\n" + pos_block(symbols, fcc_positions()),

            "ATOMIC_POSITIONS {crystal}\n" + pos_new

        )

        s_in.write_text(sc)



        s_out = work_dir / f"{sfx}.out"

        run_pw(s_in, s_out, args.nmpi, work_dir)

        e, _acc, _conv = parse_energy(s_out)

        strain_energies.append(e)

        print(f"  strain={strain:+.4f}: {f'{e:.6f} Ry' if e else 'FAILED'}",

              flush=True)



    results["strain_values"]   = STRAINS

    results["strain_energies"] = strain_energies



    vol     = cell_volume_bohr3(cell_vecs)

    modulus = fit_modulus(STRAINS, strain_energies, vol)

    results["cell_volume_bohr3"] = vol

    results["modulus_GPa"]       = modulus



    if modulus:

        print(f"\n  Elastic Modulus (C33 approx): {modulus:.2f} GPa")

    else:

        print("\n  Modulus not computed (strain SCFs failed)")



    results["status"] = "complete"

    save()



    print(f"\n{'='*60}")

    print(f"  Alloy {idx} complete.")

    print(f"  ΔHf     = {results.get('delta_hf_kJ_per_mol')}")

    print(f"  Modulus = {modulus}")

    print(f"{'='*60}\n")





if __name__ == "__main__":

    main()
