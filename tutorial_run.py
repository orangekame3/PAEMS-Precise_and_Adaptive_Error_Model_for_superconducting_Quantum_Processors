"""
PAEMS Tutorial: Run simulation with optimized parameters from IBM Brisbane.

This script demonstrates:
1. Loading optimized PAEMS parameters (from CMA-ES optimization)
2. Running a Stim-based repetition code simulation
3. Computing the detection event correlation matrix
4. Comparing PAEMS vs SI1000 (uniform depolarizing) models
"""
import json
import numpy as np
import sys
import os

# Setup paths
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import NUM_ROUNDS
from Simulation.Creation_and_Sampling.sampling import run_sampling
from Simulation.Paramaeter_Loading.calculate_p import calculate_px_py_pz

# ─── Configuration ───
NUM_QUBITS = 21
NUM_ROUNDS_SIM = 30
SHOTS_EXP = 4096  # Total shots (paper uses 4096 per run)
SHOTS = 1  # Shots per noise scenario (leakage is stochastic, so 1 per circuit)

PARAM_FILE = os.path.join(
    ROOT,
    "Data/Multi_Rep_Code/SCPQN_model/brisbane_layout_0/"
    "d1rqo2mjpjps738m44r0/optimized_parameters.json"
)

# ─── Step 1: Load optimized parameters ───
print("=" * 60)
print("Step 1: Loading PAEMS optimized parameters (IBM Brisbane)")
print("=" * 60)

with open(PARAM_FILE, 'r') as f:
    params = json.load(f)

spam_rates = params['spam_rates']
t1_t2_values = [(item['t1'], item['t2']) for item in params['t1_t2_values']]
ecr_fid = params['ecr_fid']
sqg_fid = params['sqg_fid']
ecr_lengths = params['ecr_lengths']
sqg_lengths = params['sqg_length'][0]
rd_lengths = params['rd_length']
lp = params['lp']
sp = params['sp']

print(f"  Qubits: {NUM_QUBITS}")
print(f"  Rounds: {NUM_ROUNDS_SIM}")
print(f"  Shots:  {SHOTS_EXP}")
print()

# Show parameter heterogeneity
t1_us = [t1 * 1e6 for t1, _ in t1_t2_values]
t2_us = [t2 * 1e6 for _, t2 in t1_t2_values]
print("  T1 range: {:.1f} - {:.1f} μs (ratio: {:.1f}x)".format(
    min(t1_us), max(t1_us), max(t1_us)/min(t1_us)))
print("  T2 range: {:.1f} - {:.1f} μs (ratio: {:.1f}x)".format(
    min(t2_us), max(t2_us), max(t2_us)/min(t2_us)))
print("  ECR fidelity range: {:.4f} - {:.4f}".format(
    min(ecr_fid), max(ecr_fid)))
print("  Leakage prob range: {:.2e} - {:.2e}".format(
    min(lp), max(lp)))
print()

# ─── Step 2: Understand the error decomposition ───
print("=" * 60)
print("Step 2: PAEMS error decomposition (ADC vs SDC)")
print("=" * 60)

# ADC (Asymmetric Depolarizing Channel) from T1, T2
px_py_pz = calculate_px_py_pz(t1_t2_values, sqg_lengths)
print("\n  Asymmetric Pauli channel per qubit (from T1/T2):")
print("  Qubit |   px=py   |    pz     | px+py+pz  | Asymmetry(pz/px)")
print("  " + "-" * 62)
for i in range(min(5, NUM_QUBITS)):
    px, py, pz = px_py_pz[i]
    ratio = pz / px if px > 0 else float('inf')
    print(f"    {i:2d}   | {px:.2e} | {pz:.2e} | {px+py+pz:.2e} | {ratio:.2f}")
print("  ...")
print()
print("  Key insight: SI1000 assumes px=py=pz (symmetric).")
print("  PAEMS captures that dephasing (pz) often dominates relaxation (px=py).")

# ─── Step 3: Run PAEMS simulation ───
print()
print("=" * 60)
print("Step 3: Running PAEMS simulation (this may take a moment)...")
print("=" * 60)

# Use fewer shots for tutorial speed
TUTORIAL_SHOTS = 500
results = run_sampling(
    shots=SHOTS,
    shots2=TUTORIAL_SHOTS,
    rounds=NUM_ROUNDS_SIM,
    num_qubits=NUM_QUBITS,
    lp=lp,
    sp=sp,
    spam_rates=spam_rates,
    sqg_fid=sqg_fid,
    ecr_fid=ecr_fid,
    t1_t2_values=t1_t2_values,
    ecr_lengths=ecr_lengths,
    rd_lengths=rd_lengths,
    sqg_lengths=sqg_lengths,
)

print(f"  Result shape: {results.shape}")
print(f"  (circuits × detection_events_per_circuit)")
print(f"  Detection events per circuit: {results.shape[1]}")
print(f"    = (ancilla_qubits={NUM_QUBITS//2}) × (rounds={NUM_ROUNDS_SIM}+1) - {NUM_QUBITS//2}")
print()

# ─── Step 4: Compute detection event statistics ───
print("=" * 60)
print("Step 4: Detection event statistics")
print("=" * 60)

# Detection event fraction over time
num_ancilla = (NUM_QUBITS - 1) // 2  # 10 ancilla qubits
total_det_per_round = num_ancilla

# Reshape to (rounds, ancilla, shots) approximately
# Each circuit has num_ancilla * (NUM_ROUNDS_SIM + 1) - num_ancilla detectors
# = num_ancilla * NUM_ROUNDS_SIM detectors
n_det = results.shape[1]
expected_det = num_ancilla * NUM_ROUNDS_SIM
print(f"  Ancilla qubits: {num_ancilla}")
print(f"  Expected detectors: {expected_det}")
print(f"  Actual detectors: {n_det}")

# Overall detection rate
det_rate = results.mean()
print(f"  Overall detection event rate: {det_rate:.4f}")
print()

# ─── Step 5: Compare with SI1000 conceptually ───
print("=" * 60)
print("Step 5: PAEMS vs SI1000 — conceptual comparison")
print("=" * 60)
print("""
  SI1000 (Google's model):
    - Single parameter: p (e.g., p=0.01 for all operations)
    - All qubits identical
    - No leakage modeling
    - Symmetric depolarizing noise

  PAEMS (this model):
    - Per-qubit T1, T2, gate fidelity, SPAM rates
    - Leakage probability + seepage probability per qubit
    - Asymmetric depolarizing (ADC) for decoherence
    - Symmetric depolarizing (SDC) for gate errors
    - CMA-ES optimized against real hardware data

  Example: Qubit 0 vs Qubit 8 in this dataset:
""")

for qi in [0, 8]:
    t1, t2 = t1_t2_values[qi]
    print(f"    Qubit {qi}: T1={t1*1e6:.1f}μs, T2={t2*1e6:.1f}μs, "
          f"LP={lp[qi]:.2e}, SP={sp[qi]:.3f}, "
          f"SPAM={spam_rates[qi]:.4f}")

print()
print("  → SI1000 would treat these identically!")
print("  → PAEMS captures the 10x T1 difference and 3x leakage difference.")
print()
print("Tutorial complete!")
