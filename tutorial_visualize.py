"""
PAEMS Tutorial: Visualize correlation matrices — PAEMS vs SI1000.

Generates:
  1. PAEMS correlation matrix heatmap
  2. SI1000 (uniform) correlation matrix heatmap
  3. Side-by-side comparison with correlation strength bar chart
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from Simulation.Creation_and_Sampling.sampling import run_sampling
from Simulation.Analysis_and_Plotting.analysis import compute_correlation_matrix

# ─── Config ───
NUM_QUBITS = 21
NUM_ROUNDS = 30
SHOTS = 1
SHOTS_SIM = 2000  # Increase for smoother results (paper uses 4096×25)
NUM_ANCILLA = (NUM_QUBITS - 1) // 2  # 10

PARAM_FILE = os.path.join(
    ROOT,
    "Data/Multi_Rep_Code/SCPQN_model/brisbane_layout_0/"
    "d1rqo2mjpjps738m44r0/optimized_parameters.json"
)

OUT_DIR = os.path.join(ROOT, "tutorial_output")
os.makedirs(OUT_DIR, exist_ok=True)


def load_paems_params():
    with open(PARAM_FILE, 'r') as f:
        params = json.load(f)
    return {
        'spam_rates': params['spam_rates'],
        't1_t2_values': [(item['t1'], item['t2']) for item in params['t1_t2_values']],
        'ecr_fid': params['ecr_fid'],
        'sqg_fid': params['sqg_fid'],
        'ecr_lengths': params['ecr_lengths'],
        'sqg_lengths': params['sqg_length'][0],
        'rd_lengths': params['rd_length'],
        'lp': params['lp'],
        'sp': params['sp'],
    }


def make_si1000_params(p=0.02):
    """Create SI1000-style uniform parameters."""
    # SI1000: uniform error rate p for all operations
    # Map p to physical parameters approximately
    t1_uniform = 100e-6  # 100 μs (typical)
    t2_uniform = 80e-6   # 80 μs
    ecr_fid_uniform = 1.0 - p  # gate fidelity from p
    sqg_fid_uniform = 1.0 - p / 10  # single-qubit gates ~10x better
    spam_uniform = p / 2

    return {
        'spam_rates': [spam_uniform] * NUM_QUBITS,
        't1_t2_values': [(t1_uniform, t2_uniform)] * NUM_QUBITS,
        'ecr_fid': [ecr_fid_uniform] * (NUM_QUBITS - 1),
        'sqg_fid': [sqg_fid_uniform] * NUM_QUBITS,
        'ecr_lengths': [6.6e-7] * (NUM_QUBITS - 1),
        'sqg_lengths': 6e-8,
        'rd_lengths': [1.3e-6] * NUM_QUBITS,
        'lp': [0.0] * NUM_QUBITS,  # SI1000 has no leakage
        'sp': [0.0] * NUM_QUBITS,
    }


def run_sim(params_dict, label):
    print(f"  Running {label} simulation ({SHOTS_SIM} shots)...", flush=True)
    results = run_sampling(
        shots=SHOTS,
        shots2=SHOTS_SIM,
        rounds=NUM_ROUNDS,
        num_qubits=NUM_QUBITS,
        **params_dict,
    )
    print(f"  → raw shape: {results.shape}")

    # results is (num_circuits, shots_sim * num_detectors_flat)
    # Reshape to (shots_sim, num_detectors)
    total = results.shape[1]
    n_det_per_shot = total // SHOTS_SIM
    n_rounds_eff = n_det_per_shot // NUM_ANCILLA
    print(f"  → detectors/shot: {n_det_per_shot}, effective rounds: {n_rounds_eff}")

    data = results.reshape(SHOTS_SIM, n_det_per_shot)

    corr = compute_correlation_matrix(
        data=data,
        num_qubits=NUM_QUBITS,
        rounds_=n_rounds_eff,
        shots=SHOTS_SIM,
    )
    print(f"  → correlation matrix: {corr.shape}")
    return corr, n_rounds_eff


def plot_correlation_heatmap(corr, title, filename, n_rounds=None):
    if n_rounds is None:
        n_rounds = NUM_ROUNDS
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(corr, interpolation='nearest', cmap='RdBu_r',
                     vmin=-0.02, vmax=0.08)
    fig.colorbar(cax, label=r'$p_{ij}$', shrink=0.8)

    major_ticks = np.arange(0, NUM_ANCILLA * n_rounds, n_rounds)
    major_labels = [f"Q{i+1}" for i in range(NUM_ANCILLA)]
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticklabels(major_labels, fontsize=8)
    ax.set_yticklabels(major_labels, fontsize=8)

    ax.set_xlabel('Ancilla Qubit - Round')
    ax.set_ylabel('Ancilla Qubit - Round')
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_comparison(corr_paems, corr_si1000, n_rounds=None):
    """Bar chart comparing correlation strengths by category."""
    if n_rounds is None:
        n_rounds = NUM_ROUNDS
    n = corr_paems.shape[0]

    # Classify correlations into timelike, spacelike, spacetime
    time_paems, space_paems, spacetime_paems = [], [], []
    time_si, space_si, spacetime_si = [], [], []

    for i in range(n):
        qi, ri = divmod(i, n_rounds)
        for j in range(i + 1, n):
            qj, rj = divmod(j, n_rounds)
            p_val = corr_paems[i, j]
            s_val = corr_si1000[i, j]

            if qi == qj:  # same qubit, different round → timelike
                time_paems.append(abs(p_val))
                time_si.append(abs(s_val))
            elif ri == rj:  # same round, different qubit → spacelike
                space_paems.append(abs(p_val))
                space_si.append(abs(s_val))
            else:  # different qubit AND round → spacetime
                spacetime_paems.append(abs(p_val))
                spacetime_si.append(abs(s_val))

    categories = ['Timelike\n(same qubit)', 'Spacelike\n(same round)', 'Spacetime\n(cross)']
    paems_means = [np.mean(time_paems), np.mean(space_paems), np.mean(spacetime_paems)]
    si_means = [np.mean(time_si), np.mean(space_si), np.mean(spacetime_si)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Bar chart
    ax = axes[0]
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, paems_means, w, label='PAEMS', color='#2196F3')
    ax.bar(x + w/2, si_means, w, label='SI1000', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Mean |correlation|')
    ax.set_title('(a) Correlation Strength')
    ax.legend()
    ax.set_yscale('log')

    # (b) PAEMS heatmap (small)
    ax = axes[1]
    cax = ax.matshow(corr_paems, interpolation='nearest', cmap='RdBu_r',
                     vmin=-0.02, vmax=0.08, aspect='auto')
    ax.set_title('(b) PAEMS', fontsize=12, pad=10)
    ax.set_xlabel('Detector index')
    ax.set_ylabel('Detector index')

    # (c) SI1000 heatmap (small)
    ax = axes[2]
    cax2 = ax.matshow(corr_si1000, interpolation='nearest', cmap='RdBu_r',
                      vmin=-0.02, vmax=0.08, aspect='auto')
    ax.set_title('(c) SI1000 (p=0.02)', fontsize=12, pad=10)
    ax.set_xlabel('Detector index')

    fig.colorbar(cax2, ax=axes[2], label=r'$p_{ij}$', shrink=0.6)

    plt.suptitle('PAEMS vs SI1000: Detection Event Correlations (IBM Brisbane)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: comparison.png")

    # Print summary
    print()
    print("  Correlation strength comparison:")
    print(f"  {'Category':<20} {'PAEMS':>12} {'SI1000':>12}")
    print("  " + "-" * 46)
    for cat, p, s in zip(categories, paems_means, si_means):
        cat_clean = cat.replace('\n', ' ')
        print(f"  {cat_clean:<20} {p:>12.6f} {s:>12.6f}")


# ─── Main ───
if __name__ == '__main__':
    print("=" * 60)
    print("PAEMS vs SI1000 Visualization")
    print("=" * 60)

    # 1. Run PAEMS
    print("\n[1/3] PAEMS simulation")
    paems_params = load_paems_params()
    corr_paems, n_rounds_eff = run_sim(paems_params, "PAEMS")

    # 2. Run SI1000
    print("\n[2/3] SI1000 simulation")
    si1000_params = make_si1000_params(p=0.02)
    corr_si1000, _ = run_sim(si1000_params, "SI1000")

    # Update for plotting with effective rounds
    NUM_ROUNDS_PLOT = n_rounds_eff

    # 3. Plot
    print("\n[3/3] Generating plots...")
    plot_correlation_heatmap(corr_paems, 'PAEMS Correlation Matrix (IBM Brisbane)', 'corr_paems.png', NUM_ROUNDS_PLOT)
    plot_correlation_heatmap(corr_si1000, 'SI1000 Correlation Matrix (p=0.02)', 'corr_si1000.png', NUM_ROUNDS_PLOT)
    plot_comparison(corr_paems, corr_si1000, NUM_ROUNDS_PLOT)

    print(f"\nAll plots saved to: {OUT_DIR}/")
