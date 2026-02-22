"""
Q-TRUST Pro — Quantum Circuit Trust Evaluation Engine
Paraconsistent trust evaluation for quantum hardware reliability.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np
import json
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


# =============================================================================
# Backends
# =============================================================================
def get_clean_sim(seed=42):
    return AerSimulator(seed_simulator=seed)


def get_hardware_sim(seed=None):
    """FakeSherbrooke — calibrated from real IBM Sherbrooke 127q hardware."""
    return AerSimulator.from_backend(FakeSherbrooke(), seed_simulator=seed)


def get_synthetic_sim(error_1q=0.05, error_2q=0.10, readout=0.01, seed=None):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(error_1q, 1), ['h', 'x', 'sx', 'rz']
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(min(error_2q, 1.0), 2), ['cx']
    )
    nm.add_all_qubit_readout_error(
        ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
    )
    return AerSimulator(noise_model=nm, seed_simulator=seed)


# =============================================================================
# Circuit Library
# =============================================================================
def create_qft(n):
    qc = QuantumCircuit(n)
    qc.append(QFTGate(n), range(n))
    qc.measure_all()
    return qc


def create_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure_all()
    return qc


def create_random(n, depth=5, seed=42):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        q = int(rng.integers(0, n))
        qc.h(q) if rng.random() < 0.5 else qc.x(q)
    qc.measure_all()
    return qc


# =============================================================================
# Execution
# =============================================================================
def run_circuit(circuit, backend, shots=1024):
    compiled = transpile(circuit, backend, optimization_level=1)
    result = backend.run(compiled, shots=shots).result()
    counts = dict(result.get_counts())
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# =============================================================================
# Metrics
# =============================================================================
def align(a, b):
    keys = sorted(set(a) | set(b))
    pa = np.array([a.get(k, 0.0) for k in keys])
    pb = np.array([b.get(k, 0.0) for k in keys])
    return pa, pb


def fidelity(a, b):
    """Hellinger fidelity. Bounded [0,1], symmetric."""
    p, q = align(a, b)
    return float(np.sum(np.sqrt(p * q)) ** 2)


def drift(fid_array):
    """MAD drift. Robust to outliers."""
    fid_array = np.asarray(fid_array)
    if len(fid_array) <= 1:
        return 0.0
    return float(np.mean(np.abs(fid_array - np.mean(fid_array))))


def reproducibility(trials):
    """Mean cosine similarity across consecutive trial pairs."""
    if len(trials) <= 1:
        return 1.0
    scores = []
    for i in range(len(trials) - 1):
        p, q = align(trials[i], trials[i + 1])
        denom = np.linalg.norm(p) * np.linalg.norm(q)
        scores.append(float(np.dot(p, q) / denom) if denom > 0 else 0.0)
    return float(np.mean(scores))


def stability(mean_fid, d, rep):
    """Composite stability index."""
    return float(mean_fid * (1 - min(d, 1.0)) * rep)


def confidence_interval(values, z=1.96):
    """95% CI on fidelity across trials."""
    values = np.asarray(values)
    return float(z * np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0


# =============================================================================
# Paraconsistent Trust Logic
# T = Trusted | F = Failed | B = Contradictory (requires review)
# =============================================================================
def trust_state(mean_fid, stab):
    """
    Triadic trust evaluation.
    Classical binary systems force pass/fail resolution.
    B-state holds the contradiction explicitly — flags ambiguous
    signals for human review rather than forcing a decision.
    """
    if mean_fid >= 0.92 and stab >= 0.90:
        return "T"
    if mean_fid < 0.70 and stab < 0.60:
        return "F"
    return "B"


def stability_tier(stab):
    if stab >= 0.90:
        return "Ω1_CERTIFIED"
    if stab >= 0.70:
        return "Ω2_OPERATIONAL"
    return "Ω3_UNSTABLE"


# =============================================================================
# Result
# =============================================================================
@dataclass
class EvaluationResult:
    circuit: str
    qubits: int
    mean_fidelity: float
    fidelity_ci: float
    drift: float
    reproducibility: float
    stability: float
    state: str
    tier: str
    hardware: str
    timestamp: str


# =============================================================================
# Evaluator
# =============================================================================
def evaluate(circuit, label, qubits):
    """
    Baseline: clean noiseless sim (seed=42).
    Trials: FakeSherbrooke hardware-calibrated noise.
    Seeds: primes [103, 211, 307] — avoids correlated noise patterns.
    """
    baseline = run_circuit(circuit, get_clean_sim(seed=42))

    trials = [
        run_circuit(circuit, get_hardware_sim(seed=103)),
        run_circuit(circuit, get_hardware_sim(seed=211)),
        run_circuit(circuit, get_hardware_sim(seed=307)),
    ]

    fids = np.array([fidelity(baseline, t) for t in trials])
    mean_fid = float(fids.mean())
    d = drift(fids)
    rep = reproducibility(trials)
    stab = stability(mean_fid, d, rep)

    return EvaluationResult(
        circuit=label,
        qubits=qubits,
        mean_fidelity=round(mean_fid, 4),
        fidelity_ci=round(confidence_interval(fids), 6),
        drift=round(d, 6),
        reproducibility=round(rep, 4),
        stability=round(stab, 4),
        state=trust_state(mean_fid, stab),
        tier=stability_tier(stab),
        hardware="FakeSherbrooke — IBM Sherbrooke 127q calibration",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# Failure Case Demo
# =============================================================================
def demo_failure_case():
    print("\n=== FAILURE CASE — Hardware Degradation Mid-Session ===")

    qc = create_qft(7)
    baseline = run_circuit(qc, get_clean_sim(seed=42))

    trials = [
        run_circuit(qc, get_synthetic_sim(0.001, 0.002, seed=103)),
        run_circuit(qc, get_synthetic_sim(0.08,  0.16,  seed=211)),
        run_circuit(qc, get_synthetic_sim(0.20,  0.40,  seed=307)),
        run_circuit(qc, get_hardware_sim(seed=401)),
    ]

    fids = np.array([fidelity(baseline, t) for t in trials])
    mean_fid = float(fids.mean())
    d = drift(fids)
    rep = reproducibility(trials)
    stab = stability(mean_fid, d, rep)
    state = trust_state(mean_fid, stab)
    tier = stability_tier(stab)
    naive_pass = mean_fid >= 0.80

    print(f"  Circuit        : QFT-7 (7 qubits)")
    print(f"  Mean Fidelity  : {mean_fid:.4f}  <- averages out, looks acceptable")
    print(f"  Drift (MAD)    : {d:.6f}  <- elevated")
    print(f"  Reproducibility: {rep:.4f}")
    print(f"  Stability      : {stab:.4f}")
    print(f"  Naive Check    : {'PASS' if naive_pass else 'FAIL'}"
          "  <- fidelity alone misses the problem")
    print(f"  Trust State    : {state}  (T=Trusted | B=Contradictory | F=Failed)")
    print(f"  Tier           : {tier}")

    if state in ("F", "B") and naive_pass:
        print("  >>> Paraconsistent evaluation caught what naive fidelity missed.")
    elif not naive_pass and state in ("F", "B"):
        print("  >>> Both checks caught it — fidelity degraded enough to fail both.")
    elif state == "T":
        print("  >>> NOTE: still T — noise averaging too strong for QFT-7.")


# =============================================================================
# Provider Comparison Mock (optional — run with --providers)
# =============================================================================
def provider_comparison_mock():
    print("\n=== PROVIDER COMPARISON (MOCK) — Two Backends ===")

    qc = create_ghz(4)
    baseline = run_circuit(qc, get_clean_sim(seed=42))

    profiles = {
        "Provider_A_Premium": dict(error_1q=0.005, error_2q=0.010, readout=0.003),
        "Provider_B_Budget":  dict(error_1q=0.080, error_2q=0.160, readout=0.020),
    }

    for name, cfg in profiles.items():
        trials = [
            run_circuit(qc, get_synthetic_sim(**cfg, seed=103)),
            run_circuit(qc, get_synthetic_sim(**cfg, seed=211)),
            run_circuit(qc, get_synthetic_sim(**cfg, seed=307)),
        ]
        fids = np.array([fidelity(baseline, t) for t in trials])
        mean_fid = float(fids.mean())
        d = drift(fids)
        rep = reproducibility(trials)
        stab = stability(mean_fid, d, rep)
        state = trust_state(mean_fid, stab)
        tier = stability_tier(stab)

        print(f"{name}: stability={stab:.4f} | fid={mean_fid:.4f} | "
              f"TRUST SCORE: {state} — {tier}")


# =============================================================================
# Benchmark
# =============================================================================
def benchmark(max_qubits=8):
    circuits = {
        "QFT":    create_qft,
        "GHZ":    create_ghz,
        "RANDOM": create_random,
    }

    results = []
    print("\n=== Q-TRUST Pro Benchmark — FakeSherbrooke ===\n")
    print(f"{'Circuit':<10} | {'Q':<3} | {'Fid':<7} | {'±CI':<8} | "
          f"{'Drift':<8} | {'Repro':<7} | {'Stab':<7} | {'State':<5} | Tier")
    print("-" * 90)

    for name, builder in circuits.items():
        for n in range(2, max_qubits + 1):
            qc = builder(n)
            r = evaluate(qc, name, n)
            results.append(r)
            print(
                f"{r.circuit:<10} | {r.qubits:<3} | {r.mean_fidelity:<7.4f} | "
                f"±{r.fidelity_ci:<7.4f} | {r.drift:<8.6f} | "
                f"{r.reproducibility:<7.4f} | {r.stability:<7.4f} | "
                f"{r.state:<5} | {r.tier}"
            )
            print(f"TRUST SCORE: {r.state} — {r.tier}")

    return results


# =============================================================================
# Visualization
# =============================================================================
def plot_results(results, save_path="qtrust_pro_frontier.png"):
    circuit_types = list(dict.fromkeys(r.circuit for r in results))
    colors = {"QFT": "steelblue", "GHZ": "darkorange", "RANDOM": "seagreen"}
    state_colors = {"T": "green", "F": "red", "B": "gold"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for ct in circuit_types:
        subset = [r for r in results if r.circuit == ct]
        qubits = [r.qubits for r in subset]
        stabs = [r.stability for r in subset]
        fids = [r.mean_fidelity for r in subset]
        ax1.plot(qubits, stabs, marker='o', linewidth=2,
                 color=colors.get(ct, 'gray'), label=f"{ct} Stability")
        ax1.plot(qubits, fids, marker='s', linewidth=1.5, linestyle='--',
                 color=colors.get(ct, 'gray'), alpha=0.6,
                 label=f"{ct} Fidelity")

    ax1.axhline(y=0.90, color='darkgreen', linestyle=':', linewidth=1.2,
                label='Omega1 (0.90)')
    ax1.axhline(y=0.70, color='red', linestyle='--', linewidth=1.5,
                label='Omega2 (0.70)')
    ax1.set_title(
        "Q-TRUST Pro — Stability & Fidelity by Circuit Type\n"
        "(FakeSherbrooke — IBM Sherbrooke 127q calibration)",
        fontsize=10
    )
    ax1.set_xlabel("Qubit Count")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    for r in results:
        ax2.scatter(r.qubits, r.stability,
                    color=state_colors.get(r.state, 'gray'),
                    s=120, zorder=3)

    ax2.axhline(y=0.90, color='darkgreen', linestyle=':', linewidth=1.2)
    ax2.axhline(y=0.70, color='red', linestyle='--', linewidth=1.5)
    ax2.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='T — Trusted'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
               markersize=10, label='B — Contradictory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='F — Failed'),
    ], fontsize=8)
    ax2.set_title(
        "Q-TRUST Pro — Paraconsistent Trust States\n"
        "(T=Trusted | B=Contradictory | F=Failed)",
        fontsize=10
    )
    ax2.set_xlabel("Qubit Count")
    ax2.set_ylabel("Stability Index")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Plot saved -> " + save_path)


# =============================================================================
# Export
# FIX: UTF-8 encoding on CSV to handle Omega tier characters on Windows.
# =============================================================================
def export_results(results, prefix="qtrust_pro"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{prefix}_{ts}.json"
    csv_file = f"{prefix}_{ts}.csv"

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "hardware": "FakeSherbrooke — IBM Sherbrooke 127q calibration",
            "methodology": (
                "Baseline: clean AerSimulator (seed=42). "
                "Trials: FakeSherbrooke (seeds=103, 211, 307). "
                "Stability = mean_fidelity * (1 - MAD_drift) * reproducibility. "
                "Trust state: T/F/B triadic paraconsistent logic."
            ),
            "results": [asdict(r) for r in results]
        }, f, indent=2)

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"Exported -> {json_file}, {csv_file}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Q-TRUST Pro — Quantum Circuit Trust Evaluation"
    )
    parser.add_argument("--qubits", type=int, default=8)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--providers", action="store_true",
                        help="Run provider comparison mock demo")
    args = parser.parse_args()

    print("Q-TRUST Pro — Quantum Circuit Trust Evaluation Engine")
    print("Hardware: FakeSherbrooke — IBM Sherbrooke 127q calibration")

    results = benchmark(args.qubits)
    demo_failure_case()

    if args.providers:
        provider_comparison_mock()

    if not args.no_plot:
        plot_results(results)

    if args.export:
        export_results(results)


if __name__ == "__main__":
    main()
