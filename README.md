# Q-TRUST Pro — Quantum Circuit Trust Evaluation Engine

Paraconsistent trust evaluation for quantum hardware reliability.
Built by LoopEi LLC.

## Social Impact

Quantum computing is entering production systems that influence drug 
discovery, financial optimization, and cryptographic security. As QPU 
access scales through cloud platforms, one question becomes critical:

**Should you trust the results coming off quantum hardware?**

Wrong trust decisions have real human consequences. Q-TRUST provides 
the evaluation layer that catches hardware degradation, session 
instability, and reproducibility failure before results are acted upon.

## The Problem

Standard fidelity checks average across trials — masking hardware 
drift, session instability, and reproducibility failure. A circuit 
can look acceptable on paper while your QPU is degrading in real time.

Q-TRUST answers: **should this circuit be trusted to run on QPU right 
now?**

## Approach

Four metrics drive the evaluation:

- **Hellinger Fidelity** — distribution distance from clean baseline
- **MAD Drift** — trial-to-trial instability, robust to outliers
- **Reproducibility** — cosine similarity across consecutive trial pairs
- **Stability Index** — `mean_fidelity × (1 - drift) × reproducibility`

Paraconsistent trust states:

| State | Meaning |
|-------|---------|
| **T** | Trusted |
| **F** | Failed |
| **B** | Contradictory — hold for human review |

Ω stability tiers:

| Tier | Stability |
|------|-----------|
| Ω1_CERTIFIED | ≥ 0.90 |
| Ω2_OPERATIONAL | ≥ 0.70 |
| Ω3_UNSTABLE | < 0.70 |

## Hardware

FakeSherbrooke — real IBM Sherbrooke 127-qubit calibration data.
Baseline seed=42. Trial seeds=103, 211, 307 (prime-seeded).

## Installation
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime matplotlib numpy
```

## Usage
```bash
py lqtrust.py              # Standard run
py lqtrust.py --qubits 10  # Custom qubit count
py lqtrust.py --export     # Export JSON and CSV
py lqtrust.py --no-plot    # Skip plot
py lqtrust.py --providers  # Provider comparison demo
```

## Output
```
Circuit | Q | Fid    | Drift    | Repro  | Stab   | State | Tier
QFT     | 7 | 0.9366 | 0.004229 | 0.8931 | 0.8330 | B     | Ω2_OPERATIONAL
GHZ     | 8 | 0.7872 | 0.018777 | 0.9975 | 0.7705 | B     | Ω2_OPERATIONAL
RANDOM  | 8 | 0.6012 | 0.007595 | 0.9993 | 0.5963 | F     | Ω3_UNSTABLE
```

## License
MIT
