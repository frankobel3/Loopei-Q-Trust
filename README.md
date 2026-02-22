# Q-TRUST Pro — Quantum Circuit Trust Evaluation Engine

**LoopEi LLC** | USPTO Patent Application #19/303,438  
**Inventor:** Franklyn Ernesto Beltré  
**CAGE Code:** 16GM4 | Service-Disabled Veteran-Owned Small Business (SDVOSB)  
**Web:** www.loopei.com | **Email:** frank@loopei.com

---

## The Problem

Standard fidelity checks average across trials — masking hardware drift, session instability, and reproducibility failure. A circuit can pass standard QA while your QPU is actively degrading in real time.

Quantum computing is entering production pipelines that influence drug discovery, financial optimization, and cryptographic security. Wrong trust decisions in those pipelines have real human consequences.

**Q-TRUST Pro answers one operational question:**

> *Should this circuit be trusted to run on QPU right now?*

---

## Approach

Q-TRUST Pro is built on the **LoopEi paraconsistent triadic logic framework** — a formal computational system that resolves contradictory measurement states without forcing binary collapse.

### Four Trust Metrics

| Metric | Description |
|---|---|
| **Hellinger Fidelity** | Distribution distance from clean baseline |
| **MAD Drift** | Trial-to-trial instability, robust to outliers |
| **Reproducibility** | Cosine similarity across consecutive trial pairs |
| **Stability Index** | `mean_fidelity × (1 − drift) × reproducibility` |

### Paraconsistent Trust States

| State | Meaning |
|---|---|
| **T** | Trusted — safe to act on |
| **F** | Failed — do not act on |
| **B** | Contradictory — hold for human review |

The **B-state** is Q-TRUST's key differentiator. Classical binary evaluators force T or F. Q-TRUST surfaces contradictory cases that would otherwise be silently misclassified — the exact failure mode that allows QPU degradation to propagate undetected into downstream systems.

### Ω Stability Tiers

| Tier | Threshold | Operational Meaning |
|---|---|---|
| **Ω1\_CERTIFIED** | ≥ 0.90 | Production-grade — act on results |
| **Ω2\_OPERATIONAL** | ≥ 0.70 | Proceed with caution |
| **Ω3\_UNSTABLE** | < 0.70 | Do not act on results |

---

## Validation Results

**Hardware:** FakeSherbrooke — real IBM Sherbrooke 127-qubit calibration data  
**Baseline seed:** 42 | **Trial seeds:** 103, 211, 307 (prime-seeded for independence)

| Circuit | Qubits | Fidelity | MAD Drift | Reproducibility | Stability | Trust | Tier |
|---|---|---|---|---|---|---|---|
| QFT | 7 | 0.9366 | 0.004229 | 0.8931 | 0.8330 | B | Ω2\_OPERATIONAL |
| GHZ | 8 | 0.7872 | 0.018777 | 0.9975 | 0.7705 | B | Ω2\_OPERATIONAL |
| RANDOM | 8 | 0.6012 | 0.007595 | 0.9993 | 0.5963 | F | Ω3\_UNSTABLE |

Full validation export: `qtrust_pro_20260221_115859.json` / `.csv`

---

## Social Impact — Q-volution Hackathon 2026 · Track D

Quantum hardware unreliability is not a future problem. It is happening now:

- **Drug discovery** — degraded QPUs produce silently corrupted molecular optimization results
- **Financial optimization** — unstable backends generate risk models built on noise
- **Cryptography** — drifting QPUs weaken key generation for populations that depend on it most

Q-TRUST Pro provides an accessible evaluation layer so researchers, developers, and institutions without large quantum teams can make informed, defensible decisions about when to trust their QPU results.

---

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

---

## License

Q-TRUST Pro is released under the **LoopEi Dual License Agreement**:

- **Academic & Non-Commercial Use** — Free with attribution. See `LICENSE`.
- **Commercial Use** — Requires a separate written agreement with LoopEi LLC.

Commercial use includes for-profit products, SaaS integration, production deployment, government contracting, and financial system integration.

> Unauthorized commercial use constitutes patent infringement under USPTO Application No. 19/303,438 and will be pursued to the full extent of the law.

**To obtain a commercial license:**  
frank@loopei.com | www.loopei.com | CAGE: 16GM4

---

© 2026 LoopEi LLC. All rights reserved.  
Patent Pending: USPTO Application No. 19/303,438  
Inventor: Franklyn Ernesto Beltré
