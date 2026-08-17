# P2P vs Federated Learning — Robustness Under Adversarial Attacks

A Python **simulation framework** comparing the robustness of Peer-to-Peer (P2P)
and Federated Learning (FL) systems under three categories of adversarial attack,
with **alignment-safety analysis** that goes beyond aggregate accuracy.

> **No real networking stack is used.**
> Everything runs in-process with NumPy and scikit-learn — no sockets, Docker, or GPUs required.

---

## Why This Matters for AI Alignment

Standard robustness research asks: *"Does the model stay accurate under attack?"*

This framework asks a harder question: **"Does the model stay *aligned* under attack?"**

A model can be 93% accurate and still be dangerously misaligned if:

- The 7% failures are **concentrated on specific data subgroups** that the
  evaluation metrics don't monitor (hidden failure mode / specification gaming).
- The training trajectory was **silently steered toward a different solution**
  — one with similar loss but different behavioural properties (inner misalignment).
- Adversarial nodes gained **disproportionate governance influence** over the
  model's learning direction, meaning the protocol itself failed as a coordination
  mechanism, not just a robustness mechanism.
- In P2P systems, honest nodes **fragmented into disagreeing factions**,
  each internally consistent but mutually incompatible — a protocol-level
  alignment failure invisible to single-model evaluation.

The four alignment-safety analyzers in this framework detect these failure modes:

| Analyzer | What it detects | Alignment connection |
|----------|----------------|----------------------|
| `TrainingDynamicsMonitor` | Weight drift from clean path, gradient alignment drop | Inner misalignment: did training converge to the *right* solution or just *a* solution? |
| `SubgroupHarmAnalyzer` | Per-subgroup accuracy, harm concentration index | Specification gaming: does aggregate accuracy hide subgroup-targeted failures? |
| `InfluenceTracker` | Per-node governance influence, Gini coefficient, honest influence fraction | Protocol governance: does the aggregation rule give attackers disproportionate control? |
| `ConsensusDivergenceProbe` | Honest-node weight dispersion, prediction disagreement, model fragmentation | Systemic risk: does the protocol create a single point of failure (FL) or fragmented consensus (P2P)? |

---

## Project Structure

```
p2p-fl-robustness/
├── model.py              # LogisticModel wrapper (weights get/set, clipping)
├── node.py               # Base Node class (training, evaluation)
├── attacks.py            # Adversarial subclasses + factory
├── p2p.py                # P2P gossip simulation (fully-connected & ring)
├── federated.py          # Federated Learning simulation (FedAvg)
├── metrics.py            # Per-round tracking, robustness scoring, CSV/JSON export
├── dynamics.py           # TrainingDynamicsMonitor: weight trajectory + drift detection
├── subgroup_analysis.py  # SubgroupHarmAnalyzer: per-subgroup accuracy under attack
├── influence.py          # InfluenceTracker: per-node governance influence
├── consensus.py          # ConsensusDivergenceProbe: honest-node model divergence
├── main.py               # Experiment runner + plotting + alignment dashboard
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/rudeeh/p2p-fl-robustness.git
cd p2p-fl-robustness

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Simulation

```bash
# Full run: standard metrics + alignment-safety analysis
python main.py

# Quick run (fewer nodes/rounds/trials)
python main.py --nodes 10 --rounds 10 --trials 2

# Skip standard metrics, run alignment analysis only
python main.py --skip-standard

# Enable robust median aggregation
python main.py --robust-agg

# All options
python main.py --help
```

Typical runtime: **< 5 minutes** on any modern CPU.

---

## What Gets Simulated

### Systems
| Label     | Description |
|-----------|-------------|
| `P2P-FC`  | Peer-to-peer, fully-connected topology |
| `P2P-Ring`| Peer-to-peer, ring topology |
| `FL`      | Federated Learning with central aggregator (FedAvg) |

### Attack Types
| Attack      | Behaviour |
|-------------|-----------|
| `none`      | Baseline — all nodes honest |
| `poisoning` | Malicious nodes send negated weights |
| `sybil`     | Like poisoning but with amplified vote weight (x3) |
| `noise`     | Malicious nodes send random Gaussian weights |

### Configurations
- **Malicious node fractions:** 0 %, 10 %, 30 %
- **Trials:** 5 (independent seeds)
- **Rounds:** 20 per trial
- **Dataset:** synthetic binary classification (scikit-learn `make_classification`)

---

## Output

All outputs are saved to `results/`:

| File | Description |
|------|-------------|
| `accuracy_vs_rounds.png` | Accuracy curves per attack type (P2P vs FL) |
| `robustness_vs_attack.png` | Robustness score vs. malicious fraction |
| `alignment_safety_dashboard.png` | 4-panel dashboard: weight divergence, gradient alignment, honest influence, consensus dispersion |
| `subgroup_harm_analysis.png` | Per-subgroup accuracy under attack (does the attack target specific data subgroups?) |
| `metrics.csv` | Flat per-round log of every experiment |
| `metrics.json` | Structured JSON of standard results |
| `alignment_reports.json` | Full alignment-safety analysis (dynamics, influence, consensus, subgroup harm) |

### Standard Robustness Score

```
robustness_score = mean_accuracy_with_attack / mean_accuracy_without_attack
```

### Alignment-Safety Metrics

| Metric | What it measures | Danger signal |
|--------|-----------------|---------------|
| Weight divergence (L2) | How far attacked model drifted from clean path | Growing superlinearly = active attack |
| Gradient alignment (cosine) | Are clean and attacked updates going the same direction? | Drops below 0.5 = adversarial nodes steering training |
| Cumulative drift rate | Linear vs superlinear accumulation | > 1.0 = attack is amplifying over time |
| Honest influence fraction | Do honest nodes control the model's learning? | Below 0.5 = protocol governance failure |
| Influence Gini | Concentration of effective control | > 0.3 = oligarchic control, > 0.5 = near-monopoly |
| Consensus dispersion | How much honest nodes disagree (P2P) | High + increasing = fragmentation |
| Prediction disagreement | Fraction of inputs where honest nodes disagree | > 0.1 = the 'system' has no single behaviour |
| Worst subgroup accuracy | Accuracy of the worst-performing data subgroup | < 0.7 = subgroup-targeted harm |
| Harm concentration index | Is failure concentrated in specific subgroups? | > 0.3 = attack-induced disparate impact |
| Accuracy parity gap | Max - min accuracy across subgroups | > 0.1 = specification gaming risk |

---

## Key Findings (from simulation)

### 1. P2P Ring creates consensus risk that FL does not

FL enforces a single global model — consensus dispersion is always 0. P2P ring
allows honest nodes to diverge, especially under attack. This is a structural
trade-off: FL has a **single point of failure** (one corrupted model), while P2P
ring has **fragmentation risk** (multiple incompatible models). Neither is
inherently safer — the choice depends on which failure mode is worse for the
deployment context.

### 2. Sybil attacks exploit protocol governance, not just model quality

Sybil attacks don't just reduce accuracy — they reduce the **honest influence
fraction** to as low as 0.17 in P2P ring. This means the protocol's aggregation
mechanism gave attackers 83% of effective model control. The model's failures
are then a *governance* problem, not a *robustness* problem. Fixing this requires
changing the protocol (e.g., capped influence, identity verification), not just
improving the model.

### 3. Attacks create subgroup-targeted harm invisible to aggregate metrics

Under noise attacks, the **accuracy parity gap** reaches 0.11-0.22, and the
**harm concentration index** rises to 0.28-0.50. This means the attack
disproportionately degraded specific data subgroups while aggregate accuracy
dropped only moderately. In safety-critical deployments, this is the dangerous
scenario: the system looks fine on average but fails catastrophically for
specific inputs.

---

## Extending the Framework

### Add a new attack
```python
# attacks.py
class MyAttacker(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_malicious = True

    def get_update(self):
        # your malicious logic here
        return corrupted_weights
```
Register it in the `registry` dict inside `create_attacker()`.

### Add a new alignment analyzer

Each analyzer follows the same pattern:
1. Pure observation — never modifies training behaviour
2. `observe_round()` called once per round with exposed system state
3. `get_report()` returns a dataclass with `to_dict()` for JSON serialization
4. Integrated into `main.py`'s `run_alignment_experiment()`

### Add a new topology
```python
# p2p.py  ->  P2PSystem._build_topology()
elif self.topology == "small_world":
    ...
```

---

## Critical Failure Mitigations

| Failure Mode | Mitigation |
|---|---|
| Shape mismatch in model weights | Validated before aggregation; mismatched updates discarded |
| NaN / Inf from malicious updates | `np.isfinite` check; invalid updates discarded |
| Gradient explosion | Weights clipped to `[-10, 10]` after every SGD step |
| Non-reproducibility | Global seed + per-node seed offsets; deterministic attack RNGs |
| Division by zero | Total sample count checked before weighted average |
| Hidden subgroup harm | `SubgroupHarmAnalyzer` detects disparate impact invisible to aggregate metrics |
| Protocol governance failure | `InfluenceTracker` detects when attackers gain disproportionate control |
| Consensus fragmentation | `ConsensusDivergenceProbe` detects when honest nodes diverge into factions |
| Training trajectory manipulation | `TrainingDynamicsMonitor` detects when attacked training diverges from clean |

---

## References

- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
- Blanchard et al. (2017) — *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (robust aggregation)
- Douceur (2002) — *The Sybil Attack*
- Christiano et al. — *AI Safety via Debate* (motivates training trajectory monitoring)
- Amodei et al. (2016) — *Concrete Problems in AI Safety* (motivates subgroup harm analysis)
