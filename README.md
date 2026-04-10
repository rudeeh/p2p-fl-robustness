# P2P vs Federated Learning — Robustness Under Adversarial Attacks

A Python **simulation framework** comparing the robustness of Peer-to-Peer (P2P)
and Federated Learning (FL) systems under three categories of adversarial attack.

> **No real networking stack is used.**  
> Everything runs in-process with NumPy and scikit-learn — no sockets, Docker, or GPUs required.

---

## 📁 Project Structure

```
p2p_fl_comparison/
├── model.py       # LogisticModel wrapper (weights get/set, clipping)
├── node.py        # Base Node class (training, evaluation)
├── attacks.py     # Adversarial subclasses + factory
├── p2p.py         # P2P gossip simulation (fully-connected & ring)
├── federated.py   # Federated Learning simulation (FedAvg)
├── metrics.py     # Per-round tracking, robustness scoring, CSV/JSON export
├── main.py        # Experiment runner + plotting
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/p2p-fl-robustness.git
cd p2p-fl-robustness

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Simulation

```bash
# Default run (20 nodes, 20 rounds, 5 trials, mean aggregation)
python main.py

# Custom node/round count
python main.py --nodes 30 --rounds 25

# Bonus: enable robust median aggregation
python main.py --robust-agg

# All options
python main.py --help
```

Typical runtime: **< 3 minutes** on any modern CPU.

---

## 🎯 What Gets Simulated

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
| `sybil`     | Like poisoning but with amplified vote weight (×3) |
| `noise`     | Malicious nodes send random Gaussian weights |

### Configurations
- **Malicious node fractions:** 0 %, 10 %, 30 %
- **Trials:** 5 (independent seeds)
- **Rounds:** 20 per trial
- **Dataset:** synthetic binary classification (scikit-learn `make_classification`)

---

## 📊 Output

All outputs are saved to `results/`:

| File | Description |
|------|-------------|
| `accuracy_vs_rounds.png` | Accuracy curves per attack type (P2P vs FL) |
| `robustness_vs_attack.png` | Robustness score vs. malicious fraction |
| `metrics.csv` | Flat per-round log of every experiment |
| `metrics.json` | Structured JSON of all results |

**Robustness score** is defined as:

```
robustness_score = mean_accuracy_with_attack / mean_accuracy_without_attack
```

A score of **1.0** means the attack had no measurable effect.  
A score of **0.0** means the system completely failed.

---

## 🛡️ Critical Failure Mitigations

| Failure Mode | Mitigation |
|---|---|
| Shape mismatch in model weights | Validated before aggregation; mismatched updates discarded |
| NaN / Inf from malicious updates | `np.isfinite` check; invalid updates discarded |
| Gradient explosion | Weights clipped to `[-10, 10]` after every SGD step |
| Non-reproducibility | Global seed + per-node seed offsets; deterministic attack RNGs |
| Division by zero | Total sample count checked before weighted average |

---

## 🔬 Extending the Framework

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

### Add a new topology
```python
# p2p.py  →  P2PSystem._build_topology()
elif self.topology == "small_world":
    ...
```

### Swap the ML model
Replace `LogisticModel` in `model.py` with any class exposing
`train()`, `get_weights()`, `set_weights()`, `score()`, and `loss()`.

---

## 📖 References

- McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
- Blanchard et al. (2017) — *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent* (robust aggregation)
- Douceur (2002) — *The Sybil Attack*

---

## 🧑‍💻 Author

Built as a Computer Networks course project.  
Simulation only — no real distributed systems used.
>>>>>>> 578e07a (Initial commit: P2P vs FL robustness simulation framework)
