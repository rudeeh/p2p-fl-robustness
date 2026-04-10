"""
main.py
-------
Experiment orchestrator.

What it does
------------
1. Generates a synthetic binary-classification dataset (no download needed).
2. Splits data evenly across N nodes.
3. For each combination of:
     • system      : P2P (fully-connected), P2P (ring), FL
     • attack      : none, poisoning, sybil, noise
     • attack_pct  : 0 %, 10 %, 30 %
   runs TRIALS independent repetitions and records per-round metrics.
4. Saves raw metrics as CSV + JSON logs.
5. Generates and saves two figures:
     • accuracy_vs_rounds.png  – one subplot per attack type, P2P vs FL curves
     • robustness_vs_attack.png – bar chart of robustness score vs attack %

Usage
-----
    python main.py                        # default settings
    python main.py --nodes 30 --rounds 25 # custom
    python main.py --robust-agg           # use median aggregation (bonus)

All outputs land in ./results/.
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless backend – no display required
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from attacks   import create_attacker
from federated import FederatedSystem
from metrics   import (
    ExperimentResult,
    compute_robustness_table,
    save_results_csv,
    save_results_json,
)
from model  import N_FEATURES
from node   import Node
from p2p    import P2PSystem


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_N_NODES      = 20
DEFAULT_N_ROUNDS     = 20
DEFAULT_TRIALS       = 5
DEFAULT_EPOCHS       = 3          # SGD epochs per round per node
DEFAULT_ATTACK_PCTS  = [0.0, 0.1, 0.3]
DEFAULT_ATTACK_TYPES = ["none", "poisoning", "sybil", "noise"]
DEFAULT_TOPOLOGIES   = ["fully_connected", "ring"]
RESULTS_DIR          = Path("results")

SYSTEM_COLORS = {
    "P2P-FC"  : "#2196F3",
    "P2P-Ring": "#FF9800",
    "FL"      : "#4CAF50",
    "P2P-FC (robust)" : "#1565C0",
    "P2P-Ring (robust)": "#E65100",
    "FL (robust)"      : "#1B5E20",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def generate_data(n_samples: int = 2000, random_state: int = 42):
    """
    Create a linearly-separable synthetic binary-classification dataset
    and return (X_train, X_val, y_train, y_val) after standardisation.
    """
    X, y = make_classification(
        n_samples     = n_samples,
        n_features    = N_FEATURES,
        n_informative = 10,
        n_redundant   = 5,
        n_classes     = 2,
        flip_y        = 0.05,       # slight label noise for realism
        random_state  = random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler  = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val


def split_data_for_nodes(X: np.ndarray, y: np.ndarray, n_nodes: int):
    """Split training data into n_nodes roughly equal shards."""
    indices = np.arange(len(X))
    shards  = np.array_split(indices, n_nodes)
    return [(X[s], y[s]) for s in shards]


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def build_nodes(
    n_nodes     : int,
    shards      : list,
    X_val       : np.ndarray,
    y_val       : np.ndarray,
    attack_type : str,
    attack_pct  : float,
    random_state: int,
    sybil_count : int = 3,
) -> list:
    """
    Instantiate `n_nodes` nodes, of which `ceil(n_nodes * attack_pct)` are
    malicious attackers of the given type.
    """
    n_malicious = int(np.ceil(n_nodes * attack_pct))
    nodes       = []

    # Shuffle assignment deterministically
    rng   = np.random.RandomState(random_state)
    order = rng.permutation(n_nodes)
    malicious_ids = set(order[:n_malicious])

    for i in range(n_nodes):
        X_i, y_i = shards[i]
        kwargs = dict(
            node_id      = i,
            X_train      = X_i,
            y_train      = y_i,
            X_val        = X_val,
            y_val        = y_val,
            random_state = random_state,
        )
        if i in malicious_ids and attack_type != "none":
            extra = {"sybil_count": sybil_count} if attack_type == "sybil" else {}
            nodes.append(create_attacker(attack_type, **kwargs, **extra))
        else:
            nodes.append(Node(**kwargs))

    return nodes


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    system_name : str,
    nodes       : list,
    system_obj,              # P2PSystem or FederatedSystem
    n_rounds    : int,
    epochs      : int,
    attack_type : str,
    attack_pct  : float,
    topology    : str,
    trial       : int,
) -> ExperimentResult:
    """Run `n_rounds` rounds and return an ExperimentResult."""
    result = ExperimentResult(
        system_name = system_name,
        attack_type = attack_type,
        attack_pct  = attack_pct,
        topology    = topology,
        trial       = trial,
    )

    for rnd in range(1, n_rounds + 1):
        system_obj.run_round(epochs=epochs)
        acc, loss = system_obj.evaluate()
        result.add_round(rnd, acc, loss)

    result.log_summary()
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_accuracy_vs_rounds(all_results: list, out_dir: Path, attack_types: list):
    """
    One row of subplots per attack type; each subplot shows accuracy curves
    averaged across trials for P2P-FC, P2P-Ring, and FL.
    """
    attack_types_no_none = [a for a in attack_types if a != "none"]
    n_cols = len(attack_types_no_none)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    fig.suptitle("Accuracy vs. Communication Rounds\n(averaged over trials)", fontsize=14)

    # Group results: key = (system_name, attack_type, attack_pct)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r.system_name, r.attack_type, r.attack_pct)].append(r.accuracies())

    for ax, atk in zip(axes, attack_types_no_none):
        ax.set_title(f"Attack: {atk}", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        # Draw baseline (no-attack) as dashed reference
        for sys_name in ["P2P-FC", "P2P-Ring", "FL"]:
            key_base = (sys_name, "none", 0.0)
            if key_base in grouped:
                mean_acc = np.mean(grouped[key_base], axis=0)
                rounds   = np.arange(1, len(mean_acc) + 1)
                col      = SYSTEM_COLORS.get(sys_name, "grey")
                ax.plot(rounds, mean_acc, "--", color=col, alpha=0.35, linewidth=1)

        # Draw attacked curves (0 %, 10 %, 30 %) – only 30 % for readability
        target_pct = 0.3
        for sys_name in ["P2P-FC", "P2P-Ring", "FL"]:
            key = (sys_name, atk, target_pct)
            if key in grouped:
                mean_acc = np.mean(grouped[key], axis=0)
                std_acc  = np.std(grouped[key],  axis=0)
                rounds   = np.arange(1, len(mean_acc) + 1)
                col      = SYSTEM_COLORS.get(sys_name, "grey")
                ax.plot(rounds, mean_acc, "-", color=col, label=f"{sys_name} (30 %)")
                ax.fill_between(
                    rounds,
                    mean_acc - std_acc,
                    mean_acc + std_acc,
                    alpha=0.15, color=col,
                )

        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = out_dir / "accuracy_vs_rounds.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", out)


def plot_robustness_vs_attack(all_results: list, out_dir: Path, attack_types: list):
    """
    Bar chart: robustness score vs. attack percentage, grouped by system.
    """
    attack_types_no_none = [a for a in attack_types if a != "none"]
    n_cols = len(attack_types_no_none)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    fig.suptitle("Robustness Score vs. Attack Percentage\n(1.0 = unaffected)", fontsize=14)

    rob_table = compute_robustness_table(all_results)
    pcts      = [0.1, 0.3]
    sys_names = ["P2P-FC", "P2P-Ring", "FL"]
    bar_width = 0.22
    x         = np.arange(len(pcts))

    for ax, atk in zip(axes, attack_types_no_none):
        ax.set_title(f"Attack: {atk}", fontsize=11)
        ax.set_xlabel("Malicious node fraction")
        ax.set_ylabel("Robustness score")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([f"{int(p*100)} %" for p in pcts])
        ax.grid(axis="y", alpha=0.3)

        for k, sys_name in enumerate(sys_names):
            scores = []
            for pct in pcts:
                try:
                    s = rob_table[sys_name][atk][pct]
                except KeyError:
                    s = 1.0
                scores.append(s)
            col = SYSTEM_COLORS.get(sys_name, "grey")
            ax.bar(x + k * bar_width, scores, bar_width, label=sys_name, color=col, alpha=0.85)

        ax.legend(fontsize=9)

    plt.tight_layout()
    out = out_dir / "robustness_vs_attack.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Plot saved → %s", out)


def print_summary_table(all_results: list, attack_types: list, attack_pcts: list):
    """Print a concise final summary to stdout."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r.system_name, r.attack_type, r.attack_pct)].append(r.final_accuracy())

    rob_table = compute_robustness_table(all_results)

    print("\n" + "=" * 74)
    print(f"{'FINAL SUMMARY':^74}")
    print("=" * 74)
    header = f"{'System':<12}{'Attack':<12}{'Pct':>5}  {'Acc (mean)':>10}  {'Robustness':>10}"
    print(header)
    print("-" * 74)

    for sys_name in ["P2P-FC", "P2P-Ring", "FL"]:
        for atk in attack_types:
            for pct in attack_pcts:
                key = (sys_name, atk, pct)
                if key not in grouped:
                    continue
                mean_acc = np.mean(grouped[key])
                if atk == "none":
                    rob = "-"
                else:
                    try:
                        rob = f"{rob_table[sys_name][atk][pct]:.3f}"
                    except KeyError:
                        rob = "-"
                print(f"{sys_name:<12}{atk:<12}{pct*100:>4.0f}%  {mean_acc:>10.4f}  {rob:>10}")
        print()
    print("=" * 74)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="P2P vs FL robustness simulation")
    p.add_argument("--nodes",      type=int,   default=DEFAULT_N_NODES,  help="Number of nodes")
    p.add_argument("--rounds",     type=int,   default=DEFAULT_N_ROUNDS,  help="Communication rounds")
    p.add_argument("--trials",     type=int,   default=DEFAULT_TRIALS,    help="Independent trials")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS,    help="SGD epochs per round")
    p.add_argument("--robust-agg", action="store_true",                   help="Use median aggregation (bonus)")
    p.add_argument("--seed",       type=int,   default=42,                help="Global random seed")
    return p.parse_args()


def main():
    args      = parse_args()
    agg_fn    = "median" if args.robust_agg else "mean"
    agg_label = " (robust)" if args.robust_agg else ""

    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Generate data once; nodes get fixed shards throughout all experiments
    # -----------------------------------------------------------------------
    logger.info("Generating synthetic dataset …")
    X_train, X_val, y_train, y_val = generate_data(
        n_samples=args.nodes * 100, random_state=args.seed
    )
    shards = split_data_for_nodes(X_train, y_train, args.nodes)

    all_results = []
    t0          = time.time()

    # -----------------------------------------------------------------------
    # Experiment loop
    # -----------------------------------------------------------------------
    experiment_grid = [
        # (system_label, topology_or_None)
        (f"P2P-FC{agg_label}",   "fully_connected"),
        (f"P2P-Ring{agg_label}", "ring"),
        (f"FL{agg_label}",       None),
    ]

    total = (
        len(experiment_grid)
        * len(DEFAULT_ATTACK_TYPES)
        * len(DEFAULT_ATTACK_PCTS)
        * args.trials
    )
    done = 0

    for sys_label, topology in experiment_grid:
        for atk in DEFAULT_ATTACK_TYPES:
            for pct in DEFAULT_ATTACK_PCTS:
                # Skip non-zero pct for "none" attack (all the same as pct=0)
                if atk == "none" and pct > 0.0:
                    done += args.trials
                    continue

                for trial in range(args.trials):
                    trial_seed = args.seed + trial * 1000

                    # Build fresh nodes for every trial
                    nodes = build_nodes(
                        n_nodes      = args.nodes,
                        shards       = shards,
                        X_val        = X_val,
                        y_val        = y_val,
                        attack_type  = atk,
                        attack_pct   = pct,
                        random_state = trial_seed,
                    )

                    # Instantiate simulation object
                    if topology is not None:
                        system_obj = P2PSystem(
                            nodes       = nodes,
                            topology    = topology,
                            aggregation = agg_fn,
                            random_state= trial_seed,
                        )
                    else:
                        system_obj = FederatedSystem(
                            nodes       = nodes,
                            aggregation = agg_fn,
                            random_state= trial_seed,
                        )

                    result = run_experiment(
                        system_name = sys_label,
                        nodes       = nodes,
                        system_obj  = system_obj,
                        n_rounds    = args.rounds,
                        epochs      = args.epochs,
                        attack_type = atk,
                        attack_pct  = pct,
                        topology    = topology or "central",
                        trial       = trial,
                    )
                    all_results.append(result)
                    done += 1
                    logger.info("Progress: %d / %d experiments", done, total)

    elapsed = time.time() - t0
    logger.info("All experiments finished in %.1f s", elapsed)

    # Normalise system names for robustness table (strip agg_label suffix)
    for r in all_results:
        r.system_name = r.system_name.replace(agg_label, "").strip()

    # -----------------------------------------------------------------------
    # Save logs
    # -----------------------------------------------------------------------
    save_results_csv(all_results, RESULTS_DIR / "metrics.csv")
    save_results_json(all_results, RESULTS_DIR / "metrics.json")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    plot_accuracy_vs_rounds(all_results, RESULTS_DIR, DEFAULT_ATTACK_TYPES)
    plot_robustness_vs_attack(all_results, RESULTS_DIR, DEFAULT_ATTACK_TYPES)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print_summary_table(all_results, DEFAULT_ATTACK_TYPES, DEFAULT_ATTACK_PCTS)

    print(f"\nAll outputs saved in  →  {RESULTS_DIR.resolve()}/")
    print("  • results/accuracy_vs_rounds.png")
    print("  • results/robustness_vs_attack.png")
    print("  • results/metrics.csv")
    print("  • results/metrics.json")


if __name__ == "__main__":
    main()
