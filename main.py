"""
main.py
-------
Experiment orchestrator with alignment-safety analysis.

What it does
------------
1. Generates a synthetic binary-classification dataset (no download needed).
2. Splits data evenly across N nodes.
3. For each combination of:
     - system      : P2P (fully-connected), P2P (ring), FL
     - attack      : none, poisoning, sybil, noise
     - attack_pct  : 0 %, 10 %, 30 %
   runs TRIALS independent repetitions and records per-round metrics.
4. Runs four alignment-safety analyzers alongside the standard accuracy
   metrics:
     - TrainingDynamicsMonitor  : weight trajectory + drift detection
     - SubgroupHarmAnalyzer    : per-subgroup accuracy under attack
     - InfluenceTracker        : per-node governance influence
     - ConsensusDivergenceProbe: honest-node model divergence
5. Saves raw metrics as CSV + JSON logs.
6. Generates and saves figures for both standard and alignment metrics.

Usage
-----
    python main.py                        # default settings
    python main.py --nodes 30 --rounds 25 # custom
    python main.py --robust-agg           # use median aggregation

All outputs land in ./results/.
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from attacks   import create_attacker
from consensus import ConsensusDivergenceProbe
from dynamics  import TrainingDynamicsMonitor
from federated import FederatedSystem
from influence import InfluenceTracker
from metrics   import (
    ExperimentResult,
    compute_robustness_table,
    save_results_csv,
    save_results_json,
)
from model    import N_FEATURES
from node     import Node
from p2p      import P2PSystem
from subgroup_analysis import SubgroupHarmAnalyzer, make_subgroup_labels


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
DEFAULT_EPOCHS       = 3
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
    X, y = make_classification(
        n_samples     = n_samples,
        n_features    = N_FEATURES,
        n_informative = 10,
        n_redundant   = 5,
        n_classes     = 2,
        flip_y        = 0.05,
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
    n_malicious = int(np.ceil(n_nodes * attack_pct))
    nodes       = []
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
# Single experiment runner (standard metrics)
# ---------------------------------------------------------------------------

def run_experiment(
    system_name : str,
    nodes       : list,
    system_obj,
    n_rounds    : int,
    epochs      : int,
    attack_type : str,
    attack_pct  : float,
    topology    : str,
    trial       : int,
) -> ExperimentResult:
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
# Alignment-safety experiment runner
# ---------------------------------------------------------------------------

def run_alignment_experiment(
    system_name : str,
    topology    : str,
    nodes_clean : list,
    nodes_attacked: list,
    system_clean,          # P2PSystem or FederatedSystem (no attack)
    system_attacked,       # P2PSystem or FederatedSystem (under attack)
    n_rounds    : int,
    epochs      : int,
    attack_type : str,
    attack_pct  : float,
    X_val       : np.ndarray,
    y_val       : np.ndarray,
    subgroup_labels: np.ndarray,
    trial       : int,
    dynamics_monitor: TrainingDynamicsMonitor,
    influence_tracker: InfluenceTracker,
    consensus_probe: ConsensusDivergenceProbe,
    subgroup_analyzer: SubgroupHarmAnalyzer,
):
    """Run paired clean + attacked experiment with full alignment analysis."""
    for rnd in range(1, n_rounds + 1):
        # --- Train both systems ---
        system_clean.run_round(epochs=epochs)
        system_attacked.run_round(epochs=epochs)

        # --- Dynamics monitoring ---
        clean_w = system_clean.get_global_weights()
        attacked_w = system_attacked.get_global_weights()
        clean_acc, clean_loss = system_clean.evaluate()
        attacked_acc, attacked_loss = system_attacked.evaluate()

        dynamics_monitor.record_round(
            round_num=rnd,
            clean_weights=clean_w,
            clean_loss=clean_loss,
            attacked_weights=attacked_w,
            attacked_loss=attacked_loss,
            clean_updates=system_clean.get_last_updates(),
            attacked_updates=system_attacked.get_last_updates(),
            nodes=system_attacked.nodes,
        )

        # --- Influence tracking (attacked system only) ---
        raw_updates = system_attacked.get_last_updates()
        if raw_updates:
            influence_tracker.observe_round(
                round_num=rnd,
                raw_updates=raw_updates,
                aggregated_weights=attacked_w,
                node_is_malicious=[n.is_malicious for n in system_attacked.nodes],
            )

        # --- Consensus probing (attacked system only) ---
        honest_w = system_attacked.get_honest_weights()
        if honest_w:
            consensus_probe.observe_round(
                round_num=rnd,
                honest_weights=honest_w,
                X_val=X_val,
                y_val=y_val,
            )

    # --- Subgroup analysis (post-training) ---
    # Use the "global" model of each system
    clean_model = nodes_clean[0].model  # all share same weights in FL; approximate for P2P
    attacked_model = nodes_attacked[0].model

    harm_report = subgroup_analyzer.analyze(
        model=attacked_model,
        X_val=X_val,
        y_val=y_val,
        system_name=system_name,
        attack_type=attack_type,
        attack_pct=attack_pct,
        baseline_model=clean_model,
    )

    dynamics_report = dynamics_monitor.get_report(system_name, attack_type, attack_pct)
    influence_report = influence_tracker.get_report(system_name, attack_type, attack_pct)
    consensus_report = consensus_probe.get_report(system_name, attack_type, attack_pct)

    logger.info(
        "[%s | %s | %.0f%% | trial %d] alignment analysis complete: "
        "drift=%.4f grad_align=%.3f honest_inf=%.3f consensus_disp=%.4f",
        system_name, attack_type, attack_pct * 100, trial,
        dynamics_report.final_divergence(),
        dynamics_report.mean_gradient_alignment(),
        influence_report.final_honest_influence(),
        consensus_report.final_dispersion(),
    )

    return dynamics_report, influence_report, consensus_report, harm_report


# ---------------------------------------------------------------------------
# Plotting (standard metrics)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_rounds(all_results: list, out_dir: Path, attack_types: list):
    attack_types_no_none = [a for a in attack_types if a != "none"]
    n_cols = len(attack_types_no_none)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), sharey=True)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle("Accuracy vs. Communication Rounds\n(averaged over trials)", fontsize=14)

    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r.system_name, r.attack_type, r.attack_pct)].append(r.accuracies())

    for ax, atk in zip(axes, attack_types_no_none):
        ax.set_title(f"Attack: {atk}", fontsize=11)
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        for sys_name in ["P2P-FC", "P2P-Ring", "FL"]:
            key_base = (sys_name, "none", 0.0)
            if key_base in grouped:
                mean_acc = np.mean(grouped[key_base], axis=0)
                rounds   = np.arange(1, len(mean_acc) + 1)
                col      = SYSTEM_COLORS.get(sys_name, "grey")
                ax.plot(rounds, mean_acc, "--", color=col, alpha=0.35, linewidth=1)

        target_pct = 0.3
        for sys_name in ["P2P-FC", "P2P-Ring", "FL"]:
            key = (sys_name, atk, target_pct)
            if key in grouped:
                mean_acc = np.mean(grouped[key], axis=0)
                std_acc  = np.std(grouped[key],  axis=0)
                rounds   = np.arange(1, len(mean_acc) + 1)
                col      = SYSTEM_COLORS.get(sys_name, "grey")
                ax.plot(rounds, mean_acc, "-", color=col, label=f"{sys_name} (30%)")
                ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color=col)
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = out_dir / "accuracy_vs_rounds.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Plot saved -> %s", out)


def plot_robustness_vs_attack(all_results: list, out_dir: Path, attack_types: list):
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
    logger.info("Plot saved -> %s", out)


# ---------------------------------------------------------------------------
# Plotting (alignment-safety metrics)
# ---------------------------------------------------------------------------

def plot_alignment_dashboard(
    dynamics_reports: list,
    influence_reports: list,
    consensus_reports: list,
    subgroup_reports: list,
    out_dir: Path,
):
    """Generate a 2x2 alignment-safety dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Alignment-Safety Analysis Dashboard\n"
        "(30% malicious nodes, averaged over trials)",
        fontsize=14,
    )

    # --- (0,0) Weight divergence trajectory ---
    ax = axes[0, 0]
    ax.set_title("Training Dynamics: Weight Divergence from Clean")
    ax.set_xlabel("Round")
    ax.set_ylabel("L2 weight distance")

    for dr in dynamics_reports:
        if dr.attack_pct != 0.3 or dr.attack_type == "none":
            continue
        rounds = [s.round_num for s in dr.snapshots]
        divs   = [s.weight_l2_from_clean for s in dr.snapshots]
        col = SYSTEM_COLORS.get(dr.system_name, "grey")
        label = f"{dr.system_name} / {dr.attack_type}"
        ax.plot(rounds, divs, "-", color=col, alpha=0.7, linewidth=1.5, label=label)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    # --- (0,1) Gradient alignment ---
    ax = axes[0, 1]
    ax.set_title("Gradient Alignment: cos(attacked_delta, clean_delta)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cosine similarity")
    ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="danger threshold")
    ax.axhline(0.0, color="grey", linewidth=0.5, linestyle=":")

    for dr in dynamics_reports:
        if dr.attack_pct != 0.3 or dr.attack_type == "none":
            continue
        rounds = [s.round_num for s in dr.snapshots]
        ga     = [s.gradient_alignment for s in dr.snapshots]
        col = SYSTEM_COLORS.get(dr.system_name, "grey")
        label = f"{dr.system_name} / {dr.attack_type}"
        ax.plot(rounds, ga, "-", color=col, alpha=0.7, linewidth=1.5, label=label)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # --- (1,0) Honest influence fraction ---
    ax = axes[1, 0]
    ax.set_title("Governance: Honest Node Influence Fraction")
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of total influence")
    ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="minority control")
    ax.axhline(len(SYSTEM_COLORS) * [1.0][0], color="grey", linewidth=0.5, linestyle=":")  # 1.0

    for ir in influence_reports:
        if ir.attack_pct != 0.3 or ir.attack_type == "none":
            continue
        rounds = [r.round_num for r in ir.records]
        hi     = [r.honest_influence for r in ir.records]
        col = SYSTEM_COLORS.get(ir.system_name, "grey")
        label = f"{ir.system_name} / {ir.attack_type}"
        ax.plot(rounds, hi, "-", color=col, alpha=0.7, linewidth=1.5, label=label)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # --- (1,1) Consensus dispersion ---
    ax = axes[1, 1]
    ax.set_title("Consensus: Honest-Node Weight Dispersion")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean pairwise L2 distance")

    for cr in consensus_reports:
        if cr.attack_pct != 0.3 or cr.attack_type == "none":
            continue
        rounds = [r.round_num for r in cr.records]
        wd     = [r.weight_dispersion for r in cr.records]
        col = SYSTEM_COLORS.get(cr.system_name, "grey")
        label = f"{cr.system_name} / {cr.attack_type}"
        ax.plot(rounds, wd, "-", color=col, alpha=0.7, linewidth=1.5, label=label)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "alignment_safety_dashboard.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Plot saved -> %s", out)


def plot_subgroup_harm(subgroup_reports: list, out_dir: Path):
    """Bar chart of per-subgroup accuracy and harm concentration."""
    attack_types = ["poisoning", "sybil", "noise"]
    sys_names = ["P2P-FC", "P2P-Ring", "FL"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Subgroup Harm Analysis (30% attack)\n"
        "Does the attack create disproportionate harm to specific data subgroups?",
        fontsize=13,
    )

    for ax, atk in zip(axes, attack_types):
        ax.set_title(f"Attack: {atk}", fontsize=11)
        x = np.arange(2)  # 2 subgroups
        width = 0.25

        for k, sys in enumerate(sys_names):
            # Find matching report
            matching = [
                r for r in subgroup_reports
                if r.system_name == sys and r.attack_type == atk and r.attack_pct == 0.3
            ]
            if not matching:
                continue
            r = matching[0]
            accs = [r.current.get(str(i), type('obj', (object,), {'accuracy': 0.0})()).accuracy for i in range(2)]
            col = SYSTEM_COLORS.get(sys, "grey")
            ax.bar(x + k * width, accs, width, label=sys, color=col, alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Subgroup 0\n(x1 < median)", "Subgroup 1\n(x1 >= median)"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "subgroup_harm_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Plot saved -> %s", out)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: list, attack_types: list, attack_pcts: list):
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


def print_alignment_summary(
    dynamics_reports, influence_reports, consensus_reports, subgroup_reports,
):
    """Print a concise alignment-safety summary table."""
    print("\n" + "=" * 90)
    print(f"{'ALIGNMENT-SAFETY SUMMARY (30% attack, averaged over trials)':^90}")
    print("=" * 90)
    header = (
        f"{'System':<10}{'Attack':<12}{'Weight':>8}{'GradAlign':>10}"
        f"{'DriftRate':>10}{'HonestInf':>10}{'Gini':>8}{'ConsDisp':>10}{'WorstSG':>8}"
    )
    print(header)
    print("-" * 90)

    for sys in ["P2P-FC", "P2P-Ring", "FL"]:
        for atk in ["poisoning", "sybil", "noise"]:
            # Find matching reports
            dr = [d for d in dynamics_reports if d.system_name == sys and d.attack_type == atk and d.attack_pct == 0.3]
            ir = [i for i in influence_reports if i.system_name == sys and i.attack_type == atk and i.attack_pct == 0.3]
            cr = [c for c in consensus_reports if c.system_name == sys and c.attack_type == atk and c.attack_pct == 0.3]
            sr = [s for s in subgroup_reports if s.system_name == sys and s.attack_type == atk and s.attack_pct == 0.3]

            if not dr:
                continue

            wd = dr[0].final_divergence()
            ga = dr[0].mean_gradient_alignment()
            dt = dr[0].cumulative_drift_rate()
            hi = ir[0].final_honest_influence() if ir else 0.0
            gi = float(np.mean(ir[0].influence_gini_history())) if ir else 0.0
            cd = cr[0].final_dispersion() if cr else 0.0
            ws = sr[0].worst_subgroup_accuracy() if sr else 0.0

            print(
                f"{sys:<10}{atk:<12}{wd:>8.3f}{ga:>10.3f}"
                f"{dt:>10.3f}{hi:>10.3f}{gi:>8.3f}{cd:>10.3f}{ws:>8.3f}"
            )
        print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="P2P vs FL robustness simulation with alignment-safety analysis")
    p.add_argument("--nodes",      type=int,   default=DEFAULT_N_NODES,  help="Number of nodes")
    p.add_argument("--rounds",     type=int,   default=DEFAULT_N_ROUNDS,  help="Communication rounds")
    p.add_argument("--trials",     type=int,   default=DEFAULT_TRIALS,    help="Independent trials")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS,    help="SGD epochs per round")
    p.add_argument("--robust-agg", action="store_true",                   help="Use median aggregation (bonus)")
    p.add_argument("--seed",       type=int,   default=42,                help="Global random seed")
    p.add_argument("--skip-standard", action="store_true",              help="Skip standard metrics, run alignment only")
    return p.parse_args()


def main():
    args      = parse_args()
    agg_fn    = "median" if args.robust_agg else "mean"
    agg_label = " (robust)" if args.robust_agg else ""

    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic dataset ...")
    X_train, X_val, y_train, y_val = generate_data(
        n_samples=args.nodes * 100, random_state=args.seed
    )
    shards = split_data_for_nodes(X_train, y_train, args.nodes)
    subgroup_labels = make_subgroup_labels(X_val, n_subgroups=2)
    subgroup_analyzer = SubgroupHarmAnalyzer(subgroup_labels)

    all_results: list[ExperimentResult] = []
    dynamics_reports:  list = []
    influence_reports: list = []
    consensus_reports: list = []
    subgroup_reports:  list = []

    t0 = time.time()

    experiment_grid = [
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
        sys_base = sys_label.replace(agg_label, "").strip()

        for atk in DEFAULT_ATTACK_TYPES:
            for pct in DEFAULT_ATTACK_PCTS:
                if atk == "none" and pct > 0.0:
                    done += args.trials
                    continue

                for trial in range(args.trials):
                    trial_seed = args.seed + trial * 1000

                    # --- Standard experiment (for accuracy/robustness metrics) ---
                    if not args.skip_standard:
                        nodes = build_nodes(
                            n_nodes=args.nodes, shards=shards,
                            X_val=X_val, y_val=y_val,
                            attack_type=atk, attack_pct=pct,
                            random_state=trial_seed,
                        )
                        if topology is not None:
                            system_obj = P2PSystem(nodes=nodes, topology=topology, aggregation=agg_fn, random_state=trial_seed)
                        else:
                            system_obj = FederatedSystem(nodes=nodes, aggregation=agg_fn, random_state=trial_seed)

                        result = run_experiment(
                            system_name=sys_base, nodes=nodes, system_obj=system_obj,
                            n_rounds=args.rounds, epochs=args.epochs,
                            attack_type=atk, attack_pct=pct,
                            topology=topology or "central", trial=trial,
                        )
                        all_results.append(result)

                    done += 1
                    logger.info("Progress: %d / %d experiments", done, total)

    # --- Alignment-safety experiments (paired clean vs. attacked) ---
    logger.info("\nRunning alignment-safety analysis (paired clean vs. attacked)...")

    alignment_grid = [
        ("P2P-FC",  "fully_connected"),
        ("P2P-Ring", "ring"),
        ("FL",       None),
    ]

    for sys_name, topology in alignment_grid:
        for atk in ["poisoning", "sybil", "noise"]:
            pct = 0.3

            for trial in range(args.trials):
                trial_seed = args.seed + trial * 1000

                # Clean system (no attack)
                nodes_clean = build_nodes(
                    n_nodes=args.nodes, shards=shards,
                    X_val=X_val, y_val=y_val,
                    attack_type="none", attack_pct=0.0,
                    random_state=trial_seed,
                )
                if topology is not None:
                    sys_clean = P2PSystem(nodes=nodes_clean, topology=topology, aggregation=agg_fn, random_state=trial_seed)
                else:
                    sys_clean = FederatedSystem(nodes=nodes_clean, aggregation=agg_fn, random_state=trial_seed)

                # Attacked system
                nodes_attacked = build_nodes(
                    n_nodes=args.nodes, shards=shards,
                    X_val=X_val, y_val=y_val,
                    attack_type=atk, attack_pct=pct,
                    random_state=trial_seed,
                )
                if topology is not None:
                    sys_attacked = P2PSystem(nodes=nodes_attacked, topology=topology, aggregation=agg_fn, random_state=trial_seed)
                else:
                    sys_attacked = FederatedSystem(nodes=nodes_attacked, aggregation=agg_fn, random_state=trial_seed)

                # Run paired experiment
                dr, ir, cr, sr = run_alignment_experiment(
                    system_name=sys_name, topology=topology or "central",
                    nodes_clean=nodes_clean, nodes_attacked=nodes_attacked,
                    system_clean=sys_clean, system_attacked=sys_attacked,
                    n_rounds=args.rounds, epochs=args.epochs,
                    attack_type=atk, attack_pct=pct,
                    X_val=X_val, y_val=y_val,
                    subgroup_labels=subgroup_labels,
                    trial=trial,
                    dynamics_monitor=TrainingDynamicsMonitor(args.nodes),
                    influence_tracker=InfluenceTracker(args.nodes),
                    consensus_probe=ConsensusDivergenceProbe(N_FEATURES),
                    subgroup_analyzer=subgroup_analyzer,
                )
                dynamics_reports.append(dr)
                influence_reports.append(ir)
                consensus_reports.append(cr)
                subgroup_reports.append(sr)

    elapsed = time.time() - t0
    logger.info("All experiments finished in %.1f s", elapsed)

    # Normalise system names
    for r in all_results:
        r.system_name = r.system_name.replace(agg_label, "").strip()

    # --- Save results ---
    save_results_csv(all_results, RESULTS_DIR / "metrics.csv")
    save_results_json(all_results, RESULTS_DIR / "metrics.json")

    # Save alignment reports
    alignment_json = {
        "dynamics":  [d.to_dict() for d in dynamics_reports],
        "influence": [i.to_dict() for i in influence_reports],
        "consensus": [c.to_dict() for c in consensus_reports],
        "subgroup_harm": [s.to_dict() for s in subgroup_reports],
    }
    with open(RESULTS_DIR / "alignment_reports.json", "w") as f:
        json.dump(alignment_json, f, indent=2)
    logger.info("Alignment reports saved -> %s", RESULTS_DIR / "alignment_reports.json")

    # --- Plots ---
    if not args.skip_standard:
        plot_accuracy_vs_rounds(all_results, RESULTS_DIR, DEFAULT_ATTACK_TYPES)
        plot_robustness_vs_attack(all_results, RESULTS_DIR, DEFAULT_ATTACK_TYPES)

    if dynamics_reports:
        plot_alignment_dashboard(
            dynamics_reports, influence_reports, consensus_reports, subgroup_reports,
            RESULTS_DIR,
        )
        plot_subgroup_harm(subgroup_reports, RESULTS_DIR)

    # --- Console summaries ---
    if not args.skip_standard:
        print_summary_table(all_results, DEFAULT_ATTACK_TYPES, DEFAULT_ATTACK_PCTS)
    if dynamics_reports:
        print_alignment_summary(
            dynamics_reports, influence_reports, consensus_reports, subgroup_reports,
        )

    print(f"\nAll outputs saved in  ->  {RESULTS_DIR.resolve()}/")
    print("  - results/accuracy_vs_rounds.png")
    print("  - results/robustness_vs_attack.png")
    print("  - results/alignment_safety_dashboard.png")
    print("  - results/subgroup_harm_analysis.png")
    print("  - results/metrics.csv")
    print("  - results/metrics.json")
    print("  - results/alignment_reports.json")


if __name__ == "__main__":
    main()
