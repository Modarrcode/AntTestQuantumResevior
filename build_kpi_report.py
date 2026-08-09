"""
Build derived KPI report from Excel-ready episode results.

Input:
- multifunctional_rc_ae_model/excel_episode_results.csv

Outputs:
- multifunctional_rc_ae_model/excel_kpi_results.csv
- multifunctional_rc_ae_model/excel_kpi_overall.csv
"""

import argparse
import csv
import math
import os
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sample_std(xs):
    n = len(xs)
    if n <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def median(xs):
    if not xs:
        return 0.0
    arr = sorted(xs)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def mad(xs):
    if not xs:
        return 0.0
    med = median(xs)
    dev = [abs(x - med) for x in xs]
    return median(dev)


def clipped(xs, lower=-200.0, upper=200.0):
    return [min(upper, max(lower, x)) for x in xs]


def ci95_halfwidth(xs):
    n = len(xs)
    if n <= 1:
        return 0.0
    # Normal approximation for 95% CI
    return 1.96 * sample_std(xs) / math.sqrt(n)


def load_episode_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "friction": float(r["friction"]),
                    "episode": int(r["episode"]),
                    "policy": r["policy"],
                    "reward": float(r["reward"]),
                    "forward_distance": float(r["forward_distance"]),
                    "steps": int(r["steps"]),
                }
            )
    return rows


def build_kpis(rows):
    by_friction_episode = defaultdict(dict)
    for r in rows:
        key = (r["friction"], r["episode"])
        by_friction_episode[key][r["policy"]] = r

    per_friction = defaultdict(lambda: {
        "rc_reward": [],
        "cpg_reward": [],
        "rc_distance": [],
        "cpg_distance": [],
        "gap_distance_pct": [],
        "gap_reward_pct": [],
        "gap_distance_abs": [],
        "gap_reward_abs": [],
        "rc_beats_cpg_distance": 0,
        "rc_beats_cpg_reward": 0,
        "paired_count": 0,
    })

    for (friction, _ep), rec in by_friction_episode.items():
        if "multifunctional_rc_ae" not in rec or "cpg" not in rec:
            continue

        rc = rec["multifunctional_rc_ae"]
        cpg = rec["cpg"]

        agg = per_friction[friction]
        agg["paired_count"] += 1

        agg["rc_reward"].append(rc["reward"])
        agg["cpg_reward"].append(cpg["reward"])
        agg["rc_distance"].append(rc["forward_distance"])
        agg["cpg_distance"].append(cpg["forward_distance"])
        agg["gap_distance_abs"].append(rc["forward_distance"] - cpg["forward_distance"])
        agg["gap_reward_abs"].append(rc["reward"] - cpg["reward"])

        if rc["forward_distance"] > cpg["forward_distance"]:
            agg["rc_beats_cpg_distance"] += 1
        if rc["reward"] > cpg["reward"]:
            agg["rc_beats_cpg_reward"] += 1

        if abs(cpg["forward_distance"]) > 1e-9:
            agg["gap_distance_pct"].append((rc["forward_distance"] - cpg["forward_distance"]) / abs(cpg["forward_distance"]) * 100.0)
        if abs(cpg["reward"]) > 1e-9:
            agg["gap_reward_pct"].append((rc["reward"] - cpg["reward"]) / abs(cpg["reward"]) * 100.0)

    kpi_rows = []
    overall_collect = {
        "gap_distance_pct": [],
        "gap_reward_pct": [],
        "gap_distance_abs": [],
        "gap_reward_abs": [],
        "win_distance": 0,
        "win_reward": 0,
        "count": 0,
    }

    for friction in sorted(per_friction.keys()):
        agg = per_friction[friction]
        n = agg["paired_count"]

        rc_dist_mean = mean(agg["rc_distance"])
        cpg_dist_mean = mean(agg["cpg_distance"])
        rc_rew_mean = mean(agg["rc_reward"])
        cpg_rew_mean = mean(agg["cpg_reward"])

        row = {
            "friction": friction,
            "episodes_paired": n,
            "rc_distance_mean": rc_dist_mean,
            "rc_distance_std": sample_std(agg["rc_distance"]),
            "rc_distance_ci95_half": ci95_halfwidth(agg["rc_distance"]),
            "cpg_distance_mean": cpg_dist_mean,
            "cpg_distance_std": sample_std(agg["cpg_distance"]),
            "cpg_distance_ci95_half": ci95_halfwidth(agg["cpg_distance"]),
            "distance_gap_pct_mean": mean(agg["gap_distance_pct"]),
            "distance_gap_pct_std": sample_std(agg["gap_distance_pct"]),
            "distance_gap_pct_median": median(agg["gap_distance_pct"]),
            "distance_gap_pct_mad": mad(agg["gap_distance_pct"]),
            "distance_gap_pct_clipped_mean": mean(clipped(agg["gap_distance_pct"])),
            "distance_gap_abs_mean": mean(agg["gap_distance_abs"]),
            "distance_gap_abs_std": sample_std(agg["gap_distance_abs"]),
            "rc_reward_mean": rc_rew_mean,
            "rc_reward_std": sample_std(agg["rc_reward"]),
            "rc_reward_ci95_half": ci95_halfwidth(agg["rc_reward"]),
            "cpg_reward_mean": cpg_rew_mean,
            "cpg_reward_std": sample_std(agg["cpg_reward"]),
            "cpg_reward_ci95_half": ci95_halfwidth(agg["cpg_reward"]),
            "reward_gap_pct_mean": mean(agg["gap_reward_pct"]),
            "reward_gap_pct_std": sample_std(agg["gap_reward_pct"]),
            "reward_gap_pct_median": median(agg["gap_reward_pct"]),
            "reward_gap_pct_mad": mad(agg["gap_reward_pct"]),
            "reward_gap_pct_clipped_mean": mean(clipped(agg["gap_reward_pct"])),
            "reward_gap_abs_mean": mean(agg["gap_reward_abs"]),
            "reward_gap_abs_std": sample_std(agg["gap_reward_abs"]),
            "rc_win_rate_distance_pct": (agg["rc_beats_cpg_distance"] / n * 100.0) if n > 0 else 0.0,
            "rc_win_rate_reward_pct": (agg["rc_beats_cpg_reward"] / n * 100.0) if n > 0 else 0.0,
        }
        kpi_rows.append(row)

        overall_collect["gap_distance_pct"].extend(agg["gap_distance_pct"])
        overall_collect["gap_reward_pct"].extend(agg["gap_reward_pct"])
        overall_collect["gap_distance_abs"].extend(agg["gap_distance_abs"])
        overall_collect["gap_reward_abs"].extend(agg["gap_reward_abs"])
        overall_collect["win_distance"] += agg["rc_beats_cpg_distance"]
        overall_collect["win_reward"] += agg["rc_beats_cpg_reward"]
        overall_collect["count"] += n

    overall_row = {
        "episodes_paired_total": overall_collect["count"],
        "distance_gap_pct_mean": mean(overall_collect["gap_distance_pct"]),
        "distance_gap_pct_std": sample_std(overall_collect["gap_distance_pct"]),
        "distance_gap_pct_median": median(overall_collect["gap_distance_pct"]),
        "distance_gap_pct_mad": mad(overall_collect["gap_distance_pct"]),
        "distance_gap_pct_clipped_mean": mean(clipped(overall_collect["gap_distance_pct"])),
        "distance_gap_abs_mean": mean(overall_collect["gap_distance_abs"]),
        "distance_gap_abs_std": sample_std(overall_collect["gap_distance_abs"]),
        "reward_gap_pct_mean": mean(overall_collect["gap_reward_pct"]),
        "reward_gap_pct_std": sample_std(overall_collect["gap_reward_pct"]),
        "reward_gap_pct_median": median(overall_collect["gap_reward_pct"]),
        "reward_gap_pct_mad": mad(overall_collect["gap_reward_pct"]),
        "reward_gap_pct_clipped_mean": mean(clipped(overall_collect["gap_reward_pct"])),
        "reward_gap_abs_mean": mean(overall_collect["gap_reward_abs"]),
        "reward_gap_abs_std": sample_std(overall_collect["gap_reward_abs"]),
        "rc_win_rate_distance_pct": (overall_collect["win_distance"] / overall_collect["count"] * 100.0) if overall_collect["count"] > 0 else 0.0,
        "rc_win_rate_reward_pct": (overall_collect["win_reward"] / overall_collect["count"] * 100.0) if overall_collect["count"] > 0 else 0.0,
    }

    return kpi_rows, overall_row


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build KPI report CSV from episode results")
    parser.add_argument(
        "--input",
        default=os.path.join("multifunctional_rc_ae_model", "excel_episode_results.csv"),
    )
    parser.add_argument(
        "--out-kpi",
        default=os.path.join("multifunctional_rc_ae_model", "excel_kpi_results.csv"),
    )
    parser.add_argument(
        "--out-overall",
        default=os.path.join("multifunctional_rc_ae_model", "excel_kpi_overall.csv"),
    )
    parser.add_argument(
        "--out-robust",
        default=os.path.join("multifunctional_rc_ae_model", "excel_kpi_robust.csv"),
    )
    args = parser.parse_args()

    rows = load_episode_rows(args.input)
    kpi_rows, overall_row = build_kpis(rows)

    kpi_fields = [
        "friction",
        "episodes_paired",
        "rc_distance_mean",
        "rc_distance_std",
        "rc_distance_ci95_half",
        "cpg_distance_mean",
        "cpg_distance_std",
        "cpg_distance_ci95_half",
        "distance_gap_pct_mean",
        "distance_gap_pct_std",
        "distance_gap_pct_median",
        "distance_gap_pct_mad",
        "distance_gap_pct_clipped_mean",
        "distance_gap_abs_mean",
        "distance_gap_abs_std",
        "rc_reward_mean",
        "rc_reward_std",
        "rc_reward_ci95_half",
        "cpg_reward_mean",
        "cpg_reward_std",
        "cpg_reward_ci95_half",
        "reward_gap_pct_mean",
        "reward_gap_pct_std",
        "reward_gap_pct_median",
        "reward_gap_pct_mad",
        "reward_gap_pct_clipped_mean",
        "reward_gap_abs_mean",
        "reward_gap_abs_std",
        "rc_win_rate_distance_pct",
        "rc_win_rate_reward_pct",
    ]

    overall_fields = [
        "episodes_paired_total",
        "distance_gap_pct_mean",
        "distance_gap_pct_std",
        "distance_gap_pct_median",
        "distance_gap_pct_mad",
        "distance_gap_pct_clipped_mean",
        "distance_gap_abs_mean",
        "distance_gap_abs_std",
        "reward_gap_pct_mean",
        "reward_gap_pct_std",
        "reward_gap_pct_median",
        "reward_gap_pct_mad",
        "reward_gap_pct_clipped_mean",
        "reward_gap_abs_mean",
        "reward_gap_abs_std",
        "rc_win_rate_distance_pct",
        "rc_win_rate_reward_pct",
    ]

    write_csv(args.out_kpi, kpi_fields, kpi_rows)
    write_csv(args.out_overall, overall_fields, [overall_row])
    robust_fields = [
        "friction",
        "episodes_paired",
        "distance_gap_pct_median",
        "distance_gap_pct_mad",
        "distance_gap_pct_clipped_mean",
        "distance_gap_abs_mean",
        "distance_gap_abs_std",
        "reward_gap_pct_median",
        "reward_gap_pct_mad",
        "reward_gap_pct_clipped_mean",
        "reward_gap_abs_mean",
        "reward_gap_abs_std",
        "rc_win_rate_distance_pct",
        "rc_win_rate_reward_pct",
    ]
    robust_rows = [{k: row[k] for k in robust_fields} for row in kpi_rows]
    write_csv(args.out_robust, robust_fields, robust_rows)

    print("KPI export complete")
    print(f"Input: {args.input}")
    print(f"Per-friction KPI CSV: {args.out_kpi}")
    print(f"Overall KPI CSV: {args.out_overall}")
    print(f"Robust KPI CSV: {args.out_robust}")
    print("Overall:", overall_row)


if __name__ == "__main__":
    main()
