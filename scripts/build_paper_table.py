"""
Create a paper-ready summary table from KPI CSV files.

Inputs:
- multifunctional_rc_ae_model/excel_kpi_results.csv
- multifunctional_rc_ae_model/excel_kpi_robust.csv

Output:
- multifunctional_rc_ae_model/excel_paper_table.csv
"""

import argparse
import csv
import os


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def fmt_pm(mean_val, ci_val, decimals=3):
    return f"{mean_val:.{decimals}f} ± {ci_val:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Build paper table from KPI files")
    parser.add_argument(
        "--kpi",
        default=os.path.join("multifunctional_rc_ae_model", "excel_kpi_results.csv"),
    )
    parser.add_argument(
        "--robust",
        default=os.path.join("multifunctional_rc_ae_model", "excel_kpi_robust.csv"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join("multifunctional_rc_ae_model", "excel_paper_table.csv"),
    )
    args = parser.parse_args()

    kpi_rows = load_csv(args.kpi)
    robust_rows = load_csv(args.robust)

    robust_by_fric = {float(r["friction"]): r for r in robust_rows}

    out_rows = []
    for row in sorted(kpi_rows, key=lambda r: float(r["friction"])):
        fr = float(row["friction"])
        rr = robust_by_fric.get(fr, {})

        rc_dist = fmt_pm(to_float(row["rc_distance_mean"]), to_float(row["rc_distance_ci95_half"]))
        cpg_dist = fmt_pm(to_float(row["cpg_distance_mean"]), to_float(row["cpg_distance_ci95_half"]))
        rc_rew = fmt_pm(to_float(row["rc_reward_mean"]), to_float(row["rc_reward_ci95_half"]), decimals=2)
        cpg_rew = fmt_pm(to_float(row["cpg_reward_mean"]), to_float(row["cpg_reward_ci95_half"]), decimals=2)

        out_rows.append(
            {
                "friction": f"{fr:.1f}",
                "episodes": int(float(row["episodes_paired"])),
                "rc_distance_mean_ci95": rc_dist,
                "cpg_distance_mean_ci95": cpg_dist,
                "rc_reward_mean_ci95": rc_rew,
                "cpg_reward_mean_ci95": cpg_rew,
                "distance_gap_pct_median": f"{to_float(rr.get('distance_gap_pct_median', 0.0)):.2f}",
                "reward_gap_pct_median": f"{to_float(rr.get('reward_gap_pct_median', 0.0)):.2f}",
                "distance_gap_abs_mean": f"{to_float(rr.get('distance_gap_abs_mean', 0.0)):.3f}",
                "reward_gap_abs_mean": f"{to_float(rr.get('reward_gap_abs_mean', 0.0)):.2f}",
                "rc_win_rate_distance_pct": f"{to_float(row['rc_win_rate_distance_pct']):.1f}",
                "rc_win_rate_reward_pct": f"{to_float(row['rc_win_rate_reward_pct']):.1f}",
            }
        )

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "friction",
            "episodes",
            "rc_distance_mean_ci95",
            "cpg_distance_mean_ci95",
            "rc_reward_mean_ci95",
            "cpg_reward_mean_ci95",
            "distance_gap_pct_median",
            "reward_gap_pct_median",
            "distance_gap_abs_mean",
            "reward_gap_abs_mean",
            "rc_win_rate_distance_pct",
            "rc_win_rate_reward_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print("Paper table export complete")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
