"""Export paper-ready tables from synthetic experiment CSV outputs.

Reads Table 1 and Table 3 source CSVs from results/synthetic and writes:
- Markdown tables for quick review
- LaTeX tables for manuscript inclusion
- A short artifact index README

Run:
    python -m src.experiments.export_paper_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready table artifacts")
    parser.add_argument("--synthetic-dir", type=str, default="results/synthetic")
    parser.add_argument("--out-dir", type=str, default="results/paper")
    return parser.parse_args()


def fmt3(value: float) -> str:
    return f"{value:.3f}" if pd.notna(value) else ""


def fmt_pvalue(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = [header, sep]
    for _, row in df.iterrows():
        vals = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join(rows)


def build_table1(table1_csv: Path) -> pd.DataFrame:
    src = pd.read_csv(table1_csv)

    out = pd.DataFrame(
        {
            "System": src["system"],
            "rho(S,W) mean+/-std": src.apply(
                lambda r: f"{r['rho_sw_mean']:.3f} +/- {r['rho_sw_std']:.3f}", axis=1
            ),
            "Monotonicity mean+/-std": src.apply(
                lambda r: f"{r['mono_mean']:.3f} +/- {r['mono_std']:.3f}", axis=1
            ),
            "Lead time mean+/-std": src.apply(
                lambda r: f"{r['lead_mean']:.3f} +/- {r['lead_std']:.3f}", axis=1
            ),
            "Alert rate": src["alert_rate"].map(fmt3),
        }
    )
    return out


def build_table3(table3_csv: Path) -> pd.DataFrame:
    src = pd.read_csv(table3_csv)

    preferred_order = [
        "S_t",
        "B1 Rolling Variance",
        "B2 Rolling AC1",
        "B3 Rolling Skewness",
        "B4 DFA",
        "B5 Basin Stability",
        "Cox PH (hazard)",
    ]
    order_map = {name: idx for idx, name in enumerate(preferred_order)}
    src["_order"] = src["model"].map(lambda m: order_map.get(m, 999))
    src = src.sort_values(["_order", "model"]).drop(columns=["_order"])

    out = pd.DataFrame(
        {
            "Model": src["model"],
            "rho_tau": src["rho_tau_mean"].map(fmt3),
            "lead_time": src["lead_mean"].map(fmt3),
            "separability": src["separability_mean"].map(fmt3),
            "wilcoxon_p_lead": src["wilcoxon_p_lead"].map(fmt_pvalue),
            "cliffs_delta_lead": src["cliffs_delta_lead"].map(fmt3),
            "cliffs_mag_lead": src["cliffs_mag_lead"].fillna(""),
            "wilcoxon_p_sep": src["wilcoxon_p_sep"].map(fmt_pvalue),
            "cliffs_delta_sep": src["cliffs_delta_sep"].map(fmt3),
            "cliffs_mag_sep": src["cliffs_mag_sep"].fillna(""),
        }
    )
    return out


def write_outputs(df: pd.DataFrame, md_path: Path, tex_path: Path, caption: str, label: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(dataframe_to_markdown(df))
        f.write("\n")

    tex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex)


def write_readme(out_dir: Path) -> None:
    readme = out_dir / "README.md"
    content = """# Paper Artifacts

This folder contains publication-ready table artifacts exported from synthetic experiments.

## Files

- `table1.md`: Markdown version of Table 1 (analytical S_t summary)
- `table1.tex`: LaTeX version of Table 1
- `table3.md`: Markdown version of Table 3 (main comparison)
- `table3.tex`: LaTeX version of Table 3

## Source CSVs

- `results/synthetic/table1_summary.csv`
- `results/synthetic/table3_main_comparison.csv`
"""
    with open(readme, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    args = parse_args()
    synthetic_dir = Path(args.synthetic_dir)
    out_dir = Path(args.out_dir)

    table1_csv = synthetic_dir / "table1_summary.csv"
    table3_csv = synthetic_dir / "table3_main_comparison.csv"

    if not table1_csv.exists() or not table3_csv.exists():
        missing = [str(p) for p in [table1_csv, table3_csv] if not p.exists()]
        raise FileNotFoundError(
            "Missing required source CSVs: " + ", ".join(missing)
        )

    table1 = build_table1(table1_csv)
    table3 = build_table3(table3_csv)

    write_outputs(
        df=table1,
        md_path=out_dir / "table1.md",
        tex_path=out_dir / "table1.tex",
        caption="Analytical S_t performance on synthetic systems (Table 1)",
        label="tab:table1_synthetic_st",
    )
    write_outputs(
        df=table3,
        md_path=out_dir / "table3.md",
        tex_path=out_dir / "table3.tex",
        caption="Main comparison: S_t vs baselines and Cox PH (Table 3)",
        label="tab:table3_main_comparison",
    )

    write_readme(out_dir)

    print("=" * 72)
    print("Paper-ready table export complete")
    print("=" * 72)
    print(f"output dir: {out_dir}")
    print(f"table1 markdown: {out_dir / 'table1.md'}")
    print(f"table1 latex:    {out_dir / 'table1.tex'}")
    print(f"table3 markdown: {out_dir / 'table3.md'}")
    print(f"table3 latex:    {out_dir / 'table3.tex'}")


if __name__ == "__main__":
    main()
