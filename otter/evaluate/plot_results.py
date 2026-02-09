# %%
"""
Plot results from weatherbench experiments.
"""
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd


@dataclass
class Model:
    name: str  # Name in the CSV
    plot_name: str  # Name to display in plots
    a100_days: float  # Compute cost in A100 days (0 = exclude from compute plot)
    release_date: date | None = None  # Release date for color mapping
    resolution: float = 1.5
    is_baseline: bool = False  # True if from wb_results.csv, False if from experiments
    color: str = field(default="", init=True)  # Will be computed from release_date


# =============================================================================
# CONFIGURATION - Edit these as needed
# =============================================================================

# Models to include in plots (both baselines and experiments)
MODELS = [
    # 2022
    Model("Keisler (2022) vs ERA5", "Keisler", a100_days=5.5, release_date=date(2022, 2, 15), resolution=1, is_baseline=True),
    Model("Pangu-Weather vs ERA5", "Pangu-Weather", a100_days=1440, release_date=date(2022, 11, 3), resolution=0.25, is_baseline=True),
    # 2023
    Model("FuXi vs ERA5", "FuXi", a100_days=26, release_date=date(2023, 6, 22), resolution=0.25, is_baseline=True),
    Model("GraphCast vs ERA5", "GraphCast", a100_days=1344, release_date=date(2023, 11, 14), resolution=0.25, is_baseline=True),
    Model("NeuralGCM ENS (mean) vs ERA5", "NeuralGCM 50xENS (1.4°)", a100_days=7680, release_date=date(2023, 11, 13), resolution=1.4, is_baseline=True),
    Model("NeuralGCM 0.7 vs ERA5", "NeuralGCM (0.7°)", a100_days=8064, release_date=date(2023, 11, 13), resolution=0.7, is_baseline=True),
    # 2024
    Model("ArchesWeather-S vs ERA5", "Arches-S", a100_days=2.5, release_date=date(2024, 5, 23), is_baseline=True),
    Model("ArchesWeather-M vs ERA5", "Arches-M", a100_days=5, release_date=date(2024, 5, 23), is_baseline=True),
    Model("ArchesWeather-Mx4 vs ERA5", "Arches-4M", a100_days=20, release_date=date(2024, 5, 23), is_baseline=True),
    Model("Stormer ENS (mean) vs ERA5", "Stormer ENS", a100_days=128, release_date=date(2024, 3, 1), is_baseline=True),
    # Non-ML baseline (no release date = gray)
    Model("IFS HRES vs Analysis", "IFS HRES", a100_days=0, release_date=None, resolution=0, is_baseline=True),
    # 2026 - Ours
    Model("rft_2026-01-31_15-55-elegant-buffalo", "Otter Weather (Ours)", a100_days=3, release_date=date(2026, 2, 1)),
]

# Compute colors from release dates using a colormap
def _compute_colors():
    """Map release dates to colors using a matplotlib colormap."""
    cmap = plt.cm.viridis
    
    # Get all valid dates
    dates = [m.release_date for m in MODELS if m.release_date is not None]
    if not dates:
        return
    
    min_date = min(dates)
    max_date = date(2026, 8, 1)
    date_range = (max_date - min_date).days or 1  # Avoid division by zero
    
    for m in MODELS:
        if m.release_date is None:
            m.color = "#757575"  # Gray for non-ML models
        else:
            # Normalize date to [0, 1] range
            normalized = (m.release_date - min_date).days / date_range
            m.color = mcolors.to_hex(cmap(normalized))

_compute_colors()

# Reference baseline for computing improvement (IFS HRES)
REFERENCE_BASELINE = "IFS HRES vs Analysis"
#REFERENCE_BASELINE = "rft_2026-01-31_15-55-elegant-buffalo"

# Build lookup dicts from MODELS
BASELINES_TO_INCLUDE = [m.name for m in MODELS if m.is_baseline]
A100_DAYS = {m.name: m.a100_days for m in MODELS}
PLOT_NAMES = {m.name: m.plot_name for m in MODELS}
RESOLUTIONS = {m.name: m.resolution for m in MODELS}
COLORS = {m.name: m.color for m in MODELS}

# Experiments to include in RMSE over time plots
EXPERIMENTS_FOR_RMSE_PLOT = [
    "rft_2026-01-31_15-55-elegant-buffalo",
    #"GraphCast vs ERA5",
]

# Variables to plot RMSE over time for
VARIABLES_FOR_RMSE_PLOT = ["Z500", "T850", "T2M"]

# Paths
RESULTS_CSV = "_results/weatherbench_results.csv"
BASELINES_CSV = "_baselines/weatherbench/wb_results.csv"
OUTPUT_DIR = Path("_plots")

# =============================================================================
# LOAD DATA
# =============================================================================


def load_data() -> pd.DataFrame:
    """Load and combine experiment results and baselines."""
    df_exp = pd.read_csv(RESULTS_CSV)
    df_bl = pd.read_csv(BASELINES_CSV)

    # Filter baselines to only those we want
    df_bl = df_bl[df_bl["name"].isin(BASELINES_TO_INCLUDE)]

    # Combine
    df = pd.concat([df_exp, df_bl], ignore_index=True)
    return df


# =============================================================================
# 1. COMPUTE IMPROVEMENT OVER REFERENCE BASELINE AT 24H
# =============================================================================


def compute_improvement_over_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average RMSE improvement over reference baseline at 24h for all models."""
    df_24h = df[df["lead_time"] == 24].copy()

    # Get reference baseline values
    ref_df = df_24h[df_24h["name"] == REFERENCE_BASELINE]
    ref_rmse = ref_df.set_index("variable")["rmse"].to_dict()

    results = []
    for name in df_24h["name"].unique():
        if name == REFERENCE_BASELINE:
            continue

        model_df = df_24h[df_24h["name"] == name]
        improvements = []

        for _, row in model_df.iterrows():
            var = row["variable"]
            if var in ref_rmse:
                # Positive = better than reference
                improvement = (ref_rmse[var] - row["rmse"]) / ref_rmse[var] * 100
                improvements.append(improvement)

        if improvements:
            avg_improvement = sum(improvements) / len(improvements)
            results.append({
                "name": name,
                "avg_improvement_pct": avg_improvement,
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("avg_improvement_pct", ascending=False)
    return results_df


# =============================================================================
# 2. PLOT COMPUTE COST VS IMPROVEMENT
# =============================================================================


def plot_compute_vs_improvement(improvement_df: pd.DataFrame) -> None:
    """Plot RMSE improvement over reference vs compute cost (A100 days)."""
    # Filter to models with known compute cost (exclude reference)
    plot_data = []
    for _, row in improvement_df.iterrows():
        name = row["name"]
        if name in A100_DAYS and A100_DAYS[name] > 0:
            plot_data.append({
                "name": name,
                "a100_days": A100_DAYS[name],
                "improvement": row["avg_improvement_pct"],
            })

    if not plot_data:
        print("No models with compute cost data to plot.")
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Split by resolution: circles for >= 1°, diamonds for < 1°
    for idx, row in plot_df.iterrows():
        name = row["name"]
        resolution = RESOLUTIONS[name]
        color = COLORS[name]
        marker = "o" if resolution >= 1 else "*"  # Circle vs Diamond
        size = 100
        ax.scatter(row["a100_days"], row["improvement"], s=size, c=[color], 
                   marker=marker, zorder=3)

    # Add labels for each point
    for _, row in plot_df.iterrows():
        # Shorten long names for display
        label = PLOT_NAMES[row["name"]]
        is_baseline = row["name"] in BASELINES_TO_INCLUDE
        xytext = (10, 0)
        if row["name"] == "NeuralGCM 0.7 vs ERA5":
            xytext = (-55, -15)
        if row["name"] == "NeuralGCM ENS (mean) vs ERA5":
            xytext = (-85, -15)
        ax.annotate(
            label,
            (row["a100_days"], row["improvement"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            fontweight="regular" if is_baseline else "bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Compute Cost (A100 Days)", fontsize=10)
    ax.set_ylabel("24h RMSE Improvement over IFS HRES (%)", fontsize=10)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, label="IFS HRES baseline")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "compute_vs_improvement.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved compute vs improvement plot to {path}")
    plt.close(fig)


# =============================================================================
# 3. PLOT RMSE OVER TIME
# =============================================================================


def plot_rmse_over_time(df: pd.DataFrame) -> None:
    """Plot RMSE over lead time for selected experiments and variables."""
    # Models to include: experiments + reference baseline
    models_to_plot = EXPERIMENTS_FOR_RMSE_PLOT + [REFERENCE_BASELINE]

    for var in VARIABLES_FOR_RMSE_PLOT:
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in models_to_plot:
            model_df = df[(df["name"] == model) & (df["variable"] == var)]
            if model_df.empty:
                print(f"No data for {model} / {var}")
                continue

            model_df = model_df.sort_values("lead_time")
            label = model if len(model) <= 30 else model[:27] + "..."
            ax.plot(
                model_df["lead_time"],
                model_df["rmse"],
                marker="o",
                label=label,
            )

        ax.set_xlabel("Lead Time (hours)", fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(f"RMSE over Lead Time - {var}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"rmse_over_time_{var}.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved RMSE over time plot to {path}")
        plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(f"Models: {df['name'].unique().tolist()}")

    # 1. Compute improvement over reference
    improvement_df = compute_improvement_over_reference(df)
    print(f"\n=== Improvement over {REFERENCE_BASELINE} at 24h ===")
    print(improvement_df.to_string(index=False))

    # Save to CSV
    OUTPUT_DIR.mkdir(exist_ok=True)
    improvement_df.to_csv(OUTPUT_DIR / "improvement_over_reference.csv", index=False)
    print(f"\nSaved improvement results to {OUTPUT_DIR / 'improvement_over_reference.csv'}")

    # 2. Plot compute cost vs improvement
    plot_compute_vs_improvement(improvement_df)

    # 3. Plot RMSE over time
    plot_rmse_over_time(df)

    print("\nDone!")