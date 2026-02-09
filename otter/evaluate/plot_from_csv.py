# %%
"""
Reads experiment results from CSV and generates plots and comparison statistics.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Reference values for comparison (Arches M and 4M)
REFERENCE_VALUES = {
    "Z500": {"arches_m": 44.96, "arches_4m": 41.93},
    "T850": {"arches_m": 0.615, "arches_4m": 0.593},
    "Q700": {"arches_m": 0.000527, "arches_4m": 0.000513},
    "U850": {"arches_m": 1.219, "arches_4m": 1.172},
    "V850": {"arches_m": 1.251, "arches_4m": 1.203},
    "T2M": {"arches_m": 0.539, "arches_4m": 0.517},
    "SP": {"arches_m": 56.22, "arches_4m": 52.22},
    "U10M": {"arches_m": 0.783, "arches_4m": 0.749},
    "V10M": {"arches_m": 0.819, "arches_4m": 0.783},
}

VARIABLE_ORDER = ["Z500", "T850", "Q700", "U850", "V850", "T2M", "SP", "U10M", "V10M"]


def load_results(csv_path: str = "_results/experiment_results.csv") -> pd.DataFrame:
    """Load results from CSV."""
    return pd.read_csv(csv_path)


def load_baseline(baseline_name: str) -> pd.DataFrame:
    """Load baseline data from weatherbench baselines."""
    df = pd.read_csv("_baselines/weatherbench/wb_results.csv")
    names = df.name.unique()
    if baseline_name not in names:
        raise ValueError(f"Baseline {baseline_name} not found in {names}")
    return df[df.name == baseline_name]


def plot_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    baseline_df: pd.DataFrame | None = None,
    baseline_name: str | None = None,
    baseline_time: int = 24,
    output_dir: str = "_plots",
) -> None:
    """Generate RMSE over time plots for an experiment."""
    exp_df = df[df["experiment"] == experiment_name]
    if exp_df.empty:
        raise ValueError(f"No data found for experiment '{experiment_name}'")
    
    variables = [v for v in VARIABLE_ORDER if v in exp_df["variable"].unique()]
    
    fig, axs = plt.subplots(len(variables), 1, figsize=(8, 4 * len(variables)))
    if len(variables) == 1:
        axs = [axs]
    
    baseline_improvement = 0
    
    for ax, var in zip(axs, variables):
        var_df = exp_df[exp_df["variable"] == var]
        lead_times = var_df["lead_time"].values
        rmses = var_df["rmse"].values
        
        # Get title from variable_long and level
        row = var_df.iloc[0]
        if pd.notna(row["level"]):
            title = f"{row['variable_long']} at {int(row['level'])} hPa"
        else:
            title = row["variable_long"]
        
        ax.plot(lead_times, rmses, label=title, marker="x", linestyle="--")
        
        if baseline_df is not None:
            sub_df = baseline_df[baseline_df.variable == var]
            sub_df = sub_df[sub_df["lead_time"] <= lead_times.max()]
            baseline_times = sub_df["lead_time"].values
            baseline_rmses = sub_df["rmse"].values
            ax.plot(baseline_times, baseline_rmses, label="Baseline", marker="x", linestyle="--")
            
            # Calculate improvement at baseline_time
            perf_at_time = var_df[var_df["lead_time"] == baseline_time]["rmse"].values
            baseline_perf_at_time = sub_df[sub_df["lead_time"] == baseline_time]["rmse"].values
            if len(perf_at_time) > 0 and len(baseline_perf_at_time) > 0:
                baseline_improvement += (baseline_perf_at_time[0] - perf_at_time[0]) / baseline_perf_at_time[0]
        
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel("RMSE")
        ax.legend()
    
    if baseline_df is not None:
        baseline_improvement /= len(variables)
        print(f"Improvement over baseline {baseline_name} is {baseline_improvement * 100:.2f}% at {baseline_time} hours")
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    path = output_path / f"{experiment_name}_rmse_over_time.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Plot saved to {path}")
    
    plt.close(fig)


def print_24h_comparison(df: pd.DataFrame, experiment_name: str) -> None:
    """Print 24h RMSE comparison with Arches reference values."""
    exp_df = df[df["experiment"] == experiment_name]
    exp_df = exp_df[exp_df["lead_time"] == 24]
    
    print(f"\n24h RMSE comparison for {experiment_name}:")
    print("=" * 60)
    
    for var in VARIABLE_ORDER:
        if var not in REFERENCE_VALUES:
            continue
        
        var_df = exp_df[exp_df["variable"] == var]
        if var_df.empty:
            continue
        
        rmse_24h = var_df["rmse"].values[0]
        refs = REFERENCE_VALUES[var]
        arches_m = refs["arches_m"]
        arches_4m = refs["arches_4m"]
        
        percent_m = 100 * (arches_m - rmse_24h) / arches_m
        percent_4m = 100 * (arches_4m - rmse_24h) / arches_4m
        
        row = var_df.iloc[0]
        if pd.notna(row["level"]):
            print(f"{row['variable_long']} at {int(row['level'])} hPa: {rmse_24h:.3f} (Arches M: {arches_m}, Arches 4M: {arches_4m})")
        else:
            print(f"{row['variable_long']}: {rmse_24h:.3f} (Arches M: {arches_m}, Arches 4M: {arches_4m})")
        print(f"Percent better than Arches M: {percent_m:.2f}%")
        print(f"Percent better than Arches 4M: {percent_4m:.2f}%\n")


def compare_experiments(
    df: pd.DataFrame,
    experiment_names: list[str],
    reference_baseline: str | None = None,
    lead_time: int = 24,
) -> None:
    """Compare multiple experiments against each other and optionally a baseline."""
    print(f"\nComparison at {lead_time}h lead time:")
    print("=" * 60)
    
    baseline_df = None
    if reference_baseline:
        baseline_df = load_baseline(reference_baseline)
    
    for var in VARIABLE_ORDER:
        print(f"\n{var}:")
        
        if baseline_df is not None:
            bl_df = baseline_df[(baseline_df["variable"] == var) & (baseline_df["lead_time"] == lead_time)]
            if not bl_df.empty:
                bl_rmse = bl_df["rmse"].values[0]
                print(f"  {reference_baseline}: {bl_rmse:.4f}")
        
        for exp in experiment_names:
            exp_df = df[(df["experiment"] == exp) & (df["variable"] == var) & (df["lead_time"] == lead_time)]
            if exp_df.empty:
                continue
            rmse = exp_df["rmse"].values[0]
            
            improvement_str = ""
            if baseline_df is not None:
                bl_df = baseline_df[(baseline_df["variable"] == var) & (baseline_df["lead_time"] == lead_time)]
                if not bl_df.empty:
                    bl_rmse = bl_df["rmse"].values[0]
                    improvement = 100 * (bl_rmse - rmse) / bl_rmse
                    improvement_str = f" ({improvement:+.2f}% vs {reference_baseline})"
            
            print(f"  {exp}: {rmse:.4f}{improvement_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot and compare experiment results from CSV"
    )
    parser.add_argument("experiment_name", type=str, help="Name of the experiment to plot")
    parser.add_argument(
        "--csv_path", type=str, default="_results/experiment_results.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Baseline to compare against (from weatherbench baselines)"
    )
    parser.add_argument(
        "--baseline_time", type=int, default=24,
        help="Lead time in hours to compute baseline improvement"
    )
    parser.add_argument(
        "--compare", type=str, nargs="+", default=None,
        help="Additional experiment names to compare"
    )
    parser.add_argument(
        "--output_dir", type=str, default="_plots",
        help="Directory for output plots"
    )
    args = parser.parse_args()

    # Load data
    df = load_results(args.csv_path)
    
    # Load baseline if specified
    baseline_df = None
    if args.baseline:
        baseline_df = load_baseline(args.baseline)
    
    # Generate plots
    plot_experiment(
        df,
        args.experiment_name,
        baseline_df=baseline_df,
        baseline_name=args.baseline,
        baseline_time=args.baseline_time,
        output_dir=args.output_dir,
    )
    
    # Print 24h comparison
    print_24h_comparison(df, args.experiment_name)
    
    # Compare multiple experiments if requested
    if args.compare:
        all_experiments = [args.experiment_name] + args.compare
        compare_experiments(df, all_experiments, args.baseline, args.baseline_time)


if __name__ == "__main__":
    main()

#%%
import pandas as pd

df = pd.read_csv("/scratch/js2731/otter/_results/weatherbench_results.csv")
df_bl = pd.read_csv("/scratch/js2731/otter/_baselines/weatherbench/wb_results.csv")
df = pd.concat([df, df_bl])
df