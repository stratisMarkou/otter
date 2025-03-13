import argparse
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Set,
    Union,
)

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns  # type: ignore

from otter.data.source.utils import (
    ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES,
    ALL_GRID_VARIABLE_LEVELS,
)

# Map between variables names and their abbreviation.
VAR_MAP = {
    "2m_temperature": "2t",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "ms",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "sea_surface_temperature": "sst",
    "vertical_velocity": "w",
}
# The order to respect in the score card.
VARIABLE_ORDER = list(VAR_MAP.values())


def process_index(label: str) -> Tuple[int, float]:
    """
    Functionality to obtain the right order in the score card plot.
    """
    parts = label.split()
    var = parts[0]  # First part is always the variable name
    if len(parts) > 1 and parts[1].isdigit():
        pl = float(parts[1])  # Convert pressure level to integer
    else:
        pl = float(
            "inf"
        )  # Assign inf to surface variables (e.g., "2t", "10u")
    return (
        (
            VARIABLE_ORDER.index(var)
            if var in VARIABLE_ORDER
            else len(VARIABLE_ORDER)
        ),
        pl,
    )


def get_metric(
    metrics_dict: Dict[str, float],
    metric: str,
    var: str,
    level: Union[int, None],
    time_step: str,
    largest_scale: str,
) -> float:
    """
    Extract the relevant entry from the metrics dict.
    """
    if metrics_dict is None:
        raise ValueError("metrics_dict cannot be None")

    if level is not None:
        key = f"val/{metric}/{var}/level_{level}/time_{time_step}/{largest_scale}"
    else:
        key = f"val/{metric}/{var}/time_{time_step}/{largest_scale}"
    return metrics_dict[key]


def filter_for_largest_scale(
    validation_metrics: Dict[str, float],
) -> Dict[str, float]:
    # Filter keys where the last element is the maximum scale
    max_scale = max(
        [
            int(key.split("/")[-1].replace("scale_", ""))
            for key in validation_metrics.keys()
        ]
    )

    # Filter keys that have the maximum scale
    filtered_keys = [
        key
        for key in validation_metrics.keys()
        if key.endswith(f"scale_{max_scale}")
    ]

    # Filter dictionary.
    validation_metrics = {
        k: validation_metrics[k]
        for k in filtered_keys
        if k in validation_metrics
    }

    return validation_metrics


def get_list_of_metrics(validation_metrics: Dict[str, float]) -> Set[str]:
    metrics = set()
    for key in validation_metrics.keys():
        parts = key.split("/")
        metrics.add(parts[1])
    return metrics


def get_list_of_vars(validation_metrics: Dict[str, float]) -> Set[str]:
    vars = set()
    for key in validation_metrics.keys():
        parts = key.split("/")
        vars.add(parts[2])
    return vars


def get_time_steps(validation_metrics: Dict[str, float]) -> List[str]:
    time_steps = set()
    for key in validation_metrics.keys():
        parts = key.split("/")
        if len(parts) > 4:
            if "time_" in parts[3]:
                time_steps.add(parts[3].split("time_")[-1])
            elif "time_" in parts[4]:
                time_steps.add(parts[4].split("time_")[-1])
    return sorted(time_steps)


def plot_score_card(
    base_validation_metrics: Dict[str, float],
    validation_metrics: Dict[str, float],
    lead_time: int,
    save_path: Optional[str] = None,
) -> None:
    # Checks we are passing the dicts properly.
    if base_validation_metrics is None:
        raise ValueError("base_validation_metrics cannot be None")
    if validation_metrics is None:
        raise ValueError("validation_metrics cannot be None")

    # Filter for just largest scale
    validation_metrics = filter_for_largest_scale(validation_metrics)
    base_validation_metrics = filter_for_largest_scale(base_validation_metrics)
    largest_scale = list(validation_metrics.keys())[0].split("/")[-1]

    # Get all the metric types from the metrics dictionary.
    metrics = get_list_of_metrics(validation_metrics)

    # Get all the variables from the metrics dictionary.
    vars = get_list_of_vars(validation_metrics)

    # Define variables, pressure levels, and lead times
    variables = list(VAR_MAP.keys())
    variables = [variable for variable in variables if variable in vars]
    pressure_levels = ALL_GRID_VARIABLE_LEVELS
    time_steps = get_time_steps(validation_metrics)
    temporal_resolution = lead_time * 24 // len(time_steps)

    # Initialise the figure.
    scale = 0.6 * max(
        1, 1 + 0.15 * (10 - len(time_steps))
    )  # This seems to look decent.
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(scale * len(time_steps), scale * len(time_steps)),
        gridspec_kw={"wspace": 1.2},
    )

    for ax, _metric in zip(axs, metrics):
        heatmap_data = []

        # Iterate over all variables, levels (for non-surface), and lead times.
        for _var in variables:
            levels: List[Optional[int]] = (
                list(pressure_levels)
                if _var in ALL_GRID_MULTILEVEL_DYNAMIC_VARIABLES
                else [None]
            )  # No levels for surface variables
            for _level in levels:
                for _time_step in time_steps:
                    # Retrieve the metric.
                    base_metric = get_metric(
                        base_validation_metrics,
                        _metric,
                        _var,
                        _level,
                        _time_step,
                        largest_scale,
                    )
                    metric = get_metric(
                        validation_metrics,
                        _metric,
                        _var,
                        _level,
                        _time_step,
                        largest_scale,
                    )

                    # Compute normalized RMSE difference
                    norm_diff = (metric - base_metric) / base_metric

                    # Store for visualization
                    _level_vis = (
                        _level if isinstance(_level, int) else str(_level)
                    )
                    heatmap_data.append(
                        (VAR_MAP[_var], _level_vis, int(_time_step), norm_diff)
                    )

        # Convert to DataFrame for easier plotting.
        df = pd.DataFrame(
            heatmap_data,
            columns=[
                "Variable",
                "Pressure Level",
                "Time Step",
                "Normalized RMSE Difference",
            ],
        )
        df["Lead Time"] = (df["Time Step"] + 1) * temporal_resolution

        # Pivot table for heatmap (rows: var + level, cols: lead time).
        df["Row Label"] = df.apply(
            lambda x: (
                f"{x['Variable']} {x['Pressure Level']}"
                if x["Pressure Level"]
                else x["Variable"]
            ),
            axis=1,
        )
        df["Row Label"] = df["Row Label"].str.replace(" None", "")
        pivot_table = df.pivot(
            index="Row Label",
            columns="Lead Time",
            values="Normalized RMSE Difference",
        )

        # Convert the index (Row Label) to ensure numeric sorting for
        # pressure levels.
        pivot_table["Sort_Index"] = pivot_table.index.map(process_index)
        # Sort index based on the tuple (variable, pressure level).
        pivot_table = pivot_table.sort_values(by=["Sort_Index"])
        pivot_table.drop(columns=["Sort_Index"], inplace=True)

        # Plot heatmap.
        sns.heatmap(
            pivot_table,
            cmap="RdBu_r",
            center=0,
            linewidths=0.5,
            annot=False,
            ax=ax,
        )
        # Compute percentage in which metrics have improved.
        perc_improv = round(
            df.loc[df["Normalized RMSE Difference"] < 0].shape[0]
            / df.shape[0]
            * 100,
            1,
        )
        ax.set_title(f"{_metric} - {perc_improv}%")
        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("")

    fig.supylabel("Variable & Pressure Level", x=-0.001)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return


def get_metrics_path(
    experiment_name: str,
    results_path: str,
    checkpoint_name: str,
    lead_time: int,
) -> str:
    experiment_path = os.path.join(results_path, experiment_name)
    experiment_metrics_path = os.path.join(
        experiment_path, (f"{checkpoint_name}_LT{lead_time}.pkl")
    )
    return experiment_metrics_path


def load_metrics(metrics_path: str) -> Any:
    with open(metrics_path, "rb") as f:
        validation_metrics = pickle.load(f)
    return validation_metrics


def main(
    experiment_name: str,
    base_experiment_name: str,
    lead_time: int,
    checkpoint_name: str,
    base_checkpoint_name: str,
    results_path: str = "./_results",
    base_results_path: str = "./_results",
) -> None:
    # Assert results exist.
    experiment_metrics_path = get_metrics_path(
        experiment_name=experiment_name,
        results_path=results_path,
        checkpoint_name=checkpoint_name,
        lead_time=lead_time,
    )
    assert os.path.exists(experiment_metrics_path), (
        f"Results for {experiment_name} do not exist at expected location"
    )

    base_experiment_metrics_path = get_metrics_path(
        experiment_name=base_experiment_name,
        results_path=base_results_path,
        checkpoint_name=base_checkpoint_name,
        lead_time=lead_time,
    )
    assert os.path.exists(base_experiment_metrics_path), (
        f"Results for {base_experiment_name} do not exist at expected location"
    )

    # Create path for saving score cards.
    score_card_dir = os.path.join(results_path, "score_cards/")
    os.makedirs(score_card_dir, exist_ok=True)
    save_path = os.path.join(
        score_card_dir,
        (
            f"{experiment_name}_{checkpoint_name}_vs_"
            f"{base_experiment_name}_{base_checkpoint_name}"
            f"_LT{lead_time}.png"
        ),
    )

    # Load metrics
    validation_metrics = load_metrics(experiment_metrics_path)
    base_validation_metrics = load_metrics(base_experiment_metrics_path)

    # Save score card.
    plot_score_card(
        base_validation_metrics=base_validation_metrics,
        validation_metrics=validation_metrics,
        lead_time=lead_time,
        save_path=save_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare models."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment to evaluate.",
    )
    parser.add_argument(
        "--base_experiment_name",
        type=str,
        help="Name of the base experiment to evaluate against.",
    )
    parser.add_argument(
        "--lead_time",
        type=int,
        help="Lead time in days for evaluation.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Name of the checkpoint for the experiment.",
    )
    parser.add_argument(
        "--base_checkpoint_name",
        type=str,
        default=None,
        help="Name of the checkpoint for the base experiment.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="./_results",
        help="Path from where to get the results for experiment (default: ./_results).",
    )
    parser.add_argument(
        "--base_results_path",
        type=str,
        default="./_results",
        help="Path from where to get the results for base experiment (default: ./_results).",
    )
    args = parser.parse_args()
    main(
        experiment_name=args.experiment_name,
        base_experiment_name=args.base_experiment_name,
        lead_time=args.lead_time,
        checkpoint_name=args.checkpoint_name,
        base_checkpoint_name=args.base_checkpoint_name,
        results_path=args.results_path,
        base_results_path=args.base_results_path,
    )
