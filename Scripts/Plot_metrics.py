"""Plot window-level metrics saved by KDTest.py.

This script reads the `all_windows` sheet from the Excel file and generates:
1. One accuracy line chart containing all datasets.
2. One separate chart per dataset showing accuracy (line) and unseen_ratio (bar).

Expected columns in the Excel sheet:
    dataset, window_index, acc, unseen_ratio
"""

import argparse
import os
from typing import Iterable

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
REQUIRED_COLUMNS = {
    "dataset",
    "window_index",
    "acc",
    "unseen_ratio",
    "unseen_count",
    "unseen_event_ratio",
}
'''
# We can be more lenient on the last 3 columns since some datasets may not have them, but the first 4 are essential for the main plots.
REQUIRED_COLUMNS = {
    "dataset",
    "window_index",
    "acc",
    "unseen_ratio",
    "unseen_count",
    "unseen_event_ratio",
    "global_unseen_event_ratio",
    "local_unseen_event_ratio",
    "learned_novel_event_ratio",
}

def load_window_metrics(excel_path: str, sheet_name: str = "all_windows") -> pd.DataFrame:
    """Load and validate window-level metrics from Excel."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in sheet '{sheet_name}': {sorted(missing)}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
    df["acc"] = pd.to_numeric(df["acc"], errors="coerce")
    df["unseen_ratio"] = pd.to_numeric(df["unseen_ratio"], errors="coerce")
    df["unseen_count"] = pd.to_numeric(df["unseen_count"], errors="coerce")
    df["unseen_event_ratio"] = pd.to_numeric(df["unseen_event_ratio"], errors="coerce")

    # for learned/novel event ratios, we can be more lenient since some datasets may not have them
    df["global_unseen_event_ratio"] = pd.to_numeric(
        df["global_unseen_event_ratio"], errors="coerce"
    )
    df["local_unseen_event_ratio"] = pd.to_numeric(
        df["local_unseen_event_ratio"], errors="coerce"
    )
    df["learned_novel_event_ratio"] = pd.to_numeric(
        df["learned_novel_event_ratio"], errors="coerce"
    )

    #df = df.dropna(subset=["dataset", "window_index", "acc", "unseen_ratio"])
    #df = df.dropna(subset=["dataset", "window_index", "acc", "unseen_ratio", "unseen_count",  "unseen_event_ratio"])
    df = df.dropna(
        subset=[
            "dataset",
            "window_index",
            "acc",
            "unseen_ratio",
            "unseen_count",
            "unseen_event_ratio",
            "global_unseen_event_ratio",
            "local_unseen_event_ratio",
            "learned_novel_event_ratio",
        ]
    )
    df = df.sort_values(["dataset", "window_index"]).reset_index(drop=True)
    return df


def _safe_filename(name: str) -> str:
    """Convert a dataset name to a safe filename component."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))


def plot_accuracy_all_datasets(
    df: pd.DataFrame,
    output_path: str,
    use_normalized_x: bool = False,
) -> None:
    """Plot one line chart with accuracy trends for all datasets."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for dataset, sub in df.groupby("dataset", sort=True):
        sub = sub.sort_values("window_index")
        if use_normalized_x:
            max_idx = sub["window_index"].max()
            x = sub["window_index"] / max_idx if max_idx else sub["window_index"]
            x_label = "Normalized window position"
        else:
            x = sub["window_index"]
            x_label = "Window index"

        ax.plot(
            x,
            sub["acc"],
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            label=dataset,
        )

    ax.set_title("Window-level Accuracy Across Datasets")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_unseen_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    output_path: str,
) -> None:
    """Plot one dataset's accuracy line and unseen_ratio bar chart."""
    sub = df[df["dataset"] == dataset].sort_values("window_index")
    if sub.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    x = sub["window_index"]

    fig, ax_acc = plt.subplots(figsize=(10, 5))
    ax_unseen = ax_acc.twinx()

    ax_unseen.bar(
        x,
        sub["unseen_ratio"],
        alpha=0.25,
        label="Unseen ratio",
    )
    ax_acc.plot(
        x,
        sub["acc"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Accuracy",
    )

    ax_acc.set_title(f"Accuracy and Unseen-event Ratio - {dataset}")
    ax_acc.set_xlabel("Window index")
    ax_acc.set_ylabel("Accuracy")
    ax_unseen.set_ylabel("Unseen ratio")
    ax_acc.set_ylim(0, 1.05)

    max_unseen = float(sub["unseen_ratio"].max()) if len(sub) else 0.0
    ax_unseen.set_ylim(0, max(0.05, min(1.0, max_unseen * 1.25)))

    ax_acc.grid(True, axis="y", alpha=0.3)

    line_handles, line_labels = ax_acc.get_legend_handles_labels()
    bar_handles, bar_labels = ax_unseen.get_legend_handles_labels()
    ax_acc.legend(line_handles + bar_handles, line_labels + bar_labels, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_accuracy_unseen_count_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    output_path: str,
) -> None:
    """Plot one dataset's accuracy line and unseen_count bar chart."""
    sub = df[df["dataset"] == dataset].sort_values("window_index")
    if sub.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    x = sub["window_index"]

    fig, ax_acc = plt.subplots(figsize=(10, 5))
    ax_unseen = ax_acc.twinx()

    ax_unseen.bar(
        x,
        sub["unseen_count"],
        alpha=0.25,
        label="Unseen count",
    )

    ax_acc.plot(
        x,
        sub["acc"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Accuracy",
    )

    ax_acc.set_title(f"Accuracy and Unseen-event Count - {dataset}")
    ax_acc.set_xlabel("Window index")
    ax_acc.set_ylabel("Accuracy")
    ax_unseen.set_ylabel("Unseen count")
    ax_acc.set_ylim(0, 1.05)

    max_unseen = float(sub["unseen_count"].max()) if len(sub) else 0.0
    ax_unseen.set_ylim(0, max(1.0, max_unseen * 1.25))

    ax_acc.grid(True, axis="y", alpha=0.3)

    line_handles, line_labels = ax_acc.get_legend_handles_labels()
    bar_handles, bar_labels = ax_unseen.get_legend_handles_labels()
    ax_acc.legend(
        line_handles + bar_handles,
        line_labels + bar_labels,
        loc="lower left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_unseen_event_ratio_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    output_path: str,
) -> None:
    """Plot one dataset's accuracy line and unseen_event_ratio bar chart."""
    sub = df[df["dataset"] == dataset].sort_values("window_index")
    if sub.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    x = sub["window_index"]

    fig, ax_acc = plt.subplots(figsize=(10, 5))
    ax_unseen = ax_acc.twinx()

    ax_unseen.bar(
        x,
        sub["unseen_event_ratio"],
        alpha=0.25,
        label="Unseen event ratio",
    )

    ax_acc.plot(
        x,
        sub["acc"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Accuracy",
    )

    ax_acc.set_title(f"Accuracy and Unseen-event Ratio - {dataset}")
    ax_acc.set_xlabel("Window index")
    ax_acc.set_ylabel("Accuracy")
    ax_unseen.set_ylabel("Unseen event ratio")
    ax_acc.set_ylim(0, 1.05)

    max_unseen = float(sub["unseen_event_ratio"].max()) if len(sub) else 0.0
    ax_unseen.set_ylim(0, max(0.05, min(1.0, max_unseen * 1.25)))

    ax_acc.grid(True, axis="y", alpha=0.3)

    line_handles, line_labels = ax_acc.get_legend_handles_labels()
    bar_handles, bar_labels = ax_unseen.get_legend_handles_labels()
    ax_acc.legend(
        line_handles + bar_handles,
        line_labels + bar_labels,
        loc="lower left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_novel_decomposition_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    output_path: str,
) -> None:
    """
    Plot one dataset's accuracy line and stacked bars for novel-event decomposition.

    Stacked bars:
        learned_novel_event_ratio:
            events that were unseen in pretraining but have already been learned
            by the current model before this window.

        local_unseen_event_ratio:
            events that are still unseen to the current model before this window.

    The stacked bar height should approximately equal:
        global_unseen_event_ratio
    """
    sub = df[df["dataset"] == dataset].sort_values("window_index").copy()

    if sub.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")

    # Exclude overall row if it accidentally remains in the dataframe
    if "window_id" in sub.columns:
        sub = sub[sub["window_id"].astype(str).str.lower() != "overall"]

    if sub.empty:
        raise ValueError(f"No valid window rows found for dataset: {dataset}")

    required = [
        "window_index",
        "acc",
        "learned_novel_event_ratio",
        "local_unseen_event_ratio",
        "global_unseen_event_ratio",
    ]

    missing = [col for col in required if col not in sub.columns]
    if missing:
        raise ValueError(
            f"Missing columns for novel decomposition plot: {missing}"
        )

    x = sub["window_index"]
    learned_ratio = sub["learned_novel_event_ratio"].fillna(0.0)
    local_ratio = sub["local_unseen_event_ratio"].fillna(0.0)
    global_ratio = sub["global_unseen_event_ratio"].fillna(0.0)

    fig, ax_acc = plt.subplots(figsize=(11, 5.5))
    ax_ratio = ax_acc.twinx()

    # Stacked bars
    ax_ratio.bar(
        x,
        learned_ratio,
        alpha=0.45,
        label="Learned novel event ratio",
    )

    ax_ratio.bar(
        x,
        local_ratio,
        bottom=learned_ratio,
        alpha=0.45,
        label="Still-unlearned unseen event ratio",
    )

    # Optional: draw global unseen ratio as a thin reference line.
    # This helps readers see that stacked bars match the global unseen ratio.
    ax_ratio.plot(
        x,
        global_ratio,
        linestyle="--",
        linewidth=1.2,
        marker="",
        label="Global unseen event ratio",
    )

    # Accuracy line
    ax_acc.plot(
        x,
        sub["acc"],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="Accuracy",
    )

    ax_acc.set_title(
        f"Accuracy and Novel-event Decomposition - {dataset}"
    )
    ax_acc.set_xlabel("Window index")
    ax_acc.set_ylabel("Accuracy")
    ax_ratio.set_ylabel("Event ratio")

    ax_acc.set_ylim(0, 1.05)

    max_ratio = float(
        max(
            global_ratio.max(),
            (learned_ratio + local_ratio).max(),
            0.0,
        )
    )

    ax_ratio.set_ylim(0, max(0.05, min(1.0, max_ratio * 1.25)))

    ax_acc.grid(True, axis="y", alpha=0.3)

    line_handles, line_labels = ax_acc.get_legend_handles_labels()
    bar_handles, bar_labels = ax_ratio.get_legend_handles_labels()

    ax_acc.legend(
        line_handles + bar_handles,
        line_labels + bar_labels,
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_unseen_each_dataset(
    df: pd.DataFrame,
    output_dir: str,
    datasets: Iterable[str] | None = None,
) -> None:
    """Save one accuracy/unseen-ratio figure for each dataset."""
    os.makedirs(output_dir, exist_ok=True)

    if datasets is None:
        datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        output_path = os.path.join(output_dir, f"accuracy_unseen_{_safe_filename(dataset)}.png")
        plot_accuracy_unseen_for_dataset(df, dataset, output_path)


def plot_accuracy_unseen_count_each_dataset(
    df: pd.DataFrame,
    output_dir: str,
    datasets: Iterable[str] | None = None,
) -> None:
    """Save one accuracy/unseen-count figure for each dataset."""
    os.makedirs(output_dir, exist_ok=True)

    if datasets is None:
        datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        output_path = os.path.join(
            output_dir,
            f"accuracy_unseen_count_{_safe_filename(dataset)}.png"
        )
        plot_accuracy_unseen_count_for_dataset(df, dataset, output_path)


def plot_accuracy_unseen_event_ratio_each_dataset(
    df: pd.DataFrame,
    output_dir: str,
    datasets: Iterable[str] | None = None,
) -> None:
    """Save one accuracy/unseen-event-ratio figure for each dataset."""
    os.makedirs(output_dir, exist_ok=True)

    if datasets is None:
        datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        output_path = os.path.join(
            output_dir,
            f"accuracy_unseen_event_ratio_{_safe_filename(dataset)}.png"
        )
        plot_accuracy_unseen_event_ratio_for_dataset(df, dataset, output_path)

def plot_accuracy_novel_decomposition_each_dataset(
    df: pd.DataFrame,
    output_dir: str,
    datasets: Iterable[str] | None = None,
) -> None:
    """
    Save one accuracy + novel-event decomposition figure for each dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    if datasets is None:
        datasets = sorted(df["dataset"].unique())

    for dataset in datasets:
        output_path = os.path.join(
            output_dir,
            f"accuracy_novel_decomposition_{_safe_filename(dataset)}.png"
        )

        plot_accuracy_novel_decomposition_for_dataset(
            df=df,
            dataset=dataset,
            output_path=output_path,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot window-level PPM metrics from Excel.")
    parser.add_argument("--excel_path", type=str, default="window_metrics.xlsx")
    parser.add_argument("--sheet_name", type=str, default="all_windows")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument(
        "--normalized_x",
        action="store_true",
        help="Use normalized window position for the all-dataset accuracy plot.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_window_metrics(args.excel_path, args.sheet_name)

    # 1) One figure: all datasets' accuracy lines
    #all_accuracy_path = os.path.join(args.output_dir, "accuracy_all_datasets.png")
    #plot_accuracy_all_datasets(
    #    df,
    #    output_path=all_accuracy_path,
    #    use_normalized_x=args.normalized_x,
    #)

    # 2) One figure per dataset: accuracy line + unseen_ratio bars
    #individual_dir = os.path.join(args.output_dir, "accuracy_unseen_by_dataset")
    #plot_accuracy_unseen_each_dataset(df, output_dir=individual_dir)

    #print(f"Saved all-dataset accuracy figure: {all_accuracy_path}")
    #print(f"Saved per-dataset figures to: {individual_dir}")


    # 3) One figure per dataset: accuracy line + unseen_count bars
    #count_dir = os.path.join(args.output_dir, "accuracy_unseen_count_by_dataset")
    #plot_accuracy_unseen_count_each_dataset(df, output_dir=count_dir)

    #print(f"Saved per-dataset unseen-count figures to: {count_dir}")


    # 4) One figure per dataset: accuracy line + unseen_event_ratio bars
    #event_ratio_dir = os.path.join(args.output_dir, "accuracy_unseen_event_ratio_by_dataset")
    #plot_accuracy_unseen_event_ratio_each_dataset(df, output_dir=event_ratio_dir)

    #print(f"Saved per-dataset unseen-event-ratio figures to: {event_ratio_dir}")

    # 5) One figure per dataset:
    # accuracy line + stacked bars for learned-novel/current-unseen decomposition
    novel_decomp_dir = os.path.join(args.output_dir, "accuracy_novel_decomposition_by_dataset")

    plot_accuracy_novel_decomposition_each_dataset(df, output_dir=novel_decomp_dir)

    print(f"Saved per-dataset novel-decomposition figures to: {novel_decomp_dir}")

if __name__ == "__main__":
    main()
