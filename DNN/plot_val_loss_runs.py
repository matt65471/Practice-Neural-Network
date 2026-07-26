from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path(__file__).parent / "val_loss_history.csv"
OUTPUT_PATH = Path(__file__).parent / "val_loss_all_runs.png"


def plot_all_runs(csv_path: Path = CSV_PATH, output_path: Path = OUTPUT_PATH) -> None:
    df = pd.read_csv(csv_path)

    fig, (ax_zoom, ax_full) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for run_name, group in df.groupby("run_name"):
        group = group.sort_values("epoch")
        ax_full.plot(
            group["epoch"],
            group["val_loss"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=run_name,
        )

    stable_run_names = [
        name
        for name, group in df.groupby("run_name")
        if group["val_loss"].max() < 2.5
    ]
    for run_name in stable_run_names:
        group = df[df["run_name"] == run_name].sort_values("epoch")
        if group.empty:
            continue
        ax_zoom.plot(
            group["epoch"],
            group["val_loss"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=run_name,
        )

    ax_zoom.set_ylabel("Validation Loss")
    ax_zoom.set_title("Comparable Runs (zoomed)")
    ax_zoom.legend(loc="best", fontsize=9)
    ax_zoom.grid(True, alpha=0.3)

    ax_full.set_xlabel("Epoch")
    ax_full.set_ylabel("Validation Loss")
    ax_full.set_title("All Runs")
    ax_full.legend(loc="best", fontsize=9)
    ax_full.grid(True, alpha=0.3)

    fig.suptitle("Validation Loss Across All Runs", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    plot_all_runs()
