"""Вспомогательные графики для отчёта по корреляционному анализу (часть 2).

"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from correlation_tasks import FEATURES


def load_brain_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep="\t", na_values=["NA"])


def plot_scatter_matrix(df: pd.DataFrame) -> None:
    """Scatter-графики каждого признака против MRI_Count, по полу."""
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4 * len(FEATURES), 4))
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(data=df, x=feature, y="MRI_Count", hue="Gender", ax=ax)
        ax.set_title(feature)
    fig.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap корреляций отдельно для мужчин и женщин."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gender in zip(axes, ["Male", "Female"]):
        subset = df[df["Gender"] == gender][FEATURES + ["MRI_Count"]]
        sns.heatmap(subset.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title(gender)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_path = script_dir / "brainsize.txt"
    
    data = load_brain_data(str(data_path))
    plot_scatter_matrix(data)
    plot_correlation_heatmap(data)
