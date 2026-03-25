from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .logging_utils import get_logger

logger = get_logger(__name__)

def plot_class_distribution(y: pd.Series, save_path: Path):
    logger.info("Plotting class distribution")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = y.value_counts().sort_index()
    labels = ["No Diabetes", "Diabetes"]

    sns.barplot(x=labels, y=counts.values, ax=axes[0])
    axes[0].set_title("Class Distribution")
    axes[0].set_ylabel("Count")

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=["skyblue", "salmon"],
    )
    axes[1].set_title("Class Distribution (Percentage)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved class distribution figure to {save_path}")

def plot_feature_distributions(df: pd.DataFrame, save_path: Path):
    logger.info("Plotting feature distributions")
    features = df.columns
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]
        sns.histplot(df[feature], kde=True, ax=ax, color="#1f77b4")
        ax.set_title(feature)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved feature distributions figure to {save_path}")

def plot_correlation_matrix(df: pd.DataFrame, save_path: Path):
    logger.info("Plotting correlation matrix")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved correlation matrix figure to {save_path}")

def plot_roc_curves(
    curves: Dict[str, Dict[str, List[float]]],
    save_path: Path,
):
    logger.info("Plotting ROC curves")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in curves.items():
        ax.plot(data["fpr"], data["tpr"], label=name)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved ROC curves figure to {save_path}")