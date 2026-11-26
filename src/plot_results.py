import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import LENGTHS, P_ONES, RHOS, N_TRIALS, SUMMARY_CSV, PLOTS_DIR, TARGET_P, TARGET_RHO
from src.validator import check_config

check_config(LENGTHS, P_ONES, RHOS, N_TRIALS, TARGET_P, TARGET_RHO)


METHOD_STYLES = {
    "Sample Correlation (r)": {
        "label": "Метод 1: Выборочная корреляция (r)",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
        "markersize": 8,
        "linewidth": 3.5,
        "alpha": 0.55,
    },
    "Joint Probability P(1,1)": {
        "label": "Метод 2: Совместная вероятность P(1,1)",
        "color": "#ff7f0e",
        "linestyle": "-.",
        "marker": "s",
        "markersize": 6,
        "linewidth": 2.2,
        "alpha": 0.95,
    },
    "Conditional Diff P(1|1)-P(1|0)": {
        "label": "Метод 3: Разность условных вероятностей",
        "color": "#2ca02c",
        "linestyle": "--",
        "marker": "^",
        "markersize": 6,
        "linewidth": 2.0,
        "alpha": 0.95,
    },
}


def setup_axis(ax: plt.Axes, lengths: list[int], ylabel: str, title: str) -> None:
    if len(lengths) > 1 and min(lengths) > 0:
        ax.set_xscale("log")
        ax.set_xticks(lengths)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Длина последовательности (символов)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def draw_methods(ax: plt.Axes, sub: pd.DataFrame, metric_col: str) -> bool:
    has_data = False
    for method_name, style in METHOD_STYLES.items():
        method_df = sub[sub["Estimation Method"] == method_name].copy()
        method_df[metric_col] = pd.to_numeric(method_df[metric_col], errors="coerce")
        method_df = method_df.dropna(subset=[metric_col])

        if method_df.empty:
            continue

        has_data = True
        ax.plot(
            method_df["Chain Length"],
            method_df[metric_col],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=style["markersize"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
        )
    return has_data


def plot_bias_comparison(df: pd.DataFrame, p_one: float, rho: float, lengths: list[int], out_dir: Path) -> None:
    sub = df[(df["Prob of 1 (p)"] == p_one) & (df["True Correlation (rho)"] == rho)]
    fig, ax = plt.subplots(figsize=(9, 5))

    if draw_methods(ax, sub, "Mean Bias"):
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
        setup_axis(
            ax,
            lengths,
            ylabel="Среднее смещение (Mean Bias)",
            title=f"1. Сравнение смещения оценок всех методов (p = {p_one}, rho = {rho})",
        )
        fig.savefig(out_dir / "1_bias_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_std_comparison(df: pd.DataFrame, p_one: float, rho: float, lengths: list[int], out_dir: Path) -> None:
    sub = df[(df["Prob of 1 (p)"] == p_one) & (df["True Correlation (rho)"] == rho)]
    fig, ax = plt.subplots(figsize=(9, 5))

    if draw_methods(ax, sub, "Std Dev (Spread)"):
        setup_axis(
            ax,
            lengths,
            ylabel="Разброс оценок (Std Dev)",
            title=f"2. Сравнение разброса оценок всех методов (p = {p_one}, rho = {rho})",
        )
        fig.savefig(out_dir / "2_std_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_detection_comparison(df: pd.DataFrame, p_one: float, rho: float, lengths: list[int], out_dir: Path) -> None:
    sub = df[(df["Prob of 1 (p)"] == p_one) & (df["True Correlation (rho)"] == rho)]
    fig, ax = plt.subplots(figsize=(9, 5))

    if draw_methods(ax, sub, "Detection Rate"):
        setup_axis(
            ax,
            lengths,
            ylabel="Доля обнаружений (Detection Rate)",
            title=f"3. Сравнение доли обнаруженных зависимостей (p = {p_one}, rho = {rho})",
        )
        fig.savefig(out_dir / "3_detection_rate_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_false_positive_comparison(df: pd.DataFrame, p_one: float, lengths: list[int], out_dir: Path) -> None:
    sub = df[(df["Prob of 1 (p)"] == p_one) & (df["True Correlation (rho)"] == 0.0)]
    fig, ax = plt.subplots(figsize=(9, 5))

    if draw_methods(ax, sub, "False Positive Rate"):
        setup_axis(
            ax,
            lengths,
            ylabel="Доля ложных срабатываний (False Positive Rate)",
            title=f"4. Сравнение ложных срабатываний при отсутствии связи (p = {p_one}, rho = 0.0)",
        )
        fig.savefig(out_dir / "4_false_positive_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    csv_path = Path(SUMMARY_CSV)
    if not csv_path.exists():
        print(f"Файл {SUMMARY_CSV} не найден.")
        return

    df = pd.read_csv(csv_path)

    out_dir = Path(PLOTS_DIR)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    lengths = sorted(df["Chain Length"].unique().tolist())

    plot_bias_comparison(df, TARGET_P, TARGET_RHO, lengths, out_dir)
    plot_std_comparison(df, TARGET_P, TARGET_RHO, lengths, out_dir)
    plot_detection_comparison(df, TARGET_P, TARGET_RHO, lengths, out_dir)
    plot_false_positive_comparison(df, TARGET_P, lengths, out_dir)

    print(f"Графики для p={TARGET_P}, rho={TARGET_RHO} сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
