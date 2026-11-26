import csv
from pathlib import Path
import numpy as np

from src.config import LENGTHS, P_ONES, RHOS, N_TRIALS, SEED, SUMMARY_CSV, TARGET_P, TARGET_RHO
from src.validator import check_config

from src.generator import build_transition_matrix, generate_chain
from src.estimation import (
    sample_correlation,
    estimate_joint_probs,
    estimate_conditional_probs,
)
from src.metrics import (
    calc_bias,
    calc_std,
    calc_detection_rate,
    calc_false_positive_rate,
)

check_config(LENGTHS, P_ONES, RHOS, N_TRIALS, TARGET_P, TARGET_RHO)

def summarize(estimates: np.ndarray, true_value: float, threshold: float, is_dep: bool, has_test: bool = True) -> dict:
    det_rate = "-"
    fp_rate = "-"

    if has_test:
        if is_dep:
            det_rate = calc_detection_rate(estimates, threshold)
        else:
            fp_rate = calc_false_positive_rate(estimates, threshold)

    return {
        "Mean Bias": calc_bias(estimates, true_value),
        "Std Dev (Spread)": calc_std(estimates),
        "Detection Rate": det_rate,
        "False Positive Rate": fp_rate,
    }


def run_correlation(chains: list[np.ndarray], true_rho: float) -> dict:
    estimates = np.array([sample_correlation(c) for c in chains])
    threshold = 2.0 / np.sqrt(len(chains[0]))
    return {
        "Estimation Method": "Sample Correlation (r)",
        **summarize(estimates, true_rho, threshold, true_rho != 0.0),
    }


def run_joint(chains: list[np.ndarray], p_one: float, p11: float, is_dep: bool) -> dict:
    p11_list = []
    for c in chains:
        _, _, _, p11_joint = estimate_joint_probs(c)
        p11_list.append(p11_joint)

    estimates = np.array(p11_list)
    true_p11_joint = p_one * p11
    return {
        "Estimation Method": "Joint Probability P(1,1)",
        **summarize(estimates, true_p11_joint, threshold=0.0, is_dep=is_dep, has_test=False),
    }


def run_conditional(chains: list[np.ndarray], p01: float, p11: float, is_dep: bool) -> dict:
    diffs = []
    for c in chains:
        p01_hat, p11_hat = estimate_conditional_probs(c)
        diffs.append(p11_hat - p01_hat)

    estimates = np.array(diffs)
    true_diff = p11 - p01
    threshold = 2.0 / np.sqrt(len(chains[0]))
    return {
        "Estimation Method": "Conditional Diff P(1|1)-P(1|0)",
        **summarize(estimates, true_diff, threshold, is_dep),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    output = Path(SUMMARY_CSV)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Chain Length",
        "Prob of 1 (p)",
        "True Correlation (rho)",
        "Estimation Method",
        "Mean Bias",
        "Std Dev (Spread)",
        "Detection Rate",
        "False Positive Rate",
    ]

    with open(output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for length in LENGTHS:
            for p_one in P_ONES:
                for rho in RHOS:
                    matrix = build_transition_matrix(p_one, rho)
                    (_, p01), (_, p11) = matrix
                    p01, p11 = float(p01), float(p11)
                    is_dep = (rho != 0.0)

                    chains = [
                        generate_chain(length, p_one, matrix, seed=int(rng.integers(0, 2**32 - 1)))
                        for _ in range(N_TRIALS)
                    ]

                    base = {
                        "Chain Length": length,
                        "Prob of 1 (p)": p_one,
                        "True Correlation (rho)": rho,
                    }
                    writer.writerow({**base, **run_correlation(chains, rho)})
                    writer.writerow({**base, **run_joint(chains, p_one, p11, is_dep)})
                    writer.writerow({**base, **run_conditional(chains, p01, p11, is_dep)})

            print(f"Серия для длины {length} завершена.")

    print(f"Готово! Таблица сохранена в {output}")


if __name__ == "__main__":
    main()
