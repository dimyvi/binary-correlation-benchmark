# Настройки для main.py
LENGTHS = [20, 50, 100, 500, 1000]
P_ONES = [0.1, 0.3, 0.5, 0.7, 0.9]
RHOS = [0.0, 0.1, 0.3, 0.5]
N_TRIALS = 10_000
SEED = 42
SUMMARY_CSV = "results/summary.csv"


# Настройки для plot_results.py значения должны быть из списков P_ONES и RHOS (кроме 0.0)
PLOTS_DIR = "results/plots"
TARGET_P = 0.3
TARGET_RHO = 0.3
