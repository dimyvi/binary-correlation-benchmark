import numpy as np
from src.generator import build_transition_matrix, generate_chain


def check_params(length: int, p_one: float, rho: float) -> None:
    """Проверяет корректность параметров генерации цепи Маркова."""
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)) or length < 2:
        raise ValueError("length должен быть целым числом >= 2 (нужна хотя бы 1 пара соседей)")

    if not (0.0 < p_one < 1.0):
        raise ValueError("p_one должен быть строго между 0 и 1")

    min_rho = max(-p_one / (1.0 - p_one), -(1.0 - p_one) / p_one)
    if not (min_rho <= rho <= 1.0):
        raise ValueError(f"rho должен быть в диапазоне [{min_rho:.3f}, 1.0]")


def check_matrix(matrix: np.ndarray) -> None:
    """Проверяет корректность стохастической матрицы переходов 2x2."""
    if not isinstance(matrix, np.ndarray) or matrix.shape != (2, 2):
        raise ValueError("Матрица должна быть numpy.ndarray размером 2x2")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Матрица содержит NaN или Inf")
    if np.any((matrix < 0.0) | (matrix > 1.0)) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("Некорректные вероятности в матрице (сумма по строкам должна быть 1)")


def check_chain(chain: np.ndarray) -> None:
    """Проверяет, что последовательность является одномерным бинарным массивом длины >= 2."""
    if not isinstance(chain, np.ndarray) or chain.ndim != 1:
        raise ValueError("chain должен быть одномерным массивом numpy.ndarray")
    if len(chain) < 2:
        raise ValueError("Длина последовательности должна быть >= 2 для анализа пар")
    if not np.all(np.isin(chain, (0, 1))):
        raise ValueError("Последовательность должна состоять строго из символов 0 и 1")


def check_config(lengths: list[int], p_ones: list[float], rhos: list[float], n_trials: int, target_p: float, target_rho: float) -> None:
    """Проверяет настройки из config.py перед запуском эксперимента и графиков."""
    if not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("N_TRIALS должен быть целым числом >= 1")
    if not lengths or not p_ones or not rhos:
        raise ValueError("Списки LENGTHS, P_ONES и RHOS не могут быть пустыми")

    for length in lengths:
        for p_one in p_ones:
            for rho in rhos:
                check_params(length, p_one, rho)

    if target_p not in p_ones:
        raise ValueError(f"TARGET_P={target_p} отсутствует в P_ONES={p_ones}")
    if target_rho not in rhos or target_rho == 0.0:
        raise ValueError(f"TARGET_RHO={target_rho} должен быть положительным числом из RHOS={rhos}")


def run_safe_generator(length: int, p_one: float, rho: float, seed: int | None = None) -> np.ndarray:
    """Безопасный запуск генератора с полной валидацией параметров и матрицы."""
    check_params(length, p_one, rho)
    matrix = build_transition_matrix(p_one, rho)
    check_matrix(matrix)
    chain = generate_chain(length, p_one, matrix, seed)
    check_chain(chain)
    return chain


def run_safe_estimation(estimator_func, chain: np.ndarray, *args, **kwargs):
    """Безопасный запуск любого метода оценки из estimation.py с проверкой входной цепочки."""
    check_chain(chain)
    return estimator_func(chain, *args, **kwargs)
