import numpy as np

def build_transition_matrix(p_one: float, rho: float) -> np.ndarray:
    p01 = p_one * (1 - rho)
    p11 = p_one + (1 - p_one) * rho
    p00 = 1.0 - p01
    p10 = 1.0 - p11

    return np.array([
        [p00, p01],
        [p10, p11]
    ])


def generate_chain(length: int, p_one: float, matrix: np.ndarray, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)

    chain = np.zeros(length, dtype=int)
    rand_vals = rng.random(length)

    if rand_vals[0] < p_one:
        chain[0] = 1

    for i in range(1, length):
        prev_symbol = chain[i - 1]
        prob_of_one = matrix[prev_symbol, 1]

        if rand_vals[i] < prob_of_one:
            chain[i] = 1

    return chain