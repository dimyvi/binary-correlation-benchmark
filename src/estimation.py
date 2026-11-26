import math
import numpy as np

def count_pairs(chain: np.ndarray) -> tuple[int, int, int, int]:
    curr_s = chain[:-1]
    next_s = chain[1:]
    
    n00 = int(np.sum((curr_s == 0) & (next_s == 0)))
    n01 = int(np.sum((curr_s == 0) & (next_s == 1)))
    n10 = int(np.sum((curr_s == 1) & (next_s == 0)))
    n11 = int(np.sum((curr_s == 1) & (next_s == 1)))
    return n00, n01, n10, n11


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sample_correlation(chain: np.ndarray) -> float:
    n00, n01, n10, n11 = count_pairs(chain)
    denom = (n11 + n10) * (n11 + n01) * (n01 + n00) * (n10 + n00)
    
    if denom == 0:
        return np.nan
        
    r = (n11 * n00 - n10 * n01) / math.sqrt(denom)
    return float(r)


def estimate_joint_probs(chain: np.ndarray) -> tuple[float, float, float, float]:
    n00, n01, n10, n11 = count_pairs(chain)
    n_pairs = len(chain) - 1
    
    return (n00 / n_pairs, n01 / n_pairs, n10 / n_pairs, n11 / n_pairs)


def estimate_conditional_probs(chain: np.ndarray) -> tuple[float, float]:
    n00, n01, n10, n11 = count_pairs(chain)
    
    n0 = n00 + n01
    n1 = n10 + n11
    
    p01_hat = (n01 / n0) if n0 > 0 else np.nan
    p11_hat = (n11 / n1) if n1 > 0 else np.nan
    
    return (float(p01_hat), float(p11_hat))