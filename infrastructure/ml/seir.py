from __future__ import annotations

import numpy as np


def simulate_seir(
    days: int,
    N: float,
    beta: float,
    sigma: float,
    gamma: float,
    E0: float = 10.0,
    I0: float = 10.0,
    R0: float = 0.0,
) -> np.ndarray:
    S = N - E0 - I0 - R0
    E, I, R = E0, I0, R0
    new_cases = []
    for _ in range(days):
        new_exposed = beta * S * I / N
        new_infectious = sigma * E
        new_recovered = gamma * I

        S -= new_exposed
        E += new_exposed - new_infectious
        I += new_infectious - new_recovered
        R += new_recovered

        new_cases.append(max(new_infectious, 0.0))
    return np.array(new_cases, dtype=np.float32)
