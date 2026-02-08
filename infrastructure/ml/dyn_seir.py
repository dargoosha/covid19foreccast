from __future__ import annotations

import numpy as np


def seir_simulate_dyn(
    y: np.ndarray,
    N: float,
    beta: float,
    sigma: float,
    gamma: float,
    E0_factor: float,
    I0_days: int,
    R0_init: float,
) -> dict[str, np.ndarray]:
    T = len(y)

    I0 = float(np.sum(y[: max(1, int(I0_days))]))
    E0 = float(E0_factor) * I0
    S0 = max(float(N) - E0 - I0 - float(R0_init), 0.0)

    S = np.zeros(T, dtype=np.float32)
    E = np.zeros(T, dtype=np.float32)
    I = np.zeros(T, dtype=np.float32)
    R = np.zeros(T, dtype=np.float32)

    S[0], E[0], I[0], R[0] = S0, E0, I0, float(R0_init)

    for t in range(1, T):
        inf = beta * S[t - 1] * I[t - 1] / float(N)
        inc = sigma * E[t - 1]
        rec = gamma * I[t - 1]

        S[t] = max(S[t - 1] - inf, 0.0)
        E[t] = max(E[t - 1] + inf - inc, 0.0)
        I[t] = max(I[t - 1] + inc - rec, 0.0)
        R[t] = max(R[t - 1] + rec, 0.0)

    return {
        "S": S,
        "E": E,
        "I": I,
        "R": R,
        "sigmaE": (sigma * E).astype(np.float32),
    }


def fit_seir_on_train(
    y_train: np.ndarray,
    N: float,
    beta_grid: tuple[float, float, int],
    gamma_grid: tuple[float, float, int],
    sigma: float,
    E0_factor: float,
    I0_days: int,
    R0_init: float,
    fit_target: str,
) -> tuple[float, float]:
    b0, b1, bn = beta_grid
    g0, g1, gn = gamma_grid

    betas = np.linspace(float(b0), float(b1), int(bn))
    gammas = np.linspace(float(g0), float(g1), int(gn))

    best_loss = np.inf
    best_beta = float(betas[0])
    best_gamma = float(gammas[0])

    for beta in betas:
        for gamma in gammas:
            sim = seir_simulate_dyn(
                y_train,
                float(N),
                float(beta),
                float(sigma),
                float(gamma),
                float(E0_factor),
                int(I0_days),
                float(R0_init),
            )
            target = sim[fit_target]
            loss = float(np.mean((y_train - target) ** 2))
            if loss < best_loss:
                best_loss = loss
                best_beta = float(beta)
                best_gamma = float(gamma)

    return best_beta, best_gamma
