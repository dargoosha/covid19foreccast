# seir_model.py

import numpy as np
from scipy.integrate import odeint
from typing import Dict, Sequence


def seir_ode(y, t, beta, sigma, gamma, N):
    """
    Класичне SEIR-ОДР з постійними параметрами.
    Використовується для базового (статичного) SEIR-прогнозу.
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def run_seir_simulation(
    days: int,
    beta: float,
    sigma: float,
    gamma: float,
    N: float,
    E0: float,
    I0: float,
) -> Dict[str, np.ndarray]:
    """
    Статична SEIR-модель з фіксованими параметрами.
    Повертає траєкторії S, E, I, R та new_cases = sigma * E.
    """
    R0 = 0.0
    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]

    t = np.arange(days)
    ret = odeint(seir_ode, y0, t, args=(beta, sigma, gamma, N))
    S, E, I, R = ret.T

    # new_cases ~ потік з E до I: sigma * E(t)
    new_cases = sigma * E

    return {"t": t, "S": S, "E": E, "I": I, "R": R, "new_cases": new_cases}


# -------------------------------------------------------------------------
# Time-varying SEIR: β(t), σ(t), γ(t)
# -------------------------------------------------------------------------


def seir_ode_time_varying(
    y,
    t,
    beta_series: np.ndarray,
    sigma_series: np.ndarray,
    gamma_series: np.ndarray,
    N: float,
):
    """
    SEIR-ОДР з параметрами, які змінюються в часі:
      beta(t), sigma(t), gamma(t).
    Значення параметрів беруться з дискретних рядів за індексом t (дні).
    """
    S, E, I, R = y

    # t може бути нецілим, тому округляємо до найближчого дня
    idx = int(round(float(t)))
    idx = max(0, min(idx, len(beta_series) - 1))

    beta_t = float(beta_series[idx])
    sigma_t = float(sigma_series[idx])
    gamma_t = float(gamma_series[idx])

    dSdt = -beta_t * S * I / N
    dEdt = beta_t * S * I / N - sigma_t * E
    dIdt = sigma_t * E - gamma_t * I
    dRdt = gamma_t * I
    return [dSdt, dEdt, dIdt, dRdt]


def run_seir_simulation_time_varying(
    beta_series: Sequence[float],
    sigma_series: Sequence[float],
    gamma_series: Sequence[float],
    N: float,
    E0: float,
    I0: float,
) -> Dict[str, np.ndarray]:
    """
    Багатоденна симуляція SEIR з time-varying параметрами β(t), σ(t), γ(t).

    beta_series, sigma_series, gamma_series – масиви довжини days.
    Повертає:
      t, S, E, I, R, new_cases = sigma(t) * E(t).
    """
    beta_series = np.asarray(beta_series, dtype=float)
    sigma_series = np.asarray(sigma_series, dtype=float)
    gamma_series = np.asarray(gamma_series, dtype=float)

    assert (
        beta_series.shape == sigma_series.shape == gamma_series.shape
    ), "beta, sigma, gamma series must have однакову довжину"

    days = len(beta_series)

    R0 = 0.0
    S0 = N - E0 - I0 - R0
    y0 = [S0, E0, I0, R0]

    t = np.arange(days)
    ret = odeint(
        seir_ode_time_varying,
        y0,
        t,
        args=(beta_series, sigma_series, gamma_series, N),
    )
    S, E, I, R = ret.T

    new_cases = sigma_series * E

    return {
        "t": t,
        "S": S,
        "E": E,
        "I": I,
        "R": R,
        "new_cases": new_cases,
    }
