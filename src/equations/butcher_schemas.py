from typing import Iterable

import numpy as np

from equations.methods import ButcherSchema


def load_butcher_schemas() -> Iterable[ButcherSchema]:
    euler_method = ButcherSchema(
        np.array([0], dtype=np.float64),
        np.array([[0]], dtype=np.float64),
        np.array([1], dtype=np.float64),
        "Euler method"
    )
    improved_euler_method = ButcherSchema(
        np.array([0, 1], dtype=np.float64),
        np.array([[0, 0], [1, 0]], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        "Improved euler method"
    )
    runge_kutta_method = ButcherSchema(
        np.array([0, 0.5, 0.5, 1], dtype=np.float64),
        np.array([[0, 0, 0, 0],
                  [0.5, 0, 0, 0],
                  [0, 0.5, 0, 0],
                  [0, 0, 1, 0]], dtype=np.float64),
        np.array([1 / 6, 2 / 6, 2 / 6, 1 / 6], dtype=np.float64),
        "Runge-Kutta method"
    )
    return euler_method, improved_euler_method, runge_kutta_method
