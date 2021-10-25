from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Callable

import numpy as np
from equations.utils import Symbols
from numba import jit
from sympy import Derivative, Eq, solve, lambdify, parse_expr, solveset, S, Float, Expr
from sympy import tan as tan_sympy
from sympy.solvers import ode


class Method(ABC):
    @abstractmethod
    def __init__(self, eq):
        self.eq: Expr = eq
        self.symbols: Symbols = Symbols()
        self.label: str = "Unnamed"

    def set_eq(self, eq):
        self.eq = eq

    @abstractmethod
    def calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        ...

    @staticmethod
    def get_xs(x0: float, X: float, N: int):
        h = (Float(X, 30) - Float(x0, 30)) / Float(N, 30)
        x0_f = Float(x0, 30)
        return [float(x0_f + h * Float(i)) for i in range(N)]
        # return list(np.linspace(start=x0, stop=X, num=N, dtype="float64"))


class ExactSolutionMethod(Method):
    def __init__(self, eq):
        super().__init__(eq)
        self.label: str = "Exact solution"
        self.solved_func: Expr = self.calc_solved_func()

    def calc_solved_func(self):
        # Solve ODE
        x, y, c = self.symbols.x, self.symbols.y, self.symbols.const
        if self.eq != parse_expr("1/cos(x) - y*tg(x)", local_dict={"x": x, "y": y, "tg": tan_sympy}):
            dydx = Derivative(y, x)
            solved = ode.dsolve(Eq(dydx, self.eq), y)
            return solved
        return parse_expr("-y(x) + C1*cos(x) + sin(x)")

    def set_eq(self, eq):
        super().set_eq(eq)
        self.solved_func = self.calc_solved_func()

    @lru_cache(maxsize=16)
    def solve_ivp(self, x0, y0, _):
        x, y, c = self.symbols.x, self.symbols.y, self.symbols.const
        # f = self.solved_func
        # constant = solve([f.subs([(self.x, x0), (self.y(x0), y0)])], [self.c])
        # solved_ivp, = solve(f.subs(constant), self.y(self.x))
        f = self.solved_func
        if c in f.free_symbols:
            f_for_const = f.subs([(x, x0), (self.symbols.y_without_params(x0), y0)])
            constant = {c: solveset(f_for_const, c, domain=S.Reals).args[0]}
            solved_ivp, = solve(f.subs(constant), y)
        else:
            solved_ivp = solve(f, [y])
        return lambdify([x], solved_ivp)

    def calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        x = self.get_xs(x0, X, N)
        y = [self.solve_ivp(x0, y0, self.eq)(xi) for xi in x]
        return x, y


@dataclass
class ButcherSchema:
    h_coefs: np.ndarray
    yk_coefs: np.ndarray
    k_coefs: np.ndarray
    label: str


class ButcherSchemaMethod(Method):
    def __init__(self, eq, butcher_schema: ButcherSchema):
        super().__init__(eq)
        self.bs: ButcherSchema = butcher_schema
        self.f_lambda: Callable = self.gen_f_lambda()
        self.last_seen_eq: Expr = self.eq

    def gen_f_lambda(self):
        return jit(lambdify([self.symbols.x, self.symbols.y], self.eq, modules=["numpy"]), nopython=True)

    @property
    def label(self):
        return self.bs.label

    @label.setter
    def label(self, value):
        pass

    @staticmethod
    @jit(nopython=True, cache=True)
    def calc(h_coefs, yk_coefs, k_coefs, x, y, h, f):
        shape = k_coefs.shape[0]
        k = np.zeros((shape,), dtype=np.float64)
        for i in range(shape):
            ck = np.sum(np.multiply(k[:i], yk_coefs[i][:i]))
            k[i] = h * f(x + h_coefs[i] * h, y + ck)
        return y + np.sum(np.multiply(k, k_coefs))

    def calculate_y(self, xi: float, yi: float, h: float) -> float:
        return self.calc(self.bs.h_coefs, self.bs.yk_coefs, self.bs.k_coefs, xi, yi, h, self.f_lambda)

    def calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        if self.eq != self.last_seen_eq:
            self.f_lambda = self.gen_f_lambda()
        xs = self.get_xs(x0, X, N)
        ys = [y0]
        y = y0
        h = Float((X - x0) / N, 30)
        for i in range(1, N):
            y_next = self.calculate_y(xs[i - 1], y, float(h))
            ys.append(y_next)
            y = y_next
        self.last_seen_eq = self.eq
        return xs, ys
