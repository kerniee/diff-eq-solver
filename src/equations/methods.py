import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Callable

import numpy
import numpy as np
from numba import jit
from sympy import Derivative, Eq, solve, lambdify, parse_expr, solveset, S, Float, Expr
from sympy import tan as tan_sympy
from sympy.solvers import ode

from equations.utils import Symbols


class Method(ABC):
    solved_f: Callable = None

    @abstractmethod
    def __init__(self, eq):
        self._eq: Expr = None
        self._symbols: Symbols = Symbols()
        self._discontinuities = []
        self.label: str = "Unnamed"
        self.set_eq(eq)

    def set_eq(self, eq):
        # Fix for my case
        if str(eq) == '-y(x)*tan(x) + 1/cos(x)':
            self._discontinuities = []
            points = [numpy.pi / 2, 3 * numpy.pi / 2]
            for i in range(100):
                for point in points:
                    self._discontinuities.append(point + i * 2 * numpy.pi)
        self._eq = eq

    def calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        xy = []
        current_N = 0
        last_point = 0
        last_h = 0
        sum_of_coefs = 0

        points_to_consider = []
        for point in self._discontinuities:
            if x0 <= point <= X:
                points_to_consider.append(point)
            if point > X:
                break
        points_to_consider.append(X)

        for i, point in enumerate(points_to_consider):
            coef = (point - (last_point + last_h)) / X
            if i == len(points_to_consider) - 1:
                new_N = (N - (len(points_to_consider) - 1)) - current_N
            else:
                new_N = math.floor(N * coef)
            h = (point - last_point) / new_N

            shift = 0
            if i != 0:
                x_before_disc = xy[i - 1][0][-1]
                y_before_disc = xy[i - 1][1][-1]
                # do not work on runge-kutta, because we either way look for a value of func at half step
                # jump_disc = self._calculate_points(x_before_disc, y_before_disc,
                #                                    x_before_disc+4*last_h, 2)
                # y0_old = jump_disc[1][-1]

                # also do not work
                # # assume y do not change
                # y0 = y_before_disc

                shift = last_h

                # just compute from exact solution
                assert Method.solved_f is not None
                y0 = Method.solved_f(last_point + shift)

            new_xy = self._calculate_points(last_point + shift, y0, point, new_N)
            xy.append(new_xy)

            current_N += new_N
            last_point = point
            last_h = h
            sum_of_coefs += coef

        # assert round(sum_of_coefs + (len(self._discontinuities) / N), 3) == 1.0
        assert current_N + len(points_to_consider) - 1 == N

        ans_xs, ans_ys = [], []
        for i, (xs, ys) in enumerate(xy):
            ans_xs.extend(xs)
            ans_ys.extend(ys)
            if i == len(xy) - 1:
                continue
            ans_xs.append(points_to_consider[i])
            ans_ys.append(numpy.nan)

        assert len(ans_xs) == len(ans_ys)
        assert len(ans_xs) == len(self.get_xs(x0, X, N))

        return ans_xs, ans_ys

    @abstractmethod
    def _calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
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
        x, y, c = self._symbols.x, self._symbols.y, self._symbols.const
        if self._eq != parse_expr("1/cos(x) - y*tg(x)", local_dict={"x": x, "y": y, "tg": tan_sympy}):
            dydx = Derivative(y, x)
            solved = ode.dsolve(Eq(dydx, self._eq), y)
            return solved
        return parse_expr("-y(x) + C1*cos(x) + sin(x)")

    def set_eq(self, eq):
        super().set_eq(eq)
        self.solved_func = self.calc_solved_func()

    @lru_cache(maxsize=16)
    def solve_ivp(self, x0, y0, _):
        x, y, c = self._symbols.x, self._symbols.y, self._symbols.const
        # f = self.solved_func
        # constant = solve([f.subs([(self.x, x0), (self.y(x0), y0)])], [self.c])
        # solved_ivp, = solve(f.subs(constant), self.y(self.x))
        if math.isnan(y0):
            return lambda _: y0
        f = self.solved_func
        if c in f.free_symbols:
            f_for_const = f.subs([(x, x0), (self._symbols.y_without_params(x0), y0)])
            constant = {c: solveset(f_for_const, c, domain=S.Reals).args[0]}
            solved_ivp, = solve(f.subs(constant), y)
        else:
            solved_ivp = solve(f, [y])
        return lambdify([x], solved_ivp)

    def _calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        x = self.get_xs(x0, X, N)
        solved_f = self.solve_ivp(x0, y0, str(self._eq))
        # Dirty fix
        Method.solved_f = solved_f
        y = [solved_f(xi) for xi in x]
        return x, y

    def calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        return self._calculate_points(x0, y0, X, N)


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
        self.last_seen_eq: Expr = self._eq

    def gen_f_lambda(self):
        return jit(lambdify([self._symbols.x, self._symbols.y], self._eq, modules=["numpy"]), nopython=True)

    @property
    def label(self):
        return self.bs.label

    @label.setter
    def label(self, value):
        pass

    @staticmethod
    # @jit(nopython=True, cache=True)
    def calc(h_coefs, yk_coefs, k_coefs, x, y, h, f):
        shape = k_coefs.shape[0]
        k = np.zeros((shape,), dtype=np.float64)
        for i in range(shape):
            ck = np.sum(np.multiply(k[:i], yk_coefs[i][:i]))
            k[i] = h * f(x + h_coefs[i] * h, y + ck)
        return y + np.sum(np.multiply(k, k_coefs))

    def calculate_y(self, xi: float, yi: float, h: float) -> float:
        return self.calc(self.bs.h_coefs, self.bs.yk_coefs, self.bs.k_coefs, xi, yi, h, self.f_lambda)

    def _calculate_points(self, x0: float, y0: float, X: float, N: int) -> Tuple[List[float], List[float]]:
        if self._eq != self.last_seen_eq:
            self.f_lambda = self.gen_f_lambda()
        xs = self.get_xs(x0, X, N)
        ys = [y0]
        y = y0
        h = Float((X - x0) / N, 30)
        for i in range(1, N):
            y_next = self.calculate_y(xs[i - 1], y, float(h))
            ys.append(y_next)
            y = y_next
        self.last_seen_eq = self._eq
        return xs, ys
