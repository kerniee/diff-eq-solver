from typing import List

from sympy import parse_expr, tan, Expr

from equations.methods import Method
from equations.utils import Symbols


class Model:
    def __init__(self, s: str):
        self.s: str = s
        self.symbols: Symbols = Symbols()
        self.eq: Expr = self.generate_eq(s)
        self.methods: List[Method] = []

    def generate_eq(self, s):
        x, y = self.symbols.x, self.symbols.y
        s = s.replace("sec", "1/cos").replace("^", "**")
        return parse_expr(s, local_dict={"x": x, "y": y, "tg": tan})

    def change_eq(self, s):
        self.eq = self.generate_eq(s)
        for method in self.methods:
            method.set_eq(self.eq)

    def add_method(self, method: Method.__class__, *args, **kwargs) -> Method:
        m = method(self.eq, *args, **kwargs)
        self.methods.append(m)
        return m

    def calc_error(self, right_y, y):
        return [abs(y_exact - y) for y_exact, y in zip(right_y, y)]
