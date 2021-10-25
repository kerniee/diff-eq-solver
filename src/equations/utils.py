from dataclasses import dataclass

from sympy import Symbol, Function


@dataclass
class Symbols:
    x: Symbol
    y: Function
    y_without_params: Function
    c: Symbol

    def __init__(self):
        self.x = Symbol("x")
        self.y_without_params = Function("y")
        self.y = self.y_without_params(self.x)
        self.const = Symbol("C1")
