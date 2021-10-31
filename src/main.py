from functools import partial
from typing import List, Callable


# Dirty fix for -OO compilation with NumPy


def frozen_oo():
    """Check if code is frozen with optimization=2"""
    import sys
    if frozen_oo.__doc__ is None and hasattr(sys, 'frozen'):
        from ctypes import c_int, pythonapi
        c_int.in_dll(pythonapi, 'Py_OptimizeFlag').value = 2


frozen_oo()

from matplotlib import colors

from gui import View
from equations.butcher_schemas import load_butcher_schemas
from model import Model
from equations.methods import ExactSolutionMethod, ButcherSchemaMethod, Method


class Controller:
    def __init__(self, eq: str):
        self.recalc_func: Callable = self.recalc_main

        self.view: View = View("IVP_ODE_solver")
        self.view.x0_field.bind_on_change(self.recalc)
        self.view.y0_field.bind_on_change(self.recalc)
        self.view.X_field.bind_on_change(self.recalc)
        self.view.N_field.bind_on_change(self.recalc)
        self.view.btn_change_eq.configure({"command": self.change_eq})
        self.view.menubar.add_command(label="All methods + Errors", command=self.change_to_main_plot)
        self.view.menubar.add_command(label="Error graphs", command=self.change_to_error_plot)
        self.view.n0_field.bind_on_change(self.recalc)
        self.view.N_err_field.bind_on_change(self.recalc)

        self.model: Model = Model(eq)
        self.exact: Method = self.model.add_method(ExactSolutionMethod)
        schemas = load_butcher_schemas()
        for schema in schemas:
            m = self.model.add_method(ButcherSchemaMethod, schema)
            self.view.add_comp_method_btn(m.label, partial(self.change_method_to_draw, [self.exact, m]))
        self.methods_to_draw: List[Method] = self.model.methods
        self.view.add_comp_method_btn("Draw All", partial(self.change_method_to_draw, self.model.methods))

    def change_to_main_plot(self, recalc=True):
        self.view.fields.pack()
        self.view.error_fields.pack_forget()
        self.view.N_field.pack()
        self.recalc_func = self.recalc_main
        if recalc:
            self.recalc()
        self.view.canvas.figure = self.view.main_fig
        self.view.canvas.draw()

    def change_to_error_plot(self, recalc=True):
        self.view.fields.pack()
        self.view.error_fields.pack()
        self.view.N_field.pack_forget()
        self.recalc_func = self.recalc_error
        if recalc:
            self.recalc()
        self.view.canvas.figure = self.view.error_fig
        self.view.canvas.draw()

    def change_method_to_draw(self, m: List[Method]):
        self.methods_to_draw = m
        self.recalc()

    def run(self):
        self.change_to_main_plot(recalc=False)
        self.view.update()
        self.recalc()
        self.view.mainloop()

    def change_eq(self, *args, **kwargs):
        self.model.change_eq(self.view.eq_field.get("1.0", "end-1c"))
        self.recalc()

    def recalc(self, *args, **kwargs):
        with self.view.loading_screen():
            self.recalc_func(*args, **kwargs)

    def recalc_error(self, *args, **kwargs):
        x0, y0, X, n0, N = \
            (self.view.x0_field.value(),
             self.view.y0_field.value(),
             self.view.X_field.value(),
             self.view.n0_field.value(),
             self.view.N_err_field.value(),)
        self.view.clear_plot()
        for method in self.methods_to_draw:
            ys = []
            for curr_n in range(n0, N + 1):
                x, y = method.calculate_points(x0, y0, X, curr_n)
                if method == self.exact:
                    continue
                ys.append(max(self.model.calc_error(x, y)))
            if method == self.exact:
                continue
            self.view.all_error_ax.plot(list(range(n0, N + 1)), ys, label=method.label)
        self.view.all_error_ax.legend()
        self.view.canvas.draw()

    def recalc_main(self, *args, **kwargs):
        x0, y0, X, N = \
            (self.view.x0_field.value(),
             self.view.y0_field.value(),
             self.view.X_field.value(),
             self.view.N_field.value(),)
        self.view.clear_plot()
        if len(self.methods_to_draw) > 0:
            for i, method in enumerate(self.methods_to_draw):
                x, y = method.calculate_points(x0, y0, X, N)
                self.view.main_ax.plot(x, y, label=method.label)
                if method == self.exact:
                    continue
                self.draw_eps(x, y, method.label, i)
        self.view.main_ax.legend()
        self.view.canvas.draw()

    def draw_eps(self, x, y, label, i):
        clrs = list(colors.TABLEAU_COLORS.values())
        self.view.main_err_ax.plot(x, self.model.calc_error(x, y), label=label, color=clrs[i])
        self.view.main_err_ax.legend()


if __name__ == '__main__':
    s = "sec(x) - y*tg(x)"
    # s = "3*y**(2/3)"
    # s = "5 - x**2 - y**2 + 2*x*y"
    c = Controller(s)
    c.run()
