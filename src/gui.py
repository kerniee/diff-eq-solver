import tkinter as tk
from collections import Callable
from contextlib import contextmanager
from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class NumberInputField(tk.Frame):
    def __init__(self, var_name: str, var_type: tk.Variable.__class__, value: 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var: tk.Variable.__class__ = var_type(value=value)
        self.label: tk.Label = tk.Label(self, text=var_name)
        self.entry: tk.Entry = tk.Entry(self, textvariable=self.var)

    def bind_on_change(self, callback: Callable):
        self.var.trace_add("write", callback)

    def pack(self, *args, **kwargs):
        self.label.pack(side="left")
        self.entry.pack(side="right")
        super().pack(*args, **kwargs)
        return self

    def value(self):
        return self.var.get()


class View(tk.Tk):
    def __init__(self, frame_title):
        super().__init__()
        # Create a title for the frame
        self.title(frame_title)

        self.right_panel = tk.Frame(self)

        self.eq_frame: tk.Frame = tk.Frame(self.right_panel)
        self.eq_field: tk.Text = tk.Text(height=2, width=32, master=self.eq_frame)
        self.eq_field.insert("1.0", "sec(x) - y*tg(x)")
        self.eq_field.pack(side="top")
        self.btn_change_eq: tk.Button = tk.Button(text="Change Equation", master=self.eq_frame)
        self.btn_change_eq.pack()
        self.eq_frame.pack(side="top", pady=10)

        self.fields: tk.LabelFrame = tk.LabelFrame(text="Parameters for equation", padx=10, pady=10, master=self.right_panel)
        self.y0_field: NumberInputField = NumberInputField("y0", tk.DoubleVar, 1, master=self.fields).pack()
        self.x0_field: NumberInputField = NumberInputField("x0", tk.DoubleVar, 0, master=self.fields).pack()
        self.X_field: NumberInputField = NumberInputField("X", tk.DoubleVar, 7, master=self.fields).pack()
        self.N_field: NumberInputField = NumberInputField("N", tk.IntVar, 400, master=self.fields).pack()
        self.fields.pack(pady=10)

        self.error_fields: tk.LabelFrame = tk.LabelFrame(text="Parameters for error", padx=10, pady=10, master=self.right_panel)
        self.n0_field: NumberInputField = NumberInputField("n0", tk.IntVar, 10, master=self.error_fields).pack()
        self.N_err_field: NumberInputField = NumberInputField("N", tk.IntVar, 20, master=self.error_fields).pack()
        self.error_fields.pack(pady=10)

        self.comp_method_btns: tk.Frame = tk.Frame(self.right_panel)
        self.comp_method_btns.pack(side="bottom", pady=10)

        self.right_panel.pack(side="right")

        # Plots
        plot = plt.subplots(2)
        self.main_fig: Figure = plot[0]
        self.main_axs: List[Axes] = plot[1]
        self.main_ax: Axes = self.main_axs[0]
        self.main_err_ax: Axes = self.main_axs[1]

        plot = plt.subplots()
        self.error_fig: Figure = plot[0]
        self.all_error_ax: Axes = plot[1]

        self.canvas: FigureCanvasTkAgg = FigureCanvasTkAgg(self.main_fig, master=self)
        self.canvas.get_tk_widget().pack(side='left', fill='both', expand=1)

        # Menu
        self.menubar: tk.Menu = tk.Menu(self)
        self.config(menu=self.menubar)

    @contextmanager
    def loading_screen(self):
        old_title = self.title()
        self.title("Loading...")
        yield
        self.title(old_title)

    def clear_plot(self):
        for axes in (*self.main_axs, self.all_error_ax):
            axes.clear()
            axes.grid()
        self.main_ax.set(ylabel="Function y(x)")
        self.main_err_ax.set(ylabel="Epsilon")

    def add_comp_method_btn(self, label: str, callback: Callable):
        btn = tk.Button(text=label, master=self.comp_method_btns, command=callback)
        btn.pack(side="bottom")
