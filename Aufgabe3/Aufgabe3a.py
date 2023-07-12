"""
Aufgabe:
Implementieren Sie ein Bisektionsverfahren zur Bestimmung einer Nullstelle im Intervall
[a,b] mit f(a)f(b) < 0.
"""

import numpy as np
import sympy as sp
import re
from sympy.calculus.util import continuous_domain
from sympy import Interval

def set_parameter_bisection():
    f = input("Geben Sie eine Funktion ein: ")
    a = int(input("Geben Sie a ein: "))
    b = int(input("Geben Sie b ein: "))
    n = int(input("Geben Sie die Anzahl der Iterationen ein: "))
    e = float(input("Geben Sie die Genauigkeit ein: "))

    return f, a, b, n, e

def create_functions(input_string):
    x = sp.symbols('x')
    modified_input_string = re.sub(r'(\d+)(x)', r'\1*\2', input_string)
    try:
        sympy_func = sp.sympify(modified_input_string)
    except sp.SympifyError:
        raise ValueError(f"Konnte die Funktion '{modified_input_string}' nicht interpretieren")
    numpy_func = sp.lambdify(x, sympy_func, 'numpy')
    return {"numpy": numpy_func, "sympy": sympy_func}

def is_continuous(f, a, b):
    x = sp.symbols('x')
    return continuous_domain(f, x, Interval(a, b)).is_Interval


def bisection(f, a, b, n, e):
    # Überprüfe die Vorraussetzungen
    func_dict = create_functions(f)
    func = func_dict["numpy"]
    sympy_func = func_dict["sympy"]
    # 1. f ist stetig
    if not is_continuous(sympy_func, a, b):
        raise ValueError("f ist nicht stetig")
    # 2. f(a)f(b) < 0
    if func(a) * func(b) >= 0:
        raise ValueError("f(a)f(b) >= 0")

    # ob die Parameter sinnvoll sind
    if a >= b:
        raise ValueError("a >= b")
    if n <= 0:
        raise ValueError("n <= 0")
    if e <= 0:
        raise ValueError("e <= 0")

    #Starte den Algorithmus
    for i in range(n):
        # Berechne den Mittelpunkt des Intervalls
        c = (a + b) / 2

        # Überprüfe, ob die Nullstelle gefunden wurde
        if np.abs(func(c)) < e:
            return c, i + 1

        # Wähle das neue Intervall
        # Falls f(a)f(c) < 0, dann ist die Nullstelle im Intervall [a,c]
        if func(a) * func(c) < 0:
            b = c
        # Falls f(b)f(c) < 0, dann ist die Nullstelle im Intervall [c,b]
        elif func(b) * func(c) < 0:
            a = c
    return c, n




if __name__ == "__main__":
    f, a, b, n, e = set_parameter_bisection()
    print(bisection(f, a, b, n, e))


