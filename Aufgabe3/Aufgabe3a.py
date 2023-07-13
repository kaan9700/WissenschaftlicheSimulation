import numpy as np
import sympy as sp
import re
from sympy.calculus.util import continuous_domain
from sympy import Interval

def create_functions(input_string):
    x = sp.symbols('x')
    modified_input_string = re.sub(r'(\d+)(x)', r'\1*\2', input_string)

    try:
        sympy_func = sp.sympify(modified_input_string)
        numpy_func = sp.lambdify(x, sympy_func, 'numpy')
    except (sp.SympifyError, SyntaxError, ValueError, NameError) as error:
        raise ValueError(f"Konnte die Funktion '{modified_input_string}' nicht interpretieren: {error}")

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

    if not isinstance(a, (int, float)):
        raise ValueError("a ist keine Zahl")
    if not isinstance(b, (int, float)):
        raise ValueError("b ist keine Zahl")
    if not isinstance(n, int):
        raise ValueError("n ist keine ganze Zahl")
    if not isinstance(e, (int, float)):
        raise ValueError("e ist keine Zahl")


    #Starte den Algorithmus
    for i in range(n):
        # Berechne den Mittelpunkt des Intervalls
        c = (a + b) / 2

        # Überprüfe, ob die Nullstelle gefunden wurde
        if np.abs(func(c)) < e:
            return round(c, 5), i + 1

        # Wähle das neue Intervall
        # Falls f(a)f(c) < 0, dann ist die Nullstelle im Intervall [a,c]
        if func(a) * func(c) < 0:
            b = c
        # Falls f(b)f(c) < 0, dann ist die Nullstelle im Intervall [c,b]
        elif func(b) * func(c) < 0:
            a = c
    return round(c, 5), n


if __name__ == "__main__":
    # "root": 1.13472
    test_functions = [
        {"func": "x^6-x-1", "a": 0, "b": 2},
    ]
    n = 10
    e = 1e-5
    for test_function in test_functions:
        root_bisection, iterations_bisection = bisection(test_function["func"], test_function["a"], test_function["b"], n, e)
        print(f"Bisektionsverfahren: Nullstelle bei {root_bisection}, "
              f"erreicht nach {iterations_bisection} Iterationen.")
