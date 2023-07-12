"""
Aufgabe:
Implementieren Sie das skalare Newtonverfahren zur Bestimmung einer Nullstelle.
"""
import sympy as sp
import numpy as np
import re
import matplotlib.pyplot as plt
from Aufgabe3a import is_continuous, create_functions


def set_parameter_newton():
    f = input("Geben Sie eine Funktion ein: ")
    x_0 = float(input("Geben Sie x_0 ein: "))
    n = int(input("Geben Sie die Anzahl der Iterationen ein: "))
    e = float(input("Geben Sie die Genauigkeit ein: "))

    return f, x_0, n, e

def create_derivative_function(input_string):
    x = sp.symbols('x')
    try:
        func = sp.sympify(input_string)
    except sp.SympifyError:
        raise ValueError(f"Konnte die Funktion '{input_string}' nicht interpretieren")
    derivative_func = sp.diff(func, x)
    return sp.lambdify(x, derivative_func, 'numpy')

def is_twice_differentiable(f, a, b):
    x = sp.symbols('x')
    first_derivative = sp.diff(f, x)
    second_derivative = sp.diff(first_derivative, x)
    return is_continuous(first_derivative, a, b) and is_continuous(second_derivative, a, b)


def newton(f, x_0, n, e, xd=False):
    # überprüfen, ob die Parameter sinnvoll sind
    if n <= 0:
        raise ValueError("n <= 0")
    if e <= 0:
        raise ValueError("e <= 0")
    # Füge '*' zwischen Zahlen und x ein, wenn es nicht vorhanden ist
    modified_input_string = re.sub(r'(\\d+)(x)', r'\\1*\\2', f)
    func = create_functions(modified_input_string)["numpy"]

    if not is_twice_differentiable(modified_input_string, x_0, 1):
        raise ValueError("Die Funktion ist nicht zweimal stetig differenzierbar im Intervall um x_0 mit Radius 1.")
    func_prime = create_derivative_function(modified_input_string)

    # Initialize error and eoc lists
    errors = []
    eocs = []

    # Starte den Algorithmus
    for i in range(n):
        # überprüfen, ob die Ableitung an der aktuellen Stelle null ist
        if func_prime(x_0) == 0:
            raise ValueError(
                "Die Ableitung der Funktion ist null an der aktuellen Stelle, "
                "das Newton-Verfahren kann nicht fortgesetzt werden")
        x_0 = x_0 - func(x_0)/func_prime(x_0)
        if xd is not False:
            error = np.abs(x_0 - xd)
            errors.append(error)
            if i > 0:
                eoc = np.log(errors[-1]) / np.log(errors[-2]) if errors[-2] != 0 else float('inf')
                eocs.append(eoc)
        if np.abs(func(x_0)) < e:
            if eocs != []:
                plot_eocs(eocs, modified_input_string)
            return x_0, i+1, errors
    if xd is not False:
        if eocs != []:
            plot_eocs(eocs, modified_input_string)
        return x_0, n, errors
    return x_0, n



def plot_eocs(eocs, func):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(eocs) + 2), eocs, marker='o')
    plt.title(f'Experimental order of convergence (EOC) \n for {func}')
    plt.xlabel('Iteration')
    plt.ylabel('EOC')
    plt.grid(True)
    plt.show()
