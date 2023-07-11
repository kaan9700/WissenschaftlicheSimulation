"""
Aufgabe:
Implementieren Sie ein Bisektionsverfahren zur Bestimmung einer Nullstelle im Intervall
[a,b] mit f(a)f(b) < 0.
"""

import numpy as np
import sympy as sp

def set_parameter():
    f = input("Geben Sie eine Funktion ein: ")
    a = int(input("Geben Sie a ein: "))
    b = int(input("Geben Sie b ein: "))
    n = int(input("Geben Sie die Anzahl der Iterationen ein: "))
    e = float(input("Geben Sie die Genauigkeit ein: "))

    return f, a, b, n, e

def create_function(input_string):
    x = sp.symbols('x')
    try:
        func = sp.sympify(input_string)
    except sp.SympifyError:
        raise ValueError(f"Konnte die Funktion '{input_string}' nicht interpretieren")
    return sp.lambdify(x, func, 'numpy')

def is_continuous(func, a, b, num_points=1000, tolerance=1e-5):
    x_values = np.linspace(a, b, num_points)
    for x in x_values:
        interval = np.linspace(x - tolerance, x + tolerance, num_points)
        y_interval = [func(i) for i in interval]
        if np.abs(func(x) - np.mean(y_interval)) > tolerance:
            return False
    return True


def bisection(func, a, b, n, e):
    # Überprüfe die Vorraussetzungen
    # 1. f ist stetig
    if not is_continuous(func, a, b):
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
            return c
        # Wähle das neue Intervall
        #Falls f(c) < 0, dann ist die Nullstelle im Intervall [c,b]
        if func(c) < 0:
            a = c
        #Falls f(c) > 0, dann ist die Nullstelle im Intervall [a,c]
        if func(c) > 0:
            b = c

    return c



if __name__ == "__main__":
    f, a, b, n, e = set_parameter()
    f = create_function(f)
    print(bisection(f, a, b, n, e))