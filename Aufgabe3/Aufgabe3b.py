"""
Aufgabe:
Implementieren Sie das skalare Newtonverfahren zur Bestimmung einer Nullstelle.
"""

from Aufgabe3a import is_continuous, create_functions
import sympy as sp
import numpy as np
import re


# Ermittele die Ableitung der Funktion
def create_derivative_function(input_string):
    x = sp.symbols('x')
    modified_input_string = re.sub(r'(\d+)(x)', r'\1*\2', input_string)
    try:
        func = sp.sympify(modified_input_string)
    except sp.SympifyError:
        raise ValueError(f"Konnte die Funktion '{modified_input_string}' nicht interpretieren")
    derivative_func = sp.diff(func, x)
    return sp.lambdify(x, derivative_func, 'numpy')

# Überprüfe, ob die Funktion zweimal stetig differenzierbar ist
def is_twice_differentiable(f, x0, r):
    # Definiere den Bereich, in dem die Funktion stetig sein muss
    a = x0 - r
    b = x0 + r
    x = sp.symbols('x')

    # Berechne die erste und zweite Ableitung
    first_derivative = sp.diff(f, x)
    second_derivative = sp.diff(first_derivative, x)

    # Überprüfe, ob die erste und zweite Ableitung stetig sind
    return is_continuous(first_derivative, a, b) and is_continuous(second_derivative, a, b)

# Implementiere das Newton-Verfahren
def newton(f, x_0, n, e, xd=None):
    # Überprüfe, ob die Parameter sinnvoll sind
    if n <= 0:
        raise ValueError("n <= 0")
    if e <= 0:
        raise ValueError("e <= 0")

    if not isinstance(x_0, (int, float)):
        raise ValueError("x_0 ist keine Zahl")
    if not isinstance(n, int):
        raise ValueError("n ist keine ganze Zahl")
    if not isinstance(e, (int, float)):
        raise ValueError("e ist keine Zahl")




    # Ersetze alle ax durch a*x, um die Eingabe zu vereinfachen
    modified_input_string = re.sub(r'(\\\\d+)(x)', r'\\\\1*\\\\2', f)

    # Überprüfe, ob die Funktion zweimal stetig differenzierbar ist
    if not is_twice_differentiable(modified_input_string, x_0, 1):
        raise ValueError("Die Funktion ist nicht zweimal stetig differenzierbar im Intervall um x_0 mit Radius 1.")

    # Erstelle die Funktion und ihre Ableitung
    func = create_functions(modified_input_string)["numpy"]
    func_prime = create_derivative_function(modified_input_string)

    # Initialisiere die Variablen
    errors = []
    eocs = []
    iterations = []

    x_n = x_0

    # Führe das Newton-Verfahren n-mal aus
    for i in range(n):

        # Überprüfe, ob die Ableitung der Funktion an der aktuellen Stelle null ist
        if func_prime(x_0) == 0:
            raise ValueError(
                "Die Ableitung der Funktion ist null an der aktuellen Stelle, "
                "das Newton-Verfahren kann nicht fortgesetzt werden")

        # Berechne den nächsten Näherungswert
        x_n = x_n - func(x_n) / func_prime(x_n)
        # Falls eine exakte Nullstelle bekannt ist, speichere die Iterationen und den Fehler
        if xd is not None:
            iterations.append(x_n)

            # Berechne den Fehler
            error = np.abs(x_n - xd)
            errors.append(error)

            # Berechne den EOC, falls die Iteration schon mindestens einmal durchgeführt wurde
            if i > 0:

                # Berechne den EOC, falls der Fehler nicht null ist
                if errors[-2] != 0:
                    eoc = np.log(errors[-1]) / np.log(errors[-2])
                else:
                    # Setze den EOC auf unendlich, falls der Fehler null ist
                    eoc = float('inf')
                eocs.append(eoc)

        # Überprüfe, ob der Fehler kleiner als die Toleranz ist
        if np.abs(func(x_n)) < e:
            # Plotte die Parameter, falls eine exakte Nullstelle bekannt ist
            if xd is not None:
                parameter = {'funktionswert': iterations, 'errors': errors, 'eoc': eocs}
                return round(x_n, 5), i + 1, parameter

            return round(x_n, 5), i + 1



    # Plotte die Parameter, falls eine exakte Nullstelle bekannt ist
    if xd is not None:
        parameter = {'funktionswert': iterations, 'errors': errors, 'eoc': eocs}
        return round(x_n, 5), n, parameter

    return round(x_n, 5), n



if __name__ == '__main__':

    #"root": 1.32472
    test_functions = [
        {"func": "x^3-x-1", "x_0": -1},
    ]

    n = 20
    e = 1e-5
    for test_function in test_functions:
        root_newton, iterations_newton = newton(test_function["func"], test_function["x_0"], n, e)
        print(f"Newton-Verfahren: Nullstelle bei {root_newton}, erreicht nach {iterations_newton} Iterationen.")
