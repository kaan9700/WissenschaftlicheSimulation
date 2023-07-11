"""
Aufgabe:
Implementieren Sie das skalare Newtonverfahren zur Bestimmung einer Nullstelle.
"""



# Vorraussetzungen des Algorithmus:
# f nicht linear
#f'(x) != 0
# f sei zweimal stetig differenzierbar



import sympy as sp
import numpy as np

def create_function(input_string):
    x = sp.symbols('x')
    try:
        func = sp.sympify(input_string)
    except sp.SympifyError:
        raise ValueError(f"Konnte die Funktion '{input_string}' nicht interpretieren")
    return sp.lambdify(x, func, 'numpy')

def create_derivative_function(input_string):
    x = sp.symbols('x')
    try:
        func = sp.sympify(input_string)
    except sp.SympifyError:
        raise ValueError(f"Konnte die Funktion '{input_string}' nicht interpretieren")
    derivative_func = sp.diff(func, x)
    return sp.lambdify(x, derivative_func, 'numpy')


def is_twice_differentiable(input_string, x_0, r):
    # Definiere das Symbol x
    x = sp.symbols('x')

    # Versuche die Funktion zu parsen und die zweite Ableitung zu berechnen
    try:
        func = sp.sympify(input_string)
        second_derivative = sp.diff(func, x, 2)
    except Exception as e:
        print(f"Fehler bei der Berechnung der zweiten Ableitung: {e}")
        return False

    # Lambdify die zweite Ableitung, um sie zu evaluieren
    f_second_derivative = sp.lambdify(x, second_derivative, 'numpy')

    # Überprüfe die Stetigkeit der zweiten Ableitung im gegebenen Intervall
    x_values = np.linspace(x_0 - r, x_0 + r, 1000)
    try:
        values = f_second_derivative(x_values)
    except Exception as e:
        print(f"Fehler bei der Evaluierung der zweiten Ableitung: {e}")
        return False

    if np.any(np.isnan(values) | np.isinf(values)):
        print("Die zweite Ableitung ist nicht stetig im gegebenen Intervall.")
        return False

    return True


def newton(func, func_prime, x_0, n, e):
    # überprüfen, ob die Parameter sinnvoll sind
    if n <= 0:
        raise ValueError("n <= 0")
    if e <= 0:
        raise ValueError("e <= 0")

    # Starte den Algorithmus
    for i in range(n):
        # überprüfen, ob die Ableitung an der aktuellen Stelle null ist
        if func_prime(x_0) == 0:
            raise ValueError(
                "Die Ableitung der Funktion ist null an der aktuellen Stelle, "
                "das Newton-Verfahren kann nicht fortgesetzt werden")
        x_0 = x_0 - func(x_0)/func_prime(x_0)
        if np.abs(func(x_0)) < e:
            return x_0
    return x_0


if __name__ == "__main__":
    print(is_twice_differentiable("x**2", 0, 1))  # Sollte True ausgeben
    print(is_twice_differentiable("sin(x)", 0, np.pi))  # Sollte True ausgeben
    print(is_twice_differentiable("abs(x)", 0, 1))  # Sollte False ausgeben

    f = input("Geben Sie eine Funktion ein: ")
    x_0 = float(input("Geben Sie x_0 ein: "))
    n = int(input("Geben Sie die Anzahl der Iterationen ein: "))
    e = float(input("Geben Sie die Genauigkeit ein: "))

    x = sp.symbols('x')
    func = create_function(f)
    func_prime = create_derivative_function(f)
    print(newton(func, func_prime, x_0, n, e))