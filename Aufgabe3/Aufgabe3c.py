"""
Aufgabe:
Schreiben Sie ein Skript, welches beide Methoden anhand mehrerer Beispiele testet.
Vergleichen Sie Ihre Ergebnisse. Lassen Sie sich dazu die Anzahl an Iterationen aus- geben, die benötigt wird,
um eine vorgegebene Genauigkeit ε zu erreichen. Lassen Sie sich für das Newtonverfahren nach jeder Iteration
den Fehler ∥xk − x∗∥ ausgeben, wobei x∗ die exakte Lösung ist. Bestimmen Sie außerdem ab der zweiten Iteration
die experimentelle Konvergenzordnung über EOCk = log(∥xk+1 − x∗∥) / log(∥xk − x∗∥)
Stellen Sie den Iterationsverlauf, d.h. die berechneten Näherungen sowie die Fehler, graphisch dar.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from Aufgabe3a import bisection
from Aufgabe3b import newton


if __name__ == "__main__":
    test_functions = [{"func": "-x^3-3*x^2-x+3", "root": -1}, {"func": "x^4-3*x^2+2", "root": 1}, {"func": "sin(x)", "root": 0}, {"func": "cos(x)-x", "root": 0.73908513321516}, {"func": "exp(x)-2", "root": np.log(2)}, {"func": "1/x", "root": 1}]
    a = -1
    b = 1
    x_0 = 0
    n = 100
    e = 0.0000001

    for test in test_functions:
        print(f"Test für Funktion: {test}")
        try:
            root_bisection, iterations_bisection = bisection(test["func"], a, b, n, e)
            print(f"Bisektionsverfahren: Nullstelle bei {root_bisection}, "
                  f"erreicht nach {iterations_bisection} Iterationen.")

            root_newton, iterations_newton, nst_error = newton(test["func"], x_0, n, e, test["root"])
            print(f"Newton-Verfahren: Nullstelle bei {root_newton}, erreicht nach {iterations_newton} Iterationen.")


        except ValueError as error:
            print(error)

        print('\n')
