"""
Aufgabe:
Schreiben Sie ein Skript, welches beide Methoden anhand mehrerer Beispiele testet.
Vergleichen Sie Ihre Ergebnisse. Lassen Sie sich dazu die Anzahl an Iterationen aus- geben, die benötigt wird,
um eine vorgegebene Genauigkeit ε zu erreichen. Lassen Sie sich für das Newtonverfahren nach jeder Iteration
den Fehler ∥xk − x∗∥ ausgeben, wobei x∗ die exakte Lösung ist. Bestimmen Sie außerdem ab der zweiten Iteration
die experimentelle Konvergenzordnung über EOCk = log(∥xk+1 − x∗∥) / log(∥xk − x∗∥)
Stellen Sie den Iterationsverlauf, d.h. die berechneten Näherungen sowie die Fehler, graphisch dar.
"""

import matplotlib.pyplot as plt
from Aufgabe3a import bisection
from Aufgabe3b import newton

# Plotte die Parameter
def plot_parameter(parameter, func):
    # Hole die Parameter
    iterations = parameter["funktionswert"]
    errors = parameter["errors"]
    eocs = parameter["eoc"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Plotte den Funktionswert pro Iteration
    ax1.plot(range(1, len(iterations) + 1), iterations, marker='o', color='tab:blue')
    ax1.set_title('Iterations')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Funktionswert')
    ax1.grid(True)

    # Plotte den Fehler pro Iteration
    ax2.plot(range(1, len(errors) + 1), errors, marker='o', color='tab:red')
    ax2.set_title('Errors')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error')
    ax2.grid(True)

    # Plotte den EOC pro Iteration
    ax3.plot(range(2, len(eocs) + 2), eocs, marker='o', color='tab:green')
    ax3.set_title('EOC')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('EOC')
    ax3.grid(True)

    # Setze den Titel
    fig.suptitle(f'Newton Method Analysis for {func}')
    plt.show()



if __name__ == "__main__":
    # Testfunktionen
    test_functions = [{"func": "x^3-x-1", "root": 1.32472, "x_0": 2, "a": -2, "b": 2},
                      {"func": "x^2-4", "root": 2, "x_0": 1, "a": 0, "b": 3},
                      {"func": "x*exp(-x)", "root": 0, "x_0": -0.5, "a": -1, "b": 2},
                      {"func": "x^3-2*x+2", "root": -1.76929, "x_0": -2.5, "a": -3, "b": -1},
                      ]
    # Parameter
    n = 100
    e = 1e-5

    # Teste die Funktionen
    for test in test_functions:
        print(f"Test für Funktion: {test['func']}")
        try:
            # Berechne die Nullstellen mit den beiden Verfahren
            root_bisection, iterations_bisection = bisection(test["func"], test["a"], test["b"], n, e)
            print(f"Bisektionsverfahren: Nullstelle bei {root_bisection}, "
                  f"erreicht nach {iterations_bisection} Iterationen.")

            root_newton, iterations_newton, parameter= newton(test["func"], test["x_0"], n, e, test["root"])
            print(f"Newton-Verfahren: Nullstelle bei {root_newton}, erreicht nach {iterations_newton} Iterationen.")
            print(f"Die experimentelle Konvergenzordnung für jede Iteration {parameter['eoc']}")
            print(f"Die Fehler für jede Iteration {parameter['errors']}")
            # Plotte die Parameter
            plot_parameter(parameter, test['func'])

        # Fehlerbehandlung
        except ValueError as error:
            print(error)

        print('\n')
