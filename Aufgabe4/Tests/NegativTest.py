import numpy as np
from Aufgabe4.Aufgabe4 import newton_multidimensional, calculate_convergence_order, plot_convergence_order

# Funktion und ihre Jacobimatrix
def F(x):
    return np.array([np.exp(x[0]) - x[1], x[0]**2 + x[1]**2 - 1])

def J(x):
    return np.array([[np.exp(x[0]), -1], [2*x[0], 2*x[1]]])

# Initiale Schätzung
x0 = np.array([1.0, 1.0])

# Versuch, das Newton-Verfahren anzuwenden
try:
    x, num_iter, errors = newton_multidimensional(F, J, x0)
    orders = calculate_convergence_order(errors)

    # Ausgabe der Ergebnisse und Plot
    print(f"The solution is {x} found in {num_iter} iterations.")
    print(f"The experimental orders of convergence are {orders}")
    plot_convergence_order(orders)

except Exception as e:
    print(f"Failed to find a solution: {e}")
