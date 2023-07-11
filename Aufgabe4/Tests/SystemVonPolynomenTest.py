import numpy as np
from Aufgabe4.Aufgabe4 import newton_multidimensional, calculate_convergence_order, plot_convergence_order

# System von Polynomen und ihre Jacobimatrix
def F(x):
    return np.array([x[0]**3 - 3*x[0]*x[1]**2 - 1, 3*x[0]**2*x[1] - x[1]**3])

def J(x):
    return np.array([[3*x[0]**2 - 3*x[1]**2, -6*x[0]*x[1]], [6*x[0]*x[1], 3*x[0]**2 - 3*x[1]**2]])

# Initiale Schätzung
x0 = np.array([1.0, 1.0])

# Newton Verfahren Anwenden
x, num_iter, errors = newton_multidimensional(F, J, x0)

# Berechnung der Konvergenzordnung
orders = calculate_convergence_order(errors)

# Ausgabe der Ergebnisse und Plot
print(f"The solution is {x} found in {num_iter} iterations.")
print(f"The experimental orders of convergence are {orders}")
plot_convergence_order(orders)
