import numpy as np
from Aufgabe4.Aufgabe4 import newton_multidimensional, calculate_convergence_order, plot_convergence_order

# Rosenbrock-Funktion und ihre Jacobimatrix
def F(x):
    return np.array([(1 - x[0])**2 + 100*(x[1] - x[0]**2)**2, 0])

def J(x):
    return np.array([[-400*(x[1] - x[0]**2)*x[0] + 2*x[0] - 2, 200*(x[1] - x[0]**2)], [-400*x[0]*(x[1] - x[0]**2), 200*x[1]]])

# Initiale Schätzung
x0 = np.array([2.0, 2.0])

# Newton Verfahren Anwenden
x, num_iter, errors = newton_multidimensional(F, J, x0)

# Berechnung der Konvergenzordnung
orders = calculate_convergence_order(errors)

# Ausgabe der Ergebnisse und Plot
print(f"The solution is {x} found in {num_iter} iterations.")
print(f"The experimental orders of convergence are {orders}")
plot_convergence_order(orders)
