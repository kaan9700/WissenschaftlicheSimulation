import numpy as np
from Aufgabe4.Aufgabe4 import newton_multidimensional, calculate_convergence_order, plot_convergence_order
import autograd.numpy as np
from autograd import jacobian
# System von Polynomen und ihre Jacobimatrix
def F(x):
    return np.array([x[0]**3 - 3*x[0]*x[1]**2 - 1, 3*x[0]**2*x[1] - x[1]**3])

# Initiale Schätzung
x0 = np.array([1.0, 1.0])
solution = [1, 0]
# Newton Verfahren Anwenden
x, num_iter, errors = newton_multidimensional(F, x0, solution)

# Berechnung der Konvergenzordnung
orders = calculate_convergence_order(errors)

# Ausgabe der Ergebnisse und Plot
print(f"The solution is {x} found in {num_iter} iterations.")
print(f"The experimental orders of convergence are {orders}")
plot_convergence_order(orders)
