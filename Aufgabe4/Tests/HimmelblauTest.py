import numpy as np
from Aufgabe4.Aufgabe4 import newton_multidimensional,calculate_convergence_order,plot_convergence_order
import autograd.numpy as np

# Test 1 - Himmelblau Funktion
def F(x):
    return np.array([x[0]**2 + x[1] - 11, x[0] + x[1]**2 - 7])

# Initiale Schätzung
x0 = np.array([1.0, 1.0])
solution = [3, 2]
# Newton Verfahren Anwenden
x, num_iter, errors = newton_multidimensional(F, x0, solution)

# Berechnung der Konvegenzordnung
orders = calculate_convergence_order(errors)

# Test 1 - Output
print(f"The solution is {x} found in {num_iter} iterations.")
print(f"The experimental orders of convergence are {orders}")

# Plot Test 1
plot_convergence_order(orders)