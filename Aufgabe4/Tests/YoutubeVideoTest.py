import numpy as np
from Aufgabe4.Aufgabe4 import newton_multidimensional, calculate_convergence_order, plot_convergence_order
import autograd.numpy as np

# Test 2 - Beispielfunktion
def F(x):
    return np.array([x[0] + 2*x[1] - 2, x[0]**2 + 4*x[1]**2 - 4])

# Initiale Schätzung
x0 = np.array([1.0, 2.0])
solution = [0, 1]
# Newton Verfahren Anwenden
x, num_iter, errors = newton_multidimensional(F, x0, solution)

# Berechnung der Konvegenzordnung
orders = calculate_convergence_order(errors)

# Test 2 - Output
print(f"The solution is {x} found in {num_iter} iterations.")
print(f"The experimental orders of convergence are {orders}")

# Plot Test 2
plot_convergence_order(orders)
