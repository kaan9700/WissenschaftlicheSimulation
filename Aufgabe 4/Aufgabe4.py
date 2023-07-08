"""
Aufgabe:
Schreiben Sie nun ein Programm, welches mit dem Newton-Verfahren eine Nullstelle für
mehrdimensionale Probleme F : Rn→Rn, F(x) = 0 berechnet. Nutzen Sie zum Lösen der li-
nearen Gleichungssysteme Ihr Programm aus Aufgabe 2. Schreiben Sie ein Skript, welches Ihr
Verfahren anhand mehrerer Beispiele testet. Lassen Sie sich erneut die Anzahl an benötigten
Iterationen sowie ab der zweiten Iteration die experimentelle Konvergenzordnung ausgeben
und stellen Sie den Verlauf graphisch dar.
"""

import numpy as np
import matplotlib.pyplot as plt
from Aufgabe2 import solve_linear_equation

# Funktion zur Berechnung der Jacobi-Matrix
def jacobian(f, x):
    h = 1.0e-4
    n = len(x)
    Jac = np.zeros((n,n))
    f0 = f(x)
    for i in range(n):
        temp = x[i]
        x[i] = temp + h
        f1 = f(x)
        x[i] = temp
        Jac[:,i] = (f1 - f0)/h
    return Jac, f0

# Hauptfunktion: Newton-Raphson-Verfahren
def newton_raphson(f, x, tol=1.0e-9):
    max_iter = 500
    iterations = 0
    convergence = []

    for i in range(max_iter):
        Jac, f0 = jacobian(f, x)

        # Kriterium für die Konvergenz
        if np.sqrt(np.dot(f0, f0) / len(x)) < tol: 
            return x, iterations, convergence

        dx = solve_linear_equation(Jac, f0)
        x = x - dx
        iterations += 1
        convergence.append(np.linalg.norm(dx))

    return x, iterations, convergence

# Die zu testende Funktion F
def f(x):
    return np.array([x[0]**2 + x[1]**2 - 3, x[0]*x[1] - 1, x[2]**2 - 1])

# Startwerte
x_start = np.array([1.0, 1.0])  
x_sol, iterations, convergence = newton_raphson(f, x_start)

# Graphische Darstellung der Konvergenz
plt.plot(range(iterations), convergence)
plt.xlabel('Iterationen')
plt.ylabel('Konvergenz')
plt.show()



