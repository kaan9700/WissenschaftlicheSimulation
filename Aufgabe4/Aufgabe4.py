"""
Aufgabe:
Schreiben Sie nun ein Programm, welches mit dem Newton-Verfahren eine Nullstelle für
mehrdimensionale Probleme F : Rn→Rn, F(x) = 0 berechnet. Nutzen Sie zum Lösen der li-
nearen Gleichungssysteme Ihr Programm aus Aufgabe2. Schreiben Sie ein Skript, welches Ihr
Verfahren anhand mehrerer Beispiele testet. Lassen Sie sich erneut die Anzahl an benötigten
Iterationen sowie ab der zweiten Iteration die experimentelle Konvergenzordnung ausgeben
und stellen Sie den Verlauf graphisch dar.
"""

import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from Aufgabe2.Aufgabe2 import solve_linear_equation

def newton_multidimensional(F, J, x0, tol=1e-5, max_iter=100):
    x = x0
    iteration_counter = 0
    error = np.inf
    errors = []

    while error > tol and iteration_counter < max_iter:
        delta = solve_linear_equation(J(x), -F(x))
        x_next = x + delta
        error = norm(x_next - x, 2)
        errors.append(error)
        x = x_next
        iteration_counter += 1

    return x, iteration_counter, errors

def calculate_convergence_order(errors):
    orders = []
    for i in range(2, len(errors)):
        order = np.log(errors[i-1]/errors[i]) / np.log(errors[i-2]/errors[i-1])
        orders.append(order)
    return orders

def plot_convergence_order(orders):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(orders)+2), orders, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Konvergenzordnung')
    plt.title('Experimentelle Konvergenzordnung des Newton-Verfahrens')
    plt.grid(True)
    plt.show()







