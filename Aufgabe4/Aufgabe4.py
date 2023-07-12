import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from Aufgabe2.Aufgabe2 import solve_linear_equation

# Diese Funktion implementiert das Newton-Verfahren für mehrdimensionale Probleme.
# F ist die Funktion, die die nichtlinearen Gleichungen repräsentiert.
# J ist die Jacobi-Matrix von F.
# x0 ist der Anfangswert / Schätzung.
# tol ist die Toleranz für die Konvergenz des Verfahrens.
# max_iter ist die maximale Anzahl von Iterationen, die durchgeführt werden sollen.
def newton_multidimensional(F, J, x0, tol=1e-5, max_iter=100):
    # Initialisierung der Lösungsschätzung und des Fehlers.
    x = x0
    iteration_counter = 0
    error = np.inf
    errors = []

    # Hauptiterationsschleife. Es wird so lange iteriert, bis der Fehler kleiner als die Toleranz ist oder die maximale Anzahl von Iterationen erreicht ist.
    while error > tol and iteration_counter < max_iter:
        # Lösen des linearen Gleichungssystems, um die Änderung der Lösungsschätzung zu erhalten.
        delta = solve_linear_equation(J(x), -F(x))

        # Aktualisierung der Lösungsschätzung durch Hinzufügen der Änderung zur aktuellen Schätzung.
        x_next = x + delta

        # Berechnung des Fehlers als 2-Norm der Differenz zwischen der aktuellen und der neuen Lösungsschätzung.
        error = norm(x_next - x, 2)

        # Speicherung des aktuellen Fehlers in der Fehlerliste für spätere Analyse.
        errors.append(error)

        # Aktualisierung der aktuellen Lösungsschätzung.
        x = x_next

        # Inkrementierung des Iterationszählers.
        iteration_counter += 1

    # Rückgabe der finalen Lösungsschätzung, der Anzahl der durchgeführten Iterationen und der Liste der Fehler in jeder Iteration.
    return x, iteration_counter, errors

# Diese Funktion berechnet die experimentelle Konvergenzordnung basierend auf der Liste der Fehler.
# Die Konvergenzordnung gibt an, wie schnell der Fehler in aufeinanderfolgenden Iterationen abnimmt.
def calculate_convergence_order(errors):
    # Initialisierung der Liste der Konvergenzordnungen.
    orders = []
    # Für jede Iteration, beginnend mit der dritten Iteration, wird die Konvergenzordnung berechnet.
    for i in range(2, len(errors)):
        # Die Konvergenzordnung ist das Verhältnis der Logarithmen der aufeinanderfolgenden Fehler.
        order = np.log(errors[i-1]/errors[i]) / np.log(errors[i-2]/errors[i-1])
        # Speicherung der berechneten Konvergenzordnung in der Liste der Konvergenzordnungen.
        orders.append(order)
    # Rückgabe der Liste der berechneten Konvergenzordnungen.
    return orders

# Diese Funktion erstellt ein Diagramm der berechneten Konvergenzordnungen.
def plot_convergence_order(orders):
    # Erstellung einer neuen Figur mit bestimmter Größe.
    plt.figure(figsize=(10, 6))
    # Zeichnung der Konvergenzordnungen gegen die Iterationsnummer.
    plt.plot(range(2, len(orders)+2), orders, marker='o')
    plt.xlabel('Iteration')  # Beschriftung der x-Achse.
    plt.ylabel('Konvergenzordnung')  # Beschriftung der y-Achse.
    plt.title('Experimentelle Konvergenzordnung des Newton-Verfahrens')  # Titel des Diagramms.
    plt.grid(True)  # Hinzufügen eines Gitters zum Diagramm.
    plt.show()  # Anzeigen des Diagramms.
