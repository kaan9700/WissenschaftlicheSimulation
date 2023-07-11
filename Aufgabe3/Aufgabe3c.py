"""
Aufgabe:
Schreiben Sie ein Skript, welches beide Methoden anhand mehrerer Beispiele testet.
Vergleichen Sie Ihre Ergebnisse. Lassen Sie sich dazu die Anzahl an Iterationen aus- geben, die benötigt wird,
um eine vorgegebene Genauigkeit ε zu erreichen. Lassen Sie sich für das Newtonverfahren nach jeder Iteration
den Fehler ∥xk − x∗∥ ausgeben, wobei x∗ die exakte Lösung ist. Bestimmen Sie außerdem ab der zweiten Iteration
die experimentelle Konvergenzordnung über EOCk = log(∥xk+1 − x∗∥) / log(∥xk − x∗∥)
Stellen Sie den Iterationsverlauf, d.h. die berechneten Näherungen sowie die Fehler, graphisch dar.
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Beispiele
