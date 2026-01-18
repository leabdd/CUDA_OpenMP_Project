# Parallel Cyclic Reduction (PCR)

Prototypische, nicht-parallele Implementierung des Parallel Cyclic Reduction (PCR) Algorithmus nach Hockney & Jesshope (1988).

## Beschreibung

Diese Python-Implementierung löst tridiagonale Gleichungssysteme der Form **Ax = y** mittels des PCR-Algorithmus. Der Algorithmus reduziert iterativ die Kopplung zwischen den Gleichungen, bis eine direkte Lösung möglich ist.

**Referenz:** Hockney & Jesshope, "Parallel Computers 2" (1988), S. 475-483

## Installation

```bash
pip install numpy or conda install numpy
```

## Verwendung

```python
from pcr import PCR
import numpy as np

# Tridiagonale Matrix A definieren
A = np.array([[4, 3, 0, 0],
              [8, 4, 9, 0],
              [0, 2, 8, 4],
              [0, 0, 4, 8]])

# Rechte Seite y definieren
y = np.array([1, 2, 3, 4])

# PCR-Solver initialisieren und lösen
solver = PCR(A, y)
x = solver.solve()
```

## Klassen

- **`diagonal`**: Repräsentiert eine Diagonale einer Tridiagonalmatrix mit Randbedingungen
- **`PCR`**: Hauptklasse zur Lösung tridiagonaler Systeme mit dem PCR-Algorithmus

