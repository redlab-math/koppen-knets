# FILE: tests/reference_implementation.py (VERSIÓN FIEL, SIN OPTIMIZACIONES)
#
# Esta es una traducción 1:1, no optimizada, del código de Actor en el Apéndice E,
# adaptada a Python 3. El propósito es validar la correctitud lógica contra
# una réplica exacta, sacrificando el rendimiento.

import numpy as np
import sys

# Aumentar el límite de recursión de Python para k altos, replicando un
# comportamiento implícito de entornos menos restrictivos.
sys.setrecursionlimit(2000)

# --- Funciones de soporte ---
def beta(n: int, r: int) -> float:
    """Función beta, idéntica a la lógica implícita en la tesis."""
    if n == 1:
        return float(r)
    return (n**r - 1.0) / (n - 1.0)

def extractDigits(d_int: int, k: int, gamma: int) -> list:
    """
    Función helper para extraer dígitos, replicando la estructura de Actor.
    Devuelve una lista, como en la implementación original.
    """
    digits = []
    num = d_int
    for _ in range(k):
        digits.insert(0, num % gamma)
        num //= gamma
    return digits

# --- Núcleo Recursivo (SIN MEMOIZACIÓN) ---
def koppenPhiDigits(d: list, n: int, gamma: int) -> float:
    """
    Núcleo recursivo de Köppen. Opera sobre una 'list', como en el original.
    SIN @lru_cache para una replicación exacta del flujo computacional.
    """
    k = len(d)
    if k == 1:
        return d[0] / gamma
    
    # Caso estándar
    if d[-1] < (gamma - 1):
        return (koppenPhiDigits(d[:-1], n, gamma) + 
                d[-1] * pow(gamma, -beta(n, k)))
    # Caso especial (borde)
    else:
        d_prev = d[:-1]
        
        # Lógica para encontrar el siguiente valor, idéntica a la de Actor
        d_next = list(d) # Replicando el uso de una copia
        while d_next and d_next[-1] == gamma - 1:
            d_next.pop()
        
        if not d_next:
            d_next = [gamma]
        else:
            d_next[-1] += 1
        
        correction = (gamma - 2) * pow(gamma, -beta(n, k))
        
        return 0.5 * (
            koppenPhiDigits(d_prev, n, gamma) + 
            koppenPhiDigits(d_next, n, gamma) + 
            correction
        )

# --- Wrappers ---
def koppenPhi(x: float, n: int, gamma: int, k: int) -> float:
    """Wrapper para un solo punto, replicando la estructura de Actor."""
    if x >= 1.0: return 1.0
    if x <= 0.0: return 0.0
    
    d_int = int(round(x * (gamma ** k)))
    digits_list = extractDigits(d_int, k, gamma)
    
    return koppenPhiDigits(digits_list, n, gamma)

def get_reference_output(x_array: np.ndarray, n: int, gamma: int, k: int) -> np.ndarray:
    """
    Función principal que sigue siendo compatible con la prueba.
    Usa un bucle explícito para ser más fiel al original que np.vectorize.
    """
    results = [koppenPhi(x, n, gamma, k) for x in x_array]
    return np.array(results, dtype=np.float64)