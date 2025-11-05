

import numpy as np
from functools import lru_cache
from .math import beta_cpu as beta # Use the CPU version of beta

@lru_cache(maxsize=None)
def koppen_phi_digits_recursive(digits: tuple, n: int, gamma: int) -> float:
    """
    El núcleo del algoritmo recursivo de Köppen. `digits` debe ser una tupla
    para que el caché funcione.
    """
    k = len(digits)
    
    if k == 1:
        return digits[0] / gamma
    
    if digits[-1] < (gamma - 1):
        return (koppen_phi_digits_recursive(digits[:-1], n, gamma) + 
                digits[-1] * pow(gamma, -beta(n, k)))
    else:
        d_prev = digits[:-1]
        d_next_list = list(digits)
        
        while d_next_list and d_next_list[-1] == gamma - 1:
            d_next_list.pop()
        
        if not d_next_list:
            d_next_list = [gamma]
        else:
            d_next_list[-1] += 1
        
        d_next = tuple(d_next_list)
        
        correction = (gamma - 2) * pow(gamma, -beta(n, k))
        
        return 0.5 * (
            koppen_phi_digits_recursive(d_prev, n, gamma) + 
            koppen_phi_digits_recursive(d_next, n, gamma) + 
            correction
        )

def koppen_phi_single(x: float, n: int, gamma: int, k: int) -> float:
    """Wrapper para evaluar la función para un solo punto x."""
    if x >= 1.0: return 1.0
    if x <= 0.0: return 0.0
    
    d_int = int(round(x * (gamma ** k)))
    
    digits = []
    num = d_int
    for _ in range(k):
        digits.insert(0, num % gamma)
        num //= gamma

    return koppen_phi_digits_recursive(tuple(digits), n, gamma)

def koppen_phi_batch(x_array: np.ndarray, n: int, gamma: int, k: int) -> np.ndarray:
    """Evaluación vectorizada sobre un array de NumPy."""
    v_koppen_phi = np.vectorize(koppen_phi_single, otypes=[np.float64])
    return v_koppen_phi(x_array, n, gamma, k)

def test_koppen_properties(n: int = 2, gamma: int = 6, k: int = 3):
    """Pruebas unitarias para verificar la correctitud de la recursión."""
    print(f"\n{'='*60}\nTesting Köppen recursion: n={n}, γ={gamma}, k={k}\n{'='*60}")
    
    x_test = np.linspace(0, 1, 1000)
    y_test = koppen_phi_batch(x_test, n, gamma, k)
    
    violations = np.sum(np.diff(y_test) < -1e-9) # Allow for small float errors
    print(f"\n[Test 1] Monotonicity:\n  Violations: {violations}\n  Status: {'✓ PASS' if violations == 0 else '✗ FAIL'}")
    
    y_0 = koppen_phi_single(0.0, n, gamma, k)
    y_1 = koppen_phi_single(1.0, n, gamma, k)
    print(f"\n[Test 2] Boundaries:\n  ψ̃(0) = {y_0:.6f}\n  ψ̃(1) = {y_1:.6f}\n  Status: {'✓ PASS' if abs(y_0) < 1e-9 and abs(y_1 - 1.0) < 1e-9 else '✗ FAIL'}")

