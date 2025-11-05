# FILE: tests/test_replication.py (CORREGIDO)

import numpy as np
import pytest
from pathlib import Path
import sys

# --- INICIO DE LA CORRECCIÓN DE RUTA ---
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# --- FIN DE LA CORRECCIÓN DE RUTA ---

# Importar tanto la función de batch como la función interna que tiene el caché
from koppen_knets.core.recursion import koppen_phi_batch, koppen_phi_digits_recursive

# Importar la función de REFERENCIA
from .reference_implementation import get_reference_output

# Parámetros para una comparación directa con la tesis de Actor
TEST_PARAMS = {
    "n": 2,
    "gamma": 6,
    "k": 5
}

@pytest.fixture
def test_data():
    """Genera un conjunto de datos de entrada común para ambas implementaciones."""
    return np.linspace(0.0, 1.0, 2001, dtype=np.float64)

def test_numerical_replication_is_unambiguous(test_data):
    """
    Esta prueba valida que la salida del proyecto es numéricamente idéntica
    a la implementación de referencia canónica.
    """
    print(f"\n🔬 Ejecutando prueba de replicación numérica con n={TEST_PARAMS['n']}, γ={TEST_PARAMS['gamma']}, k={TEST_PARAMS['k']}...")

    # 1. Calcular la salida de SU implementación
    # --- CORRECCIÓN ---
    # Limpiar el caché de la función interna, que es la que está decorada.
    koppen_phi_digits_recursive.cache_clear()
    # --- FIN DE LA CORRECCIÓN ---
    project_output = koppen_phi_batch(
        test_data, **TEST_PARAMS
    )

    # 2. Calcular la salida de la implementación de REFERENCIA
    reference_output = get_reference_output(
        test_data, **TEST_PARAMS
    )

    # 3. Comparar los dos resultados con una tolerancia muy estricta
    are_identical = np.allclose(project_output, reference_output, rtol=1e-12, atol=1e-12)

    if not are_identical:
        max_abs_diff = np.max(np.abs(project_output - reference_output))
        print(f"🔥 ¡FALLO! Los arrays no son idénticos.")
        print(f"   Máxima diferencia absoluta encontrada: {max_abs_diff:.2e}")

    assert are_identical, "La implementación del proyecto se desvía de la referencia canónica."

    print("✅ ¡ÉXITO! Su implementación es una réplica numérica correcta e inequívoca.")