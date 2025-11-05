import pytest
import torch
import numpy as np
import h5py
from pathlib import Path

# Asume que los módulos están en el path de python
from koppen_knets.surrogate.kst_projector import KSTProjector
from koppen_knets.surrogate.kst_sl_model import KSTSprecherLorentzModel

# --- Fixtures para crear objetos de prueba ---

@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Crea un archivo HDF5 falso para las pruebas del proyector."""
    p = tmp_path / "dummy_checkpoint.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("lipschitz_function/s_lipschitz", data=np.linspace(0, 1, 100))
        f.create_dataset("lipschitz_function/psi_lipschitz", data=np.sin(np.linspace(0, 1, 100) * np.pi))
        meta = f.create_group("metadata")
        meta.attrs['n'] = 2
    return p

@pytest.fixture
def projector(dummy_checkpoint: Path) -> KSTProjector:
    """Retorna una instancia funcional de KSTProjector."""
    return KSTProjector(n=2, psi_checkpoint=dummy_checkpoint)

@pytest.fixture
def kst_sl_model(projector: KSTProjector) -> KSTSprecherLorentzModel:
    """Retorna una instancia del modelo KST-SL."""
    return KSTSprecherLorentzModel(projector, hidden_sizes=[16, 8])

# --- Pruebas para KSTProjector ---

def test_projector_initialization(projector: KSTProjector):
    """Verifica que el proyector se inicialice con las dimensiones correctas."""
    assert projector.n == 2
    assert projector.Q == 5  # 2n + 1
    assert projector.epsilon == 0.25  # 1 / (2n)
    assert projector.lambdas.shape == (2,)
    assert np.isclose(projector.lambdas.sum(), 1.0)

def test_projector_psi_extension(projector: KSTProjector):
    """Prueba la extensión periódica de la función psi, que es crítica."""
    # psi(x) = psi(x - floor(x)) + floor(x)
    val_0_5 = projector.psi(0.5)
    val_1_5 = projector.psi(1.5)
    assert np.isclose(val_1_5, val_0_5 + 1.0)

def test_projector_batch_projection(projector: KSTProjector):
    """Verifica la forma de la salida de la proyección."""
    X = np.random.rand(10, 2)  # 10 muestras, 2 dimensiones
    Z = projector.project_batch(X)
    assert Z.shape == (10, 5)  # 10 muestras, 5 proyecciones (Q)

def test_projector_input_clipping(projector: KSTProjector, caplog):
    """Verifica que las entradas fuera de [0, 1] se recorten y generen un warning."""
    X_outside = np.array([[-0.5, 1.5]])
    projector.project_batch(X_outside)
    assert "Input X is outside the expected [0, 1] domain" in caplog.text

# --- Pruebas para KSTSprecherLorentzModel ---

def test_kst_sl_model_initialization(kst_sl_model: KSTSprecherLorentzModel):
    """Verifica la inicialización del modelo KST-SL."""
    assert isinstance(kst_sl_model, torch.nn.Module)
    assert kst_sl_model.lambdas.shape == (5, 2) # (Q, n)
    assert kst_sl_model.lambdas.requires_grad is True # Debe ser entrenable

def test_kst_sl_model_forward_pass(kst_sl_model: KSTSprecherLorentzModel):
    """Prueba el forward pass para asegurar que la forma de salida es correcta."""
    batch_size = 8
    input_tensor = torch.rand(batch_size, 2) # (batch_size, n)
    output = kst_sl_model(input_tensor)
    assert output.shape == (batch_size, 1)

def test_kst_sl_model_parameters(kst_sl_model: KSTSprecherLorentzModel):
    """Asegura que todos los parámetros esperados (lambdas y phi) sean entrenables."""
    params = list(kst_sl_model.parameters())
    # El primer parámetro debe ser la matriz lambda
    assert params[0] is kst_sl_model.lambdas
    # Debe haber más parámetros (de la red phi)
    assert len(params) > 1
