import pytest
import torch
import numpy as np
import argparse
from pathlib import Path

# Asume que los módulos están en el path de python
from koppen_knets.surrogate.kst_projector import KSTProjector
from koppen_knets.surrogate.kst_sl_model import KSTSprecherLorentzModel
from experiments.benchmark import train_pytorch_model

# Reutilizamos los fixtures de las pruebas unitarias
from tests.test_unitarios import dummy_checkpoint, projector

def test_training_step(projector):
    """
    Verifica que un paso completo de entrenamiento (forward, loss, backward, step)
    se complete sin errores para el modelo KST-SL.
    """
    model = KSTSprecherLorentzModel(projector, hidden_sizes=[4, 4]).to('cpu')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # Datos falsos
    X_batch = torch.rand(4, 2)
    y_batch = torch.rand(4, 1)

    # Paso de entrenamiento
    try:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    except Exception as e:
        pytest.fail(f"El paso de entrenamiento falló con la excepción: {e}")

    assert loss.item() is not None
    # Verificar que los gradientes se calcularon para las lambdas
    assert model.lambdas.grad is not None

def test_full_training_loop(projector):
    """
    Prueba la función de entrenamiento genérica `train_pytorch_model` con el
    modelo KST-SL en un escenario muy reducido.
    """
    model = KSTSprecherLorentzModel(projector, hidden_sizes=[8, 4]).to('cpu')
    
    # Datos de entrenamiento y validación falsos
    X_train = np.random.rand(20, 2)
    y_train = np.random.rand(20, 1)
    X_val = np.random.rand(10, 2)
    y_val = np.random.rand(10, 1)

    # Argumentos de entrenamiento falsos
    args = argparse.Namespace(
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        patience=1
    )
    
    try:
        result = train_pytorch_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            args=args,
            device=torch.device('cpu')
        )
    except Exception as e:
        pytest.fail(f"El bucle de entrenamiento completo falló con la excepción: {e}")

    # Verificar que el entrenamiento produjo un historial y un scaler
    assert 'history' in result
    assert 'scaler' in result
    assert len(result['history']['train_loss']) == 2
