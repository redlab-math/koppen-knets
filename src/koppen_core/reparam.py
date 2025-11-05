

import h5py
import numpy as np

def reparametrize_to_lipschitz_streaming(h5_input: h5py.File,
                                     h5_output: h5py.File):

    print(f"  Reparametrizing to Lipschitz function (exact resolution)...")
    
    # Load original data
    sigma = h5_input['reparametrization/sigma'][:]      # σ(x) ∈ [0,1]
    y_holder = h5_input['holder_function/y_holder'][:]  # ψ̃(x)
    
    n_points = len(sigma)
    
    # Verify dimensions match
    assert len(y_holder) == n_points, "Dimension mismatch between sigma and y_holder"
    
    print(f"    Preserving {n_points:,} original points")
    
    # The reparametrization: domain changes from x to s=σ(x)
    s_lipschitz = sigma       # New domain (arc-length)
    psi_lipschitz = y_holder  # Same function values
    
    # Save to output
    h5_output.create_dataset('lipschitz_function/s_lipschitz', data=s_lipschitz)
    h5_output.create_dataset('lipschitz_function/psi_lipschitz', data=psi_lipschitz)
    
    # Compute empirical Lipschitz constant
    ds = np.diff(s_lipschitz)
    dpsi = np.diff(psi_lipschitz)
    
    # Avoid division by zero for flat segments
    valid_mask = ds > 1e-15
    local_lipschitz = np.abs(dpsi[valid_mask] / ds[valid_mask])
    
    L_empirical = np.max(local_lipschitz) if len(local_lipschitz) > 0 else 0.0
    
    print(f"  ✓ Lipschitz function computed")
    print(f"    - Points: {n_points:,}")
    print(f"    - Domain: [{s_lipschitz[0]:.6f}, {s_lipschitz[-1]:.6f}]")
    print(f"    - Range: [{psi_lipschitz.min():.6f}, {psi_lipschitz.max():.6f}]")
    print(f"    - Empirical L: {L_empirical:.4f}")
    
    # Store metadata
    h5_output['reparametrization'].attrs['lipschitz_constant'] = L_empirical