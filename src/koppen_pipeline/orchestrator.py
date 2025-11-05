"""
Pipeline orchestration
"""

import h5py
from pathlib import Path
import time
from typing import Dict, Optional
import warnings

from config.params import KSTConfig, validate_config
from config.validation import run_full_verification
from koppen_pipeline.storage import (create_checkpoint_file, generate_holder_streaming,
                     get_memory_estimate)
from koppen_pipeline.memory import MemoryMonitor, check_resources
from koppen_core.arclength import compute_arclength_streaming
from koppen_core.reparam import reparametrize_to_lipschitz_streaming


def run_complete_pipeline(config: KSTConfig,
                          output_dir: Path,
                          checkpoint_name: Optional[str] = None,
                          skip_verification: bool = False,
                          verbose: bool = True) -> Dict:
    """
    Execute complete Köppen-Actor pipeline.
    
    Steps:
    1. Generate Hölder function ψ̃_k (streaming to HDF5)
    2. Compute arc length σ (streaming)
    3. Reparametrize to Lipschitz ψ (streaming)
    4. Verify KST conditions (Claims 17-21)
    
    Args:
        config: KST configuration
        output_dir: Output directory for checkpoints
        checkpoint_name: Custom checkpoint filename (default: auto-generated)
        skip_verification: Skip verification step (for very large k)
        verbose: Print detailed progress
    
    Returns:
        Dict with all results and metrics
    """
    
    # Validate configuration
    validate_config(config)
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_name is None:
        checkpoint_name = f'koppen_n{config.n}_k{config.k}_gamma{config.gamma}.h5'
    
    checkpoint_path = output_dir / checkpoint_name
    
    # Print header
    if verbose:
        print("\n" + "="*80)
        print("KÖPPEN-ACTOR KST PIPELINE (REENGINEERED)")
        print("="*80)
        print(f"Configuration: {config}")
        print(f"Output: {checkpoint_path}")
        print("="*80)
    
    # Resource check
    if verbose:
        try:
            check_resources(config)
        except RuntimeError as e:
            warnings.warn(str(e))
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                raise
    
    # Results tracking
    results = {
        'config': config,
        'checkpoint_path': str(checkpoint_path),
        'timings': {},
        'metrics': {}
    }
    
    pipeline_start = time.perf_counter()
    
    # ========================================================================
    # STEP 1: Generate Hölder Function ψ̃_k
    # ========================================================================
    
    with MemoryMonitor("Hölder Generation"):
        h5file = create_checkpoint_file(checkpoint_path, config)
        
        holder_time = generate_holder_streaming(
            h5file, config, verbose=verbose
        )
        results['timings']['holder_generation'] = holder_time
    
    # ========================================================================
    # STEP 2: Compute Arc Length σ
    # ========================================================================
    
    if verbose:
        print(f"\n[STEP 2/4] Computing arc length σ...")
    
    with MemoryMonitor("Arc Length"):
        arc_start = time.perf_counter()
        
        sigma, L_theoretical = compute_arclength_streaming(h5file)
        
        # Store in HDF5
        reparam_grp = h5file['reparametrization']
        reparam_grp.create_dataset('sigma', data=sigma, compression='gzip')
        reparam_grp.attrs['arc_length'] = L_theoretical
        
        arc_time = time.perf_counter() - arc_start
        results['timings']['arclength'] = arc_time
        results['metrics']['lipschitz_constant'] = L_theoretical
        
        if verbose:
            print(f"  ✓ Arc length computed in {arc_time:.3f}s")
            print(f"  ✓ Lipschitz constant L = {L_theoretical:.6f}")
    
    # ========================================================================
    # STEP 3: Reparametrize to Lipschitz ψ
    # ========================================================================
    
    if verbose:
        print(f"\n[STEP 3/4] Reparametrizing to Lipschitz ψ...")
    
    with MemoryMonitor("Reparametrization"):
        reparam_start = time.perf_counter()
        
        reparametrize_to_lipschitz_streaming(
            h5file, h5file  # Input and output same file
        )
        
        reparam_time = time.perf_counter() - reparam_start
        results['timings']['reparametrization'] = reparam_time
        
        if verbose:
            print(f"  ✓ Reparametrization complete in {reparam_time:.3f}s")
    
    # ========================================================================
    # STEP 4: Verification
    # ========================================================================
    
    if not skip_verification:
        if verbose:
            print(f"\n[STEP 4/4] Verifying KST conditions...")
        
        with MemoryMonitor("Verification"):
            verification_results = run_full_verification(
                h5file, config, L_theoretical
            )
            
            # Store verification results
            verify_grp = h5file['verification']
            for claim_num, result in verification_results.items():
                if isinstance(claim_num, int):
                    claim_grp = verify_grp.create_group(f'claim_{claim_num}')
                    claim_grp.attrs['passed'] = result.passed
                    claim_grp.attrs['name'] = result.name
                    for key, val in result.details.items():
                        if isinstance(val, (int, float, bool, str)):
                            claim_grp.attrs[key] = val
            
            results['verification'] = verification_results
    else:
        if verbose:
            print(f"\n[STEP 4/4] Verification skipped (skip_verification=True)")
    
    # ========================================================================
    # Finalize
    # ========================================================================
    
    pipeline_time = time.perf_counter() - pipeline_start
    results['timings']['total_pipeline'] = pipeline_time
    
    h5file['timings'].attrs['total_pipeline'] = pipeline_time
    h5file.flush()
    h5file.close()
    
    if verbose:
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Total time: {pipeline_time:.3f}s")
        print(f"  Hölder generation: {holder_time:.3f}s "
              f"({100*holder_time/pipeline_time:.1f}%)")
        print(f"  Arc length:        {arc_time:.3f}s "
              f"({100*arc_time/pipeline_time:.1f}%)")
        print(f"  Reparametrization: {reparam_time:.3f}s "
              f"({100*reparam_time/pipeline_time:.1f}%)")
        print(f"\nCheckpoint saved: {checkpoint_path}")
        print(f"File size: {checkpoint_path.stat().st_size / 1024**2:.2f} MB")
        print("="*80 + "\n")
    
    return results