"""
Data validation script for world datasets.

Validates that generated datasets are correct and ready for baseline training.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def validate_run(run_dir: Path) -> Dict[str, bool]:
    """Validate a single run directory."""
    checks = {
        'obs_exists': False,
        'obs_shape': False,
        'obs_dtype': False,
        'obs_no_nan': False,
        'events_exists': False,
        'events_valid': False,
        'time_to_event_exists': False,
        'time_to_event_positive': False,
        'meta_exists': False,
        'survival_record_exists': False,
    }
    
    # Check obs.npy
    obs_file = run_dir / "obs.npy"
    if obs_file.exists():
        checks['obs_exists'] = True
        obs = np.load(obs_file)
        
        # Shape check (T, F)
        if obs.ndim == 2 and obs.shape[0] > 0 and obs.shape[1] > 0:
            checks['obs_shape'] = True
        
        # Dtype check
        if obs.dtype in [np.float32, np.float64]:
            checks['obs_dtype'] = True
        
        # No NaN
        if not np.isnan(obs).any():
            checks['obs_no_nan'] = True
    
    # Check events.json
    events_file = run_dir / "events.json"
    if events_file.exists():
        checks['events_exists'] = True
        try:
            with open(events_file) as f:
                events = json.load(f)
            
            # Must have T_event
            if 'T_event' in events and isinstance(events['T_event'], (int, float)):
                checks['events_valid'] = True
        except:
            pass
    
    # Check time_to_event.npy
    tte_file = run_dir / "time_to_event.npy"
    if tte_file.exists():
        checks['time_to_event_exists'] = True
        tte = np.load(tte_file)
        
        # All values should be >= 0
        if np.all(tte >= 0):
            checks['time_to_event_positive'] = True
    
    # Check meta.json
    meta_file = run_dir / "meta.json"
    if meta_file.exists():
        checks['meta_exists'] = True
    
    # Check survival_record.csv
    survival_file = run_dir / "survival_record.csv"
    if survival_file.exists():
        checks['survival_record_exists'] = True
    
    return checks


def validate_split(world_path: Path, split: str) -> Tuple[int, int, List[str]]:
    """Validate all runs in a split."""
    split_dir = world_path / split
    
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        return 0, 0, []
    
    run_dirs = sorted(split_dir.glob("run_*"))
    
    if not run_dirs:
        logger.error(f"No runs found in {split_dir}")
        return 0, 0, []
    
    passed = 0
    failed = 0
    failures = []
    
    for run_dir in run_dirs:
        checks = validate_run(run_dir)
        
        # Critical checks
        critical = [
            'obs_exists',
            'obs_shape',
            'obs_no_nan',
            'events_exists',
            'events_valid',
        ]
        
        if all(checks[c] for c in critical):
            passed += 1
        else:
            failed += 1
            failed_checks = [k for k in critical if not checks[k]]
            failures.append(f"{run_dir.name}: {', '.join(failed_checks)}")
    
    return passed, failed, failures


def validate_aggregated_files(world_path: Path) -> Dict[str, bool]:
    """Validate aggregated CSV files."""
    checks = {
        'train_csv': False,
        'val_csv': False,
        'test_csv': False,
        'all_runs_csv': False,
        'labels_train': False,
        'labels_val': False,
        'labels_test': False,
    }
    
    # Check split CSVs
    for split in ['train', 'val', 'test']:
        csv_file = world_path / f"{split}.csv"
        if csv_file.exists():
            checks[f'{split}_csv'] = True
            
            # Check for required columns
            df = pd.read_csv(csv_file)
            if 'seed' in df.columns and 'T_event' in df.columns:
                checks[f'{split}_csv'] = True
    
    # Check all_runs.csv
    all_runs = world_path / "all_runs.csv"
    if all_runs.exists():
        checks['all_runs_csv'] = True
    
    # Check labels (try both horizon 15 and 20)
    for split in ['train', 'val', 'test']:
        for horizon in [15, 20]:
            label_file = world_path / f"labels_horizon_{horizon}_{split}.csv"
            if label_file.exists():
                checks[f'labels_{split}'] = True
                break
    
    return checks


def validate_world(world_name: str) -> None:
    """Validate entire world dataset."""
    logger.info("="*70)
    logger.info(f"VALIDATING: {world_name}")
    logger.info("="*70)
    
    world_path = Path(f"data/processed/{world_name}")
    
    if not world_path.exists():
        logger.error(f"World directory not found: {world_path}")
        return
    
    # Validate splits
    for split in ['train', 'val', 'test']:
        logger.info(f"\nValidating {split} split...")
        passed, failed, failures = validate_split(world_path, split)
        
        if failed == 0:
            logger.info(f"✅ {split}: {passed}/{passed} runs passed")
        else:
            logger.warning(f"⚠️  {split}: {passed} passed, {failed} failed")
            for failure in failures[:5]:  # Show first 5
                logger.warning(f"  - {failure}")
            if len(failures) > 5:
                logger.warning(f"  ... and {len(failures) - 5} more")
    
    # Validate aggregated files
    logger.info("\nValidating aggregated files...")
    agg_checks = validate_aggregated_files(world_path)
    
    for check, passed in agg_checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"  {status} {check}")
    
    # Summary
    all_passed = all(agg_checks.values())
    if all_passed:
        logger.info(f"\n✅ {world_name} dataset is VALID and ready for training!")
    else:
        logger.warning(f"\n⚠️  {world_name} dataset has issues, check above")


def main():
    """Validate all worlds."""
    logger.info("DATA VALIDATION - SPRINT 1")
    logger.info("="*70)
    
    worlds = ['sir_graph', 'ant_colony']
    
    for world in worlds:
        validate_world(world)
        logger.info("")
    
    logger.info("="*70)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
