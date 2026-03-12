#!/usr/bin/env python3
"""
DASHBOARD RUN — Edit run_config.py RUN_CASES, then: python run.py

Cycles through each case in RUN_CASES. Each case runs to its own folder (no duplicate case name in path):
  runcase_N/statepoints/   — OpenMC statepoints
  runcase_N/analyze/       — Plots, cu_summary, zn_summary (e.g. outer/)
  runcase_N/zn_waste/      — Waste analysis
  runcase_N.zip            — Zipped results per case (at project root)

After all cases complete, if RUN_FLARE_NPV=True:
  npv_1h_0d/               — NPV: 1 h irradiation, 0 d cooldown
  npv_1h_1d/               — NPV: 1 h irradiation, 1 d cooldown
  npv_8h_1d/               — NPV: 8 h irradiation, 1 d cooldown
  npv_8h_2d/               — NPV: 8 h irradiation, 2 d cooldown
  npv.zip                  — Zipped NPV outputs (all four folders)

INSTRUCTIONS:
 1. Edit run_config.py RUN_CASES (add/remove cases, set RUN_MODE, thicknesses, enrichments)
 2. Run: python run.py
 3. Each runcase_N.zip contains that case's statepoints, analyze, zn_waste
 4. npv.zip contains combined data-driven NPV plots from all run cases
"""

import multiprocessing as mp
import os

import run_config as C
from fusion_irradiation import run_full_pipeline


def _build_case_config(case_dict, case_name, project_base):
    """Merge shared config with case-specific overrides; set per-case dirs.
    Outputs go under project_base/case_name/ (e.g. runcase_5/statepoints, runcase_5/analyze)
    so paths are not duplicated (no runcase_5/analyze/runcase_5/).
    """
    # Shared attrs from run_config
    shared = {
        'RUN_BASE_DIR', 'TARGET_HEIGHT_CM', 'INNER_RADIUS_CM', 'SOURCE_NEUTRON_ENERGY_MEV',
        'ECON_CU64_PRICE_PER_MCI', 'ECON_CU67_PRICE_PER_MCI', 'ECON_SHINE_SOURCE_PURCHASE',
        'ECON_SHINE_SOURCE_OPERATIONS_ANNUAL', 'ECON_IRRAD_HOURS', 'ECON_COOLDOWN_DAYS',
        'ECON_MARKET_CU64_CI_PER_YEAR',
        'IRRADIATION_HOURS', 'COOLDOWN_DAYS', 'RUN_FLARE_NPV',
    }
    cfg = type('CaseConfig', (), {})()
    for k in shared:
        if hasattr(C, k):
            setattr(cfg, k, getattr(C, k))
    for k in dir(C):
        if k.startswith('ECON_') and k not in shared:
            setattr(cfg, k, getattr(C, k))
    # Case-specific overrides (skip 'name', it's only for folder naming)
    for k, v in case_dict.items():
        if k == 'name':
            continue
        setattr(cfg, k, v)
    # Per-case base: project_base/case_name (e.g. .../runcase_5) so paths are .../runcase_5/analyze/outer
    cfg.RUN_BASE_DIR = os.path.join(project_base, case_name)
    cfg.STATEPOINTS_DIR = 'statepoints'
    cfg.ANALYZE_DIR = 'analyze'
    cfg.RESULTS_DIR = cfg.ANALYZE_DIR
    cfg.ZN_WASTE_DIR = 'zn_waste'
    cfg.NPV_DIR = 'npv'
    cfg.RESULTS_ZIP = f'{case_name}.zip'
    cfg.RESULTS_ZIP_PATH = os.path.join(project_base, f'{case_name}.zip')
    return cfg


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    run_cases = getattr(C, 'RUN_CASES', None)
    if not run_cases:
        # Fallback: single case from flat config
        case = {
            'RUN_MODE': getattr(C, 'RUN_MODE', 'single_zn64'),
            'ZN64_ENRICHMENTS': getattr(C, 'ZN64_ENRICHMENTS', [0.4917]),
            'ZN67_ENRICHMENTS': getattr(C, 'ZN67_ENRICHMENTS', [0.0404]),
            'Z_INNER_THICKNESSES': getattr(C, 'Z_INNER_THICKNESSES', [0]),
            'Z_OUTER_THICKNESSES': getattr(C, 'Z_OUTER_THICKNESSES', [5]),
            'STRUCT_THICKNESSES': getattr(C, 'STRUCT_THICKNESSES', [0.5]),
            'BORON_THICKNESSES': getattr(C, 'BORON_THICKNESSES', [0]),
            'MULTI_THICKNESSES': getattr(C, 'MULTI_THICKNESSES', [0]),
            'MODERATOR_THICKNESSES': getattr(C, 'MODERATOR_THICKNESSES', [0]),
            'PARTICLES': getattr(C, 'PARTICLES', int(100e5)),
            'BATCHES': getattr(C, 'BATCHES', 20),
            'OUTPUT_PREFIX': getattr(C, 'OUTPUT_PREFIX', 'irrad_output'),
            'RUN_PARALLEL': getattr(C, 'RUN_PARALLEL', True),
            'MAX_JOBS': getattr(C, 'MAX_JOBS', 4),
            'ZN_WASTE_CASE_INDEX': getattr(C, 'ZN_WASTE_CASE_INDEX', 0),
            'ZN_WASTE_CASE_DIR': getattr(C, 'ZN_WASTE_CASE_DIR', None),
        }
        run_cases = [case]

    base = os.path.abspath(getattr(C, 'RUN_BASE_DIR', os.getcwd()))
    case_names = []
    case_names_non_complex = []  # NPV only for non-complex geometry cases

    for i, case_dict in enumerate(run_cases):
        case_name = case_dict.get('name', f'runcase_{i + 1}')
        case_names.append(case_name)
        if not case_dict.get('COMPLEX_GEOM', False):
            case_names_non_complex.append(case_name)
        print('\n' + '=' * 70)
        print(f'RUNNING CASE {i + 1}/{len(run_cases)}: {case_name} ({case_dict.get("RUN_MODE", "?")})')
        print('=' * 70)
        cfg = _build_case_config(case_dict, case_name, base)
        run_full_pipeline(cfg)

    print('\n' + '=' * 70)
    print('ALL CASES COMPLETE')
    print('=' * 70)

    if getattr(C, 'RUN_FLARE_NPV', False) and case_names_non_complex:
        print('\n[FLARE NPV] Running 4 NPV cases (1h/0d, 1h/1d, 8h/1d, 8h/2d) for non-complex cases only...')
        try:
            import zipfile
            from flare_npv import run_flare_combined, run_data_variable_analyses
            analyze_dirs = [os.path.join(base, name, 'analyze') for name in case_names_non_complex]
            npv_cases = [
                ('npv_1h_0d', 1, 0),
                ('npv_1h_1d', 1, 1),
                ('npv_8h_1d', 8, 1),
                ('npv_8h_2d', 8, 2),
            ]
            for npv_name, irrad_h, cooldown_d in npv_cases:
                npv_dir = os.path.join(base, npv_name)
                print(f"  {npv_name}: {irrad_h}h irrad, {cooldown_d}d cooldown -> {npv_dir}")
                run_flare_combined(
                    analyze_dirs,
                    output_dir=npv_dir,
                    econ_irrad_hours=irrad_h,
                    cooldown_days=cooldown_d,
                )
                print(f"    saved to {npv_dir}")
                # Variable market scenario (purity ≥99.9%, variable cap, trajectory table + plot)
                run_data_variable_analyses(output_dir=npv_dir)
            npv_zip = os.path.join(base, 'npv.zip')
            with zipfile.ZipFile(npv_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for npv_name, _, _ in npv_cases:
                    npv_dir = os.path.join(base, npv_name)
                    if os.path.isdir(npv_dir):
                        for root, _, files in os.walk(npv_dir):
                            for f in files:
                                if 'jendl' in f.lower():
                                    continue
                                zf.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), base))
            print(f"  Saved {npv_zip}")
        except Exception as e:
            print(f"  FLARE NPV error: {e}")
            import traceback
            traceback.print_exc()
