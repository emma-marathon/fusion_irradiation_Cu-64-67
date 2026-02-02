#!/usr/bin/env python3
"""
Simple Analysis for Outer Zn Layer Irradiation
-----------------------------------------------
Uses OpenMC nuclear data for half-lives and utilities.py for Bateman calculations.

Produces:
- cu_summary.csv: Cu-64/Cu-67 activity, purity for all cases
- zn_summary.csv: Zn-65 activity for waste analysis
- Plots comparing activity, purity vs irradiation/cooldown/enrichment
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openmc
import openmc.data

from utilities import (
    build_channel_rr_per_s,
    evolve_bateman_irradiation,
    apply_single_decay_step,
    compute_volumes_from_dir_name,
    get_material_density_from_statepoint,
    _half_life_seconds,
)

# ============================================
# Configuration
# ============================================
OUTER_MATERIAL_ID = 1
SOURCE_STRENGTH = 5e13  # n/s

# Zn enrichment map (same as fusion_irradiation.py)
ZN64_ENRICHMENT_MAP = {
    0.50: {'Zn66': 0.277, 'Zn67': 0.040, 'Zn68': 0.177, 'Zn70': 0.002},
    0.53: {'Zn66': 0.274, 'Zn67': 0.0398, 'Zn68': 0.155, 'Zn70': 0.001},
    0.60: {'Zn66': 0.162, 'Zn67': 0.0277, 'Zn68': 0.1038, 'Zn70': 0.0008},
    0.70: {'Zn66': 0.047, 'Zn67': 0.016, 'Zn68': 0.03, 'Zn70': 0.0007},
    0.80: {'Zn66': 0.012, 'Zn67': 0.0043, 'Zn68': 0.014, 'Zn70': 0.0006},
    0.90: {'Zn66': 0.007, 'Zn67': 0.0023, 'Zn68': 0.0091, 'Zn70': 0.0005},
    0.99: {'Zn66': 0.004, 'Zn67': 0.0011, 'Zn68': 0.0031, 'Zn70': 0.0004},
}


def calculate_enriched_zn_density(zn64_enrichment):
    """
    Calculate enriched Zn density using same formula as fusion_irradiation.py.
    
    density = 7.14 * (M_avg / 65.38)
    where M_avg is weighted average atomic mass of enriched mixture.
    """
    # Get atomic masses from OpenMC
    try:
        M_Zn64 = openmc.data.atomic_mass('Zn64')
        M_Zn66 = openmc.data.atomic_mass('Zn66')
        M_Zn67 = openmc.data.atomic_mass('Zn67')
        M_Zn68 = openmc.data.atomic_mass('Zn68')
        M_Zn70 = openmc.data.atomic_mass('Zn70')
    except:
        # Fallback to standard values
        M_Zn64, M_Zn66, M_Zn67, M_Zn68, M_Zn70 = 63.929, 65.926, 66.927, 67.925, 69.925
    
    # Natural Zn composition for fallback
    natural_frac = {'Zn66': 0.279, 'Zn67': 0.041, 'Zn68': 0.188, 'Zn70': 0.006}
    
    # Get fractions from map or calculate
    if zn64_enrichment in ZN64_ENRICHMENT_MAP:
        fracs = ZN64_ENRICHMENT_MAP[zn64_enrichment]
    elif zn64_enrichment == 0.486:
        # Natural composition
        return 7.14
    else:
        # Interpolate or scale natural fractions
        remaining = 1.0 - zn64_enrichment
        scale = remaining / (1.0 - 0.486)  # Scale from natural remaining
        fracs = {k: v * scale for k, v in natural_frac.items()}
    
    # Calculate weighted average atomic mass
    M_avg = (M_Zn64 * zn64_enrichment + 
             M_Zn66 * fracs.get('Zn66', 0) +
             M_Zn67 * fracs.get('Zn67', 0) +
             M_Zn68 * fracs.get('Zn68', 0) +
             M_Zn70 * fracs.get('Zn70', 0))
    
    # Same formula as fusion_irradiation.py
    density = 7.14 * (M_avg / 65.38)
    return density


# Analysis parameters
IRRADIATION_HOURS = [1, 2, 4, 8, 12, 24, 48, 72, 100, 138]
COOLDOWN_DAYS = [0, 1, 2, 3, 4]  # 0 = end-of-irradiation (no cooldown)

# Isotopes of interest
CU_ISOTOPES = ['Cu64', 'Cu67']
ZN_ISOTOPES = ['Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']


def get_decay_constant(nuclide):
    """Get decay constant (1/s) from OpenMC nuclear data."""
    hl = _half_life_seconds(nuclide)
    if hl is not None and hl > 0:
        return np.log(2) / hl
    return 0.0


def parse_dir_name(dir_name):
    """Parse directory name to extract simulation parameters."""
    params = {'inner': 0, 'outer': 20, 'struct': 0, 'multi': 0, 'moderator': 0, 'zn_enrichment': 0.486}
    
    try:
        if '_inner' in dir_name:
            params['inner'] = float(dir_name.split('_inner')[1].split('_')[0])
        if '_outer' in dir_name:
            params['outer'] = float(dir_name.split('_outer')[1].split('_')[0])
        if '_struct' in dir_name:
            params['struct'] = float(dir_name.split('_struct')[1].split('_')[0])
        if '_multi' in dir_name:
            params['multi'] = float(dir_name.split('_multi')[1].split('_')[0])
        if '_moderator' in dir_name:
            params['moderator'] = float(dir_name.split('_moderator')[1].split('_')[0])
        if '_zn' in dir_name:
            zn_str = dir_name.split('_zn')[1].replace('%', '')
            params['zn_enrichment'] = float(zn_str) / 100.0
    except (ValueError, IndexError):
        pass
    
    return params


def find_statepoints(base_dir, pattern='radial_output_*'):
    """
    Find all statepoint files in output directories matching the given pattern.
    
    Parameters
    ----------
    base_dir : str
        Base directory to search in
    pattern : str
        Glob pattern for output directories (e.g., 'radial_output_*')
    """
    statepoints = []
    
    for d in glob.glob(os.path.join(base_dir, pattern)):
        sp_files = glob.glob(os.path.join(d, 'statepoint.*.h5'))
        if sp_files:
            statepoints.append(sorted(sp_files)[-1])
    
    return statepoints


def get_initial_zn_atoms(volume_cm3, zn64_enrichment, density_g_cm3):
    """
    Calculate initial Zn atom counts based on volume, enrichment, and density.
    Density should come from statepoint summary materials.
    Uses OpenMC atomic mass data.
    """
    # Zn isotope fractions (natural abundance baseline)
    # Will be modified by enrichment
    zn_fractions = {
        'Zn64': zn64_enrichment,
        'Zn66': (1 - zn64_enrichment) * 0.279 / (1 - 0.486),  # Scale remaining
        'Zn67': (1 - zn64_enrichment) * 0.041 / (1 - 0.486),
        'Zn68': (1 - zn64_enrichment) * 0.188 / (1 - 0.486),
        'Zn70': (1 - zn64_enrichment) * 0.006 / (1 - 0.486),
    }
    # Normalize
    total_frac = sum(zn_fractions.values())
    zn_fractions = {k: v/total_frac for k, v in zn_fractions.items()}
    
    # Average atomic mass
    avg_mass = sum(zn_fractions[iso] * openmc.data.atomic_mass(iso) for iso in zn_fractions)
    
    # Total mass and atoms
    total_mass_g = volume_cm3 * density_g_cm3
    avogadro = 6.022e23
    total_atoms = (total_mass_g / avg_mass) * avogadro
    
    # Initial atoms by isotope
    initial_atoms = {iso: total_atoms * frac for iso, frac in zn_fractions.items()}
    # Add Cu isotopes with 0 initial atoms
    initial_atoms['Cu64'] = 0.0
    initial_atoms['Cu67'] = 0.0
    initial_atoms['Zn65'] = 0.0  # Not naturally present
    initial_atoms['Zn69'] = 0.0  # Not naturally present
    
    return initial_atoms


def analyze_case(sp_file):
    """
    Analyze a single simulation case.
    Returns dict with reaction rates, geometry parameters, and material properties.
    Density is extracted from statepoint summary (as set by fusion_irradiation.py).
    """
    dir_name = os.path.basename(os.path.dirname(sp_file))
    params = parse_dir_name(dir_name)
    
    # Get volumes from geometry
    try:
        volumes = compute_volumes_from_dir_name(dir_name)
        outer_volume = volumes.get(1, 0)
    except:
        outer_volume = 188495.56  # Default 20cm thickness
    
    # Open statepoint and get reaction rates
    sp = openmc.StatePoint(sp_file)
    rr = build_channel_rr_per_s(sp, cell_id=OUTER_MATERIAL_ID, source_strength=SOURCE_STRENGTH)
    
    # Get density from statepoint summary materials (set by fusion_irradiation.py)
    # The enriched Zn density is calculated as: 7.14 * (M_avg / 65.38)
    # where M_avg is the average atomic mass of the enriched mixture
    zn_density = get_material_density_from_statepoint(sp, OUTER_MATERIAL_ID)
    
    if zn_density is None:
        # Fallback: calculate from enrichment using same formula as fusion_irradiation.py
        zn64_enrich = params.get('zn_enrichment', 0.486)
        zn_density = calculate_enriched_zn_density(zn64_enrich)
        print(f"  Warning: Using calculated density {zn_density:.4f} g/cm3 (enrich={zn64_enrich*100:.1f}%)")
    else:
        print(f"  Density from statepoint: {zn_density:.4f} g/cm3")
    
    # Calculate mass
    zn_mass_g = outer_volume * zn_density
    
    return {
        'dir_name': dir_name,
        'zn64_enrichment': params['zn_enrichment'],
        'multi_cm': params['multi'],
        'moderator_cm': params['moderator'],
        'outer_volume_cm3': outer_volume,
        'zn_density_g_cm3': zn_density,
        'zn_mass_g': zn_mass_g,
        'reaction_rates': rr,  # Full dict for Bateman
    }


def compute_activities(case, irrad_hours, cooldown_days):
    """
    Compute Cu-64, Cu-67, Zn-65 activities using Bateman equation from utilities.
    
    Uses:
    - evolve_bateman_irradiation() for irradiation phase (production + decay)
    - apply_single_decay_step() for cooldown phase (decay only)
    - OpenMC nuclear data for half-lives
    
    Returns dict with activities (Bq, mCi) and purities.
    """
    irrad_s = irrad_hours * 3600
    cooldown_s = cooldown_days * 86400
    
    # Get initial atom counts (using density from statepoint)
    initial_atoms = get_initial_zn_atoms(
        case['outer_volume_cm3'], 
        case['zn64_enrichment'],
        case['zn_density_g_cm3']
    )
    
    # Irradiation phase: production + decay (Bateman equation)
    atoms_eoi = evolve_bateman_irradiation(initial_atoms, case['reaction_rates'], irrad_s)
    
    # Cooldown phase: decay only
    atoms_final = apply_single_decay_step(atoms_eoi, cooldown_s)
    
    # Get decay constants for activity calculation
    lam_cu64 = get_decay_constant('Cu64')
    lam_cu67 = get_decay_constant('Cu67')
    lam_zn65 = get_decay_constant('Zn65')
    
    # Activities (Bq) = N * λ
    cu64_Bq = atoms_final.get('Cu64', 0) * lam_cu64
    cu67_Bq = atoms_final.get('Cu67', 0) * lam_cu67
    zn65_Bq = atoms_final.get('Zn65', 0) * lam_zn65
    
    # Convert to mCi (1 Ci = 3.7e10 Bq)
    cu64_mCi = cu64_Bq / 3.7e7
    cu67_mCi = cu67_Bq / 3.7e7
    zn65_mCi = zn65_Bq / 3.7e7
    
    # Purity: fraction of Cu that is Cu-64 or Cu-67
    cu64_atoms = atoms_final.get('Cu64', 0)
    cu67_atoms = atoms_final.get('Cu67', 0)
    total_cu = cu64_atoms + cu67_atoms
    cu64_purity = cu64_atoms / total_cu if total_cu > 0 else 0
    cu67_purity = cu67_atoms / total_cu if total_cu > 0 else 0
    
    return {
        'cu64_mCi': cu64_mCi,
        'cu67_mCi': cu67_mCi,
        'zn65_mCi': zn65_mCi,
        'cu64_Bq': cu64_Bq,
        'cu67_Bq': cu67_Bq,
        'zn65_Bq': zn65_Bq,
        'cu64_purity': cu64_purity,
        'cu67_purity': cu67_purity,
        'cu64_atoms': cu64_atoms,
        'cu67_atoms': cu67_atoms,
        'zn65_atoms': atoms_final.get('Zn65', 0),
    }


def build_summary_dataframes(cases):
    """
    Build Cu and Zn summary DataFrames for all cases, irradiation times, and cooldown times.
    """
    cu_rows = []
    zn_rows = []
    
    for case in cases:
        # Get Zn volume, density, and mass from case (from statepoint)
        volume_cm3 = case['outer_volume_cm3']
        zn_density = case['zn_density_g_cm3']
        mass_g = case['zn_mass_g']
        mass_kg = mass_g / 1000.0
        
        for irrad_h in IRRADIATION_HOURS:
            for cool_d in COOLDOWN_DAYS:
                act = compute_activities(case, irrad_h, cool_d)
                
                cu_rows.append({
                    'zn64_enrichment': case['zn64_enrichment'],
                    'multi_cm': case['multi_cm'],
                    'mod_cm': case['moderator_cm'],
                    'zn_volume_cm3': volume_cm3,
                    'zn_density_g_cm3': zn_density,
                    'zn_mass_g': mass_g,
                    'zn_mass_kg': mass_kg,
                    'irrad_hours': irrad_h,
                    'cooldown_days': cool_d,
                    'cu64_mCi': act['cu64_mCi'],
                    'cu67_mCi': act['cu67_mCi'],
                    'cu64_Bq': act['cu64_Bq'],
                    'cu67_Bq': act['cu67_Bq'],
                    'cu64_purity': act['cu64_purity'],
                    'cu67_purity': act['cu67_purity'],
                })
                
                zn_rows.append({
                    'zn64_enrichment': case['zn64_enrichment'],
                    'multi_cm': case['multi_cm'],
                    'mod_cm': case['moderator_cm'],
                    'zn_volume_cm3': volume_cm3,
                    'zn_density_g_cm3': zn_density,
                    'zn_mass_g': mass_g,
                    'zn_mass_kg': mass_kg,
                    'irrad_hours': irrad_h,
                    'cooldown_days': cool_d,
                    'zn65_mCi': act['zn65_mCi'],
                    'zn65_Bq': act['zn65_Bq'],
                    'zn65_specific_activity_Bq_per_g': act['zn65_Bq'] / mass_g if mass_g > 0 else 0,
                })
    
    cu_df = pd.DataFrame(cu_rows)
    zn_df = pd.DataFrame(zn_rows)
    
    return cu_df, zn_df


def plot_activity_vs_variables(cu_df, output_dir):
    """
    Plot Cu-64 and Cu-67 activity vs irradiation, cooldown, and enrichment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Filter to multi=0, mod=0 if available
    df = cu_df.copy()
    multi_val, mod_val = 0, 0
    if 0 in df['multi_cm'].values and 0 in df['mod_cm'].values:
        df = df[(df['multi_cm'] == 0) & (df['mod_cm'] == 0)]
    else:
        multi_val = df['multi_cm'].iloc[0] if len(df) > 0 else 0
        mod_val = df['mod_cm'].iloc[0] if len(df) > 0 else 0
    
    enrichments = sorted(df['zn64_enrichment'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))
    
    geom_info = f"Outer: 20cm, Multi: {multi_val:.0f}cm, Mod: {mod_val:.0f}cm"
    
    # --- Plot 1: Activity vs Irradiation (cooldown=1 day) ---
    ax = axes[0]
    cool_fixed = 1
    
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['cooldown_days'] == cool_fixed)]
        sub = sub.groupby('irrad_hours').mean(numeric_only=True).reset_index().sort_values('irrad_hours')
        
        if not sub.empty:
            ax.semilogy(sub['irrad_hours'], sub['cu64_mCi'], 'o-', color=colors[i], 
                       label=f'Cu-64 Zn64={enrich*100:.0f}%', linewidth=2, markersize=5)
            ax.semilogy(sub['irrad_hours'], sub['cu67_mCi'], 's--', color=colors[i], 
                       alpha=0.6, label=f'Cu-67 Zn64={enrich*100:.0f}%', markersize=4)
    
    ax.set_xlabel('Irradiation Time (hours)', fontsize=11)
    ax.set_ylabel('Activity (mCi)', fontsize=11)
    ax.set_title(f'Activity vs Irradiation Time\nCooldown: {cool_fixed} day | {geom_info}', fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    
    # --- Plot 2: Activity vs Cooldown (irrad=8 hours) ---
    ax = axes[1]
    irrad_fixed = 8
    
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['irrad_hours'] == irrad_fixed)]
        sub = sub.groupby('cooldown_days').mean(numeric_only=True).reset_index().sort_values('cooldown_days')
        
        if not sub.empty:
            ax.semilogy(sub['cooldown_days'], sub['cu64_mCi'], 'o-', color=colors[i], 
                       label=f'Cu-64 Zn64={enrich*100:.0f}%', linewidth=2, markersize=6)
            ax.semilogy(sub['cooldown_days'], sub['cu67_mCi'], 's--', color=colors[i], 
                       alpha=0.6, label=f'Cu-67 Zn64={enrich*100:.0f}%', markersize=5)
    
    ax.set_xlabel('Cooldown Time (days)', fontsize=11)
    ax.set_ylabel('Activity (mCi)', fontsize=11)
    ax.set_title(f'Activity vs Cooldown Time\nIrradiation: {irrad_fixed}h | {geom_info}', fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    
    # --- Plot 3: Activity vs Enrichment (8h irrad, 1d cooldown) ---
    ax = axes[2]
    sub = df[(df['irrad_hours'] == irrad_fixed) & (df['cooldown_days'] == cool_fixed)]
    sub = sub.groupby('zn64_enrichment').mean(numeric_only=True).reset_index().sort_values('zn64_enrichment')
    
    if not sub.empty:
        ax.semilogy(sub['zn64_enrichment'] * 100, sub['cu64_mCi'], 'o-', 
                   color='blue', label='Cu-64', linewidth=2, markersize=8)
        ax.semilogy(sub['zn64_enrichment'] * 100, sub['cu67_mCi'], 's--', 
                   color='red', label='Cu-67', linewidth=2, markersize=8)
    
    ax.set_xlabel('Zn-64 Enrichment (%)', fontsize=11)
    ax.set_ylabel('Activity (mCi)', fontsize=11)
    ax.set_title(f'Activity vs Enrichment\nIrrad: {irrad_fixed}h, Cool: {cool_fixed}d | {geom_info}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_vs_variables.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/activity_vs_variables.png")


def plot_purity_vs_variables(cu_df, output_dir):
    """
    Plot Cu-64 and Cu-67 impurity (1-purity) vs variables.
    Y-axis shows percentage with clear labels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Filter to multi=0, mod=0 if available
    df = cu_df.copy()
    multi_val, mod_val = 0, 0
    if 0 in df['multi_cm'].values and 0 in df['mod_cm'].values:
        df = df[(df['multi_cm'] == 0) & (df['mod_cm'] == 0)]
    else:
        multi_val = df['multi_cm'].iloc[0] if len(df) > 0 else 0
        mod_val = df['mod_cm'].iloc[0] if len(df) > 0 else 0
    
    enrichments = sorted(df['zn64_enrichment'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))
    
    geom_info = f"Outer: 20cm, Multi: {multi_val:.0f}cm, Mod: {mod_val:.0f}cm"
    
    # --- Plot 1: Cu-64 Impurity vs Irradiation ---
    ax = axes[0]
    cool_fixed = 1
    
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['cooldown_days'] == cool_fixed)]
        sub = sub.groupby('irrad_hours').mean(numeric_only=True).reset_index().sort_values('irrad_hours')
        
        if not sub.empty:
            cu64_imp = np.clip((1 - sub['cu64_purity']) * 100, 0.001, 100)
            ax.semilogy(sub['irrad_hours'], cu64_imp, 'o-', color=colors[i], 
                       label=f'Zn64={enrich*100:.0f}%', linewidth=2, markersize=5)
    
    ax.set_xlabel('Irradiation Time (hours)', fontsize=11)
    ax.set_ylabel('Cu-64 Impurity [%]', fontsize=11)
    ax.set_title(f'Cu-64 Impurity vs Irradiation Time\nCooldown: {cool_fixed} day | {geom_info}', fontsize=11)
    ax.legend(fontsize=9, loc='upper left', title='Zn-64 Enrichment')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    
    # --- Plot 2: Cu-64 Impurity vs Cooldown ---
    ax = axes[1]
    irrad_fixed = 8
    
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['irrad_hours'] == irrad_fixed)]
        sub = sub.groupby('cooldown_days').mean(numeric_only=True).reset_index().sort_values('cooldown_days')
        
        if not sub.empty:
            cu64_imp = np.clip((1 - sub['cu64_purity']) * 100, 0.001, 100)
            ax.semilogy(sub['cooldown_days'], cu64_imp, 'o-', color=colors[i], 
                       label=f'Zn64={enrich*100:.0f}%', linewidth=2, markersize=6)

    ax.set_xlabel('Cooldown Time (days)', fontsize=11)
    ax.set_ylabel('Cu-64 Impurity [%]', fontsize=11)
    ax.set_title(f'Cu-64 Impurity vs Cooldown Time\nIrradiation: {irrad_fixed}h | {geom_info}', fontsize=11)
    ax.legend(fontsize=9, loc='upper left', title='Zn-64 Enrichment')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    
    # --- Plot 3: Cu-64 Impurity vs Enrichment ---
    ax = axes[2]
    sub = df[(df['irrad_hours'] == irrad_fixed) & (df['cooldown_days'] == cool_fixed)]
    sub = sub.groupby('zn64_enrichment').mean(numeric_only=True).reset_index().sort_values('zn64_enrichment')
    
    if not sub.empty:
        cu64_imp = np.clip((1 - sub['cu64_purity']) * 100, 0.001, 100)
        ax.semilogy(sub['zn64_enrichment'] * 100, cu64_imp, 'o-', 
                   color='blue', label='Cu-64', linewidth=2, markersize=8)
    
    ax.set_xlabel('Zn-64 Enrichment (%)', fontsize=11)
    ax.set_ylabel('Cu-64 Impurity [%]', fontsize=11)
    ax.set_title(f'Cu-64 Impurity vs Enrichment\nIrrad: {irrad_fixed}h, Cool: {cool_fixed}d | {geom_info}', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'impurity_vs_variables.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/impurity_vs_variables.png")


def plot_production_vs_purity(cu_df, output_dir):
    """
    Plot Cu-64 production (mCi) vs impurity (1-purity) on log scale.
    Shows only 1, 4, 8h irradiation times with no cooldown (end-of-irradiation).
    Encodes: enrichment (color), irradiation time (marker), geometry (edge color/size).
    X-axis: Impurity (%) log scale, Y-axis: Production (mCi, log scale)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filter: no cooldown (0 days) and only 1, 4, 8h irradiation
    irrad_filter = [1, 4, 8]
    cool_fixed = 0  # End of irradiation (no cooldown)
    
    df = cu_df[(cu_df['cooldown_days'] == cool_fixed) & 
               (cu_df['irrad_hours'].isin(irrad_filter))].copy()
    
    if df.empty:
        # Fallback to smallest available cooldown if 0 not available
        min_cool = cu_df['cooldown_days'].min()
        df = cu_df[(cu_df['cooldown_days'] == min_cool) & 
                   (cu_df['irrad_hours'].isin(irrad_filter))].copy()
        cool_fixed = min_cool
        print(f"  Note: No 0-day cooldown data, using {cool_fixed}-day cooldown")
    
    if df.empty:
        print("  Warning: No data for production vs purity plot")
        plt.close()
        return
    
    # Calculate impurity (1-purity) as percentage
    df['impurity_pct'] = (1 - df['cu64_purity']) * 100
    
    # Get unique values
    enrichments = sorted(df['zn64_enrichment'].unique())
    irrad_times = sorted(df['irrad_hours'].unique())
    
    # Get unique geometry configs (multi, mod)
    geom_configs = df[['multi_cm', 'mod_cm']].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1]))
    
    # Colors for enrichment (distinct colors)
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) 
                    for i, e in enumerate(enrichments)}
    
    # Markers for irradiation time (distinct shapes)
    irrad_marker = {1: 'o', 4: 's', 8: 'D'}
    
    # Edge colors for geometry (multi/mod) - use distinct colors
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1)) 
                 for i, g in enumerate(geom_configs)}
    
    # Plot all points
    for _, row in df.iterrows():
        enrich = row['zn64_enrichment']
        irrad = row['irrad_hours']
        impurity = row['impurity_pct']
        geom = (row['multi_cm'], row['mod_cm'])
        
        # Clip impurity to avoid log(0)
        impurity = max(impurity, 0.001)
        
        ax.scatter(impurity, row['cu64_mCi'],
                  c=[enrich_color[enrich]], marker=irrad_marker.get(irrad, 'o'),
                  s=140, edgecolors=[geom_edge[geom]], linewidths=2.5, alpha=0.85)
    
    ax.set_xlabel('Cu-64 Impurity (100% - Purity) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cu-64 Production [mCi]', fontsize=12, fontweight='bold')
    
    cooldown_str = "End of Irradiation (no cooldown)" if cool_fixed == 0 else f"{cool_fixed}-day Cooldown"
    ax.set_title(f'Cu-64 Production vs Impurity\n'
                 f'Irradiation: 1, 4, 8h | {cooldown_str}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # X-axis: impurity from 0.01% to 100%
    ax.set_xlim(0.01, 20)
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
    
    # Single reference line at 0.1% impurity (99.9% purity)
    ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.text(0.1, ax.get_ylim()[1]*0.7, '99.9% purity', fontsize=10, color='gray', 
            ha='center', va='bottom', rotation=90)
    
    # === LEGEND 1: Enrichment (fill colors) ===
    enrich_handles = []
    for e in enrichments:
        h = ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o', 
                      edgecolors='black', linewidths=0.5, label=f'{e*100:.0f}%')
        enrich_handles.append(h)
    
    # === LEGEND 2: Irradiation time (marker shapes) ===
    irrad_handles = []
    for t in irrad_times:
        h = ax.scatter([], [], c='lightgray', s=100, marker=irrad_marker.get(t, 'o'),
                      edgecolors='black', linewidths=1, label=f'{t}h')
        irrad_handles.append(h)
    
    # === LEGEND 3: Geometry (edge colors) ===
    geom_handles = []
    for g in geom_configs:
        multi, mod = g
        label = f'Multi={multi:.0f}cm, Mod={mod:.0f}cm'
        h = ax.scatter([], [], c='white', s=100, marker='o',
                      edgecolors=[geom_edge[tuple(g)]], linewidths=3, label=label)
        geom_handles.append(h)
    
    # Create three separate legends
    leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)', 
                     loc='upper left', fontsize=9, title_fontsize=10,
                     framealpha=0.95)
    ax.add_artist(leg1)
    
    leg2 = ax.legend(handles=irrad_handles, title='Irradiation Time\n(marker shape)',
                     loc='lower left', fontsize=9, title_fontsize=10,
                     framealpha=0.95)
    ax.add_artist(leg2)
    
    leg3 = ax.legend(handles=geom_handles, title='Geometry\n(edge color)',
                     loc='upper right', fontsize=9, title_fontsize=10,
                     framealpha=0.95)
    
    # Add summary text
    summary_text = f"Total: {len(df)} points\nBest purity: {df['cu64_purity'].max()*100:.2f}%"
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'production_vs_purity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/production_vs_purity.png")


def plot_zoomed_impurity_with_table(cu_df, output_dir, multi_cm, mod_cm):
    """
    Plot zoomed Cu-64 impurity for 8h irradiation, 1 day cooldown with purity table.
    Shows impurity evolution and table of purity values at key points.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter data
    df = cu_df[(cu_df['multi_cm'] == multi_cm) & (cu_df['mod_cm'] == mod_cm)].copy()
    if df.empty:
        plt.close()
        return
    
    irrad_fixed = 8
    cool_fixed = 1
    
    enrichments = sorted(df['zn64_enrichment'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))
    
    geom_info = f"Multi: {multi_cm:.0f}cm, Mod: {mod_cm:.0f}cm"
    
    # --- Plot 1: Impurity time evolution (irradiation + cooldown) ---
    ax = axes[0]
    
    # Build time series for each enrichment
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & 
                 (df['irrad_hours'] == irrad_fixed) & 
                 (df['cooldown_days'] == cool_fixed)]
        
        if not sub.empty:
            # Get impurity at this point
            imp = (1 - sub['cu64_purity'].values[0]) * 100
            purity = sub['cu64_purity'].values[0] * 100
            
            # Plot as a single point for now
            ax.scatter([cool_fixed], [imp], color=colors[i], s=100, 
                      label=f'Zn64={enrich*100:.0f}% (purity={purity:.2f}%)', zorder=5)
    
    # Also show impurity vs cooldown time for this irrad time
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['irrad_hours'] == irrad_fixed)]
        sub = sub.sort_values('cooldown_days')
        if not sub.empty:
            cu64_imp = np.clip((1 - sub['cu64_purity']) * 100, 0.001, 100)
            ax.semilogy(sub['cooldown_days'], cu64_imp, 'o-', color=colors[i], 
                       linewidth=2, markersize=8)
    
    ax.set_xlabel('Cooldown Time (days)', fontsize=11)
    ax.set_ylabel('Cu-64 Impurity [%]', fontsize=11)
    ax.set_title(f'Cu-64 Impurity vs Cooldown\nIrradiation: {irrad_fixed}h | {geom_info}', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    
    # Reference line at 0.1% impurity (99.9% purity)
    ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.text(ax.get_xlim()[1]*0.95, 0.12, '99.9% purity', fontsize=9, color='gray', ha='right')
    
    # --- Plot 2: Purity Table ---
    ax = axes[1]
    ax.axis('off')
    
    # Build table data
    table_data = [['Zn-64 Enrich.', 'Irrad (h)', 'Cool (d)', 'Cu-64 Purity', 'Impurity']]
    
    for enrich in enrichments:
        sub = df[(df['zn64_enrichment'] == enrich) & 
                 (df['irrad_hours'] == irrad_fixed) & 
                 (df['cooldown_days'] == cool_fixed)]
        if not sub.empty:
            purity = sub['cu64_purity'].values[0] * 100
            impurity = (1 - sub['cu64_purity'].values[0]) * 100
            table_data.append([
                f'{enrich*100:.0f}%',
                f'{irrad_fixed}',
                f'{cool_fixed}',
                f'{purity:.3f}%',
                f'{impurity:.3f}%'
            ])
    
    # Add rows for different irradiation times at 99% enrichment
    table_data.append(['', '', '', '', ''])
    table_data.append(['--- 99% Zn-64 at different irrad times ---', '', '', '', ''])
    
    enrich_99 = max(enrichments)  # Highest enrichment
    for irrad_h in [1, 4, 8, 24, 48]:
        sub = df[(df['zn64_enrichment'] == enrich_99) & 
                 (df['irrad_hours'] == irrad_h) & 
                 (df['cooldown_days'] == cool_fixed)]
        if not sub.empty:
            purity = sub['cu64_purity'].values[0] * 100
            impurity = (1 - sub['cu64_purity'].values[0]) * 100
            table_data.append([
                f'{enrich_99*100:.0f}%',
                f'{irrad_h}',
                f'{cool_fixed}',
                f'{purity:.3f}%',
                f'{impurity:.3f}%'
            ])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.22, 0.15, 0.15, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax.set_title(f'Cu-64 Purity Summary\n{geom_info}', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cu64_purity_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_dir}/cu64_purity_summary.png")


def generate_plots_for_config(cu_df, zn_df, output_dir, multi_cm, mod_cm):
    """Generate all plots for a specific geometry configuration."""
    # Filter data for this config
    cu_config = cu_df[(cu_df['multi_cm'] == multi_cm) & (cu_df['mod_cm'] == mod_cm)].copy()
    zn_config = zn_df[(zn_df['multi_cm'] == multi_cm) & (zn_df['mod_cm'] == mod_cm)].copy()
    
    if cu_config.empty:
        return
    
    # Save CSVs for this config
    cu_config.to_csv(os.path.join(output_dir, 'cu_summary.csv'), index=False)
    zn_config.to_csv(os.path.join(output_dir, 'zn_summary.csv'), index=False)
    print(f"    Saved CSVs: {len(cu_config)} Cu rows, {len(zn_config)} Zn rows")
    
    # Generate plots
    plot_activity_vs_variables(cu_config, output_dir)
    plot_purity_vs_variables(cu_config, output_dir)
    plot_zoomed_impurity_with_table(cu_df, output_dir, multi_cm, mod_cm)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple analysis for outer Zn layer irradiation")
    parser.add_argument("--base-dir", type=str, default=".", 
                        help="Base directory containing simulation output folders")
    parser.add_argument("--output-dir", type=str, default="radial_analysis_results",
                        help="Output directory for results")
    parser.add_argument("--pattern", type=str, default="radial_output_*",
                        help="Glob pattern for output directories (default: radial_output_*)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find statepoint files
    print(f"Finding statepoint files matching '{args.pattern}'...")
    statepoints = find_statepoints(args.base_dir, args.pattern)
    
    if not statepoints:
        print(f"No statepoint files found in {args.base_dir} matching '{args.pattern}'")
        return
    
    print(f"Found {len(statepoints)} statepoint files")
    
    # Analyze each case
    print("\nAnalyzing cases...")
    cases = []
    for sp_file in statepoints:
        print(f"\nAnalyzing: {sp_file}")
        try:
            case = analyze_case(sp_file)
            cases.append(case)
            
            # Print Cu production rates
            rr = case['reaction_rates']
            cu64_rate = rr.get("Zn64 (n,p) Cu64", 0)
            cu67_np = rr.get("Zn67 (n,p) Cu67", 0)
            cu67_nd = rr.get("Zn68 (n,d) Cu67", 0)
            print(f"  ✓ {case['dir_name']}: Cu64={cu64_rate:.2e}, Cu67={(cu67_np+cu67_nd):.2e} atoms/s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if not cases:
        print("No valid cases found!")
        return
    
    # Build summary tables
    print("\nBuilding summary tables...")
    cu_df, zn_df = build_summary_dataframes(cases)
    
    # Save combined CSVs to main output dir
    cu_df.to_csv(os.path.join(args.output_dir, 'cu_summary_all.csv'), index=False)
    zn_df.to_csv(os.path.join(args.output_dir, 'zn_summary_all.csv'), index=False)
    print(f"  Saved: {args.output_dir}/cu_summary_all.csv ({len(cu_df)} rows)")
    print(f"  Saved: {args.output_dir}/zn_summary_all.csv ({len(zn_df)} rows)")
    
    # Get unique geometry configurations
    configs = cu_df[['multi_cm', 'mod_cm']].drop_duplicates().values.tolist()
    print(f"\nFound {len(configs)} geometry configurations")
    
    # Generate plots for each configuration in separate folders
    print("\nGenerating plots by configuration...")
    for multi_cm, mod_cm in configs:
        config_dir = os.path.join(args.output_dir, f'multi{multi_cm:.0f}_mod{mod_cm:.0f}')
        os.makedirs(config_dir, exist_ok=True)
        print(f"\n  Config: Multi={multi_cm:.0f}cm, Mod={mod_cm:.0f}cm -> {config_dir}")
        generate_plots_for_config(cu_df, zn_df, config_dir, multi_cm, mod_cm)
    
    # Generate combined production vs purity plot (all configs)
    print("\n\nGenerating combined production vs purity plot...")
    plot_production_vs_purity(cu_df, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
