"""
Zn-65 waste packaging and disposal: drum fill by transport/storage limits or by volume only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

HALF_LIVES = {
    'Zn65': 244.0 * 86400,
    'Zn69m': 13.756 * 3600,
    'Cu64': 12.7 * 3600,
    'Cu67': 61.83 * 3600,
}

DECAY_CONSTANTS = {iso: np.log(2) / hl for iso, hl in HALF_LIVES.items()}

# External dose rate coefficient: µSv/hr per MBq at 1 m (ICRP / MIRD style). Zn62, Zn63 from user values.
EXTERNAL_DOSE_RATE = {
    'Zn62': 0.04968,    'Zn63': 0.019026,  'Zn64': 0.0,        'Zn65': 0.070992,      'Zn66': 0.0,       'Zn67': 0.0,
    'Zn68': 0.0,        'Zn69': 7.94e-7,   'Zn70': 0.0,
    'Zn69m': 0.056628,  'Zn71': 4.1796e-4,
    'Pb203': 0.098,     'Pb205': 0.386,    'Pb208': 0.0,      'Pb209': 5.85e-4,   'Pb210': 0.035,
    'Tl205': 0.0,
    'Cu64': 0.036,      'Cu67': 0.0014,    'Cu61': 0.0254,     'Cu62': 8.712e-4,
    'Cu66': 0.0125,     'Cu69': 0.06786,   'Cu70': 0.0,
    'Ni61': 0.0,        'Ni63': 0.0,       'Ni64': 0.0,        'Ni65': 0.067428,   'Ni67': 0.0,
    'Bi208': 0.3089,    'Bi210': 0.00165,  'Bi210m': 0.0437,   'Bi213': 0.01959,
    'Po210': 0.0012,
}

CAVITY_ISOTOPES = ('Zn62', 'Zn63', 'Zn65', 'Zn69m', 'Cu61', 'Cu62', 'Cu64', 'Cu66', 'Cu67', 'Cu69', 'Cu70', 'Ni61', 'Ni63', 'Ni64', 'Ni65', 'Ni67')


def get_dose_coeff(nuclide):
    return EXTERNAL_DOSE_RATE.get(nuclide, 0.0)

DOSE_LIMIT_STORAGE_UVSV_HR = 2000.0
DOSE_LIMIT_TRANSPORT_UVSV_HR = 2000.0
DOSE_LIMIT_UNSHIELDED_3M_UVSV_HR = 10_000.0
DISTANCE_3M_CM = 300.0

# Stables / no gamma use FAILSAFE.
FAILSAFE_HVL = {'HVL_Pb_cm': 2.0, 'HVL_Bi_cm': 2.4, 'HVL_concrete_cm': 8.0, 'HVL_boron_cm': 7.7, 'HVL_quartz_cm': 7.6}

# Lead thickness (cm) for buildup: B = 1 + μ*x.
SHIELDING_THICKNESS_CM = 0.5

# Lead density (g/cm³).
RHO_PB_G_CM3 = 11.35
LN2 = np.log(2.0)

# Scale factor Bi/Pb: HVL_Bi = _BI_PB_SCALE * HVL_Pb  =>  μ_Bi = μ_Pb / _BI_PB_SCALE.
_BI_PB_SCALE = 1.15
# Concrete: HVL_concrete = 4.5 * HVL_Pb  =>  μ_concrete = μ_Pb / 4.5.
_CONCRETE_PB_SCALE = 4.5
# Boron: HVL_boron = _BORON_PB_SCALE * HVL_Pb  =>  μ_boron = μ_Pb / _BORON_PB_SCALE.
_BORON_PB_SCALE = 3.85
# Quartz: HVL_quartz = _QUARTZ_PB_SCALE * HVL_Pb  =>  μ_quartz = μ_Pb / _QUARTZ_PB_SCALE.
_QUARTZ_PB_SCALE = 3.8

# Lead: Energy (MeV) -> mass attenuation coefficient μ/ρ (cm²/g). From standard tables (NIST/XCOM style).
# HVL = ln(2) / μ,  μ = (μ/ρ) * ρ_Pb.
_PB_MU_OVER_RHO_TABLE = [
    (0.003, 2146.0), (0.004, 1251), (0.005, 730), (0.008, 228), (0.01, 130.6),
    (0.015, 111.6), (0.02, 86.36), (0.03, 14.36), (0.05, 8.041), (0.06, 5.021), (0.08, 2.419),
    (0.1, 5.549), (0.15, 2.014), (0.2, 0.9985), (0.3, 0.4031), (0.4, 0.2323), (0.5, 0.1614), 
    (0.6, 0.1248), (0.8, 0.0887), (1.0, 0.07102), (1.25, 0.05876), (1.5, 0.05222), (2.0, 0.04606),
    (3.0, 0.04031), (4.0, 0.04197), (5.0, 0.04272), (6.0, 0.0439), (8.0, 0.0468),
    (10.0, 0.0497), (15.0, 0.0566), (20.0, 0.0621),
]


def _interp_mu_over_rho_pb(E_MeV):
    """Linear interpolation of μ/ρ (cm²/g) for lead at energy E_MeV. Clamp outside table range."""
    E = float(E_MeV)
    if E <= _PB_MU_OVER_RHO_TABLE[0][0]:
        return _PB_MU_OVER_RHO_TABLE[0][1]
    if E >= _PB_MU_OVER_RHO_TABLE[-1][0]:
        return _PB_MU_OVER_RHO_TABLE[-1][1]
    for i in range(len(_PB_MU_OVER_RHO_TABLE) - 1):
        e1, m1 = _PB_MU_OVER_RHO_TABLE[i]
        e2, m2 = _PB_MU_OVER_RHO_TABLE[i + 1]
        if e1 <= E <= e2:
            t = (E - e1) / (e2 - e1)
            return m1 + t * (m2 - m1)
    return _PB_MU_OVER_RHO_TABLE[-1][1]


def _buildup_B(mu_cm, x_cm=SHIELDING_THICKNESS_CM):
    """Buildup factor B = 1 + μ*x; μ = linear attenuation [cm⁻¹], x = lead thickness (0.5 cm)."""
    return 1.0 + mu_cm * x_cm


def _hvl_with_buildup(mu_cm, x_cm=SHIELDING_THICKNESS_CM):
    if mu_cm <= 0.0:
        return None
    B = _buildup_B(mu_cm, x_cm)
    # ln(1/(2*B)) = -ln(2*B); HVL = |ln(1/(2*B))/μ| = ln(2*B)/μ (B > 0.5 => ln(2*B) > 0).
    return np.log(2.0 * B) / mu_cm


def _estimate_hvl_from_energy_mev(E_MeV):
    """
    HVL (cm) from photon energy: μ = (μ/ρ)*ρ_Pb from table (linear atten coeff, mean free path 1/μ);
    B = 1 + μ*x with x = 0.5 cm lead thickness; HVL = ln(2*B)/μ. Bi and concrete μ scaled from Pb.
    """
    try:
        E = float(E_MeV)
    except (TypeError, ValueError):
        return dict(FAILSAFE_HVL)

    if E <= 0.0:
        return dict(FAILSAFE_HVL)

    mu_rho = _interp_mu_over_rho_pb(E)
    mu_pb = mu_rho * RHO_PB_G_CM3  # [cm⁻¹]
    if mu_pb <= 0.0:
        return dict(FAILSAFE_HVL)

    hvl_pb = _hvl_with_buildup(mu_pb)
    if hvl_pb is None:
        return dict(FAILSAFE_HVL)
   
    hvl_bi = _BI_PB_SCALE * hvl_pb
    hvl_concrete = _CONCRETE_PB_SCALE * hvl_pb
    hvl_boron = _BORON_PB_SCALE * hvl_pb
    hvl_quartz = _QUARTZ_PB_SCALE * hvl_pb
    return {
        'HVL_Pb_cm': round(hvl_pb, 2),
        'HVL_Bi_cm': round(hvl_bi, 2),
        'HVL_concrete_cm': round(hvl_concrete, 1),
        'HVL_boron_cm': round(hvl_boron, 2),
        'HVL_quartz_cm': round(hvl_quartz, 2),
    }


def get_gamma_hvl(nuclide):
    """
    HVL for a nuclide: if E_MeV > 0 compute from μ and B(μ*x), HVL = ln(2*B)/μ;
    otherwise (stable / no gamma) return FAILSAFE_HVL.
    """
    g = GAMMA_ENERGIES.get(nuclide, {})
    E = g.get('E_MeV', 0.0)
    try:
        E_val = float(E)
    except (TypeError, ValueError):
        E_val = 0.0

    if E_val > 0.0:
        return _estimate_hvl_from_energy_mev(E_val)
    return dict(FAILSAFE_HVL)


GAMMA_ENERGIES = {
    'Zn62': {'E_MeV': 0.6, 'intensity': 0.5},
    'Zn63': {'E_MeV': 0.67, 'intensity': 0.09},
    'Zn64': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
    'Zn65': {'E_MeV': 1.116, 'intensity': 0.506},
    'Zn66': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
    'Zn67': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
    'Zn68': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
    'Zn69': {'E_MeV': 0.44, 'intensity': 0.01},
    'Zn69m': {'E_MeV': 0.439, 'intensity': 1.0},
    'Zn70': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
    'Cu61': {'E_MeV': 0.283, 'intensity': 0.12},
    'Cu62': {'E_MeV': 0.597, 'intensity': 0.02},
    'Cu64': {'E_MeV': 1.35, 'intensity': 0.61},
    'Cu66': {'E_MeV': 1.04, 'intensity': 0.01},
    'Cu67': {'E_MeV': 0.093, 'intensity': 0.39},
    'Cu69': {'E_MeV': 0.90, 'intensity': 0.56},
    'Cu70': {'E_MeV': 0.89, 'intensity': 0.56},
    'Ni61': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
    'Ni63': {'E_MeV': 0.059, 'intensity': 0.1},
    'Ni64': {'E_MeV': 0.0, 'intensity': 0.0},   # 0.36 ms
    'Ni65': {'E_MeV': 1.11, 'intensity': 0.15},
    'Ni67': {'E_MeV': 0.0, 'intensity': 0.0},   # 21 s
    'Pb203': {'E_MeV': 0.279, 'intensity': 0.81},
    'Pb205': {'E_MeV': 0.704, 'intensity': 0.10},
    'Pb208': {'E_MeV': 0.0, 'intensity': 0.0},  # stable
    'Pb209': {'E_MeV': 0.127, 'intensity': 0.004},
    'Bi208': {'E_MeV': 2.614, 'intensity': 0.99},
    'Bi210': {'E_MeV': 0.047, 'intensity': 0.001},
    'Po210': {'E_MeV': 0.803, 'intensity': 0.001},
    'Tl205': {'E_MeV': 0.0, 'intensity': 0.0},   # stable
}


def get_gamma_hvl_details(nuclide):
    """Return dict with E_MeV, mu_Pb_cm, B, HVL_Pb_cm for nuclide (E>0), else None."""
    g = GAMMA_ENERGIES.get(nuclide, {})
    E = g.get('E_MeV', 0.0)
    try:
        E_val = float(E)
    except (TypeError, ValueError):
        return None
    if E_val <= 0.0:
        return None
    mu_rho = _interp_mu_over_rho_pb(E_val)
    mu_pb = mu_rho * RHO_PB_G_CM3
    if mu_pb <= 0.0:
        return None
    B = _buildup_B(mu_pb)
    hvl_pb = _hvl_with_buildup(mu_pb)
    if hvl_pb is None:
        return None
    return {'E_MeV': E_val, 'mu_Pb_cm': mu_pb, 'B': B, 'HVL_Pb_cm': hvl_pb}


_hvl_table_printed = False


def print_hvl_calc_table():
    """Print 'Running HVL calc' and table of isotope, E_MeV, mu_cm, B, HVL_Pb_cm for all isotopes with E>0 (once per run)."""
    global _hvl_table_printed
    if _hvl_table_printed:
        return
    isotopes_with_gamma = [(iso, g) for iso, g in sorted(GAMMA_ENERGIES.items()) if float(g.get('E_MeV') or 0) > 0]
    if not isotopes_with_gamma:
        return
    _hvl_table_printed = True
    print("\n  Running HVL calc")
    print("  " + "-" * 72)
    print(f"  {'Isotope':<8} {'E_MeV':>8} {'mu_cm':>10} {'B':>8} {'HVL_Pb_cm':>10}")
    print("  " + "-" * 72)
    for iso, _ in isotopes_with_gamma:
        d = get_gamma_hvl_details(iso)
        if d is None:
            continue
        print(f"  {iso:<8} {d['E_MeV']:>8.3f} {d['mu_Pb_cm']:>10.4f} {d['B']:>8.3f} {d['HVL_Pb_cm']:>10.2f}")
    print("  " + "-" * 72 + "\n")


CLEARANCE_LEVEL_Bq_per_g = {'Zn65': 0.1}

IRRAD_HOURS_FOR_PLOT = [1, 4, 8, 24, 72, 168, 500, 1000, 2000, 4000, 6000, 8760]
IRRAD_YEARS_20Y = 20
IRRAD_POINTS_20Y = 201


class ZnWasteAnalyzer:
    """Zn-65 waste packaging and disposal: drum fill by limits or volume-only; summary table and costs (no recycling, contingency, labor, equipment)."""

    def __init__(self, output_dir='waste_analysis', output_prefix='irrad_output', irrad_hours=8760, cooldown_days=0):
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.irrad_hours = irrad_hours
        self.cooldown_days = cooldown_days

    def run(self, from_dir=None, case=None, flux_base='.', aggregate_analyze_dir=None,
            zn65_activity_Bq=None, zn65_mass_g=1000,
            cost_no_interim_storage=True, cost_irradiation_years=None, cost_shield_cm=3.0,
            cost_packaging_per_drum=None, cost_transport_per_drum=None, cost_disposal_per_drum=None):
        """Resolve Zn-65 activity/mass from from_dir, case, or direct; then write quote package outputs.
        If aggregate_analyze_dir is set, CSV is read from there (e.g. _by_case/<dir>/zn_summary.csv or zn_summary_all.csv)
        instead of from_dir, so zn_summary_all stays in one place.
        Cost options: cost_no_interim_storage, cost_irradiation_years (e.g. 8), cost_shield_cm (3 or 4),
        cost_packaging_per_drum (e.g. 884), cost_transport_per_drum, cost_disposal_per_drum (e.g. 7125).
        """
        if case:
            parts = case.strip().split()
            if len(parts) < 6:
                raise ValueError("--case requires 'INNER OUTER STRUCT [BORON] MULTI MOD ENRICH%%' (6 or 7 values)")
            inner = float(parts[0])
            outer = float(parts[1])
            struct = float(parts[2])
            if len(parts) == 7:
                boron, multi, mod = float(parts[3]), float(parts[4]), float(parts[5])
                enrich_str = parts[6].rstrip('%')
            else:
                boron, multi, mod = 0, float(parts[3]), float(parts[4])
                enrich_str = parts[5].rstrip('%')
            enrich = float(enrich_str)
            _fmt = lambda v: int(v) if v == int(v) else v
            dir_name = f"{self.output_prefix}_inner{_fmt(inner)}_outer{_fmt(outer)}_struct{_fmt(struct)}_boron{_fmt(boron)}_multi{_fmt(multi)}_moderator{_fmt(mod)}_zn{enrich}%"
            from_dir = os.path.join(os.path.abspath(flux_base), dir_name)
            print(f"Case {case} -> {from_dir} (prefix={self.output_prefix})")
        if from_dir:
            dir_name = os.path.basename(os.path.abspath(from_dir))
            is_dual = ('dual' in dir_name) or ('_inner_zn67' in dir_name)
            csv_file = None
            if aggregate_analyze_dir:
                # Read from single aggregate location: per-case CSV or zn_summary_all filtered by dir_name
                by_case = os.path.join(aggregate_analyze_dir, '_by_case', dir_name, 'zn_summary.csv')
                if os.path.exists(by_case):
                    csv_file = by_case
                else:
                    all_csv = os.path.join(aggregate_analyze_dir, 'zn_summary_all.csv')
                    if os.path.exists(all_csv):
                        csv_file = all_csv
            if csv_file is None:
                csv_file = os.path.join(from_dir, 'zn_summary.csv')
            if not os.path.exists(csv_file):
                csv_file = os.path.join(from_dir, 'zn_summary_all.csv')
            sim_data = None
            # CSV outputs are typically outer-only; for dual, compute from statepoint and sum inner+outer.
            if (not is_dual) and os.path.exists(csv_file):
                print("=" * 70)
                print("READING ZN-65 DATA FROM simple_analyze CSV")
                print("=" * 70)
                print(f"  Reading from: {csv_file}")
                sim_data = read_zn65_from_csv(csv_file, self.irrad_hours, self.cooldown_days, dir_name=dir_name if 'zn_summary_all' in csv_file else None)
            if sim_data is None:
                print("=" * 70)
                print("READING ZN-65 DATA FROM SIMULATION")
                print("=" * 70)
                print(f"  Reading from: {from_dir}")
                sim_data = read_zn65_from_simulation(
                    from_dir,
                    irradiation_hours=self.irrad_hours,
                    cooldown_days=self.cooldown_days,
                    include_inner=is_dual,
                )
            zn65_activity_Bq = sim_data['zn65_activity_Bq']
            zn65_mass_g = sim_data['zn65_mass_g']
            print("=" * 70)
        elif zn65_activity_Bq is None:
            raise ValueError("Provide from_dir, case, or zn65_activity_Bq")
        os.makedirs(self.output_dir, exist_ok=True)
        cost_kwargs = {
            'no_interim_storage': cost_no_interim_storage,
            'scenario_irradiation_years': cost_irradiation_years,
            'shield_cm_for_drums': cost_shield_cm,
            'packaging_cost_per_drum_override': cost_packaging_per_drum,
            'transport_cost_per_drum_override': cost_transport_per_drum,
            'disposal_cost_per_drum_override': cost_disposal_per_drum,
        }
        create_comparison_plots(
            zn65_activity_Bq=zn65_activity_Bq,
            zn65_mass_g=zn65_mass_g,
            output_dir=self.output_dir,
            sim_dir=from_dir if from_dir else None,
            cooldown_days=self.cooldown_days,
            cost_kwargs=cost_kwargs,
        )


def _get_zn65_vs_irrad(sim_dir, cooldown_days=0):
    """Get [(irrad_h, zn65_Bq), ...] using simple_analyze if available."""
    try:
        import simple_analyze
        dir_name = os.path.basename(os.path.abspath(sim_dir))
        is_dual = ('dual' in dir_name) or ('_inner_zn67' in dir_name)
        sp_glob = __import__('glob').glob(os.path.join(sim_dir, 'statepoint.*.h5'))
        if not sp_glob:
            return None
        sp_file = sorted(sp_glob)[-1]
        if not is_dual:
            case = simple_analyze.analyze_case(sp_file, outer_material_id=1)
            return [(h, simple_analyze.compute_activities(case, h, cooldown_days)['zn65_Bq']) for h in IRRAD_HOURS_FOR_PLOT]
        out = []
        for h in IRRAD_HOURS_FOR_PLOT:
            bq = 0.0
            for mid in (1, 0):
                case = simple_analyze.analyze_case(sp_file, outer_material_id=mid)
                bq += float(simple_analyze.compute_activities(case, h, cooldown_days)['zn65_Bq'])
            out.append((h, bq))
        return out
    except Exception:
        return None


def read_zn65_from_csv(csv_file, irradiation_hours=8760, cooldown_days=0, dir_name=None):
    """
    Read Zn-65 data from simple_analyze output CSV (zn_summary.csv or zn_summary_all.csv).
    If dir_name is provided and the CSV has a 'dir_name' column (e.g. zn_summary_all), filter to that case.
    """
    
    if not os.path.exists(csv_file):
        return None
    
    df = pd.read_csv(csv_file)
    if dir_name is not None and 'dir_name' in df.columns:
        df = df[df['dir_name'] == dir_name].copy()
        if df.empty:
            return None
    
    # Filter to matching irradiation and cooldown times
    matching = df[
        (df['irrad_hours'] == irradiation_hours) & 
        (df['cooldown_days'] == cooldown_days)
    ]
    
    if matching.empty:
        # Use closest irradiation time
        closest_idx = (df['irrad_hours'] - irradiation_hours).abs().idxmin()
        matching = df.iloc[[closest_idx]]
        print(f"  Warning: Using closest irradiation time: {matching.iloc[0]['irrad_hours']} hours")
    
    row = matching.iloc[0]
    
    return {
        'zn65_activity_Bq': row['zn65_Bq'],
        'zn65_mass_g': row['zn_mass_g'],
        'zn69m_activity_Bq': row.get('zn69m_Bq', 0.0) if 'zn69m_Bq' in row else 0.0,
        'zn65_production_rate': 0,  # Not in CSV
        'irradiation_hours': row['irrad_hours'],
        'cooldown_days': row['cooldown_days'],
        'params': {
            'zn_enrichment': row.get('zn64_enrichment', 0.486),
            'multi': row.get('multi_cm', 0),
            'moderator': row.get('mod_cm', 0),
        },
        'volume_cm3': row.get('zn_volume_cm3', 0),
    }


def read_zn65_from_simulation(sim_dir, irradiation_hours=8760, cooldown_days=0, include_inner=False):
    """Read Zn-65 activity and Zn mass from simulation output (statepoint + Bateman).

    If include_inner=True and the case has an inner Zn chamber (material_id=0), this sums:
    - Zn-65 activity (Bq) from inner + outer
    - total Zn mass (g) from inner + outer
    """
    import glob
    
    # Find statepoint file
    sp_patterns = [
        os.path.join(sim_dir, 'statepoint.*.h5'),
        os.path.join(sim_dir, 'statepoint.h5'),
    ]
    
    sp_file = None
    for pattern in sp_patterns:
        matches = glob.glob(pattern)
        if matches:
            sp_file = sorted(matches)[-1]  # Latest statepoint
            break
    
    if sp_file is None:
        raise FileNotFoundError(f"No statepoint file found in {sim_dir}")
    
    print(f"Reading from: {sp_file}")
    
    # Import OpenMC and utilities
    import openmc
    from utilities import (
        build_channel_rr_per_s,
        compute_volumes_from_dir_name,
        get_material_density_from_statepoint,
        calculate_enriched_zn_density,
        parse_dir_name,
        SOURCE_STRENGTH,
    )
    
    dir_name = os.path.basename(sim_dir)
    params = parse_dir_name(dir_name)
    try:
        import run_config as _c
        _target_height = getattr(_c, 'TARGET_HEIGHT_CM', 100.0)
    except ImportError:
        _target_height = 100.0

    # Get volumes
    volumes = compute_volumes_from_dir_name(dir_name, target_height=_target_height)
    outer_volume_cm3 = volumes.get(1, 0)  # Outer target material ID = 1
    inner_volume_cm3 = volumes.get(0, 0)  # Inner target material ID = 0 (dual only)

    # Determine which chambers to include
    use_inner = bool(include_inner) and (inner_volume_cm3 > 0)
    mat_ids = [1] + ([0] if use_inner else [])

    # Mass = volume × density (pull density from statepoint per material_id when possible)
    total_mass_g = 0.0
    for mid in mat_ids:
        vol_cm3 = outer_volume_cm3 if mid == 1 else inner_volume_cm3
        if vol_cm3 <= 0:
            continue
        dens = get_material_density_from_statepoint(sp_file, material_id=mid)
        if dens is None:
            dens = calculate_enriched_zn_density(params.get('zn_enrichment', 0.486))
        total_mass_g += vol_cm3 * dens
    
    # Use simple_analyze to compute Zn-65 activity (avoids duplicating Bateman equations)
    # simple_analyze already handles multi-step Bateman with reaction-rate reduction
    try:
        import simple_analyze
        zn65_activity_Bq = 0.0
        zn69m_activity_Bq = 0.0
        zn65_atoms = 0.0
        for mid in mat_ids:
            case = simple_analyze.analyze_case(sp_file, outer_material_id=mid)
            activities = simple_analyze.compute_activities(case, irradiation_hours, cooldown_days)
            zn65_activity_Bq += float(activities['zn65_Bq'])
            zn69m_activity_Bq += float(activities.get('zn69m_Bq', 0.0))
            zn65_atoms += float(activities.get('zn65_atoms', 0.0))

        print("  Using simple_analyze.compute_activities (multi-step Bateman)")
        print(f"  Irradiation: {irradiation_hours} hours ({irradiation_hours/8760:.2f} years)")
        print(f"  Cooldown: {cooldown_days} days")
        print(f"  Chambers included: {'inner+outer' if use_inner else 'outer'}")
        print(f"  Zn-65 atoms (sum): {zn65_atoms:.3e}")
        print(f"  Zn-65 activity (sum): {zn65_activity_Bq:.3e} Bq ({zn65_activity_Bq/1e9:.3f} GBq)")
        if zn69m_activity_Bq > 0:
            print(f"  Zn-69m activity (sum): {zn69m_activity_Bq:.3e} Bq")
        print(f"  Zn mass (sum): {total_mass_g:.1f} g ({total_mass_g/1000:.2f} kg)")
        
    except (ImportError, Exception) as e:
        print(f"  Warning: Could not use simple_analyze ({e}), falling back to direct calculation")
        # Fallback: direct calculation (simplified, doesn't handle reaction-rate reduction)
        sp = openmc.StatePoint(sp_file)
        zn65_activity_Bq = 0.0
        zn65_production_rate = 0.0
        lam_zn65 = DECAY_CONSTANTS['Zn65']
        irrad_time_s = irradiation_hours * 3600
        cooldown_s = cooldown_days * 86400
        for mid in mat_ids:
            channel_rr = build_channel_rr_per_s(sp, cell_id=mid, source_strength=SOURCE_STRENGTH)
            pr = (channel_rr.get("Zn64 (n,gamma) Zn65", 0) or 0) + (channel_rr.get("Zn66 (n,2n) Zn65", 0) or 0)
            pr = float(np.asarray(pr).flat[0]) if hasattr(pr, "__array__") else float(pr)
            zn65_production_rate += pr
            if lam_zn65 > 0:
                zn65_atoms_eoi = (pr / lam_zn65) * (1 - np.exp(-lam_zn65 * irrad_time_s))
            else:
                zn65_atoms_eoi = pr * irrad_time_s
            zn65_activity_eoi_Bq = zn65_atoms_eoi * lam_zn65
            zn65_activity_Bq += zn65_activity_eoi_Bq * np.exp(-lam_zn65 * cooldown_s)

        print("  Fallback: Direct Bateman calculation (simplified)")
        print(f"  Chambers included: {'inner+outer' if use_inner else 'outer'}")
        print(f"  Zn-65 activity (sum): {zn65_activity_Bq:.3e} Bq ({zn65_activity_Bq/1e9:.3f} GBq)")
    
    return {
        'zn65_activity_Bq': zn65_activity_Bq,
        'zn65_mass_g': total_mass_g,
        'zn69m_activity_Bq': zn69m_activity_Bq if 'zn69m_activity_Bq' in locals() else 0.0,
        'zn65_production_rate': zn65_production_rate if 'zn65_production_rate' in locals() else 0,
        'irradiation_hours': irradiation_hours,
        'cooldown_days': cooldown_days,
        'params': params,
        'volume_cm3': outer_volume_cm3 + (inner_volume_cm3 if use_inner else 0.0),
    }


def calculate_activity_Bq(atoms, isotope):
    """Calculate activity in Bq from atom count."""
    lam = DECAY_CONSTANTS.get(isotope, 0)
    return atoms * lam


def calculate_activity_decay(initial_activity_Bq, isotope, time_days):
    """Calculate activity after decay time."""
    lam = DECAY_CONSTANTS.get(isotope, 0)
    time_s = time_days * 86400
    return initial_activity_Bq * np.exp(-lam * time_s)


ATLANTIC_MINIMUM_SHIPMENT = 1000  # USD
ATLANTIC_DOSE_MULTIPLIERS = [(0, 0.5, 1.0), (0.5, 2, 1.08), (2, 10, 1.25), (10, 200, 1.5)]  # (lo, hi mR/hr, mult)


def atlantic_compact_disposal_cost(total_weight_lbs, density_lbs_ft3, dose_rate_mR_hr=0,
                                   educational_research=False):
    """
    Atlantic Compact disposal cost (weight charges only, simple analysis).
    MINIMUM $1,000 per shipment. Price by weight; no upper weight limit from tables.
    
    Parameters
    ----------
    total_weight_lbs : float
        Total weight of waste (lbs)
    density_lbs_ft3 : float
        Package density (lbs/ft^3)
    dose_rate_mR_hr : float
        Dose rate at container surface (mR/hr) for multiplier
    educational_research : bool
        If True, rate block one lower (educational research institution)
    
    Returns
    -------
    float : Disposal cost in USD
    """
    if total_weight_lbs <= 0:
        return ATLANTIC_MINIMUM_SHIPMENT
    # Base weight rate by density
    if density_lbs_ft3 >= 120:
        rate_per_lb = 9.917
    elif density_lbs_ft3 >= 75:
        rate_per_lb = 10.910
    elif density_lbs_ft3 >= 60:
        rate_per_lb = 13.387
    elif density_lbs_ft3 >= 45:
        rate_per_lb = 17.356
    else:
        rate_per_lb = 17.356 * (45.0 / max(density_lbs_ft3, 0.1))
    # Dose multiplier (container surface dose in mR/hr)
    mult = 1.0
    for lo, hi, m in ATLANTIC_DOSE_MULTIPLIERS:
        if lo <= dose_rate_mR_hr <= hi:
            mult = m
            break
    if educational_research and mult > 1.0:
        mult = max(1.0, mult - 0.08)  # One block lower (approximate)
    base_cost = total_weight_lbs * rate_per_lb * mult
    return max(ATLANTIC_MINIMUM_SHIPMENT, base_cost)


def time_to_clearance(initial_activity_Bq, mass_g, isotope):
    """Calculate days to reach clearance level (Zn-65: 0.1 Bq/g; Lu-177m: 100 Bq/g, not in IAEA)."""
    clearance = CLEARANCE_LEVEL_Bq_per_g.get(isotope, 0.1)
    target_activity = clearance * mass_g
    
    if initial_activity_Bq <= target_activity:
        return 0
    
    lam = DECAY_CONSTANTS.get(isotope, 0)
    if lam <= 0:
        return np.inf
    
    # A(t) = A0 * exp(-λt) = target
    # t = -ln(target/A0) / λ
    t_s = -np.log(target_activity / initial_activity_Bq) / lam
    return t_s / 86400  # days


def calculate_max_activity_per_drum(isotope='Zn65', max_shielding_cm=3.0, 
                                     target_dose_rate_uSv_hr=None, drum_radius_m=0.3):
    """
    Calculate maximum activity per drum that meets storage/transport limit.
    
    Limit: 200 mrem/hr (2000 µSv/hr) at container surface (10 CFR 71.47, 34.21; restricted storage/transport).
    Dose rate at surface accounts for:
    - Distance from source to surface (inverse square law)
    - Shielding attenuation
    
    Parameters:
    -----------
    isotope : str
        Isotope name
    max_shielding_cm : float
        Maximum practical Pb shielding thickness in drum (cm)
    target_dose_rate_uSv_hr : float
        Storage/transport limit (200 mrem/hr at surface, 10 CFR 71.47)
    drum_radius_m : float
        Distance from source center to drum surface (m), default 0.3m for 55-gal drum
    
    Returns:
    --------
    float : Maximum activity in GBq per drum
    """
    if target_dose_rate_uSv_hr is None:
        target_dose_rate_uSv_hr = DOSE_LIMIT_STORAGE_UVSV_HR
    dose_coeff = get_dose_coeff(isotope) or 0.0
    if dose_coeff <= 0:
        return 0.0
    hvl = get_gamma_hvl(isotope)['HVL_Pb_cm']

    # Distance factor: surface at drum_radius vs 1m = (1/drum_radius)^2
    # For 55-gal drum: radius ~0.3m, so (1/0.3)^2 = 11.1×
    distance_factor = (1.0 / drum_radius_m)**2
    
    # Shielding factor: 2^(thickness/HVL)
    n_hvl = max_shielding_cm / hvl
    shielding_factor = 2**n_hvl
    
    # Shielded dose at surface = (dose_coeff × activity_MBq × distance_factor) / shielding_factor
    # Set equal to target: (dose_coeff × activity_MBq × distance_factor) / shielding_factor = target
    # Solve for activity: activity_MBq = (target × shielding_factor) / (dose_coeff × distance_factor)
    
    max_activity_MBq = (target_dose_rate_uSv_hr * shielding_factor) / (dose_coeff * distance_factor)
    max_activity_GBq = max_activity_MBq / 1000
    
    return max_activity_GBq


def calculate_max_activity_per_drum_concrete(isotope='Zn65', concrete_cm=15.0,
                                            target_dose_rate_uSv_hr=None, drum_radius_m=0.3):
    """Max activity per drum (GBq) with concrete liner; same logic as Pb, using concrete HVL. Uses storage/transport limit (200 mrem/hr at surface)."""
    if target_dose_rate_uSv_hr is None:
        target_dose_rate_uSv_hr = DOSE_LIMIT_STORAGE_UVSV_HR
    dose_coeff = get_dose_coeff(isotope) or 0.0
    if dose_coeff <= 0:
        return 0.0
    hvl = get_gamma_hvl(isotope)['HVL_concrete_cm']
    distance_factor = (1.0 / drum_radius_m)**2
    n_hvl = concrete_cm / hvl
    shielding_factor = 2**n_hvl
    max_activity_MBq = (target_dose_rate_uSv_hr * shielding_factor) / (dose_coeff * distance_factor)
    return max_activity_MBq / 1000


def max_activity_MBq_at_3m_below_limit(isotope='Zn65', limit_uSv_hr=None):
    """Max activity (MBq) so that unshielded dose at 3 m does not exceed limit (default 10 mSv/h = 10,000 µSv/h)."""
    if limit_uSv_hr is None:
        limit_uSv_hr = DOSE_LIMIT_UNSHIELDED_3M_UVSV_HR
    coeff = get_dose_coeff(isotope) or 0.0
    return (limit_uSv_hr * 9.0) / coeff if coeff > 0 else 0.0


# 55-gal drum: total volume ~208 L = 208,000 cm^3. Usable volume after liner (3–5 cm Pb or concrete).
def _drum_usable_volume_cm3(shield_cm, is_concrete=False):
    """Usable volume in cm^3 after shielding liner."""
    if is_concrete:
        inner_radius_cm = max(15.0, 30.0 - shield_cm)  # 15 cm concrete
    else:
        inner_radius_cm = max(20.0, 30.0 - shield_cm)  # Pb: 3-5 cm
    height_cm = 84.0
    return np.pi * (inner_radius_cm ** 2) * height_cm

ZN_DENSITY_G_CM3 = 7.14

# 55-gal drum full volume (no shielding liner) for volume-only fill
DRUM_VOLUME_CM3 = 208_000  # 208 L

# 10 CFR 61.55 Table 2: Class A limit for total of all nuclides with T½ < 5 years
CLASS_A_CI_PER_M3_LIMIT = 700  # Ci/m^3; Zn-65 (t½=244 d) qualifies

# EnergySolutions reference: 100 tons Class A waste (TABLE 1-2: 200,000 lbs) for cost/drum comparisons
ES_CLASS_A_TONS = 100
ES_CLASS_A_LBS = 200_000  # 200,000 lbs = 100 tons


def drums_by_volume_only(zn_total_mass_g, drum_volume_cm3=None):
    """
    Number of drums needed to hold total Zn by volume only (no activity limits).
    Fills drums purely by volume: ceil(total_zn_volume / drum_volume).
    """
    if drum_volume_cm3 is None:
        drum_volume_cm3 = DRUM_VOLUME_CM3
    if zn_total_mass_g <= 0 or drum_volume_cm3 <= 0:
        return 0, 0.0
    zn_volume_cm3 = zn_total_mass_g / ZN_DENSITY_G_CM3
    n_drums = max(1, int(np.ceil(zn_volume_cm3 / drum_volume_cm3)))
    return n_drums, zn_volume_cm3


def drum_storage_summary(zn_total_mass_g, zn65_activity_Bq, concrete_cm=15.0):
    """
    Waste storage: how many drums (volume-only first, then 3 cm Pb, 4 cm Pb, 5 cm Pb, concrete) and how full each is with Zn.
    First row: volume only (no activity limits)—drums = ceil(total Zn volume / drum volume).
    Other rows: drums filled to activity limit (200 mrem/hr at surface).
    Returns DataFrame: Option, Drums needed, Zn kg per drum, Fraction full, Ci/m^3 (ours), Limit Ci/m^3, Comparison.
    """
    zn65_GBq = zn65_activity_Bq / 1e9
    zn65_Ci = zn65_activity_Bq / 3.7e10  # 1 Ci = 3.7e10 Bq
    drum_vol_m3 = 0.208  # 55 gal = 208 L
    rows = []

    # Volume only (no activity limits): just fill drums by total Zn volume
    if zn_total_mass_g > 0:
        drums_vol, zn_vol_cm3 = drums_by_volume_only(zn_total_mass_g)
        zn_kg_per_drum_vol = (zn_total_mass_g / 1000.0) / drums_vol if drums_vol > 0 else 0
        zn_vol_per_drum_cm3 = zn_vol_cm3 / drums_vol if drums_vol > 0 else 0
        frac_full_vol = zn_vol_per_drum_cm3 / DRUM_VOLUME_CM3 if DRUM_VOLUME_CM3 > 0 else 0
        total_vol_m3_vol = drums_vol * drum_vol_m3
        our_ci_m3_vol = zn65_Ci / total_vol_m3_vol if total_vol_m3_vol > 0 else 0
        pct_vol = (our_ci_m3_vol / CLASS_A_CI_PER_M3_LIMIT * 100) if CLASS_A_CI_PER_M3_LIMIT > 0 else 0
        comp_vol = f'{pct_vol:.1f}% of limit' if our_ci_m3_vol <= CLASS_A_CI_PER_M3_LIMIT else f'EXCEEDS limit ({pct_vol:.1f}%)'
        rows.append({
            'Option': 'Volume only (no activity limit)', 'Drums needed': drums_vol, 'Zn kg per drum': zn_kg_per_drum_vol,
            'Fraction full': frac_full_vol,
            'Ci/m^3 (ours)': our_ci_m3_vol, 'Limit Ci/m^3 (Class A)': CLASS_A_CI_PER_M3_LIMIT,
            'Comparison': comp_vol
        })

    if zn65_GBq <= 0 or zn_total_mass_g <= 0:
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Option', 'Drums needed', 'Zn kg per drum', 'Fraction full',
                                     'Ci/m^3 (ours)', 'Limit Ci/m^3 (Class A)', 'Comparison'])

    # Calculate specific activity: Bq/g
    specific_activity_Bq_per_g = zn65_activity_Bq / zn_total_mass_g if zn_total_mass_g > 0 else 0

    max_3pb = calculate_max_activity_per_drum('Zn65', 3.0, DOSE_LIMIT_STORAGE_UVSV_HR, 0.3)
    max_4pb = calculate_max_activity_per_drum('Zn65', 4.0, DOSE_LIMIT_STORAGE_UVSV_HR, 0.3)
    max_5pb = calculate_max_activity_per_drum('Zn65', 5.0, DOSE_LIMIT_STORAGE_UVSV_HR, 0.3)
    max_conc = calculate_max_activity_per_drum_concrete('Zn65', concrete_cm, DOSE_LIMIT_STORAGE_UVSV_HR, 0.3)

    rows = []
    for label, max_GBq, shield_cm, is_conc in [('3 cm Pb', max_3pb, 3.0, False), 
                                                ('4 cm Pb', max_4pb, 4.0, False),
                                                ('5 cm Pb', max_5pb, 5.0, False), 
                                                (f'{concrete_cm:.0f} cm concrete', max_conc, concrete_cm, True)]:
        # Calculate max Zn mass per drum based on activity limit
        if specific_activity_Bq_per_g > 0 and max_GBq > 0:
            max_zn_mass_per_drum_g = (max_GBq * 1e9) / specific_activity_Bq_per_g
            max_zn_kg_per_drum = max_zn_mass_per_drum_g / 1000.0
        else:
            max_zn_kg_per_drum = zn_total_mass_g / 1000.0
        
        # Drums needed: by activity OR by mass (whichever requires more drums)
        drums_by_activity = int(np.ceil(zn65_GBq / max_GBq)) if max_GBq > 0 else 1
        drums_by_mass = int(np.ceil((zn_total_mass_g / 1000.0) / max_zn_kg_per_drum)) if max_zn_kg_per_drum > 0 else 1
        drums = max(1, max(drums_by_activity, drums_by_mass))
        
        # Fill drums: each drum gets max_zn_kg_per_drum (up to total available)
        # Average Zn per drum (last drum may be partially filled)
        zn_kg_per_drum = min(max_zn_kg_per_drum, (zn_total_mass_g / 1000.0) / drums)
        
        zn_vol_per_drum_cm3 = (zn_kg_per_drum * 1000) / ZN_DENSITY_G_CM3
        usable_vol = _drum_usable_volume_cm3(shield_cm, is_conc)
        frac_full = zn_vol_per_drum_cm3 / usable_vol if usable_vol > 0 else 0
        total_vol_m3 = drums * drum_vol_m3
        our_ci_m3 = zn65_Ci / total_vol_m3 if total_vol_m3 > 0 else 0
        pct = (our_ci_m3 / CLASS_A_CI_PER_M3_LIMIT * 100) if CLASS_A_CI_PER_M3_LIMIT > 0 else 0
        comp = f'{pct:.1f}% of limit' if our_ci_m3 <= CLASS_A_CI_PER_M3_LIMIT else f'EXCEEDS limit ({pct:.1f}%)'
        rows.append({
            'Option': label, 'Drums needed': drums, 'Zn kg per drum': zn_kg_per_drum,
            'Fraction full': frac_full,
            'Ci/m^3 (ours)': our_ci_m3, 'Limit Ci/m^3 (Class A)': CLASS_A_CI_PER_M3_LIMIT,
            'Comparison': comp
        })
    return pd.DataFrame(rows)


def _ci_m3_vs_irrad(zn65_vs_irrad, zn65_mass_g, shield_cm=3.0):
    """
    For each (irrad_hours, activity_Bq) compute drums and Ci/m^3 (Class A metric).
    Uses drums sized by storage/transport limit (200 mrem/hr at surface). Returns (hours_arr, years_arr, ci_m3_arr).
    """
    if not zn65_vs_irrad or zn65_mass_g <= 0:
        return np.array([]), np.array([]), np.array([])
    hours = np.array([p[0] for p in zn65_vs_irrad], dtype=float)
    activities_Bq = np.array([p[1] for p in zn65_vs_irrad], dtype=float)
    max_GBq = calculate_max_activity_per_drum('Zn65', float(shield_cm), DOSE_LIMIT_STORAGE_UVSV_HR, 0.3)
    drum_vol_m3 = 0.208  # 55 gal
    ci_m3_list = []
    for a_Bq in activities_Bq:
        a_GBq = a_Bq / 1e9
        a_Ci = a_Bq / 3.7e10
        drums = max(1, int(np.ceil(a_GBq / max_GBq)) if max_GBq > 0 else 1)
        total_vol_m3 = drums * drum_vol_m3
        ci_m3 = a_Ci / total_vol_m3 if total_vol_m3 > 0 else 0
        ci_m3_list.append(ci_m3)
    years = hours / 8760.0
    return hours, years, np.array(ci_m3_list)


def _ci_m3_vs_irrad_volume_only(zn65_vs_irrad, zn65_mass_g, shield_cm=3.0):
    """
    Ci/m^3 when all Zn is packed into full drums by volume only (no activity limit).
    Drums = ceil(Zn volume / usable volume per drum). Same drum count at all irradiation times.
    Returns (hours_arr, years_arr, ci_m3_arr).
    """
    if not zn65_vs_irrad or zn65_mass_g <= 0:
        return np.array([]), np.array([]), np.array([])
    zn_vol_cm3 = zn65_mass_g / ZN_DENSITY_G_CM3
    usable_cm3 = _drum_usable_volume_cm3(shield_cm, False)
    drums = max(1, int(np.ceil(zn_vol_cm3 / usable_cm3)))
    drum_vol_m3 = 0.208
    total_vol_m3 = drums * drum_vol_m3
    hours = np.array([p[0] for p in zn65_vs_irrad], dtype=float)
    activities_Bq = np.array([p[1] for p in zn65_vs_irrad], dtype=float)
    ci_m3 = (activities_Bq / 3.7e10) / total_vol_m3 if total_vol_m3 > 0 else np.zeros_like(hours)
    years = hours / 8760.0
    return hours, years, np.asarray(ci_m3)


def _irrad_hours_before_exceeding_class_a(zn65_vs_irrad, zn65_mass_g, shield_cm, limit_ci_m3=700, volume_only=False):
    """Return irrad hours at which Ci/m^3 first exceeds limit, or None if never in range."""
    if not zn65_vs_irrad or zn65_mass_g <= 0:
        return None
    if volume_only:
        hours, years, ci_m3 = _ci_m3_vs_irrad_volume_only(zn65_vs_irrad, zn65_mass_g, shield_cm)
    else:
        hours, years, ci_m3 = _ci_m3_vs_irrad(zn65_vs_irrad, zn65_mass_g, shield_cm)
    if len(ci_m3) == 0:
        return None
    over = np.where(ci_m3 > limit_ci_m3)[0]
    if len(over) == 0:
        return None  # never exceeds in sampled range
    # First index that exceeds
    i = int(over[0])
    if i == 0:
        return hours[0]  # already over at first point
    # Linear interpolation between hours[i-1] and hours[i]
    h0, h1 = hours[i - 1], hours[i]
    c0, c1 = ci_m3[i - 1], ci_m3[i]
    if c1 <= c0:
        return h1
    frac = (limit_ci_m3 - c0) / (c1 - c0)
    return h0 + frac * (h1 - h0)


def plot_class_a_700_vs_irrad(zn65_vs_irrad, zn65_mass_g, output_dir='.', cooldown_days=0):
    """
    Plot Ci/m^3 vs irradiation time when all Zn is packed into full drums (volume-only).
    Answers: After x years irrad, what is the activity concentration in those full drums?
    How long until that concentration exceeds 700 Ci/m^3 (Class A)?
    """
    if not zn65_vs_irrad or len(zn65_vs_irrad) < 2 or zn65_mass_g <= 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for shield_cm, label, color in [(3.0, '3 cm Pb', 'C0'), (5.0, '5 cm Pb', 'C1')]:
        hours, years, ci_m3 = _ci_m3_vs_irrad_volume_only(zn65_vs_irrad, zn65_mass_g, shield_cm)
        if len(years) == 0:
            continue
        ax.plot(years, ci_m3, 'o-', color=color, lw=2, ms=6, label=label)
        t_cross = _irrad_hours_before_exceeding_class_a(
            zn65_vs_irrad, zn65_mass_g, shield_cm, CLASS_A_CI_PER_M3_LIMIT, volume_only=True)
        if t_cross is not None:
            y_cross = CLASS_A_CI_PER_M3_LIMIT
            ax.axvline(x=t_cross / 8760.0, color=color, ls='--', alpha=0.6)
            ax.annotate(f'Max ~{t_cross/8760:.1f} yr\n(stay Class A)', xy=(t_cross / 8760.0, y_cross),
                       xytext=(10, 15), textcoords='offset points', fontsize=9, color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=1))
    ax.axhline(y=CLASS_A_CI_PER_M3_LIMIT, color='k', ls='-', lw=2, label=f'Class A limit ({CLASS_A_CI_PER_M3_LIMIT} Ci/m^3)')
    ax.set_xlabel('Irradiation time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ci/m^3 (activity ÷ drum volume)', fontsize=12, fontweight='bold')
    ax.set_title('Full drums (volume-only): pack all Zn into drums. What is Ci/m^3 after x years?\n'
                 'How long until you exceed 700 Ci/m^3? Ci/m^3 = activity ÷ (drums × 0.208 m^3)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(years) * 1.05 if len(years) else 10)
    _, _, c3 = _ci_m3_vs_irrad_volume_only(zn65_vs_irrad, zn65_mass_g, 3.0)
    _, _, c5 = _ci_m3_vs_irrad_volume_only(zn65_vs_irrad, zn65_mass_g, 5.0)
    y_max = max(np.max(c3) if len(c3) > 0 else 0, np.max(c5) if len(c5) > 0 else 0)
    ax.set_ylim(0, max(700 * 1.15, y_max * 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot_class_a_700_vs_irrad.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/plot_class_a_700_vs_irrad.png")


def _zn65_activity_vs_irrad_extended_to_20y(zn65_vs_irrad, lam_zn65_per_s=None):
    """
    Extend Zn-65 activity (Bq) vs irradiation time to 20 years using saturation: A(t) = A_sat * (1 - exp(-lam*t)).
    Uses the point with largest irrad time to infer A_sat. Returns (hours_arr, years_arr, activity_Bq_arr).
    """
    if not zn65_vs_irrad or len(zn65_vs_irrad) < 1:
        return np.array([]), np.array([]), np.array([])
    lam = lam_zn65_per_s if lam_zn65_per_s is not None else DECAY_CONSTANTS['Zn65']
    # Use point with largest irrad time > 0 to infer saturation activity
    points_with_t = [(p[0], p[1]) for p in zn65_vs_irrad if p[0] > 0]
    if not points_with_t:
        return np.array([]), np.array([]), np.array([])
    h_max, a_max = max(points_with_t, key=lambda x: x[0])
    t_max_s = h_max * 3600.0
    if lam <= 0:
        return np.array([]), np.array([]), np.array([])
    one_minus_exp = 1.0 - np.exp(-lam * t_max_s)
    if one_minus_exp <= 0:
        a_sat = a_max
    else:
        a_sat = a_max / one_minus_exp
    hours = np.linspace(0, IRRAD_YEARS_20Y * 8760.0, IRRAD_POINTS_20Y)
    t_s = hours * 3600.0
    activity_Bq = a_sat * (1.0 - np.exp(-lam * t_s))
    years = hours / 8760.0
    return hours, years, activity_Bq


def plot_ci_m3_vs_irrad_20y(zn65_vs_irrad, zn65_mass_g, output_dir='.', cooldown_days=0):
    """
    Plot Ci/m³ (Zn-65 activity per drum volume) vs irradiation time over 0–20 years.
    Uses volume-only drum packing; activity extended to 20 y via saturation curve.
    """
    if not zn65_vs_irrad or len(zn65_vs_irrad) < 1 or zn65_mass_g <= 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    lam_zn65 = DECAY_CONSTANTS['Zn65']
    hours, years, activity_Bq = _zn65_activity_vs_irrad_extended_to_20y(zn65_vs_irrad, lam_zn65)
    if len(years) == 0:
        return
    # Volume-only: same drum count at all times (same as drums_by_volume_only)
    drums, _ = drums_by_volume_only(zn65_mass_g)
    drum_vol_m3 = 0.208
    total_vol_m3 = drums * drum_vol_m3
    activity_Ci = activity_Bq / 3.7e10
    ci_m3 = activity_Ci / total_vol_m3 if total_vol_m3 > 0 else np.zeros_like(activity_Bq)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, ci_m3, 'b-', lw=2, label=f'Ci/m³ (Zn-65 activity ÷ {drums} drums × 0.208 m³)')
    ax.axhline(y=CLASS_A_CI_PER_M3_LIMIT, color='k', ls='--', lw=1.5, label=f'Class A limit ({CLASS_A_CI_PER_M3_LIMIT} Ci/m³)')
    ax.set_xlabel('Irradiation time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ci/m³', fontsize=12, fontweight='bold')
    ax.set_title('Zn-65 activity concentration vs irradiation time (0–20 years)\n'
                 'Ci/m³ = Zn-65 activity per drum volume (volume-only packing)', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, IRRAD_YEARS_20Y)
    y_max = np.max(ci_m3) if len(ci_m3) > 0 else CLASS_A_CI_PER_M3_LIMIT
    ax.set_ylim(0, max(CLASS_A_CI_PER_M3_LIMIT * 1.15, y_max * 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'plot_ci_m3_vs_irrad_20y.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir}/plot_ci_m3_vs_irrad_20y.png")


def shielding_thickness(activity_MBq, isotope, target_dose_rate_uSv_hr=None, material='Pb'):
    """
    Calculate shielding thickness to reduce dose rate to target.
    
    Dose rate limits:
    - Occupied areas at 1 m: often 25 µSv/hr (2.5 mrem/hr); not used for drum surface here.
    - Drum surface (storage/transport): 2000 µSv/hr (200 mrem/hr), 10 CFR 71.47, 49 CFR 173.441.
    - Facility boundary: 1000 µSv/hr (100 mrem/hr)
    
    Parameters:
    -----------
    activity_MBq : float
        Source activity in MBq
    isotope : str
        Isotope name
    target_dose_rate_uSv_hr : float
        Target dose rate (default 200 mrem/hr at drum surface for storage/transport)
    material : str
        'Pb' (lead) or 'concrete'
    
    Returns:
    --------
    float : Required thickness in cm
    """
    if target_dose_rate_uSv_hr is None:
        target_dose_rate_uSv_hr = DOSE_LIMIT_STORAGE_UVSV_HR
    c = get_dose_coeff(isotope) or 0.0
    unshielded_rate = c * activity_MBq
    
    if unshielded_rate <= target_dose_rate_uSv_hr:
        return 0
    
    ghvl = get_gamma_hvl(isotope)
    if material == 'Pb':
        hvl = ghvl['HVL_Pb_cm']
    else:
        hvl = ghvl['HVL_concrete_cm']
    
    # Number of HVLs needed
    ratio = unshielded_rate / target_dose_rate_uSv_hr
    n_hvl = np.log2(ratio)
    
    return n_hvl * hvl


def create_comparison_plots(zn65_activity_Bq, zn65_mass_g, output_dir='.',
                           zn65_vs_irrad=None, cooldown_days=0, sim_dir=None, cost_kwargs=None):
    """
    Create the Zn-65 waste quote package outputs:
    - Plot 1: Zn-65 activity vs irradiation time (with Lu-177m reference line)
    - Plot: volume-only full drums Ci/m^3 vs irrad time (how long until >700 Ci/m^3)
    - Plot 2: Long-lived decay comparison (Zn-65 vs Lu-177m reference)
    - EnergySolutions quote input CSV
    - Cost breakdown table CSV
    - Waste comparison summary CSV

    cost_kwargs : dict, optional
        Passed to create_cost_breakdown_table (e.g. no_interim_storage=True,
        scenario_irradiation_years=8, disposal_cost_per_drum_override=7125,
        packaging_cost_per_drum_override=884, shield_cm_for_drums=4).
    """
    if cost_kwargs is None:
        cost_kwargs = {}
    os.makedirs(output_dir, exist_ok=True)

    zn65_GBq = zn65_activity_Bq / 1e9
    zn65_mCi = zn65_activity_Bq / 3.7e7

    # Get Zn-65 vs irrad time for Plot 1 (from sim_dir or passed in)
    if zn65_vs_irrad is None and sim_dir:
        zn65_vs_irrad = _get_zn65_vs_irrad(sim_dir, cooldown_days)
    if zn65_vs_irrad is None:
        zn65_vs_irrad = [(8760, zn65_activity_Bq)]  # Single point

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    hours = np.array([p[0] for p in zn65_vs_irrad])
    zn65_activities_Bq = np.array([p[1] for p in zn65_vs_irrad])
    mCi_zn = zn65_activities_Bq / 3.7e7
    spec_act_zn = zn65_activities_Bq / zn65_mass_g

    ax1.plot(hours / 24, mCi_zn, 'b-o', linewidth=2, markersize=6, label='Zn-65')
    ax1.set_xlabel('Irradiation time (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Activity (mCi)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Zn-65 Activity vs Irradiation Time', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(hours) / 24 * 1.05)
    ax1.set_yscale('log')

    ax2.plot(hours / 24, spec_act_zn, 'b-o', linewidth=2, markersize=6, label='Zn-65')
    ax2.set_xlabel('Irradiation time (days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Specific Activity (Bq/g)', fontsize=12, fontweight='bold')
    ax2.set_title(f'(b) Zn-65 Specific Activity vs Irradiation Time\nTotal waste mass: {zn65_mass_g/1000:.2f} kg', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(hours) / 24 * 1.05)
    ax2.set_yscale('log')

    fig1.suptitle('Zn-65 Waste: Activity and Specific Activity vs Irradiation Time', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig1.savefig(os.path.join(output_dir, 'plot1_zn65_irrad.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_dir}/plot1_zn65_irrad.png")

    # --- Plot: volume-only full drums Ci/m^3 vs irrad (how long until >700 Ci/m^3) ---
    if zn65_vs_irrad and len(zn65_vs_irrad) >= 2:
        plot_class_a_700_vs_irrad(zn65_vs_irrad, zn65_mass_g, output_dir, cooldown_days)

    # --- Plot: Ci/m³ vs irradiation time over 20 years (Zn-65 activity per drum volume) ---
    if zn65_vs_irrad and len(zn65_vs_irrad) >= 1:
        plot_ci_m3_vs_irrad_20y(zn65_vs_irrad, zn65_mass_g, output_dir, cooldown_days)

    t_clear_zn = time_to_clearance(zn65_activity_Bq, zn65_mass_g, 'Zn65')
    x_max_years = max(12, t_clear_zn / 365.25 * 1.15)
    days = np.linspace(0, x_max_years * 365.25, 600)
    zn65_decay = calculate_activity_decay(zn65_activity_Bq, 'Zn65', days)
    zn65_decay_TBq = zn65_decay / 1e12
    clearance_zn_TBq = (CLEARANCE_LEVEL_Bq_per_g['Zn65'] * zn65_mass_g) / 1e12

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.semilogy(days / 365.25, zn65_decay_TBq, 'b-', linewidth=2.5, label='Zn-65 (t½=244 d)')
    ax2.axhline(y=clearance_zn_TBq, color='b', linestyle=':', alpha=0.5, linewidth=1, label='Clearance (0.1 Bq/g)')
    ax2.set_xlabel('Time after EOI (years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Activity (TBq)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Zn-65 Decay: Below clearance by ~{t_clear_zn/365.25:.0f} y', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlim(0, x_max_years)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'plot2_zn65_decay.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir}/plot2_zn65_decay.png")

    t_clear_zn = time_to_clearance(zn65_activity_Bq, zn65_mass_g, 'Zn65')
    shield_pb_zn = shielding_thickness(zn65_GBq * 1000, 'Zn65', 25, 'Pb')
    shield_conc_zn = shielding_thickness(zn65_GBq * 1000, 'Zn65', 25, 'concrete')
    zn65_total_mass_kg = zn65_mass_g / 1000.0

    create_cost_breakdown_table(zn65_mass_g, zn65_activity_Bq, output_dir, **cost_kwargs)
    n_drums_vol, _ = drums_by_volume_only(zn65_mass_g)
    print(f"\nZn-65 WASTE: {zn65_GBq:.3f} GBq, {zn65_mass_g:.0f} g, Class A LLRW; drums (volume only): {n_drums_vol}")

    summary_df = pd.DataFrame({
        'Metric': [
            'Activity (TBq)', 'Activity (GBq)', 'Activity (mCi)',
            'Total waste mass (kg)', 'Total waste mass (g)', 'Specific Activity (Bq/g)',
            'Half-life (days)', '10 CFR 35.92 Eligible (≤120d)', 'Inhalation Class', 'Lung Clearance Half-time',
            'IAEA 0.1 Bq/g release (years)', 'Pb Shielding (cm) for 2.5 microSv/hr', 'Concrete Shielding (cm)',
            'ALI Ingestion (microCi)', 'ALI Inhalation (microCi)', 'DAC (microCi/ml)', 'Disposal Classification',
        ],
        'Zn-65': [
            f'{zn65_GBq/1e3:.3f}', f'{zn65_GBq:.3f}', f'{zn65_mCi:,.0f}',
            f'{zn65_total_mass_kg:.2f}', f'{zn65_mass_g:.1f}', f'{zn65_activity_Bq/zn65_mass_g:.2e}',
            '244', 'NO', 'Y (Year)', '>100 days (~500d)',
            f'{t_clear_zn/365:.1f}', f'{shield_pb_zn:.1f}', f'{shield_conc_zn:.1f}',
            '400', '300', '1E-7', 'LLRW Class A (10 CFR 61.55)',
        ],
    })
    summary_df.to_csv(os.path.join(output_dir, 'waste_summary.csv'), index=False)
    print(f"Saved: {output_dir}/waste_summary.csv")


def write_zn_waste_formulas_process_file(zn_total_mass_g, zn65_activity_Bq, output_dir='.',
                                         storage_df=None, drums_needed=1,
                                         transport_cost=0, transport_cost_per_drum=0,
                                         storage_cost=0, storage_cost_per_ft3_per_year=100,
                                         packaging_cost=0, packaging_cost_per_drum=0,
                                         disposal_cost=0, disposal_cost_per_drum=0,
                                         total_cost=0, scenario_irradiation_years=None,
                                         no_interim_storage=True):
    """
    Write a text file: formulas, process steps, methods, and this-run numbers for the
    large Zn irradiation chamber → Pb storage casks → transport → storage/disposal plan.
    """
    os.makedirs(output_dir, exist_ok=True)
    zn65_GBq = zn65_activity_Bq / 1e9
    zn65_Ci = zn65_activity_Bq / 3.7e10
    zn_volume_cm3 = zn_total_mass_g / ZN_DENSITY_G_CM3
    drums_vol, _ = drums_by_volume_only(zn_total_mass_g)
    drum_vol_m3 = 0.208
    drums_volume_ft3 = drums_needed * 7.35

    lines = [
        "Zn-65 waste: formulas and process steps",
        "=" * 60,
        "",
        "OBJECTIVE: Waste after a time period; meet (1) external dose ≤10 mSv/h at 3 m from unshielded material;",
        "(2) transport limits → fill per drum, number of drums; (3) facility storage limits → fill per drum, drums to store;",
        "(4) volume-only option: stuff all Zn into N drums (3 cm / 5 cm by volume), activity levels reported.",
        "",
        "SCOPE: Irradiated Zn in Pb-shielded 55-gal drums (3 or 5 cm Pb), transport to disposal (e.g. Clive).",
        "",
        "FORMULAS",
        "-" * 40,
        "1. Activity: Zn-65 from OpenMC/Bateman; specific activity Bq/g = activity_Bq / mass_g.",
        "2. Dose vs distance (point source): D(r) = D_1m * (100/r)^2 [r in cm]. D_1m = coeff * activity_MBq; coeff = 0.070992 µSv/hr per MBq at 1 m (Zn-65 1.116 MeV).",
        "3. HVL (half-value layer): I = I0*exp(-mu*x); HVL = ln(2)/mu = 0.693/mu. Linear attenuation mu = (mu/rho)*rho; (mu/rho) from NIST. Zn-65: HVL_Pb = 1.4 cm (14 mm).",
        "4. Shielded dose: D_shielded = D_unshielded / 2^(t/HVL) for thickness t.",
        "5. Max activity per drum (storage/transport): 200 mrem/hr at container surface (10 CFR 71.47, 49 CFR 173.441).",
        "   max_activity_MBq = (25 * shielding_factor) / (coeff * distance_factor); distance_factor = (1/drum_radius_m)^2; shielding_factor = 2^(t_Pb_cm / HVL_Pb_cm).",
        "6. Drum count by activity: drums = ceil(total_activity_GBq / max_GBq_per_drum).",
        "7. Drum count by volume: drums = ceil(zn_volume_cm3 / drum_usable_volume_cm3). Usable volume after 3–5 cm Pb liner is less than 208 L (55-gal).",
        "8. Ci/m^3 (Class A): activity_Ci / (drums * 0.208 m^3). Class A limit 700 Ci/m^3 (10 CFR 61.55 Table 2, T½<5 y).",
        "9. Transport limit: 200 mrem/hr (2 mSv/h) at package surface (10 CFR 71.47, 49 CFR 173.441). Exclusive use allows 1000 mrem/hr (71.47(b)).",
        "",
        "PROCESS STEPS (plan)",
        "-" * 40,
        "Step 1: Irradiate Zn in the large irradiation chamber (fusion neutrons).",
        "Step 2: Compute Zn-65 activity at EOI (end of irradiation; with or without cooldown).",
        "Step 3: Choose cask type: 3 cm Pb or 5 cm Pb storage cylinders (55-gal drum dimensions).",
        "Step 4: Max fill per drum: limited by 200 mrem/hr at surface (10 CFR 71.47) → max GBq/drum and thus max Zn kg/drum. Usable volume per drum is reduced by the Pb liner.",
        "Step 5: Number of drums: take the larger of (drums by volume) and (drums by activity). If drums are only partially full by volume, options:",
        "   (a) Accept more drums (each drum at or below activity limit).",
        "   (b) Add inert filler (e.g. concrete, sand) to fill remaining volume and stabilize; does not change activity per drum but may simplify handling.",
        "   (c) Use thicker Pb (5 cm) to allow more activity per drum → fewer drums, but each drum heavier and more expensive.",
        "Step 6: Transport: drums must meet 200 mrem/hr at surface. Pb thickness for transport may exceed storage thickness. Transport cost: $/drum or $/mile × distance (e.g. Wisconsin → Clive ~1450 mi, ~$2.50/mi).",
        "Step 7: At Clive (or disposal facility): option to repackage into thinner-wall cylinders for long-term storage (fewer drums per unit volume in the repository), or consolidate into larger casks/drums to reduce storage space and cost.",
        "Step 8: Storage cost: $/ft^3/year for interim storage; disposal cost: $/drum or $/ft^3 for Class A LLRW.",
        "",
        "THIS RUN (summary)",
        "-" * 40,
        f"Zn mass: {zn_total_mass_g:.1f} g ({zn_total_mass_g/1000:.2f} kg). Zn-65 activity: {zn65_activity_Bq:.2e} Bq ({zn65_GBq:.3f} GBq, {zn65_Ci:.2f} Ci).",
        f"Zn volume: {zn_volume_cm3/1e6:.4f} m^3. Drums by volume only: {drums_vol}.",
        f"Drums used (this scenario): {drums_needed}. Total drum volume: {drums_volume_ft3:.1f} ft^3.",
        "",
    ]
    if storage_df is not None and not storage_df.empty:
        lines.append("Drums by option:")
        for _, r in storage_df.iterrows():
            lines.append(f"  {r['Option']}: {int(r['Drums needed'])} drums, Zn {r['Zn kg per drum']:.2f} kg/drum, {r['Fraction full']*100:.1f}% full.")
        lines.append("")
    lines.extend([
        f"Transport: ${transport_cost:,.0f} total (${transport_cost_per_drum:.0f}/drum).",
        f"Storage (interim): ${storage_cost:,.0f}" + (f" (${storage_cost_per_ft3_per_year}/ft^3/yr)" if storage_cost > 0 else " (no interim storage)."),
        f"Packaging: ${packaging_cost:,.0f} (${packaging_cost_per_drum:.0f}/drum).",
        f"Disposal: ${disposal_cost:,.0f} (${disposal_cost_per_drum:.0f}/drum).",
        f"Total estimated cost: ${total_cost:,.0f}.",
        "",
    ])
    if scenario_irradiation_years:
        lines.append(f"Scenario: {scenario_irradiation_years:.0f}-year irradiation" + (", no interim storage, then dispose." if no_interim_storage else ", then store then dispose."))
    path = os.path.join(output_dir, 'zn_waste_formulas_process.txt')
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


def _build_waste_summary_table(zn65_activity_Bq, zn65_GBq, zn65_mCi,
                               zn_total_mass_g, zn_total_mass_lbs, zn_total_mass_tons,
                               storage_df, drums_needed, meets_3m_limit, max_MBq_3m,
                               total_characterization, packaging_cost, packaging_cost_per_drum,
                               storage_cost, transport_cost, transport_cost_per_drum,
                               class_a_disposal_cost, disposal_cost_per_drum,
                               taxes_insurance, total_cost,
                               no_interim_storage, scenario_irradiation_years):
    """Build summary table rows: Parameter, Value, Unit, Notes (test.py style)."""
    drum_vol_m3 = 0.208
    rows = [
        ['Parameter', 'Value', 'Unit', 'Notes'],
        [],
        ['Objective: waste after time period', '', '', 'Meet dose limits; fill drums by transport/storage limits or by volume only.'],
        ['External dose limit (unshielded)', '10 mSv/h at 3 m', '', '1 rem/h at 10 ft from unshielded material; facility acceptance.'],
        ['Our activity vs 3 m limit', f'{zn65_GBq*1000:.1f} MBq', 'MBq', f'Max at 3 m: {max_MBq_3m:.0f} MBq. Meets limit: {"Yes" if meets_3m_limit else "No"}.'],
        [],
        ['Transport limit', '200 mrem/hr at surface', '', '10 CFR 71.47, 49 CFR 173.441. Fill per drum → number of drums.'],
    ]
    if storage_df is not None and not storage_df.empty:
        for opt in ['3 cm Pb', '5 cm Pb']:
            if opt in storage_df['Option'].values:
                r = storage_df[storage_df['Option'] == opt].iloc[0]
                max_GBq = calculate_max_activity_per_drum('Zn65', float(opt.split()[0]), DOSE_LIMIT_TRANSPORT_UVSV_HR, 0.3)
                rows.append([f'Drums ({opt}, transport)', int(r['Drums needed']), 'count', f"Zn {r['Zn kg per drum']:.2f} kg/drum, {r['Fraction full']*100:.1f}% full; max {max_GBq:.1f} GBq/drum"])
        rows.append([])
        rows.append(['Storage limit (facility)', '200 mrem/hr at surface', '', 'Restricted storage; 10 CFR 71.47. Fill per drum → number of drums.'])
        for opt in ['3 cm Pb', '5 cm Pb']:
            if opt in storage_df['Option'].values:
                r = storage_df[storage_df['Option'] == opt].iloc[0]
                rows.append([f'Drums ({opt}, storage)', int(r['Drums needed']), 'count', f"Zn {r['Zn kg per drum']:.2f} kg/drum, {r['Fraction full']*100:.1f}% full"])
    rows.append([])
    rows.append(['Volume only (ignore limits)', '', '', 'Stuff all Zn by volume; 3 cm and 5 cm liner estimates.'])
    drums_vol, zn_vol_cm3 = drums_by_volume_only(zn_total_mass_g)
    vol_3 = _drum_usable_volume_cm3(3.0, False)
    vol_5 = _drum_usable_volume_cm3(5.0, False)
    drums_3_vol = max(1, int(np.ceil(zn_vol_cm3 / vol_3)))
    drums_5_vol = max(1, int(np.ceil(zn_vol_cm3 / vol_5)))
    act_per_drum_3 = (zn65_activity_Bq / 1e9) / drums_3_vol if drums_3_vol else 0
    act_per_drum_5 = (zn65_activity_Bq / 1e9) / drums_5_vol if drums_5_vol else 0
    rows.append(['Drums (3 cm Pb, volume only)', drums_3_vol, 'count', f'Activity {act_per_drum_3:.2f} GBq/drum'])
    rows.append(['Drums (5 cm Pb, volume only)', drums_5_vol, 'count', f'Activity {act_per_drum_5:.2f} GBq/drum'])
    rows.append([])
    rows.append(['Costs (no contingency, labor, or equipment)', '', '', ''])
    rows.append(['Characterization', total_characterization, 'USD', ''])
    rows.append(['Packaging', packaging_cost, 'USD', f'${packaging_cost_per_drum:.0f}/drum'])
    rows.append(['Interim storage', storage_cost, 'USD', 'No interim storage' if no_interim_storage else ''])
    rows.append(['Transport', transport_cost, 'USD', f'${transport_cost_per_drum:.0f}/drum'])
    rows.append(['Class A disposal', class_a_disposal_cost, 'USD', f'${disposal_cost_per_drum:.0f}/drum'])
    rows.append(['Taxes & insurance', taxes_insurance, 'USD', '16% of base'])
    rows.append(['Total estimated cost', total_cost, 'USD', ''])
    return rows


def _write_waste_summary_table_png(summary_rows, output_path):
    """Render the waste summary table (Parameter, Value, Unit, Notes) as a PNG."""
    # Normalize: header + data rows; empty rows become blank 4-cell rows
    cols = ['Parameter', 'Value', 'Unit', 'Notes']
    data = []
    for row in summary_rows:
        if not row:
            data.append(['', '', '', ''])
            continue
        if len(row) >= 4 and row[0] == 'Parameter' and row[1] == 'Value':
            continue  # skip header row, we use cols
        data.append([str(x) if x is not None else '' for x in (row + ['', '', '', ''])[:4]])
    if not data:
        data = [['', '', '', '']]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * len(data))))
    ax.axis('off')
    table = ax.table(
        cellText=data,
        colLabels=cols,
        loc='center',
        cellLoc='left',
        colColours=['#e8e8e8'] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_cost_breakdown_table(zn_total_mass_g, zn65_activity_Bq, output_dir='.',
                                max_mass_per_drum_kg=None, use_atlantic_compact=True,
                                educational_research=False,
                                no_interim_storage=True,
                                scenario_irradiation_years=None,
                                shield_cm_for_drums=3.0,
                                packaging_cost_per_drum_override=None,
                                transport_cost_per_drum_override=None,
                                disposal_cost_per_drum_override=None):
    """
    Create Figure/Table 4: Cost breakdown for Zn waste (all isotopes).

    Parameters
    ----------
    zn_total_mass_g : float
        Total Zn mass (all isotopes) in grams
    zn65_activity_Bq : float
        Zn-65 activity in Bq (for activity-based cost factors)
    output_dir : str
        Output directory for the figure
    max_mass_per_drum_kg : float or None
        Unused; drum count is by volume only (no activity limits).
    use_atlantic_compact : bool
        If True, use Atlantic Compact Uniform Schedule for disposal cost (weight-based).
    educational_research : bool
        If True, educational research institution rate (one block lower).
    no_interim_storage : bool
        If True, no interim storage (storage_cost=0); e.g. "irradiate 8 years then dispose".
    scenario_irradiation_years : float or None
        If set (e.g. 8), scenario described as "X-year irradiation, then dispose".
    shield_cm_for_drums : float
        Unused for drum count (volume-only); kept for storage_df reference options.
    packaging_cost_per_drum_override : float or None
        If set (e.g. 884), use this $/drum for packaging instead of computed.
    transport_cost_per_drum_override : float or None
        If set, use this $/drum for transport instead of computed.
    disposal_cost_per_drum_override : float or None
        If set (e.g. 7125), use this $/drum for Class A disposal instead of computed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert mass to pounds and tons
    zn_total_mass_lbs = zn_total_mass_g * 0.00220462  # g to lbs
    zn_total_mass_tons = zn_total_mass_lbs / 2000.0
    
    zn65_GBq = zn65_activity_Bq / 1e9
    zn65_mCi = zn65_activity_Bq / 3.7e7

    # Max unshielded activity to meet 10 mSv/h at 3 m (facility acceptance)
    max_MBq_3m = max_activity_MBq_at_3m_below_limit('Zn65')
    meets_3m_limit = zn65_GBq * 1000 <= max_MBq_3m

    # Cost estimates (characterization, packaging, storage, transport, disposal only; no contingency, labor, equipment)
    # Reference: NIST-TR-23-003 (NRC ML2313/ML23136B278.pdf)
    # NIST DCE uses NUREG 1757 format with detailed cost breakdowns
    
    # 55-GALLON DRUM SPECIFICATIONS (per NUREG 1757, industry standards)
    # Standard 55-gal drum: 7.35 ft^3, ~22.5" diameter × 33.5" height, ~400 lbs empty
    # DOT Type A container required for Class A LLRW transport
    # Transport limit: <200 mrem/hr (2000 µSv/hr) at container surface, if set by activity 
    
    # Calculate shielding needed for transport (200 mrem/hr = 2000 µSv/hr)
    shield_pb_transport = shielding_thickness(zn65_GBq * 1000, 'Zn65', DOSE_LIMIT_TRANSPORT_UVSV_HR, 'Pb')
    shield_pb_facility = shielding_thickness(zn65_GBq * 1000, 'Zn65', DOSE_LIMIT_STORAGE_UVSV_HR, 'Pb')
    
    # Drum count: by volume only (no activity limits)—how many drums to fit total Zn volume
    drums_needed, zn_volume_cm3 = drums_by_volume_only(zn_total_mass_g)
    mass_drum_str = f'Volume only: {zn_total_mass_g/1000:.2f} kg Zn → {zn_volume_cm3/1e6:.4f} m^3 → {drums_needed} drums (55 gal each)'

    drum_volume_ft3 = 7.35  # Standard 55-gal drum volume
    drums_volume_ft3 = drums_needed * drum_volume_ft3

    # Drum storage summary: volume-only first, then 3 cm Pb, 5 cm Pb, concrete (for reference)
    storage_df = drum_storage_summary(zn_total_mass_g, zn65_activity_Bq)
    if not storage_df.empty:
        drums_3pb = int(storage_df[storage_df['Option'] == '3 cm Pb'].iloc[0]['Drums needed']) if '3 cm Pb' in storage_df['Option'].values else drums_needed
        drums_4pb = int(storage_df[storage_df['Option'] == '4 cm Pb'].iloc[0]['Drums needed']) if '4 cm Pb' in storage_df['Option'].values else drums_3pb
        drums_5pb = int(storage_df[storage_df['Option'] == '5 cm Pb'].iloc[0]['Drums needed']) if '5 cm Pb' in storage_df['Option'].values else drums_needed
        drums_conc = int(storage_df[storage_df['Option'].str.contains('concrete', na=False)].iloc[0]['Drums needed']) if storage_df['Option'].str.contains('concrete', na=False).any() else drums_needed
    else:
        drums_3pb = drums_4pb = drums_5pb = drums_conc = drums_needed

    zn65_Ci = zn65_activity_Bq / 3.7e10
    drum_vol_m3 = 0.208  # 55 gal
    # Primary: Ci/m^3 using volume-only drum count (drums sized by Zn volume only)
    ci_m3_volume_only = zn65_Ci / (drums_needed * drum_vol_m3) if (drums_needed * drum_vol_m3) > 0 else 0
    ci_m3_3pb = (zn65_Ci / (storage_df.loc[storage_df['Option'] == '3 cm Pb', 'Drums needed'].iloc[0] * drum_vol_m3)
                 if '3 cm Pb' in storage_df['Option'].values else ci_m3_volume_only)
    ci_m3_5pb = (zn65_Ci / (storage_df.loc[storage_df['Option'] == '5 cm Pb', 'Drums needed'].iloc[0] * drum_vol_m3)
                 if '5 cm Pb' in storage_df['Option'].values else ci_m3_volume_only)

    # EnergySolutions quote input: activity, mass, Class A Ci/m^3 comparison, drum options
    es_quote = pd.DataFrame([
        ['Parameter', 'Value', 'Unit', 'Notes'],
        ['Activity', zn65_activity_Bq, 'Bq', 'Zn-65 at EOI'],
        ['Activity', zn65_GBq, 'GBq', ''],
        ['Activity', zn65_mCi, 'mCi', ''],
        ['Total waste mass', zn_total_mass_g / 1000, 'kg', 'All Zn isotopes'],
        ['Total waste mass', zn_total_mass_lbs, 'lbs', ''],
        ['Half-life', 244, 'days', 'NNDC (<5 y → Table 2)'],
        ['10 CFR 61.55 Class A limit', CLASS_A_CI_PER_M3_LIMIT, 'Ci/m^3', 'Table 2: nuclides T½<5 y'],
        ['Our Ci/m^3 (volume-only drums)', ci_m3_volume_only, 'Ci/m^3', f'{ci_m3_volume_only/CLASS_A_CI_PER_M3_LIMIT*100:.1f}% of limit'],
        ['Our Ci/m^3 (3 cm Pb drums)', ci_m3_3pb, 'Ci/m^3', f'{ci_m3_3pb/CLASS_A_CI_PER_M3_LIMIT*100:.1f}% of limit'],
        ['Our Ci/m^3 (5 cm Pb drums)', ci_m3_5pb, 'Ci/m^3', f'{ci_m3_5pb/CLASS_A_CI_PER_M3_LIMIT*100:.1f}% of limit'],
        ['Disposal class', 'Class A LLRW', '', 'Zn-65 only; 10 CFR 61.55'],
        ['Primary isotope', 'Zn-65', '', '1.116 MeV γ (50.6%)'],
    ])
    for _, r in storage_df.iterrows():
        es_quote = pd.concat([es_quote, pd.DataFrame([[
            f"Drums ({r['Option']})", int(r['Drums needed']), 'count',
            f"Zn {r['Zn kg per drum']:.2f} kg/drum, {r['Fraction full']*100:.1f}% full"
        ]])], ignore_index=True)
    es_path = os.path.join(output_dir, 'energysolutions_quote_input.csv')
    es_quote.to_csv(es_path, index=False, header=False)
    print(f"Saved: {es_path}")
    
    # COST BREAKDOWN (per NUREG 1757 format, NIST DCE methodology)
    # Reference: NIST-TR-23-003 shows packaging/shipping/disposal is most costly component
    
    # 1. CHARACTERIZATION COSTS (NUREG 1757 Tab 3.5)
    # Waste characterization, sampling, analysis per 10 CFR 61
    # NIST: Laboratory costs for gamma, Sr-90, H-3, C-14, alpha emitters
    characterization_cost = 10_000  # Fixed cost for initial characterization
    lab_analysis_per_drum = 200  # $/drum for gamma analysis (NIST methodology)
    lab_analysis_cost = drums_needed * lab_analysis_per_drum
    total_characterization = characterization_cost + lab_analysis_cost
    
    # 2. PACKAGING COSTS (NUREG 1757 Tab 3.14)
    # 55-gal drums with lead shielding, labeling, documentation
    # NIST: Shipping containers, DOT compliance, documentation
    # Zn is already solid - no solidification needed
    drum_cost = 150  # $/drum (empty 55-gal drum, DOT Type A)
    
    # Lead shielding cost based on transport requirement (thicker than facility)
    # Base cost: $300 for 3-5 cm Pb, add $50/cm for additional thickness
    base_pb_cost = 300  # $/drum for 3-5 cm Pb
    extra_pb_cost = max(0, (shield_pb_transport - 3) * 50)  # Extra cost if >3 cm
    lead_shielding_cost_per_drum = base_pb_cost + extra_pb_cost
    
    labeling_doc_per_drum = 50  # $/drum (labels, manifests, DOT paperwork)
    packaging_cost_per_drum = (packaging_cost_per_drum_override if packaging_cost_per_drum_override is not None
                              else (drum_cost + lead_shielding_cost_per_drum + labeling_doc_per_drum))
    packaging_cost = drums_needed * packaging_cost_per_drum
    
    # 3. INTERIM STORAGE COSTS (NUREG 1757 Tab 3.10)
    storage_years = 0.0 if no_interim_storage else 1.0
    storage_cost_per_ft3_per_year = 100  # $/ft^3/year (mid-range: WCS, EnergySolutions, NIST DCE)
    storage_cost = 0.0 if no_interim_storage else (drums_volume_ft3 * storage_cost_per_ft3_per_year * storage_years)
    
    # 4. TRANSPORT COSTS (NUREG 1757 Tab 3.14)
    # From Wisconsin (SHINE location) to EnergySolutions (Clive, UT): 1,450 mi (NUREG-2183 Section 4.9.1)
    # Alternative: WCS (Andrews, TX): 1,305 mi (NUREG-2183 Section 2.7.1.2)
    # Use EnergySolutions distance (primary disposal site per NUREG-2183)
    transport_distance_mi = 1450  # Wisconsin to Clive, UT (NUREG-2183 Section 4.9.1)
    # Transport cost per mile from industry sources:
    # - DOT-regulated LLRW transport: $2.00-3.50/mile (includes trucking, fuel, permits, DOT compliance)
    # - NIST DCE: Similar methodology for decommissioning waste transport
    # - Industry average: ~$2.50/mile for Class A LLRW (includes fuel surcharges, overweight charges)
    transport_cost_per_mile = 2.50  # $/mile (industry average: DOT-regulated LLRW transport)
    # Cost per drum: total transport cost divided by drums, with minimum per drum
    transport_cost_total = transport_distance_mi * transport_cost_per_mile  # Total trip cost
    transport_cost_per_drum_computed = transport_cost_total / drums_needed if drums_needed > 0 else 0
    transport_cost_per_drum = (transport_cost_per_drum_override if transport_cost_per_drum_override is not None
                              else max(transport_cost_per_drum_computed, 1000))
    transport_cost = drums_needed * transport_cost_per_drum
    
    # 5. DISPOSAL COSTS
    # Option A: Atlantic Compact (weight-based, no upper weight limit from tables)
    # Option B: Volume-based (NUREG 1757)
    if disposal_cost_per_drum_override is not None:
        disposal_cost_per_drum = disposal_cost_per_drum_override
        class_a_disposal_cost = drums_needed * disposal_cost_per_drum
        disposal_cost_per_ft3 = class_a_disposal_cost / drums_volume_ft3 if drums_volume_ft3 > 0 else 150
    elif use_atlantic_compact:
        lead_lbs_per_drum = 250  # Approx for 3-5 cm Pb liner
        total_weight_lbs = zn_total_mass_lbs + drums_needed * (400 + lead_lbs_per_drum)
        density_lbs_ft3 = total_weight_lbs / drums_volume_ft3 if drums_volume_ft3 > 0 else 60
        class_a_disposal_cost = atlantic_compact_disposal_cost(
            total_weight_lbs, density_lbs_ft3, dose_rate_mR_hr=0,
            educational_research=educational_research)
        disposal_cost_per_drum = class_a_disposal_cost / drums_needed if drums_needed > 0 else 0
        disposal_cost_per_ft3 = class_a_disposal_cost / drums_volume_ft3 if drums_volume_ft3 > 0 else 150
    else:
        disposal_cost_per_ft3 = 150  # $/ft^3 (mid-range, Class A LLRW)
        disposal_cost_per_drum = disposal_cost_per_ft3 * drum_volume_ft3
        class_a_disposal_cost = drums_needed * disposal_cost_per_drum
    
    # Taxes/insurance (16% of base)
    base_cost = (total_characterization + packaging_cost + storage_cost + transport_cost + class_a_disposal_cost)
    taxes_insurance = base_cost * 0.16
    total_cost = base_cost + taxes_insurance

    # Single summary table (Parameter, Value, Unit, Notes) — primary output
    summary_rows = _build_waste_summary_table(
        zn65_activity_Bq=zn65_activity_Bq, zn65_GBq=zn65_GBq, zn65_mCi=zn65_mCi,
        zn_total_mass_g=zn_total_mass_g, zn_total_mass_lbs=zn_total_mass_lbs, zn_total_mass_tons=zn_total_mass_tons,
        storage_df=storage_df, drums_needed=drums_needed, meets_3m_limit=meets_3m_limit, max_MBq_3m=max_MBq_3m,
        total_characterization=total_characterization, packaging_cost=packaging_cost, packaging_cost_per_drum=packaging_cost_per_drum,
        storage_cost=storage_cost, transport_cost=transport_cost, transport_cost_per_drum=transport_cost_per_drum,
        class_a_disposal_cost=class_a_disposal_cost, disposal_cost_per_drum=disposal_cost_per_drum,
        taxes_insurance=taxes_insurance, total_cost=total_cost,
        no_interim_storage=no_interim_storage, scenario_irradiation_years=scenario_irradiation_years,
    )
    summary_path = os.path.join(output_dir, 'waste_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, header=False)
    print(f"Saved: {summary_path}")
    summary_png_path = os.path.join(output_dir, 'waste_summary.png')
    _write_waste_summary_table_png(summary_rows, summary_png_path)

    write_zn_waste_formulas_process_file(
        zn_total_mass_g, zn65_activity_Bq, output_dir,
        storage_df=storage_df, drums_needed=drums_needed,
        transport_cost=transport_cost, transport_cost_per_drum=transport_cost_per_drum,
        storage_cost=storage_cost, storage_cost_per_ft3_per_year=storage_cost_per_ft3_per_year,
        packaging_cost=packaging_cost, packaging_cost_per_drum=packaging_cost_per_drum,
        disposal_cost=class_a_disposal_cost, disposal_cost_per_drum=disposal_cost_per_drum,
        total_cost=total_cost, scenario_irradiation_years=scenario_irradiation_years,
        no_interim_storage=no_interim_storage,
    )
    
    # EnergySolutions reference costs (scaled from their 100 ton Class A estimate)
    # Assume their 100 ton estimate includes all costs (disposal, storage, packaging, transport)
    # Rough estimate: $500,000-2,000,000 for 100 tons Class A (all-inclusive)
    es_cost_per_ton = 10_000  # $/ton (all-inclusive estimate)
    es_reference_cost = ES_CLASS_A_TONS * es_cost_per_ton
    our_cost_scaled = zn_total_mass_tons * es_cost_per_ton
    
    # SHINE waste data from NUREG-2183 Section 4.9.1
    # SHINE produces 740 TBq Lu-177/yr, but waste is short-lived (Mo-99 t½=66h, Lu-177 t½=6.65d)
    # SHINE ships Class A waste ~annually to EnergySolutions (Clive, UT, 1,450 mi)
    # Waste types: evaporator bottoms, spent ion-exchange media (solidified)
    # Waste decays quickly before shipment (no long-lived isotopes like Zn-65)
    shine_waste_volume_estimate = '~50-200 drums/yr\n(short-lived, decays quickly)'
    shine_shipment_frequency = 'Annual\n(NUREG-2183 4.9.1)'
    shine_disposal_site = 'EnergySolutions\nClive, UT\n(1,450 mi)'
    
    # Calculate detailed explanations for each value
    drums_volume_ft3 = drums_needed * 7.35
    es_drums_estimate = ES_CLASS_A_TONS * 20
    
    # Table 1: Waste Quantities
    waste_quantities_data = [
        ['Cost Category', 'Our Case (Zn Waste)', 'SHINE (NUREG-2183)', 'EnergySolutions Reference', 'Calculation Details'],
        ['WASTE QUANTITIES', '', '', '', ''],
        ['Total waste mass', 
         f'{zn_total_mass_g:.1f} g\n({zn_total_mass_lbs:.1f} lbs)\n({zn_total_mass_tons:.4f} tons)', 
         'Evaporator bottoms\nIX media (solidified)\nShort-lived isotopes', 
         f'{ES_CLASS_A_LBS:,} lbs\n({ES_CLASS_A_TONS} tons)\nClass A waste',
         f'OUR CASE: From statepoint\nVolume × density = {zn_total_mass_g:.1f} g\nConvert: {zn_total_mass_g:.1f} g × 0.00220462 = {zn_total_mass_lbs:.1f} lbs\n{zn_total_mass_lbs:.1f} lbs ÷ 2000 = {zn_total_mass_tons:.4f} tons\n\nES REF: TABLE 1-2\n200,000 lbs = 100 tons (fixed reference)'],
        ['Primary isotope activity', 
         f'{zn65_GBq:.3f} GBq\n({zn65_mCi:,.0f} mCi)\nZn-65 (t½=244 d)', 
         'Mo-99: t½=66 h\nLu-177: t½=6.65 d\n(decay quickly)', 
         'Variable\n(Class A, B, C, GTCC)',
         f'OUR CASE: From statepoint\nZn-65 activity = {zn65_activity_Bq:.3e} Bq\nConvert: {zn65_activity_Bq:.3e} Bq ÷ 1e9 = {zn65_GBq:.3f} GBq\n{zn65_activity_Bq:.3e} Bq ÷ 3.7e7 = {zn65_mCi:,.0f} mCi\nHalf-life: 244 days (NNDC)\n\nSHINE: NUREG-2183\nMo-99: 66 hours, Lu-177: 6.65 days\nBoth decay quickly before shipment'],
        ['Estimated drums (55-gal)', 
         f'3 cm Pb: {drums_3pb:.0f} | 5 cm Pb: {drums_5pb:.0f} | concrete: {drums_conc:.0f}\n(~{drums_volume_ft3:.1f} ft^3 for 3 cm Pb)', 
         shine_waste_volume_estimate, 
         f'~{es_drums_estimate:.0f} drums\n(estimate)',
         f'OUR CASE: 3 cm Pb: {drums_3pb:.0f} drums | 5 cm Pb: {drums_5pb:.0f} drums | concrete: {drums_conc:.0f} drums\nZn kg/drum and fraction full per option.\nStorage activity limit → max GBq/drum; drums = ceil(activity / max_GBq/drum)\n55-gal drum: {drum_volume_ft3} ft^3, usable ~193,000 cm^3 after liner\nDOT Type A, <200 mrem/hr at surface\n\nES REF: {ES_CLASS_A_TONS} tons × 20 drums/ton = {es_drums_estimate:.0f} drums'],
    ]
    
    # Table 2: Cost Breakdown - Main Costs
    cost_breakdown_data = [
        ['Cost Category', 'Our Case (Zn Waste)', 'SHINE (NUREG-2183)', 'EnergySolutions Reference', 'Calculation Details'],
        ['COST BREAKDOWN', '', '', '', ''],
        ['Characterization & Analysis', 
         f'${total_characterization:,}\n(${characterization_cost:,} + ${lab_analysis_cost:,.0f})', 
         'Included in operations', 
         'Included in UCF',
         f'OUR CASE (NUREG 1757 Tab 3.5):\nInitial characterization: ${characterization_cost:,}\nLab analysis: {drums_needed:.1f} drums × ${lab_analysis_per_drum}/drum = ${lab_analysis_cost:,.0f}\nTotal: ${total_characterization:,}\n(Gamma, Sr-90, H-3, C-14, alpha analysis)\n\nNIST DCE: Similar methodology\nLab costs for gamma emitters\n\nSHINE/ES: Included in operations/UCF'],
        ['Packaging (55-gal drums)', 
         f'${packaging_cost:,.0f}\n(${packaging_cost_per_drum:.0f}/drum)', 
         'Cementation in hot cells\nStandard 55-gal drums', 
         'Included in UCF',
         f'OUR CASE (NUREG 1757 Tab 3.14):\n{drums_needed:.0f} drums × ${packaging_cost_per_drum:.0f}/drum = ${packaging_cost:,.0f}\nBreakdown:\n- Drum: ${drum_cost}/drum (DOT Type A, 55-gal)\n- Lead shielding: ${lead_shielding_cost_per_drum:.0f}/drum ({shield_pb_transport:.1f} cm Pb for transport, {shield_pb_facility:.1f} cm for facility)\n- Labeling/docs: ${labeling_doc_per_drum}/drum\n- Solidification: $0 (Zn is already solid metal)\nDrum specs: 7.35 ft^3, ~800-1200 lbs with shielding\nTransport limit: <200 mrem/hr at surface\n\nNIST DCE: Similar packaging costs\nShipping containers, DOT compliance\n\nSHINE: Hot cell cementation, standard drums\n\nES: Included in UCF'],
        ['Interim storage', 
         f'${storage_cost:,.0f}\n({storage_years:.1f} years)' if not no_interim_storage else '$0\n(no interim storage)',
         'Temporary storage\n(decay before shipment)\nShort-lived → quick decay', 
         'Included in UCF',
         f'OUR CASE (NUREG 1757 Tab 3.10):\n' + (f'{drums_volume_ft3:.1f} ft^3 × ${storage_cost_per_ft3_per_year}/ft^3/yr × {storage_years:.1f} yrs = ${storage_cost:,.0f}\nStorage period: {storage_years:.1f} years\n' if not no_interim_storage else 'No interim storage: irradiate then dispose of all waste.\n') + '(Zn-65 t½=244d)\nStorage rate: ${storage_cost_per_ft3_per_year}/ft^3/yr\n(Class A LLRW storage, mid-range)\n\nNIST DCE: $50-200/ft^3/year range\n\nSHINE: Weeks to months\n\nES: Included in UCF'],
        ['Transport', 
         f'${transport_cost:,.0f}\n(${transport_cost_per_drum:.0f}/drum)', 
         f'{shine_disposal_site}\n{shine_shipment_frequency}', 
         'Included in UCF',
         f'OUR CASE (NUREG 1757 Tab 3.14):\n{drums_needed:.0f} drums × ${transport_cost_per_drum:.0f}/drum = ${transport_cost:,.0f}\nDistance: {transport_distance_mi} mi (Wisconsin to Clive, UT)\n(NUREG-2183 Section 4.9.1: EnergySolutions, Clive, UT, 1,450 mi)\nAlternative: WCS (Andrews, TX, 1,305 mi) - NUREG-2183 Section 2.7.1.2\nCost: ${transport_cost_per_mile}/mi × {transport_distance_mi} mi = ${transport_cost_total:,.0f} total\nPer drum: ${transport_cost_per_drum:.0f}/drum (minimum $1,000/drum)\nIncludes: Trucking, permits, DOT compliance, fuel surcharges,\noverweight charges (drums ~800-1200 lbs with shielding)\n\nTransport cost per mile: ${transport_cost_per_mile}/mi\n(Industry average: $2.00-3.50/mile for DOT-regulated LLRW transport)\n\nNIST DCE: Ships to Oak Ridge, TN (525 mi)\nSimilar transport cost methodology\n\nSHINE: NUREG-2183 Section 4.9.1\nAnnual shipments to EnergySolutions\nClive, UT (1,450 mi)\n\nES: Included in UCF'],
        ['Class A disposal', 
         f'${class_a_disposal_cost:,.0f}\n(${disposal_cost_per_drum:.0f}/drum)', 
         'EnergySolutions\nNear-surface disposal\n10 CFR 61.55 Class A', 
         f'${es_reference_cost:,}\n(100 tons reference)',
         f'OUR CASE: {drums_needed:.1f} drums × ${disposal_cost_per_drum:.0f}/drum = ${class_a_disposal_cost:,.0f}. Class A LLRW (Zn-65).'],
    ]
    
    # Table 3: Taxes and total (no equipment, labor, contingency)
    additional_costs_data = [
        ['Cost Category', 'Our Case (Zn Waste)', 'SHINE (NUREG-2183)', 'EnergySolutions Reference', 'Calculation Details'],
        ['TAXES & TOTAL', '', '', '', ''],
        ['Taxes & Insurance', 
         f'${taxes_insurance:,.0f}\n(16% of base)', 
         'Included in operations', 
         'Included in UCF',
         f'Base: characterization + packaging + storage + transport + disposal. 16% taxes/insurance.'],
        ['TOTAL ESTIMATED COST', 
         f'${total_cost:,.0f}', 
         'Included in operations\n(proprietary)', 
         f'${es_reference_cost:,}\n(100 ton reference)',
         f'Characterization: ${total_characterization:,}; Packaging: ${packaging_cost:,.0f}; Storage: ${storage_cost:,.0f}; Transport: ${transport_cost:,.0f}; Disposal: ${class_a_disposal_cost:,.0f}; Taxes: ${taxes_insurance:,.0f}.'],
        ['Cost per ton', 
         f'${total_cost/max(zn_total_mass_tons,0.001):,.0f}/ton', 
         'N/A', 
         f'${es_cost_per_ton:,}/ton\n(estimated)',
         f'${total_cost:,.0f} ÷ {zn_total_mass_tons:.4f} tons'],
        ['Cost per kg', 
         f'${total_cost/(zn_total_mass_g/1000):,.0f}/kg', 
         'N/A', 
         f'${es_cost_per_ton*1000/2000:,.0f}/kg\n(estimated)',
         f'${total_cost:,.0f} ÷ {zn_total_mass_g/1000:.2f} kg'],
    ]
    
    # Table 4: Key Differences
    key_differences_data = [
        ['Cost Category', 'Our Case (Zn Waste)', 'SHINE (NUREG-2183)', 'EnergySolutions Reference', 'Calculation Details'],
        ['KEY DIFFERENCES', '', '', '', ''],
        ['Waste half-life', 
         'Zn-65: 244 days\n(Long-lived)', 
         'Mo-99: 66 hours\nLu-177: 6.65 days\n(Short-lived)', 
         'Variable',
         'OUR CASE: Zn-65 t½=244 days (NNDC)\nExceeds 10 CFR 35.92 limit (120 days)\nRequires authorized disposal\n\nSHINE: Short-lived isotopes\ndecay quickly before shipment\n\nES: Variable by waste type'],
        ['Storage requirement', 
         ('No interim storage\n(irradiate then dispose)' if no_interim_storage else f'{storage_years} years\n(before disposal)'),
         'Weeks to months\n(decay before shipment)', 
         'Variable',
         f'OUR CASE: ' + ('No interim storage. Irradiate then dispose of all waste.\n\n' if no_interim_storage else f'{storage_years} years (onsite storage until shipment)\n\n') + 'SHINE: Weeks to months\n(Short-lived waste)\n\nES: Variable'],
        ['Shielding requirement', 
         f'Facility: {shield_pb_facility:.1f} cm Pb\nTransport: {shield_pb_transport:.1f} cm Pb\n(1.116 MeV gamma)', 
         'Standard hot cell\n(<1 MeV gamma)', 
         'Variable',
         f'OUR CASE: Zn-65 gamma = 1.116 MeV (ICRP 107)\nHVL (Pb) = 1.4 cm (14 mm)\nDose const: 0.070992 microSv/hr/MBq at 1 m\nStorage/transport: 200 mrem/hr at surface (10 CFR 71.47): {shield_pb_facility:.1f} cm Pb\n\nSHINE: <1 MeV gamma\nStandard hot cell sufficient\n\nES: Variable by activity'],
        ['Shipment frequency', 
         ('Single shipment\n(after irradiation)' if no_interim_storage else 'After 5-10 yr decay\n(Long-term storage)'),
         shine_shipment_frequency, 
         'Variable',
         f'OUR CASE: ' + ('Single shipment after irradiation (no interim storage).\n\n' if no_interim_storage else f'After {storage_years}-10 years (wait for Zn-65 decay)\n\n') + 'SHINE: Annual shipments\n(NUREG-2183 Section 4.9.1)\n\nES: Variable'],
        ['Disposal pathway', 
         'EnergySolutions or WCS\nClass A LLRW', 
         shine_disposal_site, 
         'EnergySolutions\nWCS Texas',
         'OUR CASE: 10 CFR 61.55 Class A\nEnergySolutions (Clive, UT) or\nWCS (Andrews, TX)\n\nSHINE: Same pathway\nEnergySolutions, Clive, UT\n(NUREG-2183)\n\nES: Both facilities'],
    ]
    
    # Table 5: Sources
    sources_data = [
        ['Cost Category', 'Our Case (Zn Waste)', 'SHINE (NUREG-2183)', 'EnergySolutions Reference', 'Calculation Details'],
        ['SOURCES', '', '', '', ''],
        ['Our case', 
         (f"Fusion irradiation\n{scenario_irradiation_years:.0f} year production\n(no interim storage)" if no_interim_storage and scenario_irradiation_years
          else f"Fusion irradiation\n{scenario_irradiation_years or 1} year production"),
         'NUREG-2183 Section 4.9.1\nSHINE PSAR', 
         'EnergySolutions\nTABLE 1-2',
         'OUR CASE: From simulation statepoint\nFusion irradiation' + (f', {scenario_irradiation_years}-year production' if scenario_irradiation_years else ', 1 year production') + ('; no interim storage, dispose all.\n\n' if no_interim_storage else '\n\n') + 'SHINE: NUREG-2183 Section 4.9.1\nSHINE PSAR Section 4b.2\n\nES: EnergySolutions TABLE 1-2\nEstimated Material Disposal Quantities'],
        ['Cost methodology', 
         'NUREG 1757 format\n(no contingency, labor, equipment)', 
         'NUREG-2183\nSHINE PSAR', 
         'EnergySolutions UCFs',
         'OUR CASE: NUREG 1757 format. Characterization, packaging, storage, transport, disposal, taxes only.\n\nSHINE/ES: Included in operations/UCF.'],
        ['Regulatory', 
         '10 CFR 61.55\nClass A LLRW', 
         '10 CFR Part 20\nWisconsin DHS 157', 
         '10 CFR 61.55',
         'OUR CASE: 10 CFR 61.55\nLLRW Classification\nClass A disposal requirements\n\nSHINE: 10 CFR Part 20\nWisconsin Agreement State (DHS 157)\n\nES: 10 CFR 61.55'],
        ['Disposal facility', 
         'EnergySolutions (Clive, UT)\nWCS (Andrews, TX)', 
         shine_disposal_site, 
         'EnergySolutions\nWCS Texas',
         'OUR CASE: EnergySolutions (Clive, UT, 1,450 mi)\nor WCS (Andrews, TX, 1,305 mi)\n\nSHINE: EnergySolutions\nClive, UT (1,450 mi)\n(NUREG-2183)\n\nES: Both facilities available'],
    ]

    def short_calc(s):
        if not s or len(str(s)) <= 70:
            return s
        return (str(s)[:67] + '...') if len(str(s)) > 70 else s

    combined_cost_data = [waste_quantities_data[0]]
    for block in (waste_quantities_data, cost_breakdown_data, additional_costs_data, key_differences_data, sources_data):
        for row in block[1:]:
            r = list(row)
            if len(r) >= 5:
                r[4] = short_calc(r[4])
            combined_cost_data.append(r)
    pd.DataFrame(combined_cost_data[1:], columns=combined_cost_data[0]).to_csv(
        os.path.join(output_dir, 'cost_breakdown_table.csv'), index=False)
    print(f"Saved: {output_dir}/cost_breakdown_table.csv")
    # Intentionally do not write additional cost CSVs/plots beyond the quote package outputs.
    return


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Zn-65 waste: drum fill by transport/storage limits or volume-only; 10 mSv/h at 3 m; summary table and costs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zn_waste.py --case "0 5 0.5 1 1 1 49.2%%" --flux-base .
  python zn_waste.py --case "0 20 0 5 5 48.6%%"
  python zn_waste.py --from-dir irrad_output_inner0_outer5_struct0.5_boron1_multi1_moderator1_zn49.2%%
  python zn_waste.py --case "0 5 0.5 1 1 1 49.2%%" --irrad-hours 8 --cooldown-days 1
  python zn_waste.py --zn65-activity-Bq 1e12 --zn65-mass-g 1000
        """
    )
    parser.add_argument("--from-dir", type=str, default=None, help="Path to simulation output directory")
    parser.add_argument("--case", type=str, default=None, help="Geometry: 'INNER OUTER STRUCT [BORON] MULTI MOD ENRICH%%' (6 or 7 values)")
    parser.add_argument("--flux-base", type=str, default=".", help="Base directory when using --case")
    parser.add_argument("--output-prefix", type=str, default="irrad_output", help="Output directory prefix")
    parser.add_argument("--irrad-hours", type=float, default=8760, help="Irradiation time in hours")
    parser.add_argument("--cooldown-days", type=float, default=0, help="Cooldown time in days")
    parser.add_argument("--zn65-activity-Bq", type=float, default=None, help="Zn-65 activity in Bq (direct input)")
    parser.add_argument("--zn65-mass-g", type=float, default=1000, help="Mass of Zn target in grams")
    parser.add_argument("--output-dir", type=str, default="waste_analysis", help="Output directory for plots")
    # Cost scenario: 8-year irrad, then dispose, no interim storage
    parser.add_argument("--no-interim-storage", action="store_true", help="No interim storage (irradiate then dispose all waste)")
    parser.add_argument("--cost-irradiation-years", type=float, default=None, help="Scenario label, e.g. 8 for 8-year irradiation")
    parser.add_argument("--cost-shield-cm", type=float, default=3.0, choices=[3.0, 4.0], help="Drum shielding: 3 or 4 cm Pb for cost/drum count")
    parser.add_argument("--cost-packaging-per-drum", type=float, default=None, help="Override packaging $/drum (e.g. 884)")
    parser.add_argument("--cost-transport-per-drum", type=float, default=None, help="Override transport $/drum")
    parser.add_argument("--cost-disposal-per-drum", type=float, default=None, help="Override Class A disposal $/drum (e.g. 7125)")
    args = parser.parse_args()
    if not args.case and not args.from_dir and args.zn65_activity_Bq is None:
        parser.error("Either --case, --from-dir, or --zn65-activity-Bq is required")
    analyzer = ZnWasteAnalyzer(
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        irrad_hours=args.irrad_hours,
        cooldown_days=args.cooldown_days,
    )
    analyzer.run(
        from_dir=args.from_dir,
        case=args.case,
        flux_base=args.flux_base,
        zn65_activity_Bq=args.zn65_activity_Bq,
        zn65_mass_g=args.zn65_mass_g,
        cost_no_interim_storage=args.no_interim_storage,
        cost_irradiation_years=args.cost_irradiation_years,
        cost_shield_cm=args.cost_shield_cm,
        cost_packaging_per_drum=args.cost_packaging_per_drum,
        cost_transport_per_drum=args.cost_transport_per_drum,
        cost_disposal_per_drum=args.cost_disposal_per_drum,
    )


if __name__ == "__main__":

    
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        print("Error: No arguments provided")
