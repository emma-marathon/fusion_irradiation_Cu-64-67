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
from matplotlib.ticker import FuncFormatter
# Disable LaTeX (avoids FileNotFoundError when latex not installed)
plt.rcParams['text.usetex'] = False
import openmc
import openmc.data

from utilities import (
    build_channel_rr_per_s,
    parse_dir_name,
    get_zn64_enrichment_cost_per_kg,
    ZN64_COST_ANCHOR_X,
    ZN64_ENRICHMENT_MAP,
    ZN64_ENRICHMENT_COST,
    ZN67_ENRICHMENT_MAP,
    ZN67_ENRICHMENT_COST,
    evolve_bateman_irradiation,
    apply_single_decay_step,
    compute_volumes_from_dir_name,
    get_material_density_from_statepoint,
    get_initial_atoms_from_statepoint,
    get_volumetric_heating_w_cm3,
    compute_outer_surface_area_cm2_from_params,
    compute_inner_surface_area_cm2_from_params,
    get_decay_constant,
    SOURCE_STRENGTH,
    SECONDS_PER_YEAR,
    N_A,
    A_CU64_G_MOL,
    A_CU67_G_MOL,
    LAMBDA_CU64_S,
    LAMBDA_CU67_S,
    npv_from_cu_summary_row,
    annuity_factor,
    specific_activity_ci_per_g,
    print_specific_activities,
    HOURS_PER_YEAR,
)

# ============================================
# Configuration
# ============================================
OUTER_MATERIAL_ID = 1

OUTPUT_PREFIX = 'irrad_output'  # Set to None to search all output types

# Analysis parameters (overridden from run_config when run via run.py)
IRRADIATION_HOURS = [1, 2, 4, 8, 12, 16, 24, 48, 72, 98, 100, 138, 8760]  # 98 h for purity shelf-life; 16 for 8/12/16h plot; 8760 = 1 year
COOLDOWN_DAYS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7]  # 1.5 for 8h-by-cooldown plot; extended for purity fall-to-99.9%
TARGET_HEIGHT_CM = 100.0  # chamber height; set from run_config.TARGET_HEIGHT_CM when using dashboard

# Use multi-timestep evolution (rate scaling) when irradiation >= this many hours
IRRAD_MULTI_STEP_THRESHOLD_H = 72

# Isotopes of interest (Cu63, Cu65 are stable; Cu64, Cu67 are radioactive)
CU_ISOTOPES = ['Cu64', 'Cu67']
CU_ISOTOPES_ALL = ['Cu63', 'Cu64', 'Cu65', 'Cu67']  # all Cu from Zn(n,p) etc. for total mass
# All Cu isotopes for CSV (grams after irrad + cooldown); total Cu mass = sum of all
CU_ISOTOPES_CSV = ['Cu61', 'Cu62', 'Cu63', 'Cu64', 'Cu65', 'Cu66', 'Cu67', 'Cu68', 'Cu69', 'Cu70']
# Copper atomic masses [g/mol] for total copper mass (stable + radioactive)
CU_ATOMIC_MASS_G_MOL = {'Cu63': 62.9296, 'Cu64': 63.9298, 'Cu65': 64.9278, 'Cu67': 66.9277}
CU_ATOMIC_MASS_G_MOL_FULL = {
    'Cu61': 60.966, 'Cu62': 61.963, 'Cu63': 62.9296, 'Cu64': 63.9298,
    'Cu65': 64.9278, 'Cu66': 65.9289, 'Cu67': 66.9277, 'Cu68': 67.928,
    'Cu69': 68.9256, 'Cu70': 69.9254,
}
ZN_ISOTOPES = ['Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']

ECON_CU64_PRICE_PER_MCI = 15.0
ECON_CU67_PRICE_PER_MCI = 60.0
ECON_SHINE_SOURCE_PURCHASE = 5_000_000.0      # FLARE one-time cost [USD]
ECON_SHINE_SOURCE_OPERATIONS_ANNUAL = 1_000_000.0  # OPEX [USD/yr]; revenue/NPV use this
NPV_DISCOUNT_RATE = 0.1   # r = 10% for present value of cash flows over operating period
ECON_IRRAD_HOURS = 8760
ECON_COOLDOWN_DAYS = 1
ECON_MARKET_CU64_CI_PER_YEAR = 2000.0

# Known Zn-64 enrichments for normalization (handles float precision)
_ZN64_KNOWN = sorted(ZN64_ENRICHMENT_MAP.keys())
_TOL = 0.0005


def _norm_enrich(e):
    """Normalize enrichment to nearest known value (handles 0.9989999→0.999)."""
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return e
    e = float(e)
    for k in _ZN64_KNOWN:
        if abs(e - k) <= _TOL:
            return k
    # Not in map; round to nearest known or keep as-is for unknown enrichments
    best = min(_ZN64_KNOWN, key=lambda k: abs(e - k))
    return best if abs(e - best) <= 0.01 else e


def _disp_cm(v):
    """Display thickness in results: show 0.5 cm when value is 0 (for outer/struct/boron only; multi/mod show 0 as 0)."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return v
    return 0.5 if x == 0 else x


def _geom_legend_labels(outer, boron, multi, mod):
    """Labels for geometry legend: outer/boron use _disp_cm (0→0.5); multi/mod show raw so 0 stays 0."""
    o, b = _disp_cm(outer), _disp_cm(boron)
    m, md = float(multi), float(mod)
    return o, b, m, md


def _geom_legend_str(outer, boron, multi, mod):
    """Geometry legend label; omit multi/mod when both are 0."""
    o, b, m, md = _geom_legend_labels(outer, boron, multi, mod)
    base = f'Outer={o:.1f}cm, B={b:.1f}cm'
    if m != 0 or md != 0:
        base += f', Multi={m:.1f}cm, Mod={md:.1f}cm'
    return base


def _enrich_label(e):
    """Format enrichment for legend/labels: 0.999 and 1.0→'99.9%', 0.99→'99%'; one decimal where needed (e.g. 99.9 not 99.90)."""
    e = _norm_enrich(e) if e is not None else e
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return '?'
    v = float(e)
    if abs(v - 0.999) < 0.0005 or abs(v - 1.0) < 0.005:
        return '99.9%'
    if abs(v - 0.99) < 0.005:
        return '99%'
    pct = v * 100
    # One decimal for .9; otherwise strip trailing zeros
    if abs(pct - round(pct, 1)) < 0.01 and round(pct, 1) != round(pct, 0):
        s = f'{pct:.1f}'
    else:
        s = f'{pct:.2f}'.rstrip('0').rstrip('.')
    return f'{s}%'


def _enrichment_axis_label(e):
    """Format enrichment for heatmap y-axis: percent only (0.999 → 99.9%)."""
    return _enrich_label(float(e))


def _get_run_config_paths():
    """Resolve base_dir, output_dir, output_prefix from run_config when available."""
    try:
        import run_config as _c
        base = os.path.abspath(getattr(_c, 'RUN_BASE_DIR', os.getcwd()))
        statepoints = os.path.join(base, getattr(_c, 'STATEPOINTS_DIR', 'statepoints'))
        analyze = os.path.join(base, getattr(_c, 'ANALYZE_DIR', getattr(_c, 'RESULTS_DIR', 'analyze')))
        prefix = getattr(_c, 'OUTPUT_PREFIX', 'irrad_output')
        return statepoints, analyze, prefix
    except ImportError:
        return os.getcwd(), os.path.join(os.getcwd(), 'analyze'), 'irrad_output'


class IrradiationAnalyzer:
    """Runs simple analysis: find statepoints, analyze cases, build summaries, generate plots.
    base_dir, output_dir, output_prefix come from run_config when not passed.
    outer_material_id: 1 = outer (Cu-64/Zn-64 or Cu-67/Zn-67), 0 = inner (dual, Zn-67 only).
    """

    def __init__(self, base_dir=None, output_dir=None, output_prefix=None, pattern=None,
                 outer_material_id=1):
        _statepoints, _results, _prefix = _get_run_config_paths()
        self.base_dir = base_dir if base_dir is not None else _statepoints
        self.output_dir = output_dir if output_dir is not None else _results
        self.output_prefix = output_prefix if output_prefix is not None else _prefix
        self.pattern = pattern or (f'{self.output_prefix}_*' if self.output_prefix else '*')
        self.outer_material_id = int(outer_material_id)  # 1 = outer, 0 = inner (dual)

    def find_statepoints(self, allowed_dirs=None):
        """Return list of statepoint file paths in dirs matching self.pattern.
        If allowed_dirs is set (list of dir basenames), only include statepoints from those dirs."""
        statepoints = []
        seen = set()
        for d in glob.glob(os.path.join(self.base_dir, self.pattern)):
            if not os.path.isdir(d) or d in seen:
                continue
            if allowed_dirs is not None and os.path.basename(d) not in allowed_dirs:
                continue
            seen.add(d)
            sp_files = glob.glob(os.path.join(d, 'statepoint.*.h5'))
            if sp_files:
                statepoints.append(sorted(sp_files)[-1])
        return statepoints

    def run(self, aggregate_to=None, per_case_to=None, save_aggregate=True, save_per_case=True,
            layout='flat', allowed_dirs=None):
        """Find cases, build dataframes, save CSVs and plots.

        Parameters
        ----------
        aggregate_to : str or None
            Dir for aggregate files. If None, uses self.output_dir.
        per_case_to : str or None
            Parent dir for per-case folders (case_name/). If None, uses self.output_dir.
        save_aggregate : bool
            If True, write cu_summary_all, zn_summary_all, production_vs_purity, zn65_bar_by_geometry, etc.
        save_per_case : bool
            If True, write per-case files (cu_summary, activity_vs_variables, etc) under per_case_to/dir_name/.
        layout : str
            'flat' (default): legacy layout (agg in agg_dir, per-case in per_dir/dir_name).
            'geometry': write per-geometry folders under agg_dir, with per-enrichment subfolders.
        """
        agg_dir = aggregate_to if aggregate_to is not None else self.output_dir
        per_dir = per_case_to if per_case_to is not None else self.output_dir
        if save_aggregate:
            os.makedirs(agg_dir, exist_ok=True)
        if save_per_case:
            os.makedirs(per_dir, exist_ok=True)
        print(f"Finding statepoints in {self.base_dir} (pattern: {self.pattern})...")
        statepoints = self.find_statepoints(allowed_dirs=allowed_dirs)
        if not statepoints:
            print(f"No statepoint files found in {self.base_dir}")
            return
        print(f"Found {len(statepoints)} statepoint files")
        print("\nAnalyzing cases...")
        cases = []
        for sp_file in statepoints:
            print(f"\nAnalyzing: {sp_file}")
            try:
                case = analyze_case(sp_file, outer_material_id=self.outer_material_id)
                cases.append(case)
                rr = case['reaction_rates']
                cu64_rate = rr.get("Zn64 (n,p) Cu64", 0)
                cu67_np = rr.get("Zn67 (n,p) Cu67", 0)
                cu67_nd = rr.get("Zn68 (n,d) Cu67", 0)
                zn65_rate = (rr.get("Zn64 (n,gamma) Zn65", 0) or 0) + (rr.get("Zn66 (n,2n) Zn65", 0) or 0)
                print(f"  {case['dir_name']}: Cu64={cu64_rate:.2e}, Cu67={(cu67_np+cu67_nd):.2e}, Zn65={zn65_rate:.2e} atoms/s")
            except Exception as e:
                print(f"  Error: {e}")
        if not cases:
            print("No valid cases found!")
            return
        print("\nBuilding summary tables...")
        cu_df, zn_df = build_summary_dataframes(cases)
        print_specific_activities()

        # Geometry/enrichment layout (requested: outer/<geometry>/<enrichment>/...)
        if layout == 'geometry' and save_aggregate:
            # Always write a top-level aggregate for reference
            cu_df.to_csv(os.path.join(agg_dir, 'cu_summary_all.csv'), index=False)
            zn_df.to_csv(os.path.join(agg_dir, 'zn_summary_all.csv'), index=False)
            print(f"  Saved: {agg_dir}/cu_summary_all.csv ({len(cu_df)} rows)")
            print(f"  Saved: {agg_dir}/zn_summary_all.csv ({len(zn_df)} rows)")
            chamber_label = 'inner' if self.outer_material_id == 0 else 'outer'
            plot_production_vs_purity(cu_df, agg_dir, chamber_label=chamber_label)
            if chamber_label == 'outer':
                plot_production_vs_purity_8h_only(cu_df, agg_dir, chamber_label=chamber_label)
                plot_production_vs_purity_by_irradiation(cu_df, agg_dir, chamber_label=chamber_label, irrad_hours=(1, 4, 8))
                plot_production_vs_purity_by_irradiation(cu_df, agg_dir, chamber_label=chamber_label, irrad_hours=(8, 12, 16))
                plot_production_vs_purity_8h_by_cooldown(cu_df, agg_dir, chamber_label=chamber_label, cooldown_days=(0, 0.5, 1, 1.5, 2))
                plot_production_vs_purity_8h_two_cooldowns(cu_df, agg_dir, chamber_label=chamber_label)
                plot_cu64_production_vs_cooldown_by_irradiation_outer10_99pct(
                    cu_df,
                    agg_dir,
                    chamber_label=chamber_label,
                    irrad_hours=(8, 16),
                    cooldown_days=(0.5, 1.0, 1.5, 2.0),
                    purity_cut=0.999,
                )
            plot_production_vs_time_to_999_purity(cu_df, agg_dir, chamber_label=chamber_label)
            plot_production_vs_atomic_impurity(cu_df, agg_dir, chamber_label=chamber_label)
            plot_enrichment_cost_and_fractions(agg_dir)

            # Group cases by geometry (ignore enrichment)
            cases_by_dir = {c['dir_name']: c for c in cases}
            df0 = _ensure_geom_columns(cu_df.copy())
            # One row per dir_name with geometry columns
            key_cols = ['dir_name', 'chamber', 'inner_cm', 'outer_cm', 'struct_cm', 'boron_cm', 'multi_cm', 'mod_cm', 'use_zn67']
            geom_df = df0[key_cols].drop_duplicates(subset=['dir_name']).copy()

            def _fmt_num(x):
                try:
                    x = float(x)
                except Exception:
                    return str(x)
                if abs(x - round(x)) < 1e-9:
                    return str(int(round(x)))
                return f"{x:.6g}"

            def _geom_folder(row):
                prefix = self.output_prefix or 'irrad_output'
                chamber = str(row.get('chamber', 'single_cu64'))
                zn_tag = 'zn67' if bool(row.get('use_zn67', False)) else 'zn64'
                return (
                    f"{prefix}_{chamber}"
                    f"_inner{_fmt_num(row.get('inner_cm', 0))}"
                    f"_outer{_fmt_num(row.get('outer_cm', 0))}"
                    f"_struct{_fmt_num(row.get('struct_cm', 0))}"
                    f"_boron{_fmt_num(row.get('boron_cm', 0))}"
                    f"_multi{_fmt_num(row.get('multi_cm', 0))}"
                    f"_moderator{_fmt_num(row.get('mod_cm', 0))}"
                    f"_{zn_tag}"
                )

            def _enrich_folder(enrich):
                try:
                    pct = float(enrich) * 100.0
                    return f"enrich_{pct:.2f}pct".replace('.', 'p')
                except Exception:
                    return f"enrich_{enrich}"

            group_cols = ['chamber', 'inner_cm', 'outer_cm', 'struct_cm', 'boron_cm', 'multi_cm', 'mod_cm', 'use_zn67']
            for _, g in geom_df.groupby(group_cols, dropna=False):
                row0 = g.iloc[0].to_dict()
                geom_name = _geom_folder(row0)
                geom_dir = os.path.join(agg_dir, geom_name)
                os.makedirs(geom_dir, exist_ok=True)

                dir_names = g['dir_name'].tolist()
                cu_g = cu_df[cu_df['dir_name'].isin(dir_names)].copy()
                zn_g = zn_df[zn_df['dir_name'].isin(dir_names)].copy()

                cu_g.to_csv(os.path.join(geom_dir, 'cu_summary_all.csv'), index=False)
                # zn_summary_all is written only at agg_dir (one location), not per-geometry

                geom_info = geom_name
                plot_activity_vs_variables(cu_g, geom_dir, geom_info=geom_info)
                plot_purity_vs_variables(cu_g, geom_dir, geom_info=geom_info)
                plot_production_vs_purity(cu_g, geom_dir, chamber_label=chamber_label)
                if chamber_label == 'outer':
                    plot_production_vs_purity_8h_only(cu_g, geom_dir, chamber_label=chamber_label)
                    plot_production_vs_purity_by_irradiation(cu_g, geom_dir, chamber_label=chamber_label, irrad_hours=(1, 4, 8))
                    plot_production_vs_purity_by_irradiation(cu_g, geom_dir, chamber_label=chamber_label, irrad_hours=(8, 12, 16))
                    plot_production_vs_purity_8h_by_cooldown(cu_g, geom_dir, chamber_label=chamber_label, cooldown_days=(0, 0.5, 1, 1.5, 2))
                    plot_production_vs_purity_8h_two_cooldowns(cu_g, geom_dir, chamber_label=chamber_label)
                plot_production_vs_time_to_999_purity(cu_g, geom_dir, chamber_label=chamber_label)
                plot_production_vs_atomic_impurity(cu_g, geom_dir, chamber_label=chamber_label)

                # Per-enrichment folders (case-by-case): cu/zn summaries and stable paths for downstream tools
                for dn in dir_names:
                    cu_case = cu_g[cu_g['dir_name'] == dn].copy()
                    zn_case = zn_g[zn_g['dir_name'] == dn].copy()
                    enrich_val = cu_case['zn64_enrichment'].iloc[0] if not cu_case.empty else None
                    e_dir = os.path.join(geom_dir, _enrich_folder(enrich_val))
                    os.makedirs(e_dir, exist_ok=True)
                    cu_case.to_csv(os.path.join(e_dir, 'cu_summary.csv'), index=False)
                    zn_case.to_csv(os.path.join(e_dir, 'zn_summary.csv'), index=False)
                    # Also keep a stable per-case location for downstream tools (e.g. zn_waste)
                    by_case_dir = os.path.join(agg_dir, '_by_case', dn)
                    os.makedirs(by_case_dir, exist_ok=True)
                    cu_case.to_csv(os.path.join(by_case_dir, 'cu_summary.csv'), index=False)
                    zn_case.to_csv(os.path.join(by_case_dir, 'zn_summary.csv'), index=False)

            print("\nDone!")
            return

        # Aggregate files (all cases together)
        if save_aggregate:
            cu_df.to_csv(os.path.join(agg_dir, 'cu_summary_all.csv'), index=False)
            zn_df.to_csv(os.path.join(agg_dir, 'zn_summary_all.csv'), index=False)
            print(f"  Saved: {agg_dir}/cu_summary_all.csv ({len(cu_df)} rows)")
            print(f"  Saved: {agg_dir}/zn_summary_all.csv ({len(zn_df)} rows)")
            chamber_label = 'inner' if self.outer_material_id == 0 else 'outer'
            plot_production_vs_purity(cu_df, agg_dir, chamber_label=chamber_label)
            plot_production_vs_time_to_999_purity(cu_df, agg_dir, chamber_label=chamber_label)
            plot_production_vs_atomic_impurity(cu_df, agg_dir, chamber_label=chamber_label)
            plot_enrichment_cost_and_fractions(agg_dir)
        # Per-case files
        if not save_per_case:
            print("\nDone!")
            return
        case_dirs = sorted(cu_df['dir_name'].unique().tolist())
        cases_by_dir = {c['dir_name']: c for c in cases}
        print(f"\nGenerating per-case results for {len(case_dirs)} cases...")
        for dir_name in case_dirs:
            case_dir = os.path.join(per_dir, dir_name)
            os.makedirs(case_dir, exist_ok=True)
            cu_case = cu_df[cu_df['dir_name'] == dir_name].copy()
            zn_case = zn_df[zn_df['dir_name'] == dir_name].copy()
            cu_case.to_csv(os.path.join(case_dir, 'cu_summary.csv'), index=False)
            zn_case.to_csv(os.path.join(case_dir, 'zn_summary.csv'), index=False)
            print(f"\n  Case: {dir_name} -> {case_dir}")
            plot_activity_vs_variables(cu_case, case_dir)
            plot_purity_vs_variables(cu_case, case_dir, geom_info=dir_name, chamber_label=chamber_label)
        print("\nDone!")


def analyze_case(sp_file, outer_material_id=None):
    """
    Analyze a single simulation case for one Zn chamber (material_id 0=inner, 1=outer).
    Initial atoms and density from statepoint; reaction rates from tallies.
    """
    if outer_material_id is None:
        outer_material_id = OUTER_MATERIAL_ID
    dir_name = os.path.basename(os.path.dirname(sp_file))
    params = parse_dir_name(dir_name)
    volumes = compute_volumes_from_dir_name(dir_name, target_height=TARGET_HEIGHT_CM)
    volume_cm3 = volumes.get(outer_material_id, 188495.56 if outer_material_id == 1 else 1.0)

    sp = openmc.StatePoint(sp_file)
    initial_atoms = get_initial_atoms_from_statepoint(sp_file, outer_material_id, volume_cm3)
    if initial_atoms is None:
        raise RuntimeError(f"Cannot get initial atoms from statepoint for {sp_file}")

    zn_density = get_material_density_from_statepoint(sp_file, outer_material_id)
    if zn_density is None:
        raise RuntimeError(f"Cannot get density from statepoint for {sp_file}")

    rr = build_channel_rr_per_s(sp, cell_id=outer_material_id, source_strength=SOURCE_STRENGTH)
    volumetric_heating_W_cm3 = get_volumetric_heating_w_cm3(sp, outer_material_id, SOURCE_STRENGTH, volume_cm3)
    if outer_material_id == 1:
        surface_area_cm2 = compute_outer_surface_area_cm2_from_params(
            params['inner'], params['outer'], params['struct'],
            params['multi'], params['moderator'], target_height=TARGET_HEIGHT_CM)
    else:
        surface_area_cm2 = compute_inner_surface_area_cm2_from_params(
            params['inner'], params['outer'], params['struct'],
            params['multi'], params['moderator'], target_height=TARGET_HEIGHT_CM)

    enrichment = params.get('zn67_enrichment_inner') if outer_material_id == 0 else params['zn_enrichment']
    if enrichment is None:
        enrichment = params['zn_enrichment']
    return {
        'dir_name': dir_name,
        'sp_file': sp_file,
        'material_id': outer_material_id,
        'zn64_enrichment': enrichment,
        'use_zn67': params.get('use_zn67', False) or (outer_material_id == 0 and params.get('zn67_enrichment_inner') is not None),
        'inner_cm': params['inner'],
        'outer_cm': params['outer'],
        'struct_cm': params['struct'],
        'boron_cm': params.get('boron', 0),
        'multi_cm': params['multi'],
        'moderator_cm': params['moderator'],
        'chamber': params.get('chamber', 'single_cu64'),
        'outer_volume_cm3': volume_cm3,
        'zn_density_g_cm3': zn_density,
        'zn_mass_g': volume_cm3 * zn_density,
        'initial_atoms': initial_atoms,
        'reaction_rates': rr,
        'volumetric_heating_W_cm3': volumetric_heating_W_cm3,
        'surface_area_cm2': surface_area_cm2,
    }


def compute_activities(case, irrad_hours, cooldown_days):
    """
    Apply evolve_bateman_irradiation (utilities) + decay. For long irradiations
    (>= threshold), chain multiple steps and recalculate reaction rates each
    step: R(t) = R_0 * (N_parent_current / N_parent_initial) as Zn-64/Zn-66
    deplete, Zn-65 depletes/decays, Cu is produced.
    Tracks all Cu from Zn(n,p) etc.: Cu63, Cu64, Cu65, Cu67. Total copper mass
    (stable + radioactive) is used to compute effective specific activity (Ci/g)
    = activity / total_cu_mass_g (carrier-added basis).
    """
    irrad_s = irrad_hours * 3600
    cooldown_s = cooldown_days * 86400
    init = case['initial_atoms']
    rr0 = case['reaction_rates']

    if irrad_hours >= IRRAD_MULTI_STEP_THRESHOLD_H:
        n_steps = max(52, min(365, int(irrad_s / 86400)))
        dt_s = irrad_s / n_steps
        atoms = {k: float(v) for k, v in init.items()}
        for _ in range(n_steps):
            # Recalculate rr: R = R_0 * (N_parent / N_parent_initial)
            rr = {}
            for key, R0 in rr0.items():
                R0 = 0.0 if R0 is None else float(np.asarray(R0).flat[0])
                if R0 <= 0:
                    rr[key] = 0.0
                    continue
                parent = key.split()[0]
                n_init = float(init.get(parent, 0.0))
                n_curr = float(atoms.get(parent, 0.0))
                rr[key] = R0 * (n_curr / n_init) if n_init > 0 else 0.0
            atoms = evolve_bateman_irradiation(atoms, rr, dt_s)
        atoms_eoi = atoms
    else:
        atoms_eoi = evolve_bateman_irradiation(init, rr0, irrad_s)
    atoms_final = apply_single_decay_step(atoms_eoi, cooldown_s)

    lam_cu64 = get_decay_constant('Cu64')
    lam_cu67 = get_decay_constant('Cu67')
    lam_zn65 = get_decay_constant('Zn65')
    lam_zn69m = get_decay_constant('Zn69m')

    cu63_atoms = atoms_final.get('Cu63', 0)
    cu64_atoms = atoms_final.get('Cu64', 0)
    cu65_atoms = atoms_final.get('Cu65', 0)
    cu67_atoms = atoms_final.get('Cu67', 0)
    zn65_atoms = atoms_final.get('Zn65', 0)
    zn69m_atoms = atoms_final.get('Zn69m', 0)
    total_cu = cu64_atoms + cu67_atoms  # radioactive Cu for purity
    # Grams of each Cu isotope after irrad + cooldown; total Cu mass = sum of all nuclides
    cu_grams = {}
    for iso in CU_ISOTOPES_CSV:
        n_atoms = float(atoms_final.get(iso, 0) or 0)
        cu_grams[f'{iso.lower()}_g'] = n_atoms * CU_ATOMIC_MASS_G_MOL_FULL[iso] / N_A
    total_cu_mass_g = sum(cu_grams.values())
    if total_cu_mass_g < 1e-20:
        total_cu_mass_g = 0.0

    # Calculate activities
    cu64_activity = cu64_atoms * lam_cu64
    cu67_activity = cu67_atoms * lam_cu67
    total_cu_activity = cu64_activity + cu67_activity

    # Effective specific activity (Ci/g): activity per gram of total copper in sample (carrier-added)
    BQ_PER_CI = 3.7e10
    if total_cu_mass_g > 0:
        cu64_specific_activity_Ci_per_g = (cu64_activity / BQ_PER_CI) / total_cu_mass_g
        cu67_specific_activity_Ci_per_g = (cu67_activity / BQ_PER_CI) / total_cu_mass_g
    else:
        cu64_specific_activity_Ci_per_g = specific_activity_ci_per_g("64")
        cu67_specific_activity_Ci_per_g = specific_activity_ci_per_g("67")

    # Atomic purity (atoms ratio) — explicit float, same computation for all enrichments
    cu64_atomic_purity = float(cu64_atoms / total_cu) if total_cu > 0 else 0.0
    cu67_atomic_purity = float(cu67_atoms / total_cu) if total_cu > 0 else 0.0

    # Radionuclide purity (activity ratio)
    cu64_radionuclide_purity = float(cu64_activity / total_cu_activity) if total_cu_activity > 0 else 0.0
    cu67_radionuclide_purity = float(cu67_activity / total_cu_activity) if total_cu_activity > 0 else 0.0

    return {
        'cu64_mCi': cu64_atoms * lam_cu64 / 3.7e7,
        'cu67_mCi': cu67_atoms * lam_cu67 / 3.7e7,
        'zn65_mCi': zn65_atoms * lam_zn65 / 3.7e7,
        'cu64_Bq': cu64_activity,
        'cu67_Bq': cu67_activity,
        'zn65_Bq': zn65_atoms * lam_zn65,
        'zn69m_Bq': zn69m_atoms * lam_zn69m,
        'cu64_atomic_purity': cu64_atomic_purity,
        'cu67_atomic_purity': cu67_atomic_purity,
        'cu64_radionuclide_purity': cu64_radionuclide_purity,
        'cu67_radionuclide_purity': cu67_radionuclide_purity,
        'cu64_atoms': cu64_atoms,
        'cu67_atoms': cu67_atoms,
        'cu63_atoms': cu63_atoms,
        'cu65_atoms': cu65_atoms,
        'zn65_atoms': zn65_atoms,
        'total_cu_mass_g': total_cu_mass_g,
        'cu64_specific_activity_Ci_per_g': cu64_specific_activity_Ci_per_g,
        'cu67_specific_activity_Ci_per_g': cu67_specific_activity_Ci_per_g,
        **cu_grams,
    }


def build_summary_dataframes(cases):
    """
    Build Cu and Zn summary DataFrames for all cases, irradiation times, and cooldown times.
    Cu summary includes cu64_g_yr, cu67_g_yr (average production g/yr) and npv_millions (NPV in USD millions).
    NPV uses discount rate r=0.1, T=8 years, FLARE (one-time) + feedstock, OPEX $/yr; Cu64 revenue only if r-purity ≥99.9%.
    """
    cu_rows = []
    zn_rows = []
    # Price $/g and economic params from run_config or module defaults (8-year model, r=0.1, FLARE + OPEX)
    try:
        import run_config as _rc
        p64_mci = getattr(_rc, "ECON_CU64_PRICE_PER_MCI", ECON_CU64_PRICE_PER_MCI)
        p67_mci = getattr(_rc, "ECON_CU67_PRICE_PER_MCI", ECON_CU67_PRICE_PER_MCI)
        flare_usd = getattr(_rc, "ECON_SHINE_SOURCE_PURCHASE", ECON_SHINE_SOURCE_PURCHASE)
        opex_yr = getattr(_rc, "ECON_SHINE_SOURCE_OPERATIONS_ANNUAL", ECON_SHINE_SOURCE_OPERATIONS_ANNUAL)
    except ImportError:
        p64_mci = ECON_CU64_PRICE_PER_MCI
        p67_mci = ECON_CU67_PRICE_PER_MCI
        flare_usd = ECON_SHINE_SOURCE_PURCHASE
        opex_yr = ECON_SHINE_SOURCE_OPERATIONS_ANNUAL
    price_cu64_usd_per_g = float(p64_mci) * 1000.0 * specific_activity_ci_per_g("64")
    price_cu67_usd_per_g = float(p67_mci) * 1000.0 * specific_activity_ci_per_g("67")

    for case in cases:
        # Get Zn volume, density, and mass from case (from statepoint)
        volume_cm3 = case['outer_volume_cm3']
        zn_density = case['zn_density_g_cm3']
        mass_g = case['zn_mass_g']
        mass_kg = mass_g / 1000.0
        enrich = _norm_enrich(case['zn64_enrichment'])  # Normalize 0.9989999→0.999, etc.

        for irrad_h in IRRADIATION_HOURS:
            for cool_d in COOLDOWN_DAYS:
                act = compute_activities(case, irrad_h, cool_d)
                irrad_s = max(irrad_h * 3600.0, 1.0)
                cu64_g_yr = (float(act['cu64_Bq']) * A_CU64_G_MOL * SECONDS_PER_YEAR /
                             (LAMBDA_CU64_S * irrad_s * N_A))
                cu67_g_yr = (float(act['cu67_Bq']) * A_CU67_G_MOL * SECONDS_PER_YEAR /
                             (LAMBDA_CU67_S * irrad_s * N_A))

                zn_cost_per_kg = get_zn64_enrichment_cost_per_kg(enrich)
                loading = zn_cost_per_kg * mass_kg
                row = {
                    'dir_name': case['dir_name'],
                    'zn64_enrichment': enrich,
                    'zn_feedstock_cost': loading,
                    'use_zn67': case.get('use_zn67', False),
                    'inner_cm': case.get('inner_cm', 0),
                    'outer_cm': case.get('outer_cm', 20),
                    'struct_cm': case.get('struct_cm', 0),
                    'boron_cm': case.get('boron_cm', 0),
                    'multi_cm': case['multi_cm'],
                    'mod_cm': case['moderator_cm'],
                    'chamber': case.get('chamber', 'single_cu64'),
                    'zn_volume_cm3': volume_cm3,
                    'zn_density_g_cm3': zn_density,
                    'zn_mass_g': mass_g,
                    'zn_mass_kg': mass_kg,
                    'irrad_hours': irrad_h,
                    'cooldown_days': cool_d,
                    'cu64_mCi': float(act['cu64_mCi']),
                    'cu67_mCi': float(act['cu67_mCi']),
                    'cu64_Bq': float(act['cu64_Bq']),
                    'cu67_Bq': float(act['cu67_Bq']),
                    'cu64_g_yr': cu64_g_yr,
                    'cu67_g_yr': cu67_g_yr,
                    'cu64_atomic_purity': float(act['cu64_atomic_purity']),
                    'cu67_atomic_purity': float(act['cu67_atomic_purity']),
                    'cu64_radionuclide_purity': float(act['cu64_radionuclide_purity']),
                    'cu67_radionuclide_purity': float(act['cu67_radionuclide_purity']),
                    'total_cu_mass_g': float(act.get('total_cu_mass_g', 0) or 0),
                    'cu64_specific_activity_Ci_per_g': float(act['cu64_specific_activity_Ci_per_g']),
                    'cu67_specific_activity_Ci_per_g': float(act['cu67_specific_activity_Ci_per_g']),
                    # Grams of each Cu isotope after irrad + cooldown (total_cu_mass_g = sum of all)
                    'cu61_g': float(act.get('cu61_g', 0) or 0),
                    'cu62_g': float(act.get('cu62_g', 0) or 0),
                    'cu63_g': float(act.get('cu63_g', 0) or 0),
                    'cu64_g': float(act.get('cu64_g', 0) or 0),
                    'cu65_g': float(act.get('cu65_g', 0) or 0),
                    'cu66_g': float(act.get('cu66_g', 0) or 0),
                    'cu67_g': float(act.get('cu67_g', 0) or 0),
                    'cu68_g': float(act.get('cu68_g', 0) or 0),
                    'cu69_g': float(act.get('cu69_g', 0) or 0),
                    'cu70_g': float(act.get('cu70_g', 0) or 0),
                }
                npv = npv_from_cu_summary_row(
                    row, price_cu64_usd_per_g, price_cu67_usd_per_g,
                    sell_fraction=1.0, cap_usd_per_yr=None, purity_cap_64=True,
                    r=NPV_DISCOUNT_RATE, T_years=REVENUE_PLOT_YEARS,
                    capex_usd=float(flare_usd), opex_fixed_usd_per_yr=float(opex_yr),
                )
                row['npv_millions'] = npv / 1e6 if not np.isnan(npv) else np.nan
                cu_rows.append(row)
                
                zn_rows.append({
                    'dir_name': case['dir_name'],
                    'zn64_enrichment': enrich,
                    'zn_feedstock_cost': zn_cost_per_kg * mass_kg,
                    'use_zn67': case.get('use_zn67', False),
                    'inner_cm': case.get('inner_cm', 0),
                    'outer_cm': case.get('outer_cm', 20),
                    'struct_cm': case.get('struct_cm', 0),
                    'boron_cm': case.get('boron_cm', 0),
                    'multi_cm': case['multi_cm'],
                    'mod_cm': case['moderator_cm'],
                    'chamber': case.get('chamber', 'single_cu64'),
                    'zn_volume_cm3': volume_cm3,
                    'zn_density_g_cm3': zn_density,
                    'zn_mass_g': mass_g,
                    'zn_mass_kg': mass_kg,
                    'irrad_hours': irrad_h,
                    'cooldown_days': cool_d,
                    'zn65_mCi': act['zn65_mCi'],
                    'zn65_Bq': act['zn65_Bq'],
                    'zn65_specific_activity_Bq_per_g': act['zn65_Bq'] / mass_g if mass_g > 0 else 0,
                    'zn69m_Bq': act.get('zn69m_Bq', 0.0),
                })
    
    cu_df = pd.DataFrame(cu_rows)
    zn_df = pd.DataFrame(zn_rows)
    
    return cu_df, zn_df


def plot_activity_vs_variables(cu_df, output_dir, geom_info=None):
    """
    Plot Cu-64 and Cu-67 activity vs irradiation, cooldown, and enrichment.
    Uses all cases (all enrichments, all geometry: outer, struct, boron, multi, mod).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    df = _ensure_geom_columns(cu_df.copy())
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    enrichments = sorted(df['zn64_enrichment'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))
    geom_info = geom_info or "All geometries (outer, struct, boron, multi, mod)"
    
    # --- Plot 1: Activity vs Irradiation (cooldown=1 day), max 72h ---
    ax = axes[0]
    cool_fixed = 1
    irrad_max_h = 72

    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['cooldown_days'] == cool_fixed) & (df['irrad_hours'] <= irrad_max_h)]
        sub = sub.groupby('irrad_hours').mean(numeric_only=True).reset_index().sort_values('irrad_hours')
        
        if not sub.empty:
            ax.semilogy(sub['irrad_hours'], sub['cu64_mCi'], 'o-', color=colors[i], 
                       label=f'Cu-64 Zn64={_enrich_label(enrich)}', linewidth=2, markersize=5)
            ax.semilogy(sub['irrad_hours'], sub['cu67_mCi'], 's--', color=colors[i], 
                       alpha=0.6, label=f'Cu-67 Zn64={_enrich_label(enrich)}', markersize=4)
    
    ax.set_xlim(0, irrad_max_h)
    ax.set_xlabel('Irradiation Time (hours)', fontsize=11)
    ax.set_ylabel('Activity (mCi)', fontsize=11)
    ax.set_title(f'Activity vs Irradiation Time (cooldown: {cool_fixed} d)', fontsize=11)
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
                       label=f'Cu-64 Zn64={_enrich_label(enrich)}', linewidth=2, markersize=6)
            ax.semilogy(sub['cooldown_days'], sub['cu67_mCi'], 's--', color=colors[i], 
                       alpha=0.6, label=f'Cu-67 Zn64={_enrich_label(enrich)}', markersize=5)
    
    ax.set_xlabel('Cooldown Time (days)', fontsize=11)
    ax.set_ylabel('Activity (mCi)', fontsize=11)
    ax.set_title(f'Activity vs Cooldown Time (irradiation: {irrad_fixed} h)', fontsize=11)
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
    ax.set_title(f'Activity vs Enrichment (irradiation: {irrad_fixed} h, cooldown: {cool_fixed} d)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_vs_variables.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/activity_vs_variables.png")


def plot_purity_vs_variables(cu_df, output_dir, geom_info=None, chamber_label=None):
    """
    Plot radionuclide impurity (1 - activity fraction) vs variables.
    chamber_label: 'inner' forces Cu-67 (dual inner), 'outer' uses data.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    df = _ensure_geom_columns(cu_df.copy())
    if 'cu64_radionuclide_purity' not in df.columns:
        df['cu64_radionuclide_purity'] = df.get('cu64_atomic_purity', 0)
    if 'cu67_radionuclide_purity' not in df.columns:
        df['cu67_radionuclide_purity'] = df.get('cu67_atomic_purity', 0)
    # Dual inner → Cu-67; otherwise use use_zn67 from data
    use_cu67 = True if chamber_label == 'inner' else (df['use_zn67'].any() if 'use_zn67' in df.columns else False)
    purity_col = 'cu67_radionuclide_purity' if use_cu67 else 'cu64_radionuclide_purity'
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    enrich_col = 'zn64_enrichment'
    df[enrich_col] = df[enrich_col].apply(_norm_enrich)
    enrich_label = 'Zn-67 Enrichment (%)' if use_cu67 else 'Zn-64 Enrichment (%)'
    enrichments = sorted(df[enrich_col].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))
    cool_fixed = 1
    irrad_fixed = 8
    irrad_max_h = 72
    geom_info = geom_info or "All geometries (outer, struct, boron, multi, mod)"

    for ax_idx, (x_var, x_label, fix_var, fix_val) in enumerate([
        ('irrad_hours', 'Irradiation Time (hours)', 'cooldown_days', cool_fixed),
        ('cooldown_days', 'Cooldown Time (days)', 'irrad_hours', irrad_fixed),
        (enrich_col, enrich_label, None, None),
    ]):
        ax = axes[ax_idx]
        for i, enrich in enumerate(enrichments):
            if fix_var is None:
                sub = df[(df['irrad_hours'] == irrad_fixed) & (df['cooldown_days'] == cool_fixed)]
            else:
                mask = (df[enrich_col] == enrich) & (df[fix_var] == fix_val)
                if x_var == 'irrad_hours':
                    mask = mask & (df['irrad_hours'] <= irrad_max_h)
                sub = df[mask]
            if sub.empty:
                continue
            grp = sub.groupby(x_var).mean(numeric_only=True).reset_index().sort_values(x_var)
            x_vals = grp[x_var].values
            rad_imp = np.clip((1 - grp[purity_col].values) * 100, 0.001, 100)
            ax.semilogy(x_vals, rad_imp, 'o-', color=colors[i], linewidth=1.5, markersize=4, alpha=0.9,
                        label=f'Zn64={_enrich_label(enrich)}' if fix_var is not None else None)
        if fix_var is not None:
            leg_title = 'Zn-67 Enrichment' if use_cu67 else 'Zn-64 Enrichment'
            ax.legend([_enrich_label(e) for e in enrichments], fontsize=7, ncol=2, title=leg_title)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(f'{isotope} Radionuclide Impurity [%]', fontsize=11)
        ax.set_title(f'{isotope} Impurity vs {x_label.split(" ")[0]}', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_ylim(0.01, 100)
        ax.set_yticks([0.01, 0.1, 1, 10, 100])
        ax.set_yticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])

    # Single fullcase/geom title at top (no per-subplot overlap)
    fig.suptitle(geom_info, fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'impurity_vs_variables_{isotope.replace("-", "")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/impurity_vs_variables_{isotope.replace('-', '')}.png")


def _ensure_geom_columns(df):
    """Ensure geometry columns exist for backward compatibility (outer, struct, boron, multi, mod)."""
    df = df.copy()
    if 'outer_cm' not in df.columns:
        df['outer_cm'] = 20
    if 'struct_cm' not in df.columns:
        df['struct_cm'] = 0
    if 'boron_cm' not in df.columns:
        df['boron_cm'] = 0
    if 'multi_cm' not in df.columns:
        df['multi_cm'] = 0
    if 'mod_cm' not in df.columns:
        df['mod_cm'] = 0
    return df


def plot_production_vs_purity(cu_df, output_dir, chamber_label=None):
    """
    Plot Cu-64 or Cu-67 production (mCi) vs radionuclide impurity (1-radionuclide purity).
    Cu-64: by irradiation (1,4,8h) at EOI, log x-axis. Cu-67: by cooldown (fixed 8h irrad), linear x-axis.
    chamber_label: 'outer' or 'inner' for proper labeling.
    """
    # Dual: inner → Cu-67, outer → Cu-64. Single: use use_zn67 from data.
    use_cu67 = cu_df['use_zn67'].any() if 'use_zn67' in cu_df.columns else (chamber_label == 'inner')
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    prod_col = 'cu67_mCi' if use_cu67 else 'cu64_mCi'
    purity_col = 'cu67_radionuclide_purity' if use_cu67 else 'cu64_radionuclide_purity'
    chamber = chamber_label or 'outer'
    chamber_str = f" ({chamber} chamber)"

    # Cu-64: fix cooldown=0, use 8 h irradiation only. Cu-67: fix irradiation=8h, vary cooldown.
    if use_cu67:
        irrad_fixed = 8
        cool_filter = sorted(cu_df['cooldown_days'].unique().tolist())
        df = cu_df[(cu_df['irrad_hours'] == irrad_fixed) & (cu_df['cooldown_days'].isin(cool_filter))].copy()
    else:
        # Only keep 8 h irradiation points to simplify the plot
        irrad_filter = [8]
        cool_fixed = 0
        df = cu_df[(cu_df['cooldown_days'] == cool_fixed) & (cu_df['irrad_hours'].isin(irrad_filter))].copy()
        if df.empty:
            min_cool = cu_df['cooldown_days'].min()
            df = cu_df[(cu_df['cooldown_days'] == min_cool) & (cu_df['irrad_hours'].isin(irrad_filter))].copy()
            cool_fixed = min_cool

    df = _ensure_geom_columns(df)
    if df.empty:
        print("  Warning: No data for production vs purity plot")
        plt.close()
        return

    # Radionuclide impurity
    if purity_col in df.columns:
        df['impurity_pct'] = (1 - df[purity_col]) * 100
    elif 'cu64_radionuclide_purity' in df.columns:
        df['impurity_pct'] = (1 - df['cu64_radionuclide_purity']) * 100
    else:
        df['impurity_pct'] = (1 - df.get('cu64_radionuclide_purity', 0)) * 100

    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment', 'irrad_hours', 'cooldown_days']
    plot_cols = [prod_col, 'impurity_pct'] + agg_cols
    df_plot = df[plot_cols].groupby(agg_cols, as_index=False).mean()
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))

    fig, ax = plt.subplots(figsize=(14, 10))

    # Colors for enrichment
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5)
                    for i, e in enumerate(enrichments)}

    # Cu-64: marker by irradiation (8 h only after filtering). Cu-67: marker by cooldown.
    if use_cu67:
        cooldown_times = sorted(df_plot['cooldown_days'].unique())
        var_marker = {c: ['o', 's', 'D', '^', 'v'][i % 5] for i, c in enumerate(cooldown_times)}
    else:
        irrad_times = sorted(df_plot['irrad_hours'].unique())
        var_marker = {1: 'o', 4: 's', 8: 'D'}

    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1))
                 for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}

    for _, row in df_plot.iterrows():
        enrich = row['zn64_enrichment']
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        if use_cu67:
            cool_d = row['cooldown_days']
            impurity = float(row['impurity_pct'])
            marker = var_marker.get(cool_d, 'o')
        else:
            irrad = row['irrad_hours']
            impurity = max(float(row['impurity_pct']), 0.001)
            marker = var_marker.get(irrad, 'o')
        ax.scatter(impurity, row[prod_col],
                  c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                  marker=marker, s=140, edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                  linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85)

    ax.set_xlabel(f'{isotope} Radionuclide Impurity (100% − Radionuclide Purity) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{isotope} Production [mCi]{chamber_str}', fontsize=12, fontweight='bold')

    if use_cu67:
        ax.set_title(f'{isotope} Production vs Radionuclide Impurity (by Cooldown){chamber_str}\n'
                     f'Irradiation: {irrad_fixed} h | Cooldown: {", ".join(map(str, cooldown_times))} days',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xticks([5, 10, 30, 50])
        ax.set_xticklabels(['5%', '10%', '30%', '50%'])
    else:
        cool_fixed = df_plot['cooldown_days'].iloc[0]
        cooldown_str = "End of Irradiation (no cooldown)" if cool_fixed == 0 else f"{cool_fixed}-day Cooldown"
        irrad_times = sorted(df_plot['irrad_hours'].unique())
        irrad_desc = ", ".join(f"{t} h" for t in irrad_times)
        ax.set_title(f'{isotope} Production vs Radionuclide Impurity{chamber_str}\n'
                     f'Irradiation: {irrad_desc} | {cooldown_str}',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([0.01, 0.1, 1, 10])
        ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
        ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)

    # === LEGEND 1: Enrichment (fill colors) ===
    enrich_label = 'Zn-67 Enrichment\n(fill color)' if use_cu67 else 'Zn-64 Enrichment\n(fill color)'
    enrich_handles = []
    for e in enrichments:
        h = ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o',
                      edgecolors='black', linewidths=0.5, label=_enrich_label(e))
        enrich_handles.append(h)

    # === LEGEND 2: Irradiation (Cu-64) or Cooldown (Cu-67) (marker shapes) ===
    if use_cu67:
        var_handles = []
        for c in cooldown_times:
            h = ax.scatter([], [], c='lightgray', s=100, marker=var_marker.get(c, 'o'),
                          edgecolors='black', linewidths=1, label=f'{c}d cooldown')
            var_handles.append(h)
        var_legend_title = 'Cooldown\n(marker shape)'
    else:
        irrad_times = sorted(df_plot['irrad_hours'].unique())
        var_handles = []
        for t in irrad_times:
            h = ax.scatter([], [], c='lightgray', s=100, marker=var_marker.get(t, 'o'),
                          edgecolors='black', linewidths=1, label=f'{t}h')
            var_handles.append(h)
        var_legend_title = 'Irradiation Time\n(marker shape)'
    
    # === LEGEND 3: Geometry (edge color: outer, boron; multi/mod only if non-zero) ===
    geom_handles = []
    for g in geom_configs:
        label = _geom_legend_str(g[0], g[1], g[2], g[3])
        h = ax.scatter([], [], c='white', s=100, marker='o',
                      edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=label)
        geom_handles.append(h)
    
    # Create three separate legends
    leg1 = ax.legend(handles=enrich_handles, title=enrich_label,
                     loc='center right', fontsize=9, title_fontsize=10,
                     framealpha=0.95)
    ax.add_artist(leg1)
    
    leg2 = ax.legend(handles=var_handles, title=var_legend_title,
                     loc='lower left', fontsize=9, title_fontsize=10,
                     framealpha=0.95)
    ax.add_artist(leg2)
    
    leg3 = ax.legend(handles=geom_handles, title='Geometry\n(edge color)',
                     loc='upper right', fontsize=9, title_fontsize=10,
                     framealpha=0.95)
    
    # Gray dashed line at top production + annotation: scaled to 1 yr (8h→8760h, no cooldown) vs global demand
    top_prod_mCi = float(df_plot[prod_col].max())
    ref_irrad_h = 8  # scale 8h runs out to full year
    scale_factor = 8760.0 / ref_irrad_h  # 1095
    scaled_yr_mCi = top_prod_mCi * scale_factor
    global_demand_mCi_yr = ECON_MARKET_CU64_CI_PER_YEAR * 1000.0  # Ci -> mCi
    times_demand = scaled_yr_mCi / global_demand_mCi_yr if global_demand_mCi_yr > 0 else 0
    ax.axhline(y=top_prod_mCi, color='gray', linestyle='--', alpha=0.7, zorder=0)
    ax.text(0.02, 0.98, f"Scaled to 1 yr (8h→8760h, no cooldown): ~{times_demand:.0f}× global demand/yr",
            transform=ax.transAxes, fontsize=8, color='gray', verticalalignment='top',
            horizontalalignment='left')

    plt.tight_layout()
    fname = f'production_vs_purity_{isotope.replace("-", "")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


def _irrad_marker_map(irrad_hours_list):
    """Return dict irrad_h -> marker for consistent shapes (circle, square, diamond, triangle up/down)."""
    markers = ['o', 's', 'D', '^', 'v', 'p', 'h']
    return {h: markers[i % len(markers)] for i, h in enumerate(sorted(irrad_hours_list))}


def plot_production_vs_purity_8h_only(cu_df, output_dir, chamber_label=None):
    """
    Cu-64 production vs radionuclide impurity, 8 h irradiation only (one marker shape), EOI (no cooldown).
    Same style as production_vs_purity: enrichment fill, geometry edge, vertical lines at 99.9%/99.99%.
    """
    chamber = chamber_label or 'outer'
    chamber_str = f" ({chamber} chamber)"
    df_all = cu_df.copy()
    df_all = _ensure_geom_columns(df_all)
    if 'use_zn67' in df_all.columns and df_all['use_zn67'].any():
        return
    isotope = 'Cu-64'
    prod_col = 'cu64_mCi'
    purity_col = 'cu64_radionuclide_purity'
    cool_fixed = 0
    irrad_filter = [8]
    df = df_all[(df_all['cooldown_days'] == cool_fixed) & (df_all['irrad_hours'].isin(irrad_filter))].copy()
    if df.empty:
        min_cool = df_all['cooldown_days'].min()
        df = df_all[(df_all['cooldown_days'] == min_cool) & (df_all['irrad_hours'].isin(irrad_filter))].copy()
        cool_fixed = min_cool
    if df.empty:
        print("  Warning: No 8 h data for production_vs_purity_8h_only")
        return
    if purity_col in df.columns:
        df['impurity_pct'] = (1 - df[purity_col]) * 100
    else:
        df['impurity_pct'] = (1 - df.get('cu64_radionuclide_purity', 0)) * 100
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment', 'irrad_hours', 'cooldown_days']
    df_plot = df[[prod_col, 'impurity_pct'] + agg_cols].groupby(agg_cols, as_index=False).mean()
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) for i, e in enumerate(enrichments)}
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1)) for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
    for _, row in df_plot.iterrows():
        enrich = row['zn64_enrichment']
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        impurity = max(float(row['impurity_pct']), 0.001)
        ax.scatter(impurity, row[prod_col],
                  c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                  marker='o', s=140, edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                  linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85)
    ax.set_xlabel(f'{isotope} Radionuclide Impurity (100% − Radionuclide Purity) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{isotope} Production [mCi]{chamber_str}', fontsize=12, fontweight='bold')
    cooldown_str = "End of Irradiation (no cooldown)" if cool_fixed == 0 else f"{cool_fixed}-day Cooldown"
    ax.set_title(f'{isotope} Production vs Radionuclide Impurity{chamber_str}\n8 h Irradiation | {cooldown_str}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
    ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.axvline(x=0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o', edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
    geom_handles = [ax.scatter([], [], c='white', s=100, marker='o', edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=_geom_legend_str(g[0], g[1], g[2], g[3])) for g in geom_configs]
    leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)', loc='center right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
    top_prod_mCi = float(df_plot[prod_col].max())
    ref_irrad_h = 8
    scale_factor = 8760.0 / ref_irrad_h
    scaled_yr_mCi = top_prod_mCi * scale_factor
    global_demand_mCi_yr = ECON_MARKET_CU64_CI_PER_YEAR * 1000.0
    times_demand = scaled_yr_mCi / global_demand_mCi_yr if global_demand_mCi_yr > 0 else 0
    ax.axhline(y=top_prod_mCi, color='gray', linestyle='--', alpha=0.7, zorder=0)
    ax.text(0.02, 0.98, f"Scaled to 1 yr (8h=8760h, no cooldown): ~{times_demand:.0f}× global demand/yr", transform=ax.transAxes, fontsize=8, color='gray', verticalalignment='top', horizontalalignment='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'production_vs_purity_Cu64_8h_only.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/production_vs_purity_Cu64_8h_only.png")


def plot_production_vs_purity_by_irradiation(cu_df, output_dir, chamber_label=None, irrad_hours=(1, 4, 8)):
    """
    Cu-64 production vs radionuclide impurity with multiple irradiation times: different marker shapes per irrad (circle, square, diamond, etc.).
    irrad_hours: e.g. (1, 4, 8) or (8, 12, 16). EOI (no cooldown). Same enrichment/geometry legend style as reference plot.
    For (1, 4, 8) or (8, 12, 16) h plots, only 10 cm outer points are included.
    """
    chamber = chamber_label or 'outer'
    chamber_str = f" ({chamber} chamber)"
    df_all = cu_df.copy()
    df_all = _ensure_geom_columns(df_all)
    if 'use_zn67' in df_all.columns and df_all['use_zn67'].any():
        return
    isotope = 'Cu-64'
    prod_col = 'cu64_mCi'
    purity_col = 'cu64_radionuclide_purity'
    irrad_hours = sorted(irrad_hours)
    cool_fixed = 0
    df = df_all[(df_all['cooldown_days'] == cool_fixed) & (df_all['irrad_hours'].isin(irrad_hours))].copy()
    if df.empty:
        min_cool = df_all['cooldown_days'].min()
        df = df_all[(df_all['cooldown_days'] == min_cool) & (df_all['irrad_hours'].isin(irrad_hours))].copy()
        cool_fixed = min_cool
    if df.empty:
        print(f"  Warning: No data for irrad {irrad_hours} in plot_production_vs_purity_by_irradiation")
        return
    if tuple(irrad_hours) in ((1, 4, 8), (8, 12, 16)):
        df = df[df['outer_cm'] == 10].copy()
        if df.empty:
            print(f"  Warning: No data for irrad {irrad_hours} with outer_cm=10 in plot_production_vs_purity_by_irradiation")
            return
    if purity_col in df.columns:
        df['impurity_pct'] = (1 - df[purity_col]) * 100
    else:
        df['impurity_pct'] = (1 - df.get('cu64_radionuclide_purity', 0)) * 100
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment', 'irrad_hours', 'cooldown_days']
    df_plot = df[[prod_col, 'impurity_pct'] + agg_cols].groupby(agg_cols, as_index=False).mean()
    irrad_times = sorted(df_plot['irrad_hours'].unique())
    if not irrad_times:
        return
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    var_marker = _irrad_marker_map(irrad_times)
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) for i, e in enumerate(enrichments)}
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1)) for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
    for _, row in df_plot.iterrows():
        enrich = row['zn64_enrichment']
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        irrad = row['irrad_hours']
        impurity = max(float(row['impurity_pct']), 0.001)
        marker = var_marker.get(irrad, 'o')
        ax.scatter(impurity, row[prod_col],
                  c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                  marker=marker, s=140, edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                  linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85)
    ax.set_xlabel(f'{isotope} Radionuclide Impurity (100% − Radionuclide Purity) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{isotope} Production [mCi]{chamber_str}', fontsize=12, fontweight='bold')
    cooldown_str = "End of Irradiation (no cooldown)" if cool_fixed == 0 else f"{cool_fixed}-day Cooldown"
    irrad_desc = ", ".join(f"{t} h" for t in irrad_times)
    ax.set_title(f'{isotope} Production vs Radionuclide Impurity{chamber_str}\nIrradiation: {irrad_desc} | {cooldown_str}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
    ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.axvline(x=0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o', edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
    var_handles = [ax.scatter([], [], c='lightgray', s=100, marker=var_marker.get(t, 'o'), edgecolors='black', linewidths=1, label=f'{t}h') for t in irrad_times]
    geom_handles = [ax.scatter([], [], c='white', s=100, marker='o', edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=_geom_legend_str(g[0], g[1], g[2], g[3])) for g in geom_configs]
    leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)', loc='center right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=var_handles, title='Irradiation Time\n(marker shape)', loc='lower left', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg2)
    ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ref_irrad_h = max(irrad_times) if irrad_times else 8
    top_prod_mCi = float(df_plot[prod_col].max())
    scale_factor = 8760.0 / ref_irrad_h
    scaled_yr_mCi = top_prod_mCi * scale_factor
    global_demand_mCi_yr = ECON_MARKET_CU64_CI_PER_YEAR * 1000.0
    times_demand = scaled_yr_mCi / global_demand_mCi_yr if global_demand_mCi_yr > 0 else 0
    ax.axhline(y=top_prod_mCi, color='gray', linestyle='--', alpha=0.7, zorder=0)
    ax.text(0.02, 0.98, f"Scaled to 1 yr ({ref_irrad_h}h=8760h, no cooldown): ~{times_demand:.0f}× global demand/yr", transform=ax.transAxes, fontsize=8, color='gray', verticalalignment='top', horizontalalignment='left')
    plt.tight_layout()
    tag = "_".join(str(h) for h in irrad_times) + "h"
    fname = f'production_vs_purity_Cu64_{tag}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


def _cooldown_marker_map(cooldown_days_list):
    """Return dict cooldown_days -> marker for distinct shapes (circle, square, diamond, triangle up, triangle down, etc.)."""
    markers = ['o', 's', 'D', '^', 'v', 'p', 'h', 'H', '*']
    sorted_cool = sorted(cooldown_days_list, key=lambda x: (float(x), x))
    return {c: markers[i % len(markers)] for i, c in enumerate(sorted_cool)}


def plot_production_vs_purity_8h_by_cooldown(cu_df, output_dir, chamber_label=None, cooldown_days=(0, 0.5, 1, 1.5, 2)):
    """
    Cu-64 production vs radionuclide impurity: 8 h irradiation, multiple cooldown periods (0, 0.5, 1, 1.5, 2 d).
    Each cooldown is a distinct marker shape (circle, square, diamond, triangle up, triangle down).
    """
    chamber = chamber_label or 'outer'
    chamber_str = f" ({chamber} chamber)"
    df_all = cu_df.copy()
    df_all = _ensure_geom_columns(df_all)
    if 'use_zn67' in df_all.columns and df_all['use_zn67'].any():
        return
    isotope = 'Cu-64'
    prod_col = 'cu64_mCi'
    purity_col = 'cu64_radionuclide_purity'
    cooldown_days = tuple(sorted(set(cooldown_days), key=float))
    df = df_all[(df_all['irrad_hours'] == 8) & (df_all['cooldown_days'].isin(cooldown_days))].copy()
    if df.empty:
        print(f"  Warning: No 8 h data for cooldown_days={cooldown_days} in plot_production_vs_purity_8h_by_cooldown")
        return
    if purity_col in df.columns:
        df['impurity_pct'] = (1 - df[purity_col]) * 100
    else:
        df['impurity_pct'] = (1 - df.get('cu64_radionuclide_purity', 0)) * 100
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment', 'irrad_hours', 'cooldown_days']
    df_plot = df[[prod_col, 'impurity_pct'] + agg_cols].groupby(agg_cols, as_index=False).mean()
    cool_times = sorted(df_plot['cooldown_days'].unique(), key=lambda x: (float(x), x))
    if not cool_times:
        return
    var_marker = _cooldown_marker_map(cool_times)
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) for i, e in enumerate(enrichments)}
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1)) for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
    for _, row in df_plot.iterrows():
        enrich = row['zn64_enrichment']
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        cool_d = row['cooldown_days']
        impurity = max(float(row['impurity_pct']), 0.001)
        marker = var_marker.get(cool_d, 'o')
        ax.scatter(impurity, row[prod_col],
                  c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                  marker=marker, s=140, edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                  linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85)
    ax.set_xlabel(f'{isotope} Radionuclide Impurity (100% − Radionuclide Purity) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{isotope} Production [mCi]{chamber_str}', fontsize=12, fontweight='bold')
    cool_labels = [f'{c}d' if float(c) == int(c) else f'{float(c):.1f}d' for c in cool_times]
    cool_desc = ", ".join(cool_labels)
    ax.set_title(f'{isotope} Production vs Radionuclide Impurity{chamber_str}\n8 h Irradiation | Cooldown: {cool_desc}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
    ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    ax.axvline(x=0.01, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o', edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
    var_handles = [ax.scatter([], [], c='lightgray', s=100, marker=var_marker.get(c, 'o'), edgecolors='black', linewidths=1,
                              label=f'{c}d' if float(c) == int(c) else f'{float(c):.1f}d') for c in cool_times]
    geom_handles = [ax.scatter([], [], c='white', s=100, marker='o', edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=_geom_legend_str(g[0], g[1], g[2], g[3])) for g in geom_configs]
    leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)', loc='center right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=var_handles, title='Cooldown\n(marker shape)', loc='lower left', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg2)
    ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
    top_prod_mCi = float(df_plot[prod_col].max())
    scale_factor = 8760.0 / 8.0
    scaled_yr_mCi = top_prod_mCi * scale_factor
    global_demand_mCi_yr = ECON_MARKET_CU64_CI_PER_YEAR * 1000.0
    times_demand = scaled_yr_mCi / global_demand_mCi_yr if global_demand_mCi_yr > 0 else 0
    ax.axhline(y=top_prod_mCi, color='gray', linestyle='--', alpha=0.7, zorder=0)
    ax.text(0.02, 0.98, f"Scaled to 1 yr (8h=8760h): ~{times_demand:.0f}× global demand/yr", transform=ax.transAxes, fontsize=8, color='gray', verticalalignment='top', horizontalalignment='left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'production_vs_purity_Cu64_8h_by_cooldown.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/production_vs_purity_Cu64_8h_by_cooldown.png")


def plot_cu64_production_vs_cooldown_by_irradiation_outer10_99pct(
    cu_df,
    output_dir,
    chamber_label=None,
    irrad_hours=(8, 16),
    cooldown_days=(0.5, 1.0, 1.5, 2.0),
    purity_cut=0.999,
):
    """
    Plot Cu-64 production (mCi) vs cooldown time (days) for a 10 cm outer Zn-64 blanket
    at 99% Zn-64 enrichment, with different colored lines for irradiation times.

    Only points with Cu-64 radionuclide purity >= purity_cut (default 99.9%) are shown.
    Intended to represent batch cycles where we irradiate for a fixed number of hours,
    remove copper, let it cooldown outside the machine, and keep irradiating Zn.
    """
    chamber = chamber_label or 'outer'
    df = cu_df.copy()
    df = _ensure_geom_columns(df)

    # Restrict to Cu-64 (outer chamber / no Zn-67-only cases)
    if 'use_zn67' in df.columns:
        df = df[~df['use_zn67']].copy()

    # Normalize enrichment and pick 99% Zn-64, outer thickness = 10 cm
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    df = df[(df['outer_cm'] == 10) & (df['zn64_enrichment'] == 0.99)].copy()
    if df.empty:
        print("  Warning: No outer=10 cm, Zn64=99% data for Cu-64 production vs cooldown plot")
        return

    # Apply radionuclide purity cut (Cu-64 activity fraction >= 99.9%)
    purity_col = 'cu64_radionuclide_purity'
    if purity_col in df.columns:
        df = df[df[purity_col] >= purity_cut].copy()
    else:
        df = df[df.get('cu64_radionuclide_purity', 0) >= purity_cut].copy()
    if df.empty:
        print(f"  Warning: No Cu-64 data with radionuclide purity ≥{purity_cut * 100:.1f}% for outer=10 cm, Zn64=99%")
        return

    irrad_hours = sorted(set(irrad_hours))
    cooldown_days = sorted(set(cooldown_days), key=float)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(irrad_hours)))

    any_line = False
    for color, irrad in zip(colors, irrad_hours):
        sub = df[(df['irrad_hours'] == irrad) & (df['cooldown_days'].isin(cooldown_days))].copy()
        if sub.empty:
            continue
        # Average over any duplicate geometry/enrichment entries at same cooldown
        sub = sub.groupby('cooldown_days', as_index=False)['cu64_mCi'].mean()
        sub = sub.sort_values('cooldown_days')
        if sub.empty:
            continue
        ax.semilogy(
            sub['cooldown_days'],
            sub['cu64_mCi'],
            'o-',
            color=color,
            linewidth=2.0,
            markersize=6,
            label=f'{irrad} h irradiation',
        )
        any_line = True

    if not any_line:
        print("  Warning: No Cu-64 points to plot for production vs cooldown (outer=10 cm, Zn64=99%)")
        plt.close()
        return

    ax.set_xlabel('Cooldown Time (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cu-64 Production [mCi]\n(radionuclide purity ≥ 99.9%)', fontsize=12, fontweight='bold')
    irrad_desc = ", ".join(f"{h} h" for h in irrad_hours)
    cool_desc = ", ".join(str(c) for c in cooldown_days)
    ax.set_title(
        f'Cu-64 Production vs Cooldown ({chamber} chamber)\n'
        f'Outer thickness = 10 cm, Zn-64 enrichment = 99%\n'
        f'Irradiation times: {irrad_desc}; cooldown days: {cool_desc}',
        fontsize=14,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(title='Irradiation time', fontsize=9, title_fontsize=10)
    plt.tight_layout()

    fname = 'cu64_production_vs_cooldown_outer10_99pct.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


def plot_production_vs_purity_8h_two_cooldowns(cu_df, output_dir, chamber_label=None, cooldown_days=(1.0, 2.0)):
    """
    Two-panel figure: Cu-64 production vs radionuclide impurity for 8 h irradiation,
    comparing 1-day and 2-day cooldown (one subplot per cooldown).
    Each subplot also annotates the factor of global Cu-64 market supply if the
    highest-production point were scaled to a full year at 8 h per cycle.
    """
    chamber = chamber_label or 'outer'
    # This plot is intended for outer Cu-64 cases only.
    df_all = cu_df.copy()
    df_all = _ensure_geom_columns(df_all)
    if 'use_zn67' in df_all.columns and df_all['use_zn67'].any():
        # Mixed / Cu-67 mode – skip with a warning instead of raising.
        print("  Warning: plot_production_vs_purity_8h_two_cooldowns is defined for Cu-64 (outer) only; skipping.")
        return

    isotope = 'Cu-64'
    prod_col = 'cu64_mCi'
    purity_col = 'cu64_radionuclide_purity'

    # Normalize enrichment for grouping/legend
    df_all['zn64_enrichment'] = df_all['zn64_enrichment'].apply(_norm_enrich)
    # Restrict to 8 h irradiation and the requested cooldown days
    df_all = df_all[(df_all['irrad_hours'] == 8) & (df_all['cooldown_days'].isin(cooldown_days))].copy()
    if df_all.empty:
        print(f"  Warning: No 8 h data for cooldown days={cooldown_days} in production_vs_purity_8h_two_cooldowns")
        return

    # Add impurity percentage
    if purity_col in df_all.columns:
        df_all['impurity_pct'] = (1 - df_all[purity_col].astype(float)) * 100.0
    elif 'cu64_radionuclide_purity' in df_all.columns:
        df_all['impurity_pct'] = (1 - df_all['cu64_radionuclide_purity'].astype(float)) * 100.0
    else:
        df_all['impurity_pct'] = (1 - df_all.get('cu64_radionuclide_purity', 0).astype(float)) * 100.0

    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment', 'irrad_hours', 'cooldown_days']
    plot_cols = [prod_col, 'impurity_pct'] + agg_cols
    df_plot_all = df_all[plot_cols].groupby(agg_cols, as_index=False).mean()

    if df_plot_all.empty:
        print("  Warning: No grouped data for production_vs_purity_8h_two_cooldowns")
        return

    enrichments = sorted(df_plot_all['zn64_enrichment'].unique())
    geom_configs = df_plot_all[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))

    # Color/edge/style maps shared across both subplots
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {
        e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5)
        for i, e in enumerate(enrichments)
    }
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {
        tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1))
        for i, g in enumerate(geom_configs)
    }
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    chamber_str = f" ({chamber} chamber)"

    for ax, cd in zip(axes, cooldown_days):
        sub = df_plot_all[np.isclose(df_plot_all['cooldown_days'].astype(float), float(cd), atol=1e-6)]
        if sub.empty:
            ax.set_visible(False)
            continue
        for _, row in sub.iterrows():
            enrich = row['zn64_enrichment']
            geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
            imp = max(float(row['impurity_pct']), 0.001)
            ax.scatter(
                imp,
                row[prod_col],
                c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                marker='o',
                s=110,
                edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                linewidths=geom_lw.get(tuple(geom), 2.0),
                alpha=0.9,
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlabel(f'{isotope} Radionuclide Impurity (100% − Purity) [%]', fontsize=11, fontweight='bold')
        if ax is axes[0]:
            ax.set_ylabel(f'{isotope} Production [mCi]{chamber_str}', fontsize=12, fontweight='bold')
        ax.set_xticks([0.01, 0.1, 1, 10])
        ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
        ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)

        # Market supply factor: top production at this cooldown, scaled to 1 yr at 8 h per cycle
        top_prod_mCi = float(sub[prod_col].max())
        ref_irrad_h = 8.0
        scale_factor = 8760.0 / ref_irrad_h  # 8 h per cycle → 1 year
        scaled_yr_mCi = top_prod_mCi * scale_factor
        global_demand_mCi_yr = ECON_MARKET_CU64_CI_PER_YEAR * 1000.0  # Ci → mCi
        times_demand = scaled_yr_mCi / global_demand_mCi_yr if global_demand_mCi_yr > 0 else 0.0
        ax.axhline(y=top_prod_mCi, color='gray', linestyle='--', alpha=0.7, zorder=0)
        # Label on the horizontal dashed line (data coords), like the other production vs purity plot
        x_right = ax.get_xlim()[1]
        ax.text(
            x_right,
            top_prod_mCi,
            f" ~{times_demand:.0f}× global demand/yr (8 h cycles)",
            fontsize=8,
            color='gray',
            verticalalignment='center',
            horizontalalignment='right',
        )
        ax.set_title(f'{isotope} Production vs Radionuclide Impurity\n8 h Irradiation, {cd:g} d Cooldown',
                     fontsize=12, fontweight='bold')

    fig.suptitle(f'{isotope} Production vs Radionuclide Impurity — 8 h, 1 d vs 2 d cooldown{chamber_str}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = f'production_vs_purity_8h_1d_vs_2d_{isotope.replace("-", "")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


def plot_production_vs_purity_by_cooldown(cu_outer, output_dir, cu_inner_df=None):
    """Production vs purity for outer (Cu-64) and optionally inner (Cu-67)."""
    plot_production_vs_purity(cu_outer, output_dir, chamber_label='outer')
    if cu_inner_df is not None and not cu_inner_df.empty:
        plot_production_vs_purity(cu_inner_df, output_dir, chamber_label='inner')


def plot_production_vs_time_to_999_purity(cu_df, output_dir, chamber_label=None, irrad_hours=(1, 4, 8)):
    """
    Plot Cu-64 or Cu-67 production (mCi) at EOI vs cooldown days until purity FALLS to 99.9% (unfit for sale).
    Y-axis: production at EOI (cooldown=0) after irradiation. X-axis: days until purity falls to 99.9%.
    irrad_hours: e.g. (1, 4, 8) — different marker shapes per irradiation time.
    chamber_label: 'outer' or 'inner' for proper labeling.
    """
    irrad_filter = list(irrad_hours)
    purity_target = 0.999

    df = cu_df[(cu_df['irrad_hours'].isin(irrad_filter))].copy()
    df = _ensure_geom_columns(df)
    if df.empty:
        print(f"  Warning: No {irrad_filter} h irradiation data for production vs time-to-unfit plot")
        return

    if chamber_label == 'inner':
        use_cu67 = True
    else:
        use_cu67 = df['use_zn67'].any() if 'use_zn67' in df.columns else False
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    prod_col = 'cu67_mCi' if use_cu67 else 'cu64_mCi'
    purity_col = 'cu67_radionuclide_purity' if use_cu67 else 'cu64_radionuclide_purity'
    if purity_col not in df.columns and 'cu64_radionuclide_purity' in df.columns:
        purity_col = 'cu67_radionuclide_purity' if use_cu67 else 'cu64_radionuclide_purity'

    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment']
    cases = df.groupby(agg_cols, as_index=False).first()[agg_cols]

    rows = []
    for _, case in cases.iterrows():
        geom = (case['outer_cm'], case['boron_cm'], case['multi_cm'], case['mod_cm'])
        enrich = case['zn64_enrichment']
        # Process each irradiation time separately
        for irrad_h in irrad_filter:
            sub = df[(df['outer_cm'] == case['outer_cm']) & (df['boron_cm'] == case['boron_cm']) &
                     (df['multi_cm'] == case['multi_cm']) & (df['mod_cm'] == case['mod_cm']) &
                     (df['zn64_enrichment'] == enrich) & (df['irrad_hours'] == irrad_h)].sort_values('cooldown_days').reset_index(drop=True)
            if sub.empty:
                continue
            cool = sub['cooldown_days'].values
            pur = sub[purity_col].values.astype(float)
            prod = sub[prod_col].values.astype(float)
            
            # Production at EOI (cooldown=0) - this is what we plot on y-axis
            prod_at_eoi = prod[0] if len(prod) > 0 else 0.0

            # Purity DECREASES with cooldown. Find when it FALLS to 99.9% (unfit for sale).
            if pur[0] < purity_target:
                # Already below 99.9% at EOI (cooldown=0) → unfit immediately
                time_to_unfit = 0.0
            else:
                # Start >99.9%; find first cooldown where purity <= 99.9%
                found = False
                for i in range(1, len(cool)):
                    if pur[i] <= purity_target:
                        t0, t1 = float(cool[i - 1]), float(cool[i])
                        p0, p1 = pur[i - 1], pur[i]
                        if abs(p1 - p0) > 1e-12:
                            frac = (purity_target - p0) / (p1 - p0)
                        else:
                            frac = 1.0
                        time_to_unfit = t0 + frac * (t1 - t0)
                        found = True
                        break
                if not found:
                    # Still above 99.9% at max cooldown → lower bound on shelf life
                    time_to_unfit = float(cool[-1]) + 0.5

            rows.append({
                'outer_cm': case['outer_cm'], 'boron_cm': case['boron_cm'],
                'multi_cm': case['multi_cm'], 'mod_cm': case['mod_cm'],
                'zn64_enrichment': enrich, 'irrad_hours': irrad_h,
                'time_to_unfit_days': time_to_unfit,
                'production_mCi': prod_at_eoi,
            })

    if not rows:
        print("  Warning: No cases for production vs time-to-99.9%-purity plot")
        return

    df_plot = pd.DataFrame(rows)
    # Exclude cases already <99.9% at EOI (time_to_unfit_days==0) — do not plot them
    df_plot = df_plot[df_plot['time_to_unfit_days'] > 0].reset_index(drop=True)
    if df_plot.empty:
        print("  Warning: All cases already <99.9% at EOI; no points to plot for time-to-unfit")
        return
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    irrad_times = sorted(df_plot['irrad_hours'].unique())
    var_marker = _irrad_marker_map(irrad_times)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.colormaps.get_cmap('plasma')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5)
                    for i, e in enumerate(enrichments)}
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1))
                 for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
    chamber = chamber_label or 'outer'
    chamber_str = f" ({chamber} chamber)"

    for _, row in df_plot.iterrows():
        enrich = row['zn64_enrichment']
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        irrad_h = row['irrad_hours']
        marker = var_marker.get(irrad_h, 'o')
        ax.scatter(row['time_to_unfit_days'], row['production_mCi'],
                  c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                  s=140, marker=marker,
                  edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                  linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85)

    ax.set_xlabel('Cooldown until purity falls to 99.9% (unfit for sale)\n'
                  '(only cases that reach 99.9% at EOI are shown)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{isotope} Production [mCi] at EOI{chamber_str}', fontsize=12, fontweight='bold')
    irrad_desc = ", ".join(f"{h} h" for h in irrad_times)
    ax.set_title(f'{isotope} Production vs Cooldown Until Unfit for Sale (99.9% purity){chamber_str}\n'
                 f'Irradiation: {irrad_desc} | All cases', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    x_min = df_plot['time_to_unfit_days'].min()
    x_max = df_plot['time_to_unfit_days'].max()
    # X-axis starts at minutes so scale reads: minutes → hours → days
    _m = 1.0 / (60 * 24)
    _h = 1.0 / 24
    x_left = min(10 * _m, x_min * 0.5) if x_min > 0 else 10 * _m
    ax.set_xlim(max(0.003, x_left), x_max * 1.2)
    # Tick positions in days, labels in min / h / d
    tick_days_labels = [
        (10 * _m, '10 m'),
        (30 * _m, '30 m'),
        (1 * _h, '1 h'),
        (2 * _h, '2 h'),
        (6 * _h, '6 h'),
        (12 * _h, '12 h'),
        (1.0, '1 d'),
        (2.0, '2 d'),
        (3.0, '3 d'),
        (4.0, '4 d'),
        (5.0, '5 d'),
        (7.0, '7 d'),
    ]
    x_lo, x_hi = ax.get_xlim()
    tick_days = [d for d, _ in tick_days_labels if x_lo <= d <= x_hi]
    tick_labels = [lbl for d, lbl in tick_days_labels if x_lo <= d <= x_hi]
    if tick_days:
        ax.set_xticks(tick_days)
        ax.set_xticklabels(tick_labels)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o',
                      edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
    irrad_handles = [ax.scatter([], [], c='gray', s=100, marker=var_marker.get(h, 'o'),
                      edgecolors='black', linewidths=1, label=f'{h} h') for h in irrad_times]
    geom_handles = []
    for g in geom_configs:
        label = _geom_legend_str(g[0], g[1], g[2], g[3])
        geom_handles.append(ax.scatter([], [], c='white', s=100, marker='o',
                      edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=label))
    leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)' if not use_cu67 else 'Zn-67 Enrichment\n(fill color)',
                     loc='center left', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=irrad_handles, title='Irradiation\n(marker)', loc='lower left', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg2)
    ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.text(0.02, 0.02, f"Total: {len(df_plot)} cases", transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    fname = f'production_vs_time_to_999purity_{isotope.replace("-", "")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


def plot_production_at_eoi_by_enrichment(cu_df, output_dir, chamber_label=None):
    """
    Plot Cu-64 or Cu-67 production (mCi) at EOI (cooldown=0) vs enrichment for 1h and 8h irradiation.
    Simple plot: production at EOI, grouped by enrichment and geometry.
    """
    if chamber_label == 'inner':
        use_cu67 = True
    else:
        use_cu67 = cu_df['use_zn67'].any() if 'use_zn67' in cu_df.columns else False
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    prod_col = 'cu67_mCi' if use_cu67 else 'cu64_mCi'
    
    df = cu_df[(cu_df['cooldown_days'] == 0)].copy()
    df = _ensure_geom_columns(df)
    if df.empty:
        print(f"  Warning: No cooldown=0 data for {isotope} production at EOI plot")
        return
    
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    
    for irrad_hours in [1, 8]:
        df_irrad = df[df['irrad_hours'] == irrad_hours].copy()
        if df_irrad.empty:
            print(f"  Warning: No {irrad_hours}h irradiation data for {isotope} production at EOI plot")
            continue
        
        geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
        enrichments = sorted(df_irrad['zn64_enrichment'].unique())
        geom_configs = df_irrad[geom_cols].drop_duplicates().values.tolist()
        geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        cmap = plt.colormaps.get_cmap('plasma')
        enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5)
                        for i, e in enumerate(enrichments)}
        geom_edge_colors = plt.colormaps.get_cmap('Set1')
        geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1))
                     for i, g in enumerate(geom_configs)}
        geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
        chamber_str = f" ({chamber_label} chamber)" if chamber_label else ""
        
        x_positions = {}
        x_pos = 0
        for enrich in enrichments:
            x_positions[enrich] = x_pos
            x_pos += 1
        
        for _, row in df_irrad.iterrows():
            enrich = row['zn64_enrichment']
            geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
            prod = float(row[prod_col])
            x = x_positions[enrich]
            ax.scatter(x, prod,
                      c=[enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray'))],
                      s=180, marker='o',
                      edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                      linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85, zorder=5)
        
        ax.set_xlabel('Zn-64 Enrichment (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{isotope} Production [mCi] at EOI{chamber_str}', fontsize=13, fontweight='bold')
        ax.set_title(f'{isotope} Production at EOI (cooldown=0) vs Enrichment{chamber_str}\n'
                     f'{irrad_hours} hour irradiation', fontsize=14, fontweight='bold')
        
        ax.set_xticks([x_positions[e] for e in enrichments])
        ax.set_xticklabels([_enrich_label(e) for e in enrichments], rotation=45, ha='right')
        ax.set_yscale('linear')
        
        prod_values = df_irrad[prod_col].astype(float)
        y_min, y_max = float(prod_values.min()), float(prod_values.max())
        y_pad = (y_max - y_min) * 0.1
        ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
        
        enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=120, marker='o',
                          edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
        geom_handles = []
        for g in geom_configs:
            label = _geom_legend_str(g[0], g[1], g[2], g[3])
            geom_handles.append(ax.scatter([], [], c='white', s=120, marker='o',
                          edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=label))
        
        leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)' if not use_cu67 else 'Zn-67 Enrichment\n(fill color)',
                         loc='upper left', fontsize=9, title_fontsize=10, framealpha=0.95)
        ax.add_artist(leg1)
        ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
        
        ax.text(0.98, 0.02, f"Total: {len(df_irrad)} cases", transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        fname = f'production_at_eoi_{isotope.replace("-", "")}_{irrad_hours}h.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_dir}/{fname}")


def plot_production_vs_atomic_impurity(cu_df, output_dir, chamber_label=None):
    """
    Plot Cu-64 or Cu-67 production (mCi) vs atomic impurity (1 - atomic purity) on log or linear scale.
    Same structure as production vs radionuclide impurity but uses atomic purity (atom fraction).
    chamber_label: 'outer' or 'inner' for proper labeling.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    # Only 8 h irradiation points are used in this plot
    irrad_filter = [8]
    cool_fixed = 0
    
    df = cu_df[(cu_df['cooldown_days'] == cool_fixed) &
               (cu_df['irrad_hours'].isin(irrad_filter))].copy()
    df = _ensure_geom_columns(df)
    if df.empty:
        min_cool = cu_df['cooldown_days'].min()
        df = cu_df[(cu_df['cooldown_days'] == min_cool) &
                   (cu_df['irrad_hours'].isin(irrad_filter))].copy()
        df = _ensure_geom_columns(df)
        cool_fixed = min_cool
    if df.empty:
        print("  Warning: No data for production vs atomic impurity plot")
        plt.close()
        return

    # Dual inner → Cu-67; otherwise use use_zn67 from data
    use_cu67 = True if chamber_label == 'inner' else (df['use_zn67'].any() if 'use_zn67' in df.columns else False)
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    prod_col = 'cu67_mCi' if use_cu67 else 'cu64_mCi'
    atomic_purity_col = 'cu67_atomic_purity' if use_cu67 else 'cu64_atomic_purity'
    if atomic_purity_col not in df.columns:
        atomic_purity_col = 'cu67_purity' if use_cu67 else 'cu64_purity'  # backward compat for old CSVs
    df['atomic_impurity_pct'] = (1 - df[atomic_purity_col].astype(float)) * 100
    df['atomic_impurity_pct'] = np.clip(df['atomic_impurity_pct'], 1e-4, 100.0)
    # Normalize enrichments once before groupby to ensure exact matching (handles float precision)
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    agg_cols = geom_cols + ['zn64_enrichment', 'irrad_hours', 'cooldown_days']
    plot_cols = [prod_col, 'atomic_impurity_pct'] + agg_cols
    df_plot = df[plot_cols].groupby(agg_cols, as_index=False).mean()
    
    # Filter out rows with NaN/invalid values that can't be plotted
    df_plot = df_plot.dropna(subset=[prod_col, 'atomic_impurity_pct'])
    
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    irrad_times = sorted(df_plot['irrad_hours'].unique())
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))

    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5)
                    for i, e in enumerate(enrichments)}
    irrad_marker = {8: 'D'}
    geom_edge_colors = plt.colormaps.get_cmap('Set1')
    geom_edge = {tuple(g): geom_edge_colors(i / max(len(geom_configs) - 1, 1))
                 for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
    chamber = chamber_label or 'outer'
    chamber_str = f" ({chamber} chamber)"

    # Enrichments are normalized exactly by _norm_enrich; use exact equality checks (no tolerance)
    # If enrichments are different after normalization, they are distinct and will plot separately

    # Sort by enrichment (descending) so higher enrichments plot first (lower zorder), then lower enrichments on top
    df_plot_sorted = df_plot.sort_values('zn64_enrichment', ascending=False).reset_index(drop=True)

    for _, row in df_plot_sorted.iterrows():
        enrich = row['zn64_enrichment']
        enrich_val = float(enrich)  # Use exact normalized value
        irrad = row['irrad_hours']
        imp = max(float(row['atomic_impurity_pct']), 0.001)
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        # zorder: higher enrichments plot first (lower zorder), lower enrichments on top (higher zorder)
        # This ensures if points overlap, lower enrichments are visible
        # Use enrichment value directly: 0.999 -> zorder ~1, 0.99 -> zorder ~10, 0.81 -> zorder ~19
        zorder = int((1.0 - enrich_val) * 1000)  # Higher enrichment = lower zorder
        # enrich is already normalized from groupby; use directly
        ax.scatter(imp, row[prod_col],
                   c=[enrich_color.get(enrich, 'gray')], 
                   marker=irrad_marker.get(irrad, 'o'),
                   s=140, edgecolors=[geom_edge.get(tuple(geom), 'gray')],
                   linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85, zorder=zorder)

    ax.set_xlabel(f'{isotope} Atomic Impurity (100% − Atomic Purity) [%]', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{isotope} Production [mCi]{chamber_str}', fontsize=12, fontweight='bold')
    cooldown_str = "End of Irradiation (no cooldown)" if cool_fixed == 0 else f"{cool_fixed}-day Cooldown"
    irrad_desc = ", ".join(f"{h} h" for h in irrad_times)
    ax.set_title(f'{isotope} Production vs Atomic Impurity{chamber_str}\n'
                 f'Irradiation: {irrad_desc} | {cooldown_str}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    # Consistent: impurity (1-purity) on log scale for Cu-64 and Cu-67; no axis limits
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([0.01, 0.1, 1, 10])
    ax.set_xticklabels(['0.01%\n(99.99%)', '0.1%\n(99.9%)', '1%\n(99%)', '10%\n(90%)'])
    ax.axvline(x=0.1, color='gray', linestyle=':', linewidth=2, alpha=0.8)

    enrich_leg = 'Zn-67 Enrichment\n(fill color)' if use_cu67 else 'Zn-64 Enrichment\n(fill color)'
    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o',
                     edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
    irrad_handles = [ax.scatter([], [], c='lightgray', s=100, marker=irrad_marker.get(t, 'o'),
                    edgecolors='black', linewidths=1, label=f'{t}h') for t in irrad_times]
    geom_handles = []
    for g in geom_configs:
        label = _geom_legend_str(g[0], g[1], g[2], g[3])
        geom_handles.append(ax.scatter([], [], c='white', s=100, marker='o',
                      edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=label))
    leg1 = ax.legend(handles=enrich_handles, title=enrich_leg, loc='center right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=irrad_handles, title='Irradiation Time\n(marker shape)', loc='lower left', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg2)
    ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
    plt.tight_layout()
    fname = f'production_vs_atomic_impurity_{isotope.replace("-", "")}.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


# 8-year revenue model (used by revenue vs cooldown and ideal NPV vs cooldown)
REVENUE_PLOT_YEARS = 8

# Default irradiation (h) and radionuclide purity threshold for ideal-NPV vs cooldown plot
IDEAL_NPV_IRRAD_HOURS = 8
IDEAL_NPV_PURITY_THRESHOLD = 0.999  # 99.9% radionuclide purity

# Revenue vs cooldown with 99.9% r-purity gate: same irrad basis
REVENUE_VS_COOLDOWN_IRRAD_HOURS = 8
REVENUE_VS_COOLDOWN_PURITY_THRESHOLD = 0.999  # Cu-64 sellable only if ≥99.9%


def plot_revenue_vs_cooldown_purity_threshold(cu_df, output_dir, chamber_label=None, irrad_hours=None, purity_threshold=None):
    """
    Revenue vs cooldown time with radionuclide purity limit: Cu-64 can only be sold if r-purity >= 99.9%.
    Plot all (enrichment, geometry) that meet the threshold at each cooldown and their net revenue.
    Legend: only enrichments that appear in the plotted data (same style as other enrichment legends).

    Why fewer points and lower revenue at longer cooldown: Cu-64 decays (t½ ~12.7 h) and short-lived
    impurities change with time, so radionuclide purity drops after EOI. At longer cooldown only
    higher-enrichment / better-geometry cases still meet 99.9%; revenue falls because less Cu-64
    is sellable and/or fewer cycles fit in 8 years. This is consistent with the model.
    Y-axis: revenue [USD millions], not forced down to 0.
    """
    irrad_hours = irrad_hours if irrad_hours is not None else REVENUE_VS_COOLDOWN_IRRAD_HOURS
    purity_threshold = purity_threshold if purity_threshold is not None else REVENUE_VS_COOLDOWN_PURITY_THRESHOLD
    df = _ensure_geom_columns(cu_df.copy())
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    mask_irrad = np.isclose(df['irrad_hours'].astype(float), float(irrad_hours), atol=0.01)
    df = df[mask_irrad].copy()
    if df.empty:
        print("  Warning: No data for revenue vs cooldown (irrad_hours=%s)" % irrad_hours)
        return
    use_cu67 = True if chamber_label == 'inner' else (df['use_zn67'].any() if 'use_zn67' in df.columns else False)
    purity_col = 'cu67_radionuclide_purity' if use_cu67 else 'cu64_radionuclide_purity'
    if purity_col not in df.columns:
        purity_col = 'cu64_radionuclide_purity'
    try:
        import run_config as _rc
        p64 = getattr(_rc, "ECON_CU64_PRICE_PER_MCI", ECON_CU64_PRICE_PER_MCI)
        p67 = getattr(_rc, "ECON_CU67_PRICE_PER_MCI", ECON_CU67_PRICE_PER_MCI)
    except ImportError:
        p64, p67 = ECON_CU64_PRICE_PER_MCI, ECON_CU67_PRICE_PER_MCI
    hours_8yr = REVENUE_PLOT_YEARS * HOURS_PER_YEAR
    scale_8yr = hours_8yr / float(irrad_hours)
    try:
        import run_config as _rc
        _flare = getattr(_rc, 'ECON_SHINE_SOURCE_PURCHASE', ECON_SHINE_SOURCE_PURCHASE)
        _opex_yr = getattr(_rc, 'ECON_SHINE_SOURCE_OPERATIONS_ANNUAL', ECON_SHINE_SOURCE_OPERATIONS_ANNUAL)
    except ImportError:
        _flare = ECON_SHINE_SOURCE_PURCHASE
        _opex_yr = ECON_SHINE_SOURCE_OPERATIONS_ANNUAL
    opex_8yr = REVENUE_PLOT_YEARS * _opex_yr

    # Only count Cu-64 revenue when purity >= threshold; Cu-67 always (or gate same way for inner)
    df['purity_ok'] = df[purity_col].astype(float) >= purity_threshold
    cu64_rev = df['cu64_mCi'].astype(float) * scale_8yr * p64
    cu67_rev = df['cu67_mCi'].astype(float) * scale_8yr * p67
    df['cu64_revenue_8yr'] = np.where(df['purity_ok'], cu64_rev, 0.0)
    if use_cu67:
        df['cu67_revenue_8yr'] = np.where(df['purity_ok'], cu67_rev, 0.0)
    else:
        df['cu67_revenue_8yr'] = cu67_rev
    one_time = _flare + df['zn_feedstock_cost'].astype(float)
    df['revenue_8yr'] = df['cu64_revenue_8yr'] + df['cu67_revenue_8yr'] - opex_8yr - one_time

    # Only plot rows that meet purity threshold (sellable Cu-64)
    df_plot = df[df['purity_ok']].copy()
    if df_plot.empty:
        print("  Warning: No rows meet r-purity >= %.1f%% for revenue vs cooldown" % (purity_threshold * 100))
        return

    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    # Legend: only enrichments that appear in the plotted data (normalized so 0.76 and 0.760 match)
    enrichments_in_plot = sorted(set(_norm_enrich(e) for e in df_plot['zn64_enrichment']))
    geom_configs = df_plot[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    # Standard enrichment colors (match flare_npv: 99.9% lime green, etc.)
    try:
        from flare_npv import _get_enrichment_color
        enrich_color = {e: _get_enrichment_color(e, is_cu67=use_cu67) for e in enrichments_in_plot}
    except ImportError:
        cmap = plt.colormaps.get_cmap('viridis')
        enrich_color = {e: cmap(i / (len(enrichments_in_plot) - 1) if len(enrichments_in_plot) > 1 else 0.5) for i, e in enumerate(enrichments_in_plot)}
    geom_edge = {tuple(g): plt.colormaps.get_cmap('Set1')(i / max(len(geom_configs) - 1, 1)) for i, g in enumerate(geom_configs)}
    geom_lw = {tuple(g): 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for _, row in df_plot.iterrows():
        enrich = _norm_enrich(row['zn64_enrichment'])
        geom = (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm'])
        rev_m = float(row['revenue_8yr']) / 1e6
        ax.scatter(row['cooldown_days'], rev_m,
                  c=[enrich_color.get(enrich, 'gray')], marker='o', s=140,
                  edgecolors=[geom_edge.get(tuple(geom), 'gray')], linewidths=geom_lw.get(tuple(geom), 2.5), alpha=0.85)
    ax.set_xlabel('Cooldown time [days]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net revenue after 8 years [USD millions]\n(Cu-64 only if r-purity ≥ 99.9%; minus OPEX and FLARE + Zn cost)', fontsize=12, fontweight='bold')
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    ax.set_title(f'Revenue vs cooldown (r-purity ≥ {purity_threshold*100:.1f}% to sell {isotope})\n'
                 f'Only enrichments that meet threshold at each cooldown are shown; fewer at longer cooldown\n'
                 f'Production basis: {irrad_hours} h irradiation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    r_vals = df_plot['revenue_8yr'].astype(float) / 1e6
    r_min, r_max = float(r_vals.min()), float(r_vals.max())
    margin = max(0.08 * (r_max - r_min), 0.01) if r_max > r_min else 0.1
    # Y-axis not down to 0: use data range so scale is visible
    ax.set_ylim(r_min - margin, r_max + margin)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.2f}M'))
    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o', edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments_in_plot]
    geom_handles = []
    for g in geom_configs:
        label = _geom_legend_str(g[0], g[1], g[2], g[3])
        h = ax.scatter([], [], c='white', s=100, marker='o', edgecolors=[geom_edge[tuple(g)]], linewidths=geom_lw.get(tuple(g), 3), label=label)
        geom_handles.append(h)
    # Standard enrichment legend (fill color) — draw second so it stays on top
    leg_enrich = ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg_enrich)
    ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)' if not use_cu67 else 'Zn-67 Enrichment\n(fill color)', loc='center right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.text(0.98, 0.02, f"Points: {len(df_plot)} (purity ≥ {purity_threshold*100:.1f}%)", transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'revenue_vs_cooldown_purity_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/revenue_vs_cooldown_purity_threshold.png")


def plot_ideal_npv_vs_cooldown(cu_df, output_dir, chamber_label=None, irrad_hours=None, purity_threshold=None):
    """
    For each cooldown time in the CSV: select the ideal (enrichment, thickness) that meets the
    radionuclide purity threshold and maximizes NPV (or revenue). Plot one point per cooldown:
    x = cooldown_days, y = NPV (millions USD). Uses fixed irrad_hours (e.g. 8 h) so cycles are
    comparable (irrad + cooldown). Legend: enrichment (fill color), geometry (edge color), same
    style as production_vs_purity.
    """
    irrad_hours = irrad_hours if irrad_hours is not None else IDEAL_NPV_IRRAD_HOURS
    purity_threshold = purity_threshold if purity_threshold is not None else IDEAL_NPV_PURITY_THRESHOLD
    df = _ensure_geom_columns(cu_df.copy())
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    # Restrict to chosen irradiation time
    mask_irrad = np.isclose(df['irrad_hours'].astype(float), float(irrad_hours), atol=0.01)
    df = df[mask_irrad].copy()
    if df.empty:
        print("  Warning: No data for ideal NPV vs cooldown (irrad_hours=%s)" % irrad_hours)
        return
    use_cu67 = True if chamber_label == 'inner' else (df['use_zn67'].any() if 'use_zn67' in df.columns else False)
    purity_col = 'cu67_radionuclide_purity' if use_cu67 else 'cu64_radionuclide_purity'
    if purity_col not in df.columns:
        purity_col = 'cu64_radionuclide_purity'
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    y_col = 'npv_millions' if 'npv_millions' in df.columns else None
    try:
        import run_config as _rc
        _p64 = getattr(_rc, "ECON_CU64_PRICE_PER_MCI", ECON_CU64_PRICE_PER_MCI)
        _p67 = getattr(_rc, "ECON_CU67_PRICE_PER_MCI", ECON_CU67_PRICE_PER_MCI)
    except ImportError:
        _p64, _p67 = ECON_CU64_PRICE_PER_MCI, ECON_CU67_PRICE_PER_MCI
    _hours_8yr = REVENUE_PLOT_YEARS * HOURS_PER_YEAR
    _scale_8yr = _hours_8yr / float(irrad_hours)
    try:
        import run_config as _rc
        _flare = getattr(_rc, 'ECON_SHINE_SOURCE_PURCHASE', ECON_SHINE_SOURCE_PURCHASE)
        _opex_yr = getattr(_rc, 'ECON_SHINE_SOURCE_OPERATIONS_ANNUAL', ECON_SHINE_SOURCE_OPERATIONS_ANNUAL)
    except ImportError:
        _flare = ECON_SHINE_SOURCE_PURCHASE
        _opex_yr = ECON_SHINE_SOURCE_OPERATIONS_ANNUAL
    _opex_8yr = REVENUE_PLOT_YEARS * _opex_yr
    cooldown_vals = sorted(df['cooldown_days'].dropna().unique())
    rows_ideal = []
    for cool_d in cooldown_vals:
        sub = df[np.isclose(df['cooldown_days'].astype(float), float(cool_d), atol=0.01)]
        sub = sub[sub[purity_col].astype(float) >= purity_threshold]
        if sub.empty:
            continue
        if y_col and y_col in sub.columns:
            best_idx = sub[y_col].astype(float).idxmax()
        else:
            # Fallback: compute NPV with r=0.1, 8 yr, FLARE + feedstock, OPEX; Cu64 only if purity >= threshold
            af = annuity_factor(NPV_DISCOUNT_RATE, REVENUE_PLOT_YEARS)
            one_time = _flare + sub['zn_feedstock_cost'].astype(float)
            cu64_ok = sub[purity_col].astype(float) >= purity_threshold if purity_col == 'cu64_radionuclide_purity' else np.ones(len(sub), dtype=bool)
            annual_rev = (np.where(cu64_ok, sub['cu64_mCi'].astype(float) * (_scale_8yr / REVENUE_PLOT_YEARS) * _p64, 0.0) +
                          sub['cu67_mCi'].astype(float) * (_scale_8yr / REVENUE_PLOT_YEARS) * _p67)
            annual_net = annual_rev - _opex_yr
            npv_undisc = -one_time + af * annual_net
            best_idx = npv_undisc.idxmax()
        row = sub.loc[best_idx]
        if y_col and y_col in sub.columns:
            npv_val = float(sub.loc[best_idx, y_col])
        else:
            one_time = _flare + float(row['zn_feedstock_cost'])
            cu64_ok = float(row[purity_col]) >= purity_threshold if purity_col == 'cu64_radionuclide_purity' else True
            annual_rev = ((float(row['cu64_mCi']) * (_scale_8yr / REVENUE_PLOT_YEARS) * _p64 if cu64_ok else 0.0) +
                         float(row['cu67_mCi']) * (_scale_8yr / REVENUE_PLOT_YEARS) * _p67)
            annual_net = annual_rev - _opex_yr
            npv_val = (-one_time + af * annual_net) / 1e6
        rows_ideal.append({
            'cooldown_days': float(cool_d),
            'npv_millions': npv_val,
            'zn64_enrichment': row['zn64_enrichment'],
            'geom': (row['outer_cm'], row['boron_cm'], row['multi_cm'], row['mod_cm']),
        })
    if not rows_ideal:
        print("  Warning: No rows meet purity >= %.2f for ideal NPV vs cooldown" % purity_threshold)
        return
    df_plot = pd.DataFrame(rows_ideal)
    enrichments = sorted(df_plot['zn64_enrichment'].unique())
    geom_configs = sorted([tuple(g) for g in df_plot['geom'].unique()], key=lambda x: (x[0], x[1], x[2], x[3]))
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) for i, e in enumerate(enrichments)}
    geom_edge = {g: plt.colormaps.get_cmap('Set1')(i / max(len(geom_configs) - 1, 1)) for i, g in enumerate(geom_configs)}
    geom_lw = {g: 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_configs)}
    fig, ax = plt.subplots(figsize=(12, 8))
    for _, row in df_plot.iterrows():
        enrich = row['zn64_enrichment']
        geom = tuple(row['geom'])
        ax.scatter(row['cooldown_days'], row['npv_millions'],
                  c=[enrich_color.get(_norm_enrich(enrich), 'gray')], marker='o', s=140,
                  edgecolors=[geom_edge.get(geom, 'gray')], linewidths=geom_lw.get(geom, 2.5), alpha=0.85)
    ax.set_xlabel('Cooldown time [days]', fontsize=12, fontweight='bold')
    ax.set_ylabel('NPV [USD millions]', fontsize=12, fontweight='bold')
    isotope = 'Cu-67' if use_cu67 else 'Cu-64'
    ax.set_title(f'Ideal NPV vs cooldown (purity ≥ {purity_threshold*100:.1f}%)\n'
                 f'One point per cooldown: best (enrichment, thickness) at {irrad_hours} h irradiation',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.2f}M' if abs(x) >= 1 else f'${x:.3f}M'))
    enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o',
                                edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
    geom_handles = []
    for g in geom_configs:
        label = _geom_legend_str(g[0], g[1], g[2], g[3])
        h = ax.scatter([], [], c='white', s=100, marker='o',
                      edgecolors=[geom_edge[g]], linewidths=geom_lw.get(g, 3), label=label)
        geom_handles.append(h)
    leg1 = ax.legend(handles=enrich_handles, title='Zn-64 Enrichment\n(fill color)', loc='center right', fontsize=9, title_fontsize=10, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper right', fontsize=9, title_fontsize=10, framealpha=0.95)
    summary_text = f"Purity ≥ {purity_threshold*100:.1f}% | {irrad_hours} h irrad | {len(df_plot)} cooldown points"
    ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ideal_npv_vs_cooldown.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/ideal_npv_vs_cooldown.png")


def plot_zn65_case_comparison(zn_df, output_dir):
    """
    Case-by-case comparison of Zn-65 production at 1-year irradiation.
    Each case (dir_name) shown with its Zn-65 mCi.
    """
    irrad_1yr = ECON_IRRAD_HOURS
    cool_fixed = 0
    df = zn_df[(zn_df['cooldown_days'] == cool_fixed) & 
               (zn_df['irrad_hours'] == irrad_1yr)].copy()
    if df.empty:
        print("  Warning: No 1-year Zn-65 data for case comparison")
        return
    df = _ensure_geom_columns(df)
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    df = df.sort_values('zn65_mCi', ascending=False).reset_index(drop=True)
    cases = df['dir_name'].unique()
    zn65_vals = [df[df['dir_name'] == c]['zn65_mCi'].iloc[0] for c in cases]
    enrich_vals = [float(_norm_enrich(df[df['dir_name'] == c]['zn64_enrichment'].iloc[0])) * 100 for c in cases]
    fig, ax = plt.subplots(figsize=(12, max(6, len(cases) * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(np.unique(enrich_vals))))
    enrich_to_color = {e: colors[i] for i, e in enumerate(sorted(set(enrich_vals)))}
    bars = ax.barh(range(len(cases)), zn65_vals, color=[enrich_to_color.get(e, 'gray') for e in enrich_vals], alpha=0.85)
    ax.set_yticks(range(len(cases)))
    labels = [str(c)[:60] + ('...' if len(str(c)) > 60 else '') for c in cases]
    ax.set_yticklabels(labels, fontsize=8, ha='right')
    ax.set_xlabel('Zn-65 Production [mCi] (1-year irradiation)', fontsize=12, fontweight='bold')
    ax.set_title('Zn-65 Production by Case (1-year irradiation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=enrich_to_color[e], alpha=0.85, label=_enrich_label(e / 100.0)) 
               for e in sorted(enrich_to_color)]
    ax.legend(handles=handles, title='Zn-64 Enrichment', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zn65_case_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/zn65_case_comparison.png")


def plot_zn65_bar_by_geometry(zn_df, output_dir):
    """
    Horizontal bar plot: Y-axis = geometry cases, X-axis = Zn-65 production at 1 year.
    Bars colored by enrichment; highest enrichment drawn first (back), others overlaid on top.
    """
    irrad_1yr = ECON_IRRAD_HOURS
    cool_fixed = 0
    df = zn_df[(zn_df['cooldown_days'] == cool_fixed) & 
               (zn_df['irrad_hours'] == irrad_1yr)].copy()
    if df.empty:
        print("  Warning: No 1-year Zn-65 data for geometry bar plot")
        return
    df = _ensure_geom_columns(df)
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    geom_cols = ['outer_cm', 'boron_cm', 'multi_cm', 'mod_cm']
    enrichments = sorted(df['zn64_enrichment'].unique(), reverse=True)
    geom_configs = df[geom_cols].drop_duplicates().values.tolist()
    geom_configs = sorted(geom_configs, key=lambda x: (x[0], x[1], x[2], x[3]))
    
    def _geom_label(g):
        outer, boron, multi, mod = g[0], g[1], g[2], g[3]
        o, b = _disp_cm(outer), _disp_cm(boron)
        s = f'o={o:.0f} b={b:.0f}'
        if float(multi) != 0 or float(mod) != 0:
            s += f'\nMul={multi:.0f} mod={mod:.0f}'
        return s
    
    n_geom = len(geom_configs)
    bar_height = 0.6
    fig, ax = plt.subplots(figsize=(12, max(6, n_geom * 0.7)))
    cmap = plt.colormaps.get_cmap('viridis')
    enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) 
                    for i, e in enumerate(reversed(enrichments))}
    
    for i, geom in enumerate(geom_configs):
        sub = df[(df['outer_cm'] == geom[0]) & (df['boron_cm'] == geom[1]) & 
                 (df['multi_cm'] == geom[2]) & (df['mod_cm'] == geom[3])]
        for z, enrich in enumerate(enrichments):
            row = sub[sub['zn64_enrichment'] == enrich]
            if row.empty:
                continue
            zn65 = row['zn65_mCi'].iloc[0]
            ax.barh(i, zn65, height=bar_height, left=0, color=enrich_color.get(_norm_enrich(enrich), enrich_color.get(enrich, 'gray')), 
                    alpha=0.85, zorder=len(enrichments) - z)
    
    ax.set_yticks(range(n_geom))
    ax.set_yticklabels([_geom_label(g) for g in geom_configs], fontsize=10)
    ax.set_xlabel('Zn-65 Production [mCi] (1-year irradiation)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Geometry', fontsize=11, fontweight='bold')
    ax.set_title('Zn-65 production only', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    seen = set()
    handles = []
    for e in enrichments:
        if e not in seen:
            seen.add(e)
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=enrich_color[e], alpha=0.85, label=_enrich_label(e)))
    ax.legend(handles=handles, title='Zn-64 Enrichment', fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zn65_bar_by_geometry.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/zn65_bar_by_geometry.png")


def _plot_one_enrichment_figure(iso_label, enrichment_cost_dict, enrichment_map, output_dir, cost_ylabel_iso):
    """
    Two heatmaps: (1) Cost vs enrichment — piecewise linear interpolation between median anchor values ($/kg→$/g).
    (2) Zn isotope fractions vs enrichment. For Zn64, y-axis = ZN64_COST_ANCHOR_X (49.17%–99.9%); no Zn64 column.
    """
    # Single enrichment list: for Zn64 use canonical anchor grid so cost and y-axis align (no 0.995 or 1.0)
    if iso_label == 'Zn64':
        enrichments = sorted(ZN64_COST_ANCHOR_X)
        # Costs from linear interpolation at anchor points (enrichment_cost_dict is built from same grid)
        cost_per_kg = np.array([enrichment_cost_dict.get(e, get_zn64_enrichment_cost_per_kg(e)) for e in enrichments])
    else:
        enrichments = sorted(enrichment_cost_dict.keys())
        cost_per_kg = np.array([enrichment_cost_dict[e] for e in enrichments])
    cost_per_g = cost_per_kg / 1000.0  # $/kg → $/g
    cost_matrix = cost_per_g.reshape(-1, 1)

    if iso_label == 'Zn64':
        iso_columns = ['Zn66', 'Zn67', 'Zn68', 'Zn70']  # omit Zn64 (redundant with y-axis enrichment)
        frac_rows = [[enrichment_map[e]['Zn66'], enrichment_map[e]['Zn67'], enrichment_map[e]['Zn68'], enrichment_map[e]['Zn70']] for e in enrichments]
    else:
        iso_columns = ['Zn64', 'Zn66', 'Zn68', 'Zn70']
        frac_rows = []
        for e in enrichments:
            fracs = enrichment_map[e]
            frac_rows.append([fracs['Zn64'], fracs['Zn66'], fracs['Zn68'], fracs['Zn70']])
    frac_matrix = np.array(frac_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    cmap_cost = plt.colormaps.get_cmap('YlOrRd')

    # Left: Cost vs Enrichment — linear interpolation between median anchor values
    im1 = ax1.imshow(cost_matrix, aspect='auto', cmap=cmap_cost, origin='lower',
                     extent=[-0.5, 0.5, -0.5, len(enrichments) - 0.5])
    ax1.set_yticks(range(len(enrichments)))
    ax1.set_yticklabels([_enrichment_axis_label(e) for e in enrichments], fontsize=9)
    ax1.set_ylabel('Enrichment Fraction', fontsize=11)
    ax1.set_xticks([0])
    ax1.set_xticklabels([cost_ylabel_iso + ' Isotope'])
    ax1.set_title(f'{iso_label} Enrichment Cost vs Enrichment (fraction)', fontsize=12, fontweight='bold')
    cbar1 = fig.colorbar(im1, ax=ax1, label='Cost ($/g)')
    for i, (e, val) in enumerate(zip(enrichments, cost_per_g)):
        ax1.text(0, i, f'{val:,.1f}', ha='center', va='center', fontsize=9, color='black' if val < (cost_per_g.max() + cost_per_g.min()) / 2 else 'white')

    # Right: Isotope fractions — same y-axis; Zn64 column omitted for Zn64
    im2 = ax2.imshow(frac_matrix, aspect='auto', cmap=cmap_cost, vmin=0, vmax=0.7, origin='lower')
    ax2.set_yticks(range(len(enrichments)))
    ax2.set_yticklabels([_enrichment_axis_label(e) for e in enrichments], fontsize=9)
    ax2.set_ylabel(f'{iso_label} Enrichment Fraction', fontsize=11)
    ax2.set_xticks(range(len(iso_columns)))
    ax2.set_xticklabels(iso_columns)
    ax2.set_xlabel('Isotope', fontsize=11)
    ax2.set_title(f'Zn Isotope Fractions vs {iso_label} Enrichment (fraction)', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=ax2, label='Isotope Fraction')
    for i in range(len(enrichments)):
        for j in range(len(iso_columns)):
            ax2.text(j, i, f'{frac_matrix[i, j]:.3f}', ha='center', va='center', fontsize=8, color='black' if frac_matrix[i, j] < 0.35 else 'white')

    if iso_label == 'Zn64':
        fig.text(0.5, -0.02, 'Costs: piecewise linear interpolation between median anchor values ($/kg → $/g). Enrichments: 49.17%–99.9%.',
                 ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    elif iso_label == 'Zn67':
        fig.text(0.5, -0.02, 'Fraction anchors: 4.04%, 7.3%, 10%, 14%, 17.7%. Others linearly interpolated.', ha='center', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = os.path.join(output_dir, f'{iso_label.lower()}_enrichment_cost_and_fractions.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_enrichment_cost_and_fractions(output_dir):
    """
    Create two figures from utilities data (no hardcoding):
    - Zn64: enrichment cost vs enrichment (fraction), and Zn isotope fractions vs Zn-64 enrichment (fraction).
    - Zn67: enrichment cost vs enrichment (fraction), and Zn isotope fractions vs Zn-67 enrichment (fraction).
    """
    _plot_one_enrichment_figure(
        'Zn64',
        ZN64_ENRICHMENT_COST,
        ZN64_ENRICHMENT_MAP,
        output_dir,
        cost_ylabel_iso='Zn64',
    )
    _plot_one_enrichment_figure(
        'Zn67',
        ZN67_ENRICHMENT_COST,
        ZN67_ENRICHMENT_MAP,
        output_dir,
        cost_ylabel_iso='Zn67',
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple analysis for outer Zn layer irradiation")
    parser.add_argument("--base-dir", type=str, default=None, help="Base dir (default: run_config RUN_BASE_DIR/STATEPOINTS_DIR)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir (default: run_config RUN_BASE_DIR/RESULTS_DIR)")
    parser.add_argument("--pattern", type=str, default=None, help="Glob pattern for case dirs (default: OUTPUT_PREFIX_*)")
    args = parser.parse_args()
    analyzer = IrradiationAnalyzer(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        output_prefix=None,
        pattern=args.pattern,
    )
    analyzer.run()


if __name__ == '__main__':
    main()
