#!/usr/bin/env python3
"""
Sphere: 14.1 MeV point source at center of Zn spherical shell (r 5.5–15.5 cm).
Only zinc in model. Total reaction-rate tallies (no cell filter); rate = tally_mean * source_strength (use .mean not .sum).
Initial atoms from summary; constant-R Bateman for Cu64/Cu67/Zn65. Output: cu_summary.csv, zn_summary.csv per run.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openmc

from utilities import (
    SOURCE_STRENGTH,
    NEUTRON_ENERGY_MEV,
    get_zn_fractions,
    get_zn_fractions_log,
    get_initial_atoms_from_statepoint,
    get_initial_zn_atoms_fallback,
    get_material_density_from_statepoint,
    calculate_enriched_zn_density,
    get_zn64_enrichment_cost_per_kg,
    plot_zn64_enrichment,
    plot_zn64_enrichment_log,
)

# Geometry
SPHERE_INNER_R_CM = 5.5
SPHERE_OUTER_R_CM = 15.5
# Source energy and run parameters; flux = (tally/volume_cell) * SOURCE_STRENGTH (same as fusion_irradiation)
SOURCE_ENERGY_MEV = NEUTRON_ENERGY_MEV
OUTPUT_DIR = 'sphere_output'
PARTICLES = int(1e5)
BATCHES = 10

# Align with utilities cost anchors (linearly interpolated); 99% and 99.9% (no 100%)
ZN64_ENRICHMENTS = [0.4917, 0.53, 0.71, 0.76, 0.81, 0.86, 0.91, 0.96, 0.97, 0.98, 0.99, 0.999]
IRRADIATION_HOURS = [1, 2, 4, 8, 24]
COOLDOWN_DAYS = [0, 1, 2]

NUCLIDES_ZN = ['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']

# Analytical Cu-64 hand-calculation (cross-section only, no reaction-rate tallies)
SPHERE_THICKNESS_CM = SPHERE_OUTER_R_CM - SPHERE_INNER_R_CM
N_A = 6.02214076e23  # per mol
M_ZN_G_MOL = 65.38  # average Zn molar mass
# Zn64(n,p)Cu64 at 14.1 MeV: ~0.2 barn (ENDF/literature); 1 b = 1e-24 cm²
SIGMA_ZN64_NP_CM2 = 0.2e-24
T_HALF_CU64_S = 12.701 * 3600  # Cu-64 half-life [s]
LAMBDA_CU64_PER_S = np.log(2) / T_HALF_CU64_S


def _phi_avg_sphere_shell(source_strength_per_s, volume_cm3):
    """Simple average flux in spherical shell: phi_avg = S * thickness / V [n/(cm²·s)]."""
    if volume_cm3 <= 0:
        return 0.0
    return source_strength_per_s * SPHERE_THICKNESS_CM / volume_cm3


def _N_Zn64_from_volume_density(volume_cm3, density_g_cm3, zn64_enrichment):
    """Number of Zn-64 atoms: (V*rho/M)*N_A*enrichment."""
    mass_g = volume_cm3 * density_g_cm3
    n_mol = mass_g / M_ZN_G_MOL
    return n_mol * N_A * zn64_enrichment


def _analytical_simple_cu64_mci(volume_cm3, density_g_cm3, zn64_enrichment, source_strength_per_s, irrad_hours, verbose=False):
    """
    Cu-64 mCi from cross section only: R = sigma * phi_avg * N_Zn64, then
    N_Cu64(t) = (R/lambda)(1 - exp(-lambda*t)), mCi = N*lambda/3.7e7.
    No OpenMC reaction-rate data; hand calculation for comparison.
    """
    phi_avg = _phi_avg_sphere_shell(source_strength_per_s, volume_cm3)
    N_zn64 = _N_Zn64_from_volume_density(volume_cm3, density_g_cm3, zn64_enrichment)
    R = SIGMA_ZN64_NP_CM2 * phi_avg * N_zn64  # atoms/s
    lam = LAMBDA_CU64_PER_S
    t_s = irrad_hours * 3600.0
    if lam <= 0:
        N_cu64 = R * t_s
    else:
        N_cu64 = (R / lam) * (1.0 - np.exp(-lam * t_s))
    mci = N_cu64 * lam / 3.7e7
    if verbose:
        print("  [analytical simple] Cross-section-based Cu-64:")
        print(f"    phi_avg = S * thickness / V = {source_strength_per_s:.3e} * {SPHERE_THICKNESS_CM} / {volume_cm3:.2f} = {phi_avg:.3e} n/(cm^2 s)")
        print(f"    N_Zn64 = (V*rho/M)*N_A*enrich = {N_zn64:.3e}")
        print(f"    R = sigma * phi_avg * N_Zn64 = {SIGMA_ZN64_NP_CM2:.2e} * {phi_avg:.3e} * {N_zn64:.3e} = {R:.3e} atoms/s")
        print(f"    N_Cu64(t) = (R/lambda)(1-exp(-lambda*t)); t = {irrad_hours} h -> N_Cu64 = {N_cu64:.3e}, mCi = {mci:.4f}")
    return mci


def _half_life_s(nuclide):
    try:
        hl = openmc.data.half_life(nuclide)
        return float(hl) if hl is not None and np.isfinite(hl) and hl > 0 else None
    except Exception:
        return None


def _get_decay_constant(nuclide):
    hl = _half_life_s(nuclide)
    return (np.log(2) / hl) if hl and hl > 0 else 0.0


def create_geometry(zn64_enrichment, use_log=False):
    """Inner vacuum, Zn shell, outer vacuum. Single Zn material. use_log: log-interpolate isotope fractions (natural→99.18%)."""
    e = float(zn64_enrichment)
    density = calculate_enriched_zn_density(e)
    fracs = get_zn_fractions_log(e) if use_log else get_zn_fractions(e)
    mat = openmc.Material(material_id=1, name='zn')
    mat.set_density('g/cm3', density)
    for iso, f in fracs.items():
        mat.add_nuclide(iso, f)
    mat.temperature = 294

    si = openmc.Sphere(r=SPHERE_INNER_R_CM)
    so = openmc.Sphere(r=SPHERE_OUTER_R_CM, boundary_type='vacuum')
    inner = openmc.Cell(cell_id=1, name='inner', fill=None, region=-si)
    zn_cell = openmc.Cell(cell_id=2, name='zn', fill=mat, region=+si & -so)
    outer = openmc.Cell(cell_id=3, name='outer', fill=None, region=+so)
    geometry = openmc.Geometry(openmc.Universe(cells=[inner, zn_cell, outer]))
    return geometry, openmc.Materials([mat]), zn_cell


def create_source():
    """Point at center, 14.1 MeV, isotropic. Strength=1 so tally is per source particle; scale by SOURCE_STRENGTH in post only (avoid double scaling)."""
    s = openmc.IndependentSource()
    s.space = openmc.stats.Point((0.0, 0.0, 0.0))
    s.angle = openmc.stats.Isotropic()
    s.energy = openmc.stats.Discrete([SOURCE_ENERGY_MEV * 1e6], [1.0])
    s.strength = 1.0  # do not set SOURCE_STRENGTH here; flux and reaction rates scale by SOURCE_STRENGTH in post
    return s


def create_tallies(zn_cell):
    """Total reaction rates plus flux in Zn cell for plotting. Flux spectra: same filter order as fusion_irradiation [neutron, energy, cell]."""
    tallies = openmc.Tallies()
    t_cu = openmc.Tally(name='Total_Cu_Production_rxn_rate')
    t_cu.scores = ['(n,p)', '(n,d)']
    t_cu.nuclides = NUCLIDES_ZN
    tallies.append(t_cu)
    t_zn = openmc.Tally(name='Total_Zn_rxn_rate')
    t_zn.scores = ['(n,gamma)', '(n,2n)']
    t_zn.nuclides = NUCLIDES_ZN
    tallies.append(t_zn)
    neutron_filter = openmc.ParticleFilter(['neutron'])
    energy_filter = openmc.EnergyFilter.from_group_structure('CCFE-709')
    single_cell_filter = openmc.CellFilter([zn_cell])
    t_flux = openmc.Tally(name='flux')
    t_flux.filters = [single_cell_filter, neutron_filter]
    t_flux.scores = ['flux']
    tallies.append(t_flux)
    t_spectra = openmc.Tally(name='zn_spectra')
    t_spectra.filters = [neutron_filter, energy_filter, single_cell_filter]
    t_spectra.scores = ['flux']
    tallies.append(t_spectra)
    return tallies


def get_reaction_rates_per_s(sp):
    """
    Reaction rates [atoms/s] from Total tallies. Only zinc in model.
    Use tally mean (per source particle), not sum: rate = tally_mean * SOURCE_STRENGTH.
    OpenMC tally.mean = sum/num_realizations (num_realizations = batches), so using .sum
    would inflate rates by num_batches (e.g. 10x) and make mCi too large.
    """
    rr = {}

    def _read(tally_name, channels):
        try:
            t = sp.get_tally(name=tally_name)
        except LookupError:
            return
        arr = np.asarray(t.mean)
        if arr.ndim == 3:
            arr = arr[0]
        for parent, score, product in channels:
            if parent not in t.nuclides or score not in t.scores:
                continue
            i = t.get_nuclide_index(parent)
            j = t.get_score_index(score)
            rr[f'{parent} {score} {product}'] = float(arr[i, j]) * SOURCE_STRENGTH

    _read('Total_Cu_Production_rxn_rate', [
        ('Zn64', '(n,p)', 'Cu64'), ('Zn67', '(n,p)', 'Cu67'), ('Zn68', '(n,d)', 'Cu67'),
    ])
    _read('Total_Zn_rxn_rate', [
        ('Zn63', '(n,gamma)', 'Zn64'), ('Zn64', '(n,gamma)', 'Zn65'), ('Zn65', '(n,gamma)', 'Zn66'),
        ('Zn66', '(n,gamma)', 'Zn67'), ('Zn67', '(n,gamma)', 'Zn68'), ('Zn68', '(n,gamma)', 'Zn69'),
        ('Zn69', '(n,gamma)', 'Zn70'),
        ('Zn65', '(n,2n)', 'Zn64'), ('Zn66', '(n,2n)', 'Zn65'), ('Zn67', '(n,2n)', 'Zn66'),
        ('Zn68', '(n,2n)', 'Zn67'), ('Zn69', '(n,2n)', 'Zn68'), ('Zn70', '(n,2n)', 'Zn69'),
    ])
    return rr


def plot_flux_spectrum(sp, zn_cell, volume_cm3, run_dir):
    """Plot neutron flux per unit lethargy vs energy in Zn cell; save to run_dir/zn_flux_spectrum.png.
    Same flux extraction format as fusion_irradiation: (tally_mean/volume)*source_strength, norm_flux = flux/lethargy_bin_width.
    Uses analytical volume (cm³) to avoid OpenMC cell.volume units/availability issues.
    """
    try:
        cell_tally = sp.get_tally(name='zn_spectra')
    except LookupError:
        return
    # Filter order: [neutron_filter, energy_filter, cell_filter] (same as fusion_irradiation)
    energy_filt = cell_tally.filters[1]
    energy_bins = np.asarray(energy_filt.bins)
    if energy_bins.ndim == 2:
        energy_edges_eV = np.concatenate([energy_bins[:, 0], [energy_bins[-1, 1]]])
    else:
        energy_edges_eV = energy_bins
    lethargy_bin_width = np.log(energy_edges_eV[1:] / energy_edges_eV[:-1])

    # Extract flux (same as fusion_irradiation):
    #   flux per bin = (tally_mean / volume) * source_strength  [n/cm²/s]
    #   norm_flux    = flux / lethargy_bin_width
    openmc_flux = cell_tally.mean.flatten()
    volume_of_cell = float(volume_cm3)
    if zn_cell.volume is not None and zn_cell.volume > 0:
        cell_vol = float(zn_cell.volume)
        ratio = cell_vol / volume_of_cell
        if ratio < 0.01 or ratio > 100:
            print(f"  [sphere] Warning: cell.volume ({cell_vol:.2e}) vs analytical ({volume_of_cell:.2e} cm³) ratio={ratio:.2e}; using analytical.")
    flux = (openmc_flux / volume_of_cell) * SOURCE_STRENGTH
    norm_flux = flux / lethargy_bin_width

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(energy_edges_eV[:-1], norm_flux, where='post', lw=2, color='#8c564b', label='Zn')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Flux per unit lethargy [n/(cm²·s)]')
    ax.set_title(f'Zn cell flux spectrum — {SOURCE_ENERGY_MEV} MeV, {SOURCE_STRENGTH:.1e} n/s')
    ax.legend()
    ax.grid(True, which='both', ls=':', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(run_dir, 'zn_flux_spectrum.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [sphere] Saved {path}")


def bateman_constant_r(rr, irrad_s, cooldown_s):
    """Constant R, no parent depletion. Cu64, Cu67, Zn65 only."""
    R64 = float(rr.get('Zn64 (n,p) Cu64', 0) or 0)
    R67 = float(rr.get('Zn67 (n,p) Cu67', 0) or 0) + float(rr.get('Zn68 (n,d) Cu67', 0) or 0)
    R65 = float(rr.get('Zn64 (n,gamma) Zn65', 0) or 0) + float(rr.get('Zn66 (n,2n) Zn65', 0) or 0)
    lam64 = _get_decay_constant('Cu64')
    lam67 = _get_decay_constant('Cu67')
    lam65 = _get_decay_constant('Zn65')

    def _n(R, lam, t_irr, t_cool):
        if R <= 0:
            return 0.0
        if lam and lam > 0:
            n = (R / lam) * (1.0 - np.exp(-lam * t_irr))
            return n * np.exp(-lam * t_cool)
        return R * t_irr

    return (
        _n(R64, lam64, irrad_s, cooldown_s),
        _n(R67, lam67, irrad_s, cooldown_s),
        _n(R65, lam65, irrad_s, cooldown_s),
        lam64, lam67, lam65,
    )


def compute_activities(case, irrad_hours, cooldown_days):
    """Constant-R Bateman; return mCi, Bq, purities."""
    irrad_s = irrad_hours * 3600.0
    cooldown_s = cooldown_days * 86400.0
    rr = case.get('reaction_rates') or {}
    cu64, cu67, zn65, lam64, lam67, lam65 = bateman_constant_r(rr, irrad_s, cooldown_s)
    total_cu = cu64 + cu67
    a64 = cu64 * lam64
    a67 = cu67 * lam67
    total_a = a64 + a67
    return {
        'cu64_mCi': cu64 * lam64 / 3.7e7,
        'cu67_mCi': cu67 * lam67 / 3.7e7,
        'zn65_mCi': zn65 * lam65 / 3.7e7,
        'cu64_Bq': a64, 'cu67_Bq': a67, 'zn65_Bq': zn65 * lam65,
        'cu64_atoms': cu64, 'cu67_atoms': cu67, 'zn65_atoms': zn65,
        'cu64_atomic_purity': cu64 / total_cu if total_cu > 0 else 0.0,
        'cu67_atomic_purity': cu67 / total_cu if total_cu > 0 else 0.0,
        'cu64_radionuclide_purity': a64 / total_a if total_a > 0 else 0.0,
        'cu67_radionuclide_purity': a67 / total_a if total_a > 0 else 0.0,
    }


def build_summary_dataframes(case, zn64_enrichment):
    """cu_summary and zn_summary over IRRADIATION_HOURS x COOLDOWN_DAYS."""
    vol = case['outer_volume_cm3']
    rho = case['zn_density_g_cm3']
    mass_g = case.get('zn_mass_g', vol * rho)
    mass_kg = mass_g / 1000.0
    cost_kg = get_zn64_enrichment_cost_per_kg(zn64_enrichment)
    dir_name = case.get('dir_name', '')

    cu_rows = []
    zn_rows = []
    for irrad_h in IRRADIATION_HOURS:
        for cool_d in COOLDOWN_DAYS:
            act = compute_activities(case, irrad_h, cool_d)
            cu_rows.append({
                'dir_name': dir_name, 'zn64_enrichment': zn64_enrichment,
                'zn_feedstock_cost': cost_kg * mass_kg, 'use_zn67': False,
                'zn_volume_cm3': vol, 'zn_density_g_cm3': rho, 'zn_mass_g': mass_g, 'zn_mass_kg': mass_kg,
                'irrad_hours': irrad_h, 'cooldown_days': cool_d,
                'cu64_mCi': act['cu64_mCi'], 'cu67_mCi': act['cu67_mCi'],
                'cu64_Bq': act['cu64_Bq'], 'cu67_Bq': act['cu67_Bq'],
                'cu64_atomic_purity': act['cu64_atomic_purity'], 'cu67_atomic_purity': act['cu67_atomic_purity'],
                'cu64_radionuclide_purity': act['cu64_radionuclide_purity'], 'cu67_radionuclide_purity': act['cu67_radionuclide_purity'],
            })
            zn_rows.append({
                'dir_name': dir_name, 'zn64_enrichment': zn64_enrichment,
                'zn_feedstock_cost': cost_kg * mass_kg, 'use_zn67': False,
                'zn_volume_cm3': vol, 'zn_density_g_cm3': rho, 'zn_mass_g': mass_g, 'zn_mass_kg': mass_kg,
                'irrad_hours': irrad_h, 'cooldown_days': cool_d,
                'zn65_mCi': act['zn65_mCi'], 'zn65_Bq': act['zn65_Bq'],
                'zn65_specific_activity_Bq_per_g': act['zn65_Bq'] / mass_g if mass_g > 0 else 0.0,
            })
    return pd.DataFrame(cu_rows), pd.DataFrame(zn_rows)


def run_one_sphere_case(zn64_enrichment, run_dir, use_log=False):
    """Run OpenMC, read total tallies, get reaction rates and initial atoms from summary, write cu_summary.csv and zn_summary.csv."""
    os.makedirs(run_dir, exist_ok=True)
    cwd = os.getcwd()

    geometry, materials, zn_cell = create_geometry(zn64_enrichment, use_log=use_log)
    settings = openmc.Settings()
    settings.source = create_source()
    settings.particles = PARTICLES
    settings.batches = BATCHES
    settings.run_mode = 'fixed source'

    model = openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=create_tallies(zn_cell))
    try:
        os.chdir(run_dir)
        model.export_to_xml()
        model.run()
    finally:
        os.chdir(cwd)

    volume_cm3 = (4.0 / 3.0) * np.pi * (SPHERE_OUTER_R_CM**3 - SPHERE_INNER_R_CM**3)
    sp_path = os.path.join(run_dir, f'statepoint.{BATCHES}.h5')
    if not os.path.isfile(sp_path):
        print(f"  [sphere] No statepoint at {sp_path}")
        return None

    sp = openmc.StatePoint(sp_path)
    rr = get_reaction_rates_per_s(sp)
    plot_flux_spectrum(sp, zn_cell, volume_cm3, run_dir)
    sp.close()

    zn_density = get_material_density_from_statepoint(sp_path, material_id=1)
    if zn_density is None:
        zn_density = 7.14
    initial_atoms = get_initial_atoms_from_statepoint(sp_path, material_id=1, volume_cm3=volume_cm3)
    if initial_atoms is None:
        initial_atoms = get_initial_zn_atoms_fallback(volume_cm3, zn64_enrichment, zn_density)
    else:
        zn_keys = ['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']
        tot = sum(float(initial_atoms.get(k, 0) or 0) for k in zn_keys)
        if tot <= 0 or float(initial_atoms.get('Zn64', 0) or 0) <= 0:
            initial_atoms = get_initial_zn_atoms_fallback(volume_cm3, zn64_enrichment, zn_density)

    case = {
        'reaction_rates': rr,
        'initial_atoms': initial_atoms,
        'outer_volume_cm3': volume_cm3,
        'zn_density_g_cm3': zn_density,
        'zn_mass_g': volume_cm3 * zn_density,
        'dir_name': os.path.basename(run_dir.rstrip(os.sep)),
    }

    cu_df, zn_df = build_summary_dataframes(case, zn64_enrichment)
    cu_df.to_csv(os.path.join(run_dir, 'cu_summary.csv'), index=False)
    zn_df.to_csv(os.path.join(run_dir, 'zn_summary.csv'), index=False)
    print(f"  [sphere] {run_dir}: cu_summary.csv ({len(cu_df)} rows), zn_summary.csv ({len(zn_df)} rows)")
    out = {**case, 'zn64_enrichment': zn64_enrichment, 'cu64_mCi': None, 'cu64_radionuclide_purity': None}
    sub = cu_df[(cu_df['irrad_hours'] == 1) & (cu_df['cooldown_days'] == 0)]
    if not sub.empty:
        out['cu64_mCi'] = float(sub['cu64_mCi'].iloc[0])
        out['cu64_radionuclide_purity'] = float(sub['cu64_radionuclide_purity'].iloc[0])
    return out


def run_sphere_linear():
    """Run all enrichments with linear isotope interpolation; save to sphere_linear/."""
    base = os.path.join(OUTPUT_DIR, 'sphere_linear')
    os.makedirs(base, exist_ok=True)
    plot_zn64_enrichment(os.path.join(base, 'zn64_enrichment_linear.png'), max_enrichment=0.999)
    results = []
    for e in ZN64_ENRICHMENTS:
        run_dir = os.path.join(base, f'enrich_{e*100:.2f}pct'.replace('.', 'p'))
        print(f"  [linear] {e*100:.2f}% -> {run_dir}")
        r = run_one_sphere_case(e, run_dir, use_log=False)
        if r:
            r['case_type'] = 'linear'
            results.append(r)
    return results


def run_sphere_log():
    """Run all enrichments with log isotope interpolation; save to sphere_log/."""
    base = os.path.join(OUTPUT_DIR, 'sphere_log')
    os.makedirs(base, exist_ok=True)
    plot_zn64_enrichment_log(os.path.join(base, 'zn64_enrichment_log.png'), max_enrichment=0.999)
    results = []
    for e in ZN64_ENRICHMENTS:
        run_dir = os.path.join(base, f'enrich_{e*100:.2f}pct'.replace('.', 'p'))
        print(f"  [log] {e*100:.2f}% -> {run_dir}")
        r = run_one_sphere_case(e, run_dir, use_log=True)
        if r:
            r['case_type'] = 'log'
            results.append(r)
    return results


def _enrich_label(e):
    """Format enrichment for plot labels: 0.999→'99.9%' (one sig fig), 0.99→'99%'."""
    v = float(e)
    if abs(v - 0.999) < 0.0005:
        return '99.9%'
    if abs(v - 0.99) < 0.005:
        return '99%'
    s = f'{v * 100:.2f}'.rstrip('0').rstrip('.')
    return f'{s}%'


def plot_production_vs_purity_vs_enrichment(linear_results, log_results):
    """Production vs purity: circle=linear, square=log; color=enrichment. Enrichment in colorbar (right margin), no labels on plot."""
    base = OUTPUT_DIR
    df_lin = pd.DataFrame(linear_results)
    df_log = pd.DataFrame(log_results)
    if df_lin.empty and df_log.empty:
        return
    all_enr = []
    for df in (df_lin, df_log):
        if not df.empty and 'zn64_enrichment' in df.columns:
            all_enr.extend(df['zn64_enrichment'].astype(float).tolist())
    vmin = (min(all_enr) * 100) if all_enr else 49.0
    vmax = (max(all_enr) * 100) if all_enr else 100.0

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.colormaps.get_cmap('viridis')
    sc = None
    for df, label, marker in [
        (df_lin, 'Linear (utilities)', 'o'),
        (df_log, 'Log (natural→99.18%)', 's'),
    ]:
        if df.empty or 'cu64_radionuclide_purity' not in df.columns:
            continue
        purity = df['cu64_radionuclide_purity'].astype(float) * 100
        prod = df['cu64_mCi'].astype(float)
        enr_pct = df['zn64_enrichment'].astype(float) * 100
        sc = ax.scatter(purity, prod, c=enr_pct, marker=marker, s=80, cmap=cmap, vmin=vmin, vmax=vmax,
                        label=label, alpha=0.85, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Cu-64 radionuclide purity (%)')
    ax.set_ylabel('Cu-64 production (mCi)')
    ax.set_title('Sphere: production vs purity (1 h irrad, 0 d cooldown)\nLinear vs log enrichment interpolation')
    ax.legend(loc='lower right', title='Interpolation')
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label='Zn-64 enrichment (%)', location='right')
        tick_vals = [vmin]
        for v in [71, 91, 99, 99.9]:
            if vmin <= v <= vmax:
                tick_vals.append(v)
        if vmax not in tick_vals:
            tick_vals.append(vmax)
        tick_vals = sorted(set(tick_vals))
        cbar.ax.set_yticks(tick_vals)
        cbar.ax.set_yticklabels([_enrich_label(t / 100.0).replace('%', '') for t in tick_vals])
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    path = os.path.join(base, 'sphere_production_vs_purity_linear_vs_log.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [sphere] Saved {path}")

    # Production vs enrichment and purity vs enrichment (enrichment on x-axis)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for df, label, marker, color in [
        (df_lin, 'Linear', 'o', 'C0'),
        (df_log, 'Log', 's', 'C1'),
    ]:
        if df.empty:
            continue
        enr = df['zn64_enrichment'].astype(float) * 100
        prod = df['cu64_mCi'].astype(float)
        purity = df['cu64_radionuclide_purity'].astype(float) * 100
        ax1.plot(enr, prod, marker=marker, color=color, label=label, linestyle='--', alpha=0.8)
        ax2.plot(enr, purity, marker=marker, color=color, label=label, linestyle='--', alpha=0.8)
    ax1.set_xlabel('Zn-64 enrichment (%)')
    ax1.set_ylabel('Cu-64 production (mCi)')
    ax1.set_title('Production vs enrichment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Zn-64 enrichment (%)')
    ax2.set_ylabel('Cu-64 radionuclide purity (%)')
    ax2.set_title('Radionuclide purity vs enrichment')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.suptitle('Sphere: linear vs log interpolation (1 h irrad, 0 d cooldown)', y=1.02)
    plt.tight_layout()
    path2 = os.path.join(base, 'sphere_production_and_purity_vs_enrichment.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [sphere] Saved {path2}")


def plot_and_table_analytical_vs_openmc(linear_results, output_dir=None):
    """
    Compare hand-calculation (cross-section) Cu-64 mCi to OpenMC for natural Zn (linear case).
    Writes a comparison table (CSV) and a plot (OpenMC vs Analytical vs irradiation time).
    """
    NATURAL_ZN_ENRICHMENT = 0.4917
    nat = next((r for r in linear_results if r is not None and abs(float(r.get('zn64_enrichment', 0)) - NATURAL_ZN_ENRICHMENT) < 1e-6), None)
    if nat is None:
        print("  [sphere] No natural Zn (49.17%) result in linear_results; skip analytical comparison.")
        return
    out_dir = output_dir or os.path.join(OUTPUT_DIR, 'sphere_linear')
    os.makedirs(out_dir, exist_ok=True)

    volume_cm3 = float(nat['outer_volume_cm3'])
    density = float(nat['zn_density_g_cm3'])
    rows = []
    for irrad_h in IRRADIATION_HOURS:
        act = compute_activities(nat, irrad_h, 0.0)
        openmc_mci = act['cu64_mCi']
        analytical_mci = _analytical_simple_cu64_mci(volume_cm3, density, NATURAL_ZN_ENRICHMENT, SOURCE_STRENGTH, irrad_h, verbose=False)
        ratio = openmc_mci / analytical_mci if analytical_mci > 0 else np.nan
        rows.append({
            'irrad_hours': irrad_h,
            'openmc_cu64_mCi': openmc_mci,
            'analytical_cu64_mCi': analytical_mci,
            'ratio_openmc_to_analytical': ratio,
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'analytical_vs_openmc_natural_zn.csv')
    df.to_csv(csv_path, index=False)
    print(f"  [sphere] Saved comparison table: {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['irrad_hours'], df['openmc_cu64_mCi'], 'o-', label='OpenMC (linear, natural Zn)', color='C0', lw=2)
    ax.plot(df['irrad_hours'], df['analytical_cu64_mCi'], 's--', label='Analytical (σ×φ×N, 14.1 MeV)', color='C1', lw=2)
    ax.set_xlabel('Irradiation time [h]')
    ax.set_ylabel('Cu-64 [mCi]')
    ax.set_title('Sphere: Cu-64 production — OpenMC vs hand calculation (natural Zn, 0 d cooldown)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'analytical_vs_openmc_natural_zn.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [sphere] Saved comparison plot: {plot_path}")


def main():
    """Run sphere: linear and log enrichment interpolation; plot enrichment curves and production vs purity comparison; compare to analytical Cu-64 (natural Zn)."""
    print("Sphere: linear + log enrichment interpolation; compare production vs r-purity vs enrichment")
    linear_results = run_sphere_linear()
    log_results = run_sphere_log()
    plot_production_vs_purity_vs_enrichment(linear_results, log_results)
    plot_and_table_analytical_vs_openmc(linear_results)
    print("Done.")


if __name__ == '__main__':
    main()
