#!/usr/bin/env python3
"""
FLARE NPV — data-driven financial analysis for Cu-64/Cu-67 production.

All calculations and plots use ONLY the cu_summary CSV from simple_analyze:
  - Rows: (thickness outer_cm, enrichment zn64_enrichment, loading zn_feedstock_cost, …)
  - Values: production (cu64_g_yr, cu67_g_yr) from utilities.load_run_data_from_cu_summary.
Production g/yr = (mCi at EOI) * (HOURS_PER_YEAR / irrad_hours) / (mCi/g); set run_config.NPV_IRRAD_HOURS (e.g. 1 or 8).
Interpolation (enrichment, thickness, budget) is done over these CSV values only.
Pricing from run_config only.

Entry: run_flare_combined(analyze_dirs, output_dir) or run_data_driven_analyses(output_dir).
"""

import csv
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from utilities import (
    load_run_data_from_cu_summary,
    find_cu_summary_csv,
    ZN64_CONTINGENCY_ENRICHMENTS,
    get_zn64_enrichment_cost_per_kg_contingency,
)

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fallback": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.2,
})

r = 0.1
T_years = 20
capex_usd = 2.0e7
opex_fixed_usd_per_yr = 2.0e5

reload_fraction_per_year = 0.0

# Product pricing: set from run_config (set_pricing_from_run_config)
price_cu64_usd_per_g = None
price_cu67_usd_per_g = None

# Purity lookup built from run data (zn64_enrichment -> cu64_radionuclide_purity)
cu64_purity_lookup = {}

# Data-driven mode: run data from fusion_irradiation cu_summary (production, mass, volume per case)
run_data_df = None
# Thickness grid: from run data (unique outer_cm) when available; else default run_config-style
THICKNESSES_CM = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
THICKNESSES_M = THICKNESSES_CM / 100.0
N_A = 6.02214076e23
A_cu64 = 63.930
A_cu67 = 66.928
t_half_cu64_s = 12.701 * 3600.0
t_half_cu67_s = 61.83 * 3600.0

# Contingency mode: only natural, 71%, 99% Zn-64; fixed $/kg (no interpolation). Set via run_config.FLARE_NPV_CONTINGENCY.
def _contingency_mode():
    try:
        import run_config as _rc
        return getattr(_rc, "FLARE_NPV_CONTINGENCY", False)
    except ImportError:
        return False


def _run_data_cu64_only_contingency_enrichments(df, tol=0.005):
    """When in contingency mode, filter to rows whose zn64_enrichment is one of ZN64_CONTINGENCY_ENRICHMENTS (for Cu-64 / Zn cost)."""
    if df is None or df.empty or not _contingency_mode() or "zn64_enrichment" not in df.columns:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        e = float(df.iloc[i].get("zn64_enrichment", np.nan))
        if get_zn64_enrichment_cost_per_kg_contingency(e, tol=tol) is not None:
            mask[i] = True
    return df.loc[mask].copy()


def _apply_contingency_to_run_data(df, tol=0.005):
    """Filter to Zn-64 enrichments (natural, 71%, 99%) only; set zn_feedstock_cost from contingency $/kg. No interpolation."""
    if df is None or df.empty or "zn64_enrichment" not in df.columns or "zn_mass_g" not in df.columns:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        e = float(df.iloc[i].get("zn64_enrichment", np.nan))
        if get_zn64_enrichment_cost_per_kg_contingency(e, tol=tol) is not None:
            mask[i] = True
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    mass_kg = sub["zn_mass_g"].astype(float).values / 1000.0
    loading = np.array(
        [
            get_zn64_enrichment_cost_per_kg_contingency(float(sub.iloc[j]["zn64_enrichment"]), tol=tol) * mass_kg[j]
            for j in range(len(sub))
        ]
    )
    sub["zn_feedstock_cost"] = loading
    return sub


# Output layout: run_data/{scenario}/ for plots; run_data/csv/ for interpolated CSVs
FOLDER_RUN_DATA = "run_data"
FOLDER_RUN_DATA_VARIABLE = "run_data_variable"
FOLDER_CSV = "csv"
RUN_DATA_SELL_ALL = "sell_all"
RUN_DATA_MARKET_CAP = "market_cap"
RUN_DATA_PURITY_CAP = "purity_cap"           # Cu-64 revenue only if radionuclide purity ≥99.9%
RUN_DATA_MARKET_PURITY_CAP = "market_purity_cap"  # Market cap + Cu-64 purity ≥99.9%
FOLDER_ADDITIONAL_PLOTS = "additional_plots"


def _title_suffix(tag, purity_cap_64):
    """Title suffix for plot labels. Handles all four scenario tags."""
    if tag == RUN_DATA_SELL_ALL:
        return " — Sell all"
    if tag == RUN_DATA_MARKET_CAP:
        return " — Market cap"
    if tag == RUN_DATA_PURITY_CAP:
        return " — Purity ≥99.9%"
    if tag == RUN_DATA_MARKET_PURITY_CAP:
        return " — Cap + purity"
    return ""

# Cu-64: cool gradient — deep navy/purple → blue → teal → green → lime green
_COOL_HEX = [
    "#0d0221", "#1a0a2e", "#16213e", "#1a237e", "#283593", "#3949ab", "#5c6bc0",
    "#0097a7", "#00838f", "#00695c", "#1b5e20", "#2e7d32", "#43a047", "#66bb6a",
    "#81c784", "#9ccc65", "#aed581", "#c5e1a5", "#cddc39", "#d4e157", "#dcedc8", "#e8f5e9",
]
# Cu-67: warm gradient — dark red → bright red → orange → yellow
_WARM_HEX = [
    "#3d0a0a", "#5c0a0a", "#8b0000", "#a52a2a", "#b22222", "#c0392b", "#dc143c",
    "#e74c3c", "#ff4444", "#ff6347", "#ff7f50", "#ff8c00", "#ffa500", "#ffb347",
    "#ffc107", "#ffd700", "#ffec8b", "#fff176", "#fff59d", "#fff9c4", "#fffde7",
]
# Revenue-ceiling palettes: vibrant only (no pale/neon ends). Cool: navy→green; warm: red→amber.
_COOL_HEX_VIBRANT = [
    "#0d0221", "#1a237e", "#3949ab", "#0097a7", "#00695c", "#1b5e20", "#2e7d32", "#43a047", "#66bb6a",
]
_WARM_HEX_VIBRANT = [
    "#8b0000", "#b22222", "#c0392b", "#dc143c", "#e74c3c", "#ff6347", "#ff8c00", "#ffa500", "#ffb347",
]


def _cool_palette_cu64(n=12):
    """Return (n, 4) rgba: evenly sampled from deep navy/purple → lime green."""
    rgba = np.array([mcolors.to_rgba(h) for h in _COOL_HEX])
    if n <= len(rgba):
        idx = np.linspace(0, len(rgba) - 1, n, dtype=int)
        return rgba[idx]
    return np.array([rgba[i % len(rgba)] for i in range(n)])


def _warm_palette_cu67(n=12):
    """Return (n, 4) rgba: evenly sampled from dark red → yellow."""
    rgba = np.array([mcolors.to_rgba(h) for h in _WARM_HEX])
    if n <= len(rgba):
        idx = np.linspace(0, len(rgba) - 1, n, dtype=int)
        return rgba[idx]
    return np.array([rgba[i % len(rgba)] for i in range(n)])


def _cool_palette_vibrant(n):
    """Return (n, 4) rgba for revenue ceiling etc.: vibrant cool only (no pale/neon green)."""
    rgba = np.array([mcolors.to_rgba(h) for h in _COOL_HEX_VIBRANT])
    idx = np.linspace(0, len(rgba) - 1, n, dtype=int)
    return rgba[idx]


def _warm_palette_vibrant(n):
    """Return (n, 4) rgba for revenue ceiling etc.: vibrant warm only (no pale yellow)."""
    rgba = np.array([mcolors.to_rgba(h) for h in _WARM_HEX_VIBRANT])
    idx = np.linspace(0, len(rgba) - 1, n, dtype=int)
    return rgba[idx]


def _get_enrichment_color(enrichment, is_cu67=False):
    """
    Get color for a specific enrichment level. Ensures 99% (and 99.9% if present) gets lime green (#cddc39).
    Uses vibrant colors only, never pale.
    """
    e = float(enrichment)
    
    # Special case: 99% (contingency high) or 99.9% always gets lime green
    if abs(e - 0.99) < 0.001 or abs(e - 0.999) < 0.001:
        return mcolors.to_rgba('#cddc39')  # Bright lime green
    
    if is_cu67:
        # Cu-67: vibrant warm colors (reds, oranges, yellows) - avoid pale yellows
        warm_vibrant = ['#b41f24', '#c0392b', '#e74c3c', '#dc143c', '#ff4444', 
                       '#ff6347', '#ff7f50', '#ff8c00', '#ffa500', '#ffb347',
                       '#ffc107', '#ffd700']  # Stop before pale yellows
        # Map enrichment to index (assuming enrichments like 0.017, 0.0404, 0.117)
        if e < 0.05:
            idx = int((e / 0.05) * 4) % len(warm_vibrant)
        else:
            idx = min(4 + int((e - 0.05) / 0.15 * 8), len(warm_vibrant) - 1)
        return mcolors.to_rgba(warm_vibrant[idx])
    else:
        # Cu-64: vibrant cool colors (blues, teals, greens) - ensure 99%/99.9% handled above
        cool_vibrant = ['#0047ba', '#0066cc', '#0080ff', '#1e90ff', '#4169e1', 
                       '#6495ed', '#0097a7', '#00838f', '#00695c', '#1b5e20',
                       '#2e7d32', '#43a047', '#66bb6a', '#81c784', '#9ccc65']
        # Map enrichment to index (assuming enrichments like 0.4917, 0.71, 0.81, 0.91, 0.99)
        if e < 0.5:
            idx = int((e / 0.5) * 3) % len(cool_vibrant)
        elif e < 0.8:
            idx = 3 + int((e - 0.5) / 0.3 * 4) % (len(cool_vibrant) - 3)
        elif e < 0.95:
            idx = 7 + int((e - 0.8) / 0.15 * 4) % (len(cool_vibrant) - 7)
        else:  # 0.95-0.99 (99%/99.9% handled separately)
            idx = 11 + int((e - 0.95) / 0.04 * 3) % (len(cool_vibrant) - 11)
        return mcolors.to_rgba(cool_vibrant[min(idx, len(cool_vibrant) - 1)])


def _plot_path(output_dir, subfolder, filename):
    """Return full path for plot; ensure subfolder exists under output_dir."""
    if not output_dir:
        return filename
    path = os.path.join(output_dir, subfolder)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def _csv_path(output_dir, filename):
    """Return path for CSV under output_dir/run_data/csv; ensure folder exists."""
    if not output_dir:
        return filename
    path = os.path.join(output_dir, FOLDER_RUN_DATA, FOLDER_CSV)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def _get_unique_enrichments(rows, tol=0.001):
    """Unique enrichments from row dicts (tolerance-based grouping)."""
    raw = [r["enrichment"] for r in rows if not np.isnan(r.get("enrichment", np.nan))]
    if not raw:
        return []
    raw = sorted(set(raw))
    out = []
    for e in raw:
        if not any(abs(e - u) <= tol for u in out):
            out.append(e)
    return sorted(out)



def _specific_activity_ci_per_g(isotope):
    """Specific activity in Ci/g (physical constant)."""
    if isotope == "64":
        return N_A * np.log(2) / (A_cu64 * t_half_cu64_s * 3.7e10)
    return N_A * np.log(2) / (A_cu67 * t_half_cu67_s * 3.7e10)


def set_pricing_from_run_config():
    global price_cu64_usd_per_g, price_cu67_usd_per_g
    try:
        import run_config as _c
        p64_mci = getattr(_c, "ECON_CU64_PRICE_PER_MCI", None)
        p67_mci = getattr(_c, "ECON_CU67_PRICE_PER_MCI", None)
    except ImportError:
        p64_mci = p67_mci = None
    if p64_mci is None or p67_mci is None:
        raise ValueError(
            "NPV pricing must come from run_config. Set ECON_CU64_PRICE_PER_MCI and ECON_CU67_PRICE_PER_MCI."
        )
    # $/g = $/mCi * mCi/g; 1 Ci = 1000 mCi so mCi/g = 1000 * Ci/g
    price_cu64_usd_per_g = float(p64_mci) * 1000.0 * _specific_activity_ci_per_g("64")
    price_cu67_usd_per_g = float(p67_mci) * 1000.0 * _specific_activity_ci_per_g("67")



def _default_npv_irrad_hours():
    """Production basis: run_config.NPV_IRRAD_HOURS (e.g. 1 or 8); g/yr = mCi * (24*365/irrad_h) / (mCi/g)."""
    try:
        import run_config as _c
        return getattr(_c, 'NPV_IRRAD_HOURS', getattr(_c, 'ECON_IRRAD_HOURS', 8760))
    except ImportError:
        return 8760


def set_run_data_from_csv(csv_path, irrad_hours=None, cooldown_days=0):
    """
    Load full run data from cu_summary for data-driven NPV.
    irrad_hours: which run length to use (e.g. run_config.NPV_IRRAD_HOURS). If None, uses NPV_IRRAD_HOURS.
    g/yr from utilities: mCi at EOI * (HOURS_PER_YEAR / irrad_hours) / (mCi/g). Call before run_data_driven_analyses.
    """
    global run_data_df, cu64_purity_lookup
    if irrad_hours is None:
        irrad_hours = _default_npv_irrad_hours()
    run_data_df = load_run_data_from_cu_summary(
        csv_path, irrad_hours=irrad_hours, cooldown_days=cooldown_days
    )
    if run_data_df is not None and not run_data_df.empty and _contingency_mode():
        run_data_df = _apply_contingency_to_run_data(run_data_df)
        if run_data_df is not None and not run_data_df.empty:
            print(f"  Contingency: filtered to enrichments {ZN64_CONTINGENCY_ENRICHMENTS}, loading = $/kg × zn_mass_kg (no interpolation)")
    if run_data_df is not None and not run_data_df.empty:
        cu64_purity_lookup = {
            float(row['zn64_enrichment']): float(row.get('cu64_radionuclide_purity', 0) or 0)
            for _, row in run_data_df.drop_duplicates('zn64_enrichment').iterrows()
        }
        n_rows = len(run_data_df)
        geom_keys = run_data_df['dir_name'].nunique() if 'dir_name' in run_data_df.columns else 1
        cu64_min = run_data_df['cu64_g_yr'].min() if 'cu64_g_yr' in run_data_df.columns else 0
        cu64_max = run_data_df['cu64_g_yr'].max() if 'cu64_g_yr' in run_data_df.columns else 0
        print(f"  Loaded {n_rows} run cases ({geom_keys} geometries) at {irrad_hours}h, {cooldown_days}d cooldown")
        print(f"  Cu-64 production: {cu64_min*1e3:.4f}–{cu64_max*1e3:.4f} mg/yr")
        _update_thicknesses_from_run_data()
        print(f"  Thickness grid from run data: {THICKNESSES_CM.tolist()} cm")
    else:
        run_data_df = None
        print("  No run data loaded (CSV missing or no matching rows)")


def run_flare_from_openmc(analyze_dir, econ_irrad_hours=None, cooldown_days=0, output_dir=None):
    """
    Load run data from cu_summary under analyze_dir, run data-driven NPV analyses.
    econ_irrad_hours: basis for g/yr (e.g. run_config.NPV_IRRAD_HOURS). If None, uses NPV_IRRAD_HOURS.
    output_dir: npv folder.
    """
    if econ_irrad_hours is None:
        econ_irrad_hours = _default_npv_irrad_hours()
    csv_path = find_cu_summary_csv(analyze_dir)
    if not csv_path:
        raise FileNotFoundError(
            f"No cu_summary CSV found under {analyze_dir}. "
            "Run fusion_irradiation and simple_analyze first (per run_config)."
        )
    print(f"  Loading production from {csv_path} (irrad={econ_irrad_hours}h, cooldown={cooldown_days}d)")
    set_pricing_from_run_config()
    set_run_data_from_csv(csv_path, irrad_hours=econ_irrad_hours, cooldown_days=cooldown_days)
    if run_data_df is None or run_data_df.empty:
        raise ValueError(
            f"cu_summary at {csv_path} has no rows for {econ_irrad_hours}h, {cooldown_days}d. "
            "Check run_config IRRADIATION_HOURS/COOLDOWN_DAYS vs ECON_IRRAD_HOURS/ECON_COOLDOWN_DAYS."
        )

    run_data_driven_analyses(output_dir=output_dir)


def run_flare_combined(analyze_dirs, output_dir, econ_irrad_hours=None, cooldown_days=0):
    """
    Load run data from multiple analyze dirs (e.g. runcase_4 Cu-64 and runcase_5 Cu-67),
    combine into one dataset, and run data-driven NPV analyses only.
    econ_irrad_hours: which run length to use for g/yr (e.g. run_config.NPV_IRRAD_HOURS). If None, uses NPV_IRRAD_HOURS.
    g/yr = mCi at EOI * (HOURS_PER_YEAR / irrad_hours) / (mCi/g) from utilities.
    """
    global run_data_df, cu64_purity_lookup
    if econ_irrad_hours is None:
        econ_irrad_hours = _default_npv_irrad_hours()
    set_pricing_from_run_config()
    print(f"  NPV production basis: {econ_irrad_hours} h irradiation (g/yr = mCi * {24*365}/irrad_h / (mCi/g))")
    dfs = []
    for adir in analyze_dirs:
        csv_path = find_cu_summary_csv(adir)
        if not csv_path:
            continue
        df = load_run_data_from_cu_summary(
            csv_path, irrad_hours=econ_irrad_hours, cooldown_days=cooldown_days
        )
        if df is not None and not df.empty:
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {csv_path}")
    if not dfs:
        raise FileNotFoundError(
            "No cu_summary CSV found under any of: " + ", ".join(analyze_dirs) + ". "
            "Run fusion_irradiation and simple_analyze for each case first."
        )
    run_data_df = pd.concat(dfs, ignore_index=True)
    if _contingency_mode():
        run_data_df = _apply_contingency_to_run_data(run_data_df)
        if run_data_df is not None and not run_data_df.empty:
            print(f"  Contingency: only enrichments {ZN64_CONTINGENCY_ENRICHMENTS}, loading = contingency $/kg × zn_mass_kg (no interpolation)")
        elif run_data_df is None or run_data_df.empty:
            raise ValueError("Contingency mode: no run data left after filtering to enrichments (0.4917, 0.71, 0.99). Ensure cu_summary has rows for these.")
    cu64_purity_lookup = {}
    if 'zn64_enrichment' in run_data_df.columns and 'cu64_radionuclide_purity' in run_data_df.columns:
        for _, row in run_data_df.drop_duplicates('zn64_enrichment').iterrows():
            e = float(row['zn64_enrichment'])
            p = float(row.get('cu64_radionuclide_purity', 0) or 0)
            cu64_purity_lookup[e] = p
    n_rows = len(run_data_df)
    n_cu64 = (run_data_df['use_zn67'] == False).sum() if 'use_zn67' in run_data_df.columns else 0
    n_cu67 = (run_data_df['use_zn67'] == True).sum() if 'use_zn67' in run_data_df.columns else 0
    print(f"  Combined {n_rows} run cases (Cu-64: {n_cu64}, Cu-67: {n_cu67}) at {econ_irrad_hours}h, {cooldown_days}d cooldown")
    _update_thicknesses_from_run_data()
    print(f"  Thickness grid from run data: {THICKNESSES_CM.tolist()} cm")
    run_data_driven_analyses(output_dir=output_dir)


def annuity_factor(rate, n_years):
    """Present value annuity factor: sum of 1/(1+r)^t for t = 1..n."""
    if rate == 0:
        return float(n_years)
    return (1.0 - (1.0 + rate)**(-n_years)) / rate




def _npv_from_run_row(row, sell_fraction, cap_usd_per_yr, is_cu67=False, price_override_usd_per_g=None, purity_cap_64=False, loading_multiplier=1.0):
    """NPV for one run-data row.
    Uses only summary CSV data: production (cu64_g_yr/cu67_g_yr) and loading (zn_feedstock_cost).
    loading_multiplier: scale Zn cost (1.0 = expected; 1.5 = Zn costs 1.5× expected)."""
    base_price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
    if base_price is None and price_override_usd_per_g is None:
        raise ValueError("Pricing not set. Run set_pricing_from_run_config() first.")
    price = price_override_usd_per_g if price_override_usd_per_g is not None else base_price
    prod_g_yr = float(row['cu67_g_yr'] if is_cu67 else row['cu64_g_yr'])
    purity = float(row.get('cu64_radionuclide_purity', 0) or 0) if not is_cu67 else 1.0
    if row.get('zn_feedstock_cost') is None:
        raise ValueError("NPV requires zn_feedstock_cost from OpenMC/simple_analyze CSV. No fallback.")
    _load = row['zn_feedstock_cost']
    if (isinstance(_load, (float, np.floating)) and np.isnan(_load)) or str(_load) == 'nan':
        raise ValueError("zn_feedstock_cost is NaN; CSV must contain real loading cost from run.")
    loading = float(_load) * float(loading_multiplier)
    rev = sell_fraction * prod_g_yr * price
    if not is_cu67 and purity_cap_64:
        rev = rev if purity >= 0.999 else 0.0
    if cap_usd_per_yr is not None:
        rev = min(rev, float(cap_usd_per_yr))
    af = annuity_factor(r, T_years)
    annual_net = rev - opex_fixed_usd_per_yr - reload_fraction_per_year * loading
    return -capex_usd - loading + af * annual_net


def _payback_from_run_row(row, cap_usd_per_yr=None, is_cu67=False, purity_cap_64=False):
    """Undiscounted payback period in years for one run-data row. Returns np.inf if annual_net <= 0.
    purity_cap_64: if True and Cu-64, revenue only when purity >= 99.9%."""
    if price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        raise ValueError("Pricing not set. Run set_pricing_from_run_config() first.")
    prod_g_yr = float(row['cu67_g_yr'] if is_cu67 else row['cu64_g_yr'])
    purity = float(row.get('cu64_radionuclide_purity', 0) or 0) if not is_cu67 else 1.0
    if row.get('zn_feedstock_cost') is None:
        raise ValueError("Payback requires zn_feedstock_cost from CSV.")
    _load = row['zn_feedstock_cost']
    if (isinstance(_load, (float, np.floating)) and np.isnan(_load)) or str(_load) == 'nan':
        return np.inf
    loading = float(_load)
    price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
    rev = prod_g_yr * price
    if not is_cu67 and purity_cap_64:
        rev = rev if purity >= 0.999 else 0.0
    if cap_usd_per_yr is not None:
        rev = min(rev, float(cap_usd_per_yr))
    annual_net = rev - opex_fixed_usd_per_yr - reload_fraction_per_year * loading
    total_upfront = capex_usd + loading
    if annual_net <= 0:
        return np.inf
    return total_upfront / annual_net


def _irr_from_run_row(row, sell_fraction, cap_usd_per_yr, is_cu67=False, purity_cap_64=False):
    """IRR for one run-data row. Uses only real OpenMC results from CSV: production, loading cost. No physics scaling.
    purity_cap_64: if True and Cu-64, revenue only when purity >= 99.9%."""
    if price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        raise ValueError("Pricing not set. Run set_pricing_from_run_config() first.")
    prod_g_yr = float(row['cu67_g_yr'] if is_cu67 else row['cu64_g_yr'])
    purity = float(row.get('cu64_radionuclide_purity', 0) or 0) if not is_cu67 else 1.0
    if row.get('zn_feedstock_cost') is None:
        raise ValueError("IRR requires zn_feedstock_cost from OpenMC/simple_analyze CSV. No fallback.")
    _load = row['zn_feedstock_cost']
    if (isinstance(_load, (float, np.floating)) and np.isnan(_load)) or str(_load) == 'nan':
        raise ValueError("zn_feedstock_cost is NaN; CSV must contain real loading cost from run.")
    loading = float(_load)
    if is_cu67:
        rev = sell_fraction * prod_g_yr * price_cu67_usd_per_g
    else:
        rev = sell_fraction * prod_g_yr * price_cu64_usd_per_g
    if not is_cu67 and purity_cap_64:
        rev = rev if purity >= 0.999 else 0.0
    if cap_usd_per_yr is not None:
        rev = min(rev, float(cap_usd_per_yr))
    annual_net = rev - opex_fixed_usd_per_yr - reload_fraction_per_year * loading
    if annual_net <= 0:
        return np.nan
    total_upfront = capex_usd + loading
    def npv_at_rate(rate):
        return -total_upfront + annual_net * annuity_factor(rate, T_years)
    if npv_at_rate(0.0) < 0:
        return np.nan
    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if npv_at_rate(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def run_data_driven_scenario(tag, sell_fraction=1.0, cap64=None, cap67=None, is_cu67=None, purity_cap_64=False, output_dir=None):
    """
    NPV scenario using run data: production, mass, volume from fusion_irradiation CSV.
    Pricing from run_config. Saves to run_data/{tag}/.
    purity_cap_64: if True, Cu-64 revenue only when purity >= 99.9%.
    """
    if run_data_df is None or run_data_df.empty:
        print("  No run data; skipping data-driven scenario")
        return
    if price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        raise ValueError("Pricing not set. Run set_pricing_from_run_config() first (e.g. via run_flare_from_openmc).")
    df = run_data_df.copy()
    keys = ['dir_name', 'zn64_enrichment']
    df = df.drop_duplicates(subset=[c for c in keys if c in df.columns])
    has_zn67 = df['use_zn67'].any() if 'use_zn67' in df.columns else False
    has_zn64 = not df['use_zn67'].all() if 'use_zn67' in df.columns else True
    modes = []
    if is_cu67 is True or (is_cu67 is None and has_zn67):
        modes.append(True)
    if is_cu67 is False or (is_cu67 is None and has_zn64):
        modes.append(False)
    for is_cu67_mode in modes:
        sub = df[df['use_zn67'] == is_cu67_mode] if 'use_zn67' in df.columns else df
        if sub.empty:
            continue
        if not is_cu67_mode and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            continue
        isotope_label = "Cu-67 (Zn-67)" if is_cu67_mode else "Cu-64"
        cap = cap67 if is_cu67_mode else cap64
        print("\n" + "=" * 80)
        print(f"  DATA-DRIVEN SCENARIO: {tag} ({isotope_label})")
        print(f"  Source: fusion_irradiation run (production, mass, volume from CSV)")
        print(f"  Sell fraction: {sell_fraction:.0%}")
        cap_s = "None" if cap is None else f"${cap/1e6:.1f}M"
        purity_s = " (Cu-64 revenue only if purity>=99.9%)" if (purity_cap_64 and not is_cu67_mode) else ""
        print(f"  Cap: {cap_s}{purity_s}")
        print("=" * 80)
        sep = "─" * 115
        print(f"\n  {sep}")
        print(f"  {'Geometry':<30}  {'Enrich':>8}  {'Prod (mg/yr)':>12}  {'mCi/1h':>10}  {'Mass (kg)':>10}  {'Revenue ($/yr)':>16}  {'Loading ($)':>14}  {'NPV ($M)':>10}  {'IRR':>8}")
        print(f"  {sep}")
        iso = "67" if is_cu67_mode else "64"
        sa_mci_per_g = 1000.0 * _specific_activity_ci_per_g(iso)
        for _, row in sub.iterrows():
            geom = str(row.get('dir_name', row.get('outer_cm', '?')))[:28]
            enrich = float(row['zn64_enrichment'])
            prod = float(row['cu67_g_yr'] if is_cu67_mode else row['cu64_g_yr'])
            # mCi at 1h: from CSV EOI mCi when irrad=1h, else back-calc from g/yr (mCi_yr/8760 = prod*sa_mci_per_g/8760)
            _hours_yr = 24.0 * 365.0
            irrad_h = float(row.get('irrad_hours', 1))
            if irrad_h > 0 and row.get('cu67_mCi' if is_cu67_mode else 'cu64_mCi') is not None:
                try:
                    mci_eoi = float(row['cu67_mCi'] if is_cu67_mode else row['cu64_mCi'])
                    mci_1h = mci_eoi / irrad_h  # activity per hour (same as EOI mCi when irrad_h=1)
                except (TypeError, ValueError):
                    mci_1h = prod * sa_mci_per_g / _hours_yr
            else:
                mci_1h = prod * sa_mci_per_g / _hours_yr
            mass = float(row.get('zn_mass_kg', row.get('zn_mass_g', 0) / 1000.0))
            purity = float(row.get('cu64_radionuclide_purity', 0) or 0) if not is_cu67_mode else 1.0
            # Loading cost from CSV only (no physics fallback)
            if row.get('zn_feedstock_cost') is None:
                raise ValueError("NPV requires zn_feedstock_cost in CSV. No fallback.")
            _load = row['zn_feedstock_cost']
            if (isinstance(_load, (float, np.floating)) and np.isnan(_load)) or str(_load) == 'nan':
                raise ValueError("zn_feedstock_cost is NaN in CSV.")
            load = float(_load)
            if is_cu67_mode:
                rev = sell_fraction * prod * price_cu67_usd_per_g
            else:
                rev = sell_fraction * prod * price_cu64_usd_per_g
                if purity_cap_64:
                    rev = rev if purity >= 0.999 else 0.0
            if cap is not None:
                rev = min(rev, float(cap))
            n = _npv_from_run_row(row, sell_fraction, cap, is_cu67_mode, purity_cap_64=purity_cap_64)
            ir = _irr_from_run_row(row, sell_fraction, cap, is_cu67_mode, purity_cap_64=purity_cap_64)
            ir_s = f"{ir*100:.1f}%" if not np.isnan(ir) else "N/A"
            prod_mg = prod * 1e3
            prod_str = f"{prod_mg:12.3f}" if prod_mg >= 0.001 else f"{prod_mg:12.2e}"
            mci_str = f"{mci_1h:10.2f}" if mci_1h >= 0.01 else f"{mci_1h:10.2e}"
            print(f"  {geom:<30}  {enrich*100:7.2f}%  {prod_str}  {mci_str}  {mass:10.1f}  {rev:16,.0f}  {load:14,.0f}  {n/1e6:10.2f}  {ir_s:>8}")

    # Combined plots only (enrichment, thickness, payback, IRR, investor) — produced per tag in run_data_driven_analyses; no _cu64/_cu67-only JPEGs.


def run_data_driven_enrichment_plots_combined(output_dir=None, tag="sell_all", cap64=None, cap67=None, purity_cap_64=False, n_interp=60):
    """NPV, production, revenue vs enrichment from run data — one JPEG with 6 subplots: 3 for Cu-64, 3 for Cu-67.
    One point per enrichment at a single reference thickness (max thickness).
    purity_cap_64: Cu-64 revenue only when purity >= 99.9%.
    In contingency mode: no interpolation between enrichments (only the 3 points: natural, 71%, 99%)."""
    if _contingency_mode():
        n_interp = 3
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    df = run_data_df.copy()
    sf = os.path.join(FOLDER_RUN_DATA, tag)
    _sub = _title_suffix(tag, purity_cap_64)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    # Row 0: Cu-64 (NPV, Production, Revenue). Row 1: Cu-67 (NPV, Production, Revenue).
    y_config = [
        ("npv", "NPV (USD, millions)", "NPV vs Enrichment"),
        ("prod_mg", "Production (mg/yr)", "Production vs Enrichment"),
        ("rev", "Revenue (USD millions/yr)", "Revenue vs Enrichment"),
    ]
    for is_cu67, row_idx in [(False, 0), (True, 1)]:
        label = "Cu-67" if is_cu67 else "Cu-64"
        line_color = _warm_palette_cu67(1)[0] if is_cu67 else _cool_palette_cu64(1)[0]
        sub = df[df['use_zn67'] == is_cu67] if 'use_zn67' in df.columns else df
        if sub.empty:
            for col_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            for col_idx in range(3):
                axes[row_idx, col_idx].set_visible(False)
            continue
        if 'outer_cm' in sub.columns:
            ref_t = float(sub['outer_cm'].astype(float).max())
            sub = sub[np.isclose(sub['outer_cm'].astype(float), ref_t, atol=0.02)].copy()
        cap = (cap67 if is_cu67 else cap64)
        enr = sub['zn64_enrichment'].astype(float).values
        for col_idx, (ykey, ylabel, title_base) in enumerate(y_config):
            ax = axes[row_idx, col_idx]
            if ykey == "npv":
                vals = np.array([_npv_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64) / 1e6 for _, row in sub.iterrows()])
            elif ykey == "prod_mg":
                vals = sub['cu67_g_yr' if is_cu67 else 'cu64_g_yr'].astype(float).values * 1e3
            else:
                prod = sub['cu67_g_yr' if is_cu67 else 'cu64_g_yr'].astype(float).values
                price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
                rev = prod * price
                if not is_cu67 and purity_cap_64:
                    purity_arr = np.array([float(row.get('cu64_radionuclide_purity', 0) or 0) for _, row in sub.iterrows()])
                    rev = np.where(purity_arr >= 0.999, rev, 0.0)
                if cap is not None:
                    rev = np.minimum(rev, float(cap))
                vals = rev / 1e6
            unq_enr, idx_inv = np.unique(enr, return_inverse=True)
            mean_vals = np.array([np.mean(vals[idx_inv == i]) for i in range(len(unq_enr))])
            enr_plot, vals_plot = unq_enr, mean_vals
            order = np.argsort(enr_plot)
            enr_plot, vals_plot = enr_plot[order], vals_plot[order]
            ax.scatter(enr_plot * 100, vals_plot, s=70, marker="s" if is_cu67 else "o", zorder=3, edgecolors="k", linewidths=0.5, color=line_color, label=label)
            if len(enr_plot) >= 2:
                enr_fine = np.linspace(enr_plot.min(), enr_plot.max(), n_interp)
                vals_fine = np.interp(enr_fine, enr_plot, vals_plot)
                ax.plot(enr_fine * 100, vals_fine, "-", lw=1.8, alpha=0.8, color=line_color)
            ax.set_xlabel("Enrichment (%)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{label}: {title_base}{_sub}")
            ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if output_dir:
        plt.savefig(_plot_path(output_dir, sf, "npv_production_revenue_vs_enrichment_combined_run_data.jpeg"), dpi=500)
    plt.close(fig)

    # CSV: interpolated enrichment curves
    if output_dir:
        has_purity = "cu64_radionuclide_purity" in df.columns
        for is_cu67, iso_name in [(False, "cu64"), (True, "cu67")]:
            sub = df[df['use_zn67'] == is_cu67] if 'use_zn67' in df.columns else df
            if sub.empty or len(sub) < 2:
                continue
            if 'outer_cm' in sub.columns:
                ref_t = float(sub['outer_cm'].astype(float).max())
                sub = sub[np.isclose(sub['outer_cm'].astype(float), ref_t, atol=0.02)].copy()
            cap = (cap67 if is_cu67 else cap64)
            enr = sub['zn64_enrichment'].astype(float).values
            load = np.array([float(row.get('zn_feedstock_cost', 0) or 0) for _, row in sub.iterrows()])
            npv_ = np.array([_npv_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64) / 1e6 for _, row in sub.iterrows()])
            prod_g = sub['cu67_g_yr' if is_cu67 else 'cu64_g_yr'].astype(float).values
            prod_mg = prod_g * 1e3
            price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
            rev = prod_g * price
            if not is_cu67 and purity_cap_64:
                purity_pct = np.array([float(row.get('cu64_radionuclide_purity', 0) or 0) * 100 for _, row in sub.iterrows()])
                rev = np.where(purity_pct >= 99.9, rev, 0.0)
            if cap is not None:
                rev = np.minimum(rev, float(cap))
            rev_m = rev / 1e6
            purity_pct = np.array([float(row.get('cu64_radionuclide_purity', 0) or 0) * 100 for _, row in sub.iterrows()]) if (has_purity and not is_cu67) else None
            prod_999 = np.where((purity_pct >= 99.9) if purity_pct is not None else np.zeros_like(prod_mg), prod_mg, 0.0) if (has_purity and not is_cu67) else None
            # One point per enrichment (at ref thickness; mean only if duplicate enrichments at same thickness)
            unq_enr, idx_inv = np.unique(enr, return_inverse=True)
            enr = unq_enr.copy()
            load = np.array([np.mean(load[idx_inv == i]) for i in range(len(unq_enr))])
            npv_ = np.array([np.mean(npv_[idx_inv == i]) for i in range(len(unq_enr))])
            prod_mg = np.array([np.mean(prod_mg[idx_inv == i]) for i in range(len(unq_enr))])
            rev_m = np.array([np.mean(rev_m[idx_inv == i]) for i in range(len(unq_enr))])
            if purity_pct is not None:
                purity_pct = np.array([np.mean(purity_pct[idx_inv == i]) for i in range(len(unq_enr))])
                prod_999 = np.array([np.mean(prod_999[idx_inv == i]) for i in range(len(unq_enr))])
            order = np.argsort(enr)
            enr, load, npv_, prod_mg, rev_m = enr[order], load[order], npv_[order], prod_mg[order], rev_m[order]
            if purity_pct is not None:
                purity_pct, prod_999 = purity_pct[order], prod_999[order]
            enr_fine = np.linspace(enr.min(), enr.max(), n_interp)
            load_f = np.interp(enr_fine, enr, load)
            npv_f = np.interp(enr_fine, enr, npv_)
            prod_f = np.interp(enr_fine, enr, prod_mg)
            rev_f = np.interp(enr_fine, enr, rev_m)
            purity_f = np.interp(enr_fine, enr, purity_pct) if purity_pct is not None else None
            prod_999_f = np.interp(enr_fine, enr, prod_999) if prod_999 is not None else None
            path = _csv_path(output_dir, f"enrichment_{tag}_{iso_name}_interpolated.csv")
            add_mci = (not is_cu67)  # Cu-64: add mCi/yr for sanity check (e.g. 81% enrich, 20 cm)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                headers = ["enrichment_pct", "loading_cost_usd", "production_mg_yr", "revenue_millions_yr", "npv_millions"]
                if add_mci:
                    headers += ["production_mCi_yr"]
                if purity_f is not None:
                    headers += ["purity_pct", "production_above_99_9pct_mg_yr"]
                w.writerow(headers)
                for i in range(len(enr_fine)):
                    row = [round(enr_fine[i] * 100, 4), round(load_f[i], 2), round(prod_f[i], 4), round(rev_f[i], 4), round(npv_f[i], 4)]
                    if add_mci:
                        prod_g_i = prod_f[i] / 1e3
                        mci_yr = prod_g_i * _specific_activity_ci_per_g("64") * 1000.0
                        row += [round(mci_yr, 2)]
                    if purity_f is not None:
                        row += [round(purity_f[i], 2), round(prod_999_f[i], 4)]
                    w.writerow(row)


def run_data_driven_thickness_plots(output_dir=None, n_interp=80):
    """
    Revenue vs thickness from run data: one figure with Cu-64 above, Cu-67 below.
    Uses linear interpolation and marks actual data points. Colors are consistent and vibrant.
    99% (and 99.9% if present) enrichment gets lime green, no pale colors.
    """
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    if 'outer_cm' not in run_data_df.columns:
        return
    df = run_data_df.copy()
    thick_col = 'outer_cm'
    t_cm = np.sort(df[thick_col].dropna().unique().astype(float))
    if len(t_cm) < 1:
        return
    
    t_min = float(t_cm.min())
    t_max = float(t_cm.max())
    t_fine = np.linspace(t_min, t_max, n_interp) if len(t_cm) >= 2 else t_cm
    
    # Marker shapes for different enrichments
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+', 'x', '1', '2', '3', '4']
    
    sf = os.path.join(FOLDER_RUN_DATA, RUN_DATA_SELL_ALL)
    fig, (ax64, ax67) = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax, is_cu67_mode, suffix in [(ax64, False, "_cu64"), (ax67, True, "_cu67")]:
        sub = df[df['use_zn67'] == is_cu67_mode] if 'use_zn67' in df.columns else df
        if sub.empty:
            ax.set_visible(False)
            continue
        
        isotope_label = "Cu-67 (Zn-67)" if is_cu67_mode else "Cu-64"
        enrichments = sorted(sub['zn64_enrichment'].dropna().unique(), key=float)
        all_rev = []
        
        for ei, enrich in enumerate(enrichments):
            e = float(enrich)
            enrich_sub = sub[np.abs(sub['zn64_enrichment'].astype(float) - e) < 0.01]
            if enrich_sub.empty:
                continue
            
            # Get actual data points (group by thickness, mean if duplicates)
            by_t = enrich_sub.groupby(thick_col, as_index=False).agg({
                'cu67_g_yr' if is_cu67_mode else 'cu64_g_yr': 'mean'
            })
            t_data = by_t[thick_col].astype(float).values
            prod_data = by_t['cu67_g_yr' if is_cu67_mode else 'cu64_g_yr'].astype(float).values
            
            # Sort by thickness
            order = np.argsort(t_data)
            t_data = t_data[order]
            prod_data = prod_data[order]
            
            if len(t_data) < 2:
                continue
            
            # Calculate revenue for data points
            price = price_cu67_usd_per_g if is_cu67_mode else price_cu64_usd_per_g
            rev_data = prod_data * price / 1e6  # millions USD/yr
            all_rev.extend(rev_data[np.isfinite(rev_data)])
            
            # Linear interpolation
            prod_interp = np.interp(t_fine, t_data, prod_data)
            prod_interp = np.clip(prod_interp, np.min(prod_data), np.max(prod_data))
            
            rev_interp = prod_interp * price / 1e6  # millions USD/yr
            
            # Get color (ensures 99%/99.9% is lime green, no pale colors)
            c = _get_enrichment_color(e, is_cu67_mode)
            marker = markers[ei % len(markers)]
            linestyle = '-'
            markersize = 6
            enrich_label = f"{e*100:.2f}".rstrip('0').rstrip('.') + '%'
            
            # Plot interpolated line
            ax.plot(t_fine, rev_interp, linestyle=linestyle, linewidth=1.5, 
                   color=c, alpha=0.8, zorder=1)
            
            # Plot actual data points with markers
            ax.scatter(t_data, rev_data, marker=marker, s=markersize**2, 
                      color=c, edgecolors='black', linewidths=0.5, 
                      alpha=0.9, zorder=2)
            
            # Dummy plot for legend (like example) - off-screen, shows marker+linestyle
            ax.plot([9e8, 9e9], [9e8, 9e9], marker=marker, linestyle=linestyle,
                   markersize=markersize, linewidth=1.5, color=c,
                   label=enrich_label)
        
        # Set axes properties with data-driven y-limits
        ax.set_xlabel("Blanket thickness (cm)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Annual revenue (USD millions/yr)", fontsize=12, fontweight='bold')
        ax.set_title(f"Revenue vs Thickness — {isotope_label}",
                    fontsize=13, fontweight='bold')
        ax.set_xlim(0, 21)  # fixed thickness range so legend dummy plot doesn't scale axis to 1e9
        ax.grid(True, alpha=0.25)
        
        # Data-driven y-axis limits
        if all_rev:
            arr = np.array(all_rev, dtype=float)
            fin = arr[np.isfinite(arr) & (arr >= 0)]
            if len(fin) > 0:
                rev_max = float(np.max(fin)) * 1.15
                ax.set_ylim(0, max(rev_max, 1e-3))
            else:
                ax.set_ylim(0, 1000)
        else:
            ax.set_ylim(0, 1000)
        
        ax.legend(frameon=True, fontsize=9, loc='best', ncol=2 if len(enrichments) > 6 else 1)
        
        if output_dir:
            path = _csv_path(output_dir, f"revenue_vs_thickness{suffix}_interpolated.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                headers = ["thickness_cm"] + [f"revenue_millions_yr_enrich_{e*100:.2f}pct" for e in sorted(enrichments)]
                w.writerow(headers)
                for j, t in enumerate(t_fine):
                    row = [round(float(t), 4)]
                    for enrich in sorted(enrichments):
                        e = float(enrich)
                        enrich_sub = sub[np.abs(sub['zn64_enrichment'].astype(float) - e) < 0.01]
                        if enrich_sub.empty:
                            row.append(0.0)
                            continue
                        by_t = enrich_sub.groupby(thick_col, as_index=False).agg({
                            'cu67_g_yr' if is_cu67_mode else 'cu64_g_yr': 'mean'
                        })
                        t_data = by_t[thick_col].astype(float).values
                        prod_data = by_t['cu67_g_yr' if is_cu67_mode else 'cu64_g_yr'].astype(float).values
                        order = np.argsort(t_data)
                        t_data = t_data[order]
                        prod_data = prod_data[order]
                        if len(t_data) >= 2:
                            prod_at_t = np.interp(t, t_data, prod_data)
                            prod_at_t = np.clip(prod_at_t, np.min(prod_data), np.max(prod_data))
                        else:
                            prod_at_t = prod_data[0] if len(prod_data) > 0 else 0.0
                        price = price_cu67_usd_per_g if is_cu67_mode else price_cu64_usd_per_g
                        row.append(round(prod_at_t * price / 1e6, 4))
                    w.writerow(row)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(_plot_path(output_dir, sf, "revenue_vs_thickness_run_data.jpeg"), dpi=500)
    plt.close(fig)
    return


def run_data_driven_thickness_plots_for_scenario(output_dir=None, tag="sell_all", cap64=None, cap67=None, purity_cap_64=False, n_interp=40):
    """NPV, Production, Revenue vs thickness — one JPEG: 1 row × 3 cols (Cu-64 only)."""
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None:
        return
    if 'outer_cm' not in run_data_df.columns:
        return
    df = run_data_df.copy()
    sub = df[df['use_zn67'] == False] if 'use_zn67' in df.columns else df
    if sub.empty:
        return
    thick_col = 'outer_cm'
    t_cm = np.sort(sub[thick_col].dropna().unique().astype(float))
    if len(t_cm) < 1:
        return
    sf = os.path.join(FOLDER_RUN_DATA, tag)
    _sub = _title_suffix(tag, purity_cap_64)
    cool_pal = _cool_palette_cu64(12)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes_config = [
        ("npv", "NPV (USD, millions)", "NPV vs Thickness"),
        ("production_mg", "Production (mg/yr)", "Production vs Thickness"),
        ("revenue", "Revenue (USD millions/yr)", "Revenue vs Thickness"),
    ]
    for col_idx, (ykey, ylabel, title_suffix) in enumerate(axes_config):
        ax = axes[col_idx]
        enrichments = sorted(sub['zn64_enrichment'].dropna().unique(), key=float)
        for ei, enrich in enumerate(enrichments):
            e = float(enrich)
            sub_e = sub[np.abs(sub['zn64_enrichment'].astype(float) - e) < 1e-6]
            if sub_e.empty:
                continue
            th = sub_e[thick_col].astype(float).values
            if ykey == "npv":
                vals = np.array([_npv_from_run_row(row, 1.0, cap64, False, purity_cap_64=purity_cap_64) / 1e6 for _, row in sub_e.iterrows()])
            elif ykey == "production_mg":
                vals = sub_e['cu64_g_yr'].astype(float).values * 1e3
            else:
                prod = sub_e['cu64_g_yr'].astype(float).values
                rev = prod * price_cu64_usd_per_g
                if purity_cap_64:
                    pur = np.array([float(r.get('cu64_radionuclide_purity', 0) or 0) for _, r in sub_e.iterrows()])
                    rev = np.where(pur >= 0.999, rev, 0.0)
                if cap64 is not None:
                    rev = np.minimum(rev, float(cap64))
                vals = rev / 1e6
            order = np.argsort(th)
            th, vals = th[order], vals[order]
            c = cool_pal[ei % len(cool_pal)]
            leg = f"{e*100:.1f}%" if e >= 0.1 else f"{e*100:.2f}%"
            ax.plot(th, vals, "o-", color=c, lw=1.2, ms=5, alpha=0.8, label=leg)
            if ykey == "npv" and len(vals) > 0:
                i_opt = np.argmax(vals)
                ax.plot(th[i_opt], vals[i_opt], "*", color="black", ms=12, zorder=5)
        ax.set_xlabel("Thickness (cm)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Cu-64: {title_suffix}{_sub}")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=8, ncol=2)
    plt.tight_layout()
    if output_dir:
        plt.savefig(_plot_path(output_dir, sf, "npv_production_revenue_vs_thickness.jpeg"), dpi=500)
    plt.close(fig)
    return


def run_data_driven_npv_vs_price_figure(output_dir=None, sf=None, n_budget=80, cap64=None, cap67=None, purity_cap_64=False, tag=None):
    """
    NPV vs loading budget for 4 price scenarios: regular, 2×, 4×, 10× lower.
    Layout: 2 rows × 4 cols — row 0 Cu-64 (cool tones), row 1 Cu-67 (warm tones).
    cap64, cap67, purity_cap_64: constraint for this scenario (applied to NPV at each price).
    """
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    if 'zn_feedstock_cost' not in run_data_df.columns or 'outer_cm' not in run_data_df.columns:
        return
    thick_col = 'outer_cm'
    price_mults = [1.0, 0.5, 0.25, 0.1]  # regular, 2× lower, 4× lower, 10× lower

    def _build_rows_price(is_cu67):
        sub = run_data_df[run_data_df['use_zn67'] == is_cu67].copy()
        if sub.empty:
            return []
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            return []
        rows = []
        base_price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
        cap = cap67 if is_cu67 else cap64
        for _, row in sub.iterrows():
            try:
                load = float(row['zn_feedstock_cost'])
                thick = float(row[thick_col])
            except (TypeError, KeyError):
                continue
            if np.isnan(load) or load < 0:
                continue
            npvs = [_npv_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64, price_override_usd_per_g=base_price * m) for m in price_mults]
            rows.append({'loading': load, 'thickness_cm': thick, 'npvs': npvs})
        return rows

    rows64 = _build_rows_price(False)
    rows67 = _build_rows_price(True)
    if not rows64 and not rows67:
        return
    all_loads = [r['loading'] for r in rows64] + [r['loading'] for r in rows67]
    load_lo = max(1e4, min(all_loads) * 0.5)
    load_hi = max(all_loads) * 1.05
    budget_grid = np.logspace(np.log10(load_lo), np.log10(load_hi), n_budget)
    budget_m = budget_grid / 1e6

    def _max_npv_at_budget(rows, budget, price_idx):
        best = -np.inf
        for r in rows:
            if r['loading'] > budget:
                continue
            if r['npvs'][price_idx] > best:
                best = r['npvs'][price_idx]
        return best if best > -np.inf else np.nan

    cool = _cool_palette_cu64(4)
    warm = _warm_palette_cu67(4)
    # Use darker, more saturated colors for 10× lower (col=3) instead of pale end-of-gradient colors
    cool_10x = np.array([0.0, 0.4, 0.2, 1.0])  # Dark green for Cu-64 10× lower
    warm_10x = np.array([0.8, 0.3, 0.0, 1.0])  # Dark orange-red for Cu-67 10× lower
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    labels = ["Regular price", "2× lower", "4× lower", "10× lower"]
    for col in range(4):
        ax64 = axes[0, col]
        ax67 = axes[1, col]
        if rows64:
            npvs = np.array([_max_npv_at_budget(rows64, b, col) for b in budget_grid])
            npvs = np.nan_to_num(npvs, nan=0.0)
            color_64 = cool_10x if col == 3 else cool[col]
            ax64.plot(budget_m, npvs / 1e6, "-", color=color_64, lw=2, label=labels[col])
        ax64.set_xlabel("Loading budget (USD, millions)")
        ax64.set_ylabel("NPV (USD, millions)")
        ax64.set_title(f"Cu-64: {labels[col]}")
        ax64.set_xscale("log")
        ax64.grid(True, alpha=0.25, which="both")
        if rows67:
            npvs = np.array([_max_npv_at_budget(rows67, b, col) for b in budget_grid])
            npvs = np.nan_to_num(npvs, nan=0.0)
            color_67 = warm_10x if col == 3 else warm[col]
            ax67.plot(budget_m, npvs / 1e6, "-", color=color_67, lw=2, label=labels[col])
        ax67.set_xlabel("Loading budget (USD, millions)")
        ax67.set_ylabel("NPV (USD, millions)")
        ax67.set_title(f"Cu-67: {labels[col]}")
        ax67.set_xscale("log")
        ax67.grid(True, alpha=0.25, which="both")
    _sub = _title_suffix(tag, purity_cap_64)
    plt.suptitle("NPV vs Loading Budget — price sensitivity" + _sub, fontsize=13, y=1.02)
    plt.tight_layout()
    if output_dir and sf:
        plt.savefig(_plot_path(output_dir, sf, "npv_vs_loading_budget_by_price.jpeg"), dpi=500)
    plt.close(fig)


def _set_payback_ylim(ax, y_vals, payback_max_yr):
    """Set y-axis to fit min/max data with small margin; less margin on top so data fills the plot."""
    y_vals = np.asarray(y_vals)
    valid = np.isfinite(y_vals)
    if not np.any(valid):
        ax.set_ylim(0, payback_max_yr)
        return
    y_min, y_max = float(np.min(y_vals[valid])), float(np.max(y_vals[valid]))
    span = y_max - y_min
    pad_bottom = max(0.05, span * 0.08) if span > 0 else 0.05
    pad_top = max(0.05, span * 0.05)  # smaller top margin
    y_lo = max(0.0, y_min - pad_bottom)
    y_hi = min(payback_max_yr, y_max + pad_top)
    if y_max <= 0.1:
        y_hi = max(0.1, y_max + 0.05)
    ax.set_ylim(y_lo, y_hi)


# Payback figure: only plot 0-3 yr (cut out outliers)
PAYBACK_PLOT_YMAX = 3.0


def run_data_driven_payback_plots(output_dir=None, n_budget=80, payback_max_yr=60,
                                  tag=None, cap64=None, cap67=None, purity_cap_64=False):
    """
    Payback period (years) vs thickness, vs enrichment, vs loading budget.
    One figure: 2 rows (Cu-64 top, Cu-67 bottom) × 3 cols. Only 0-3 yr shown (outliers excluded).
    When tag is set, saves under run_data/{tag}/ (e.g. sell_all, market_cap, purity_cap).
    """
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    if 'zn_feedstock_cost' not in run_data_df.columns or 'outer_cm' not in run_data_df.columns:
        return
    sf = os.path.join(FOLDER_RUN_DATA, tag if tag else FOLDER_ADDITIONAL_PLOTS)
    thick_col = 'outer_cm'
    _sub = _title_suffix(tag, purity_cap_64)

    def _safe_payback(row, is_cu67):
        try:
            cap = cap67 if is_cu67 else cap64
            pb = _payback_from_run_row(row, cap_usd_per_yr=cap, is_cu67=is_cu67, purity_cap_64=purity_cap_64)
            return min(pb, payback_max_yr) if np.isfinite(pb) else payback_max_yr
        except (ValueError, TypeError, KeyError):
            return np.nan

    def _rows_with_payback(is_cu67):
        sub = run_data_df[run_data_df['use_zn67'] == is_cu67].copy()
        if sub.empty:
            return []
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            return []
        rows = []
        for _, row in sub.iterrows():
            try:
                load = float(row['zn_feedstock_cost'])
                thick = float(row[thick_col])
                enr = float(row['zn64_enrichment'])
            except (TypeError, KeyError):
                continue
            if np.isnan(load) or load < 0:
                continue
            pb = _safe_payback(row, is_cu67)
            if np.isnan(pb):
                continue
            rows.append({'loading': load, 'thickness_cm': thick, 'enrichment': enr, 'payback_yr': pb})
        return rows

    def _min_payback_at_budget(rows, budget):
        best = np.inf
        for r in rows:
            if r['loading'] > budget:
                continue
            if r['payback_yr'] < best:
                best = r['payback_yr']
        return min(best, payback_max_yr) if np.isfinite(best) else np.nan

    rows64 = _rows_with_payback(False)
    rows67 = _rows_with_payback(True)
    if not rows64 and not rows67:
        return

    all_loads = [r['loading'] for r in rows64] + [r['loading'] for r in rows67]
    load_lo = max(1e4, min(all_loads) * 0.5)
    load_hi = max(all_loads) * 1.05
    budget_grid = np.logspace(np.log10(load_lo), np.log10(load_hi), n_budget)
    budget_m = budget_grid / 1e6

    cool = _cool_palette_cu64(8)
    warm = _warm_palette_cu67(8)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row_idx, (rows, label, palette) in enumerate([
        (rows64, "Cu-64", cool),
        (rows67, "Cu-67", warm),
    ]):
        if not rows:
            for col in range(3):
                axes[row_idx, col].set_visible(False)
            continue
        thick_ax, enr_ax, budget_ax = axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2]

        # Col 0: Payback vs thickness (one series per enrichment); only plot 0-3 yr (exclude outliers)
        enrichments = _get_unique_enrichments(rows, tol=0.001)
        for ei, enr in enumerate(enrichments):
            pts = [(r['thickness_cm'], r['payback_yr']) for r in rows if abs(r['enrichment'] - enr) < 0.001 and r['payback_yr'] <= PAYBACK_PLOT_YMAX]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            c = palette[ei % len(palette)]
            thick_ax.plot(x, y, "o-", color=c, lw=1.5, ms=6, label=f"{enr*100:.1f}%")
        thick_ax.set_xlabel("Thickness (cm)")
        thick_ax.set_ylabel("Payback period (yr)")
        thick_ax.set_title(f"{label}: Payback vs Thickness")
        thick_ax.grid(True, alpha=0.25)
        if thick_ax.get_legend_handles_labels()[0]:
            thick_ax.legend(frameon=False, fontsize=8)
        pb_thick = [r['payback_yr'] for r in rows if r['payback_yr'] <= PAYBACK_PLOT_YMAX]
        _set_payback_ylim(thick_ax, pb_thick if pb_thick else [0, PAYBACK_PLOT_YMAX], PAYBACK_PLOT_YMAX)

        # Col 1: Payback vs enrichment (one series per thickness); only plot 0-3 yr (exclude outliers)
        thicknesses = sorted(set(r['thickness_cm'] for r in rows))
        for ti, th in enumerate(thicknesses):
            pts = [(r['enrichment'] * 100, r['payback_yr']) for r in rows if abs(r['thickness_cm'] - th) < 0.01 and r['payback_yr'] <= PAYBACK_PLOT_YMAX]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            c = palette[ti % len(palette)]
            display_th = 0.5 if th == 0 else th
            enr_ax.plot(x, y, "s-", color=c, lw=1.5, ms=6, label=f"{display_th:.1f} cm")
        enr_ax.set_xlabel("Enrichment (%)")
        enr_ax.set_ylabel("Payback period (yr)")
        enr_ax.set_title(f"{label}: Payback vs Enrichment")
        enr_ax.grid(True, alpha=0.25)
        if enr_ax.get_legend_handles_labels()[0]:
            enr_ax.legend(frameon=False, fontsize=8)
        pb_enr = [r['payback_yr'] for r in rows if r['payback_yr'] <= PAYBACK_PLOT_YMAX]
        _set_payback_ylim(enr_ax, pb_enr if pb_enr else [0, PAYBACK_PLOT_YMAX], PAYBACK_PLOT_YMAX)

        # Col 2: Payback vs loading budget (best payback at each budget); cap at 0-3 yr for display
        payback_at_budget = np.array([_min_payback_at_budget(rows, b) for b in budget_grid])
        payback_at_budget = np.nan_to_num(payback_at_budget, nan=payback_max_yr)
        payback_plot = np.minimum(payback_at_budget, PAYBACK_PLOT_YMAX)
        c = palette[len(palette) // 2]
        budget_ax.plot(budget_m, payback_plot, "-", color=c, lw=2, label=label)
        budget_ax.set_xlabel("Loading budget (USD, millions)")
        budget_ax.set_ylabel("Payback period (yr)")
        budget_ax.set_title(f"{label}: Payback vs Budget")
        budget_ax.set_xscale("log")
        budget_ax.grid(True, alpha=0.25, which="both")
        _set_payback_ylim(budget_ax, payback_plot, PAYBACK_PLOT_YMAX)
        budget_ax.legend(frameon=False)

    plt.suptitle("Payback vs thickness, enrichment, budget" + _sub, fontsize=12, y=1.02)
    plt.tight_layout()
    if output_dir:
        plt.savefig(_plot_path(output_dir, sf, "payback_vs_thickness_enrichment_loading_budget.jpeg"), dpi=500)
    plt.close(fig)


# IRR axis can go up to 1000% (or as high as data) in IRR vs thickness/enrichment/budget and scatter
IRR_DISPLAY_CAP_PCT = 1000.0


def _set_irr_ylim(ax, y_vals):
    """Set y-axis to fit min/max data + margin; allow up to IRR_DISPLAY_CAP_PCT (1000%) or as high as needed."""
    y_vals = np.asarray(y_vals)
    valid = np.isfinite(y_vals) & (y_vals >= 0)
    if not np.any(valid):
        ax.set_ylim(0, 100.0)
        return
    y_min, y_max = float(np.min(y_vals[valid])), float(np.max(y_vals[valid]))
    span = y_max - y_min
    pad = max(1.0, span * 0.12) if span > 0 else 1.0
    y_lo = max(0.0, y_min - pad)
    y_hi = min(y_max + pad, IRR_DISPLAY_CAP_PCT)
    ax.set_ylim(y_lo, y_hi)


def run_data_driven_irr_plots(output_dir=None, n_budget=80, irr_max_pct=None,
                              tag=None, cap64=None, cap67=None, purity_cap_64=False):
    """IRR (%) vs thickness, vs enrichment, vs loading budget. 2 rows (Cu-64, Cu-67) × 3 cols. When tag set, saves to run_data/{tag}/."""
    if irr_max_pct is None:
        irr_max_pct = IRR_DISPLAY_CAP_PCT
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    if 'zn_feedstock_cost' not in run_data_df.columns or 'outer_cm' not in run_data_df.columns:
        return
    sf = os.path.join(FOLDER_RUN_DATA, tag if tag else FOLDER_ADDITIONAL_PLOTS)
    thick_col = 'outer_cm'
    _sub = _title_suffix(tag, purity_cap_64)

    def _safe_irr(row, is_cu67):
        try:
            cap = cap67 if is_cu67 else cap64
            ir = _irr_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64)
            return ir * 100 if not np.isnan(ir) else np.nan
        except (ValueError, TypeError, KeyError):
            return np.nan

    def _rows_with_irr(is_cu67):
        sub = run_data_df[run_data_df['use_zn67'] == is_cu67].copy()
        if sub.empty:
            return []
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            return []
        rows = []
        for _, row in sub.iterrows():
            try:
                load = float(row['zn_feedstock_cost'])
                thick = float(row[thick_col])
                enr = float(row['zn64_enrichment'])
            except (TypeError, KeyError):
                continue
            if np.isnan(load) or load < 0:
                continue
            ir = _safe_irr(row, is_cu67)
            if np.isnan(ir):
                continue
            rows.append({'loading': load, 'thickness_cm': thick, 'enrichment': enr, 'irr_pct': min(ir, irr_max_pct)})
        return rows

    def _max_irr_at_budget(rows, budget):
        best = -np.inf
        for r in rows:
            if r['loading'] > budget:
                continue
            if r['irr_pct'] > best:
                best = r['irr_pct']
        return best if best > -np.inf else np.nan

    rows64 = _rows_with_irr(False)
    rows67 = _rows_with_irr(True)
    if not rows64 and not rows67:
        return
    all_loads = [r['loading'] for r in rows64] + [r['loading'] for r in rows67]
    load_lo = max(1e4, min(all_loads) * 0.5)
    load_hi = max(all_loads) * 1.05
    budget_grid = np.logspace(np.log10(load_lo), np.log10(load_hi), n_budget)
    budget_m = budget_grid / 1e6

    cool = _cool_palette_cu64(8)
    warm = _warm_palette_cu67(8)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row_idx, (rows, label, palette) in enumerate([(rows64, "Cu-64", cool), (rows67, "Cu-67", warm)]):
        if not rows:
            for col in range(3):
                axes[row_idx, col].set_visible(False)
            continue
        thick_ax, enr_ax, budget_ax = axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2]
        # Extract enrichments with tolerance to ensure 0.117 etc. are included
        enrichments = _get_unique_enrichments(rows, tol=0.001)
        for ei, enr in enumerate(enrichments):
            pts = [(r['thickness_cm'], r['irr_pct']) for r in rows if abs(r['enrichment'] - enr) < 0.001]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            thick_ax.plot(x, y, "o-", color=palette[ei % len(palette)], lw=1.5, ms=6, label=f"{enr*100:.1f}%")
        thick_ax.set_xlabel("Thickness (cm)")
        thick_ax.set_ylabel("IRR (%)")
        thick_ax.set_title(f"{label}: IRR vs Thickness")
        thick_ax.grid(True, alpha=0.25)
        if thick_ax.get_legend_handles_labels()[0]:
            thick_ax.legend(frameon=False, fontsize=8)
        t_vals = [r['thickness_cm'] for r in rows]
        t_min, t_max = min(t_vals), max(t_vals)
        pad_t = max(0.5, (t_max - t_min) * 0.05)
        thick_ax.set_xlim(max(0.0, t_min - pad_t), t_max + pad_t)
        _set_irr_ylim(thick_ax, [r['irr_pct'] for r in rows])
        thicknesses = sorted(set(r['thickness_cm'] for r in rows))
        for ti, th in enumerate(thicknesses):
            pts = [(r['enrichment'] * 100, r['irr_pct']) for r in rows if abs(r['thickness_cm'] - th) < 0.01]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            display_th = 0.5 if th == 0 else th
            enr_ax.plot(x, y, "s-", color=palette[ti % len(palette)], lw=1.5, ms=6, label=f"{display_th:.1f} cm")
        enr_vals = [r['enrichment'] * 100 for r in rows]
        enr_min = max(0, min(enr_vals) - 2)
        enr_max = min(100, max(enr_vals) + 2)
        enr_ax.set_xlabel("Enrichment (%)")
        enr_ax.set_ylabel("IRR (%)")
        enr_ax.set_title(f"{label}: IRR vs Enrichment")
        enr_ax.grid(True, alpha=0.25)
        if enr_ax.get_legend_handles_labels()[0]:
            enr_ax.legend(frameon=False, fontsize=8)
        enr_ax.set_xlim(enr_min, enr_max)
        _set_irr_ylim(enr_ax, [r['irr_pct'] for r in rows])
        irr_at_budget = np.array([_max_irr_at_budget(rows, b) for b in budget_grid])
        irr_at_budget = np.nan_to_num(irr_at_budget, nan=0.0)
        budget_ax.plot(budget_m, irr_at_budget, "-", color=palette[len(palette) // 2], lw=2, label=label)
        budget_ax.set_xlabel("Loading budget (USD, millions)")
        budget_ax.set_ylabel("IRR (%)")
        budget_ax.set_title(f"{label}: IRR vs Budget")
        budget_ax.set_xscale("log")
        budget_ax.grid(True, alpha=0.25, which="both")
        _set_irr_ylim(budget_ax, irr_at_budget)
        budget_ax.legend(frameon=False)
    plt.suptitle("IRR vs thickness, enrichment, budget" + _sub, fontsize=12, y=1.02)
    plt.tight_layout()
    if output_dir:
        plt.savefig(_plot_path(output_dir, sf, "irr_vs_thickness_enrichment_loading_budget.jpeg"), dpi=500)
    plt.close(fig)


def run_data_driven_investor_plots(output_dir=None, tag=None, cap64=None, cap67=None, purity_cap_64=False):
    """Cashflow, risk/return (NPV vs payback, NPV vs IRR). When tag set, saves to run_data/{tag}/."""
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    if 'zn_feedstock_cost' not in run_data_df.columns or 'outer_cm' not in run_data_df.columns:
        return
    sf = os.path.join(FOLDER_RUN_DATA, tag if tag else FOLDER_ADDITIONAL_PLOTS)
    _sub = _title_suffix(tag, purity_cap_64)

    # Build rows with NPV, payback, IRR per run (using scenario cap/purity when provided)
    def _row_metrics(is_cu67):
        sub = run_data_df[run_data_df['use_zn67'] == is_cu67].copy()
        if sub.empty:
            return []
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            return []
        out = []
        cap = cap67 if is_cu67 else cap64
        for _, row in sub.iterrows():
            try:
                load = float(row['zn_feedstock_cost'])
                if np.isnan(load) or load < 0:
                    continue
                npv = _npv_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64)
                pb = _payback_from_run_row(row, cap, is_cu67, purity_cap_64=purity_cap_64)
                ir = _irr_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64)
                prod = float(row['cu67_g_yr' if is_cu67 else 'cu64_g_yr'])
                price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
                rev = prod * price
                if not is_cu67 and purity_cap_64:
                    purity = float(row.get('cu64_radionuclide_purity', 0) or 0)
                    rev = rev if purity >= 0.999 else 0.0
                if cap is not None:
                    rev = min(rev, float(cap))
                annual_net = rev - opex_fixed_usd_per_yr - reload_fraction_per_year * load
                geom = str(row.get('dir_name', row.get('outer_cm', '?')))[:24]
                thick = row.get('outer_cm', '')
                thick_s = "?"
                if thick != '' and str(thick) != 'nan':
                    try:
                        tv = float(thick)
                        thick_s = f"{(0.5 if tv == 0 else tv):.1f} cm"
                    except (TypeError, ValueError):
                        pass
                enr = float(row.get('zn64_enrichment', 0) or 0)
                case_label = f"{geom} | {thick_s} | {enr*100:.1f}%"
                outer = float(thick) if thick != '' and str(thick) != 'nan' else 0
                boron = float(row.get('boron_cm', 0) or 0)
                multi = float(row.get('multi_cm', 0) or 0)
                mod = float(row.get('mod_cm', row.get('moderator_cm', 0)) or 0)
                irrad_h = float(row.get('irrad_hours', 1) or 1)
                out.append({
                    'npv': npv, 'payback_yr': min(pb, 50) if np.isfinite(pb) else 50,
                    'irr_pct': ir*100 if not np.isnan(ir) else np.nan,
                    'loading': load, 'annual_net': annual_net, 'label': case_label,
                    'enrichment': enr, 'outer_cm': outer, 'boron_cm': boron, 'multi_cm': multi, 'mod_cm': mod,
                    'irrad_hours': irrad_h,
                })
            except (ValueError, TypeError, KeyError):
                continue
        return out

    rows64 = _row_metrics(False)
    rows67 = _row_metrics(True)
    if not rows64 and not rows67:
        return

    # 1) Cumulative cashflow over time for best-NPV run (Cu-64 and Cu-67)
    fig, (ax64, ax67) = plt.subplots(2, 1, figsize=(10, 8))
    for ax, rows, label, color in [(ax64, rows64, "Cu-64", _cool_palette_cu64(5)[2]), (ax67, rows67, "Cu-67", _warm_palette_cu67(5)[2])]:
        if not rows:
            ax.set_visible(False)
            continue
        best = max(rows, key=lambda r: r['npv'])
        n_y = int(T_years) + 1
        years = np.arange(n_y, dtype=float)
        cum = np.zeros(n_y)
        cum[0] = -capex_usd - best['loading']
        for i in range(1, n_y):
            cum[i] = cum[i-1] + best['annual_net']
        ax.fill_between(years, 0, cum, alpha=0.4, color=color)
        ax.plot(years, cum, "-", color=color, lw=2)
        ax.axhline(0, ls="--", color="gray")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative cash flow (USD)")
        ax.set_title(f"{label}: Cumulative cash flow{_sub}")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if output_dir:
        plt.savefig(_plot_path(output_dir, sf, "cumulative_cashflow_best_npv.jpeg"), dpi=500)
    plt.close(fig)

    def _geom_key(r):
        return (round(r['outer_cm'], 2), round(r['boron_cm'], 2), round(r['multi_cm'], 2), round(r['mod_cm'], 2))

    def _scatter_legend_encoding(ax, rows, label, x_vals, y_vals):
        """Scatter plot with enrichment (fill), irradiation (marker), geometry (edge) and three legends like simple_analyze."""
        if not rows:
            return
        enrichments = sorted(set(round(r['enrichment'], 4) for r in rows))
        geom_tuples = sorted(set(_geom_key(r) for r in rows), key=lambda x: (x[0], x[1], x[2], x[3]))
        irrad_times = sorted(set(round(r['irrad_hours'], 1) for r in rows))
        var_marker = {1: 'o', 4: 's', 8: 'D', 24: '^', 72: 'v', 98: 'p', 8760: 'H', 26280: '*'}
        cmap = plt.colormaps.get_cmap('viridis')
        enrich_color = {e: cmap(i / (len(enrichments) - 1) if len(enrichments) > 1 else 0.5) for i, e in enumerate(enrichments)}
        geom_edge_colors = plt.colormaps.get_cmap('Set1')
        geom_edge = {g: geom_edge_colors(i / max(len(geom_tuples) - 1, 1)) for i, g in enumerate(geom_tuples)}
        geom_lw = {g: 2.0 + (i % 2) * 1.0 for i, g in enumerate(geom_tuples)}

        for i, r in enumerate(rows):
            enr = round(r['enrichment'], 4)
            geom = _geom_key(r)
            irh = r['irrad_hours']
            marker = var_marker.get(irh, 'o')
            c = enrich_color.get(enr, list(enrich_color.values())[0] if enrich_color else 'gray')
            ec = geom_edge.get(geom, 'k')
            lw = geom_lw.get(geom, 2)
            ax.scatter(x_vals[i], y_vals[i], s=80, c=[c], alpha=0.9, marker=marker, edgecolors=[ec], linewidths=lw)

        def _enrich_label(e):
            v = float(e)
            if abs(v - 0.999) < 0.0005 or abs(v - 1.0) < 0.005:
                return '99.9%'
            if abs(v - 0.99) < 0.005:
                return '99%'
            pct = v * 100
            if abs(pct - round(pct, 1)) < 0.01 and round(pct, 1) != round(pct, 0):
                s = f'{pct:.1f}'
            else:
                s = f'{pct:.2f}'.rstrip('0').rstrip('.')
            return f'{s}%'
        enrich_handles = [ax.scatter([], [], c=[enrich_color[e]], s=100, marker='o', edgecolors='black', linewidths=0.5, label=_enrich_label(e)) for e in enrichments]
        irrad_handles = [ax.scatter([], [], c='gray', s=100, marker=var_marker.get(h, 'o'), edgecolors='black', linewidths=1, label=f'{int(h)} h') for h in irrad_times]
        geom_handles = []
        for g in geom_tuples:
            outer, boron, multi, mod = g
            lbl = f'Outer={outer:.1f}cm, B={boron:.1f}cm'
            if multi != 0 or mod != 0:
                lbl += f', M={multi:.1f}cm, Mod={mod:.1f}cm'
            geom_handles.append(ax.scatter([], [], c='white', s=100, marker='o', edgecolors=[geom_edge[g]], linewidths=geom_lw.get(g, 3), label=lbl))
        enrich_title = 'Zn-67 Enrichment\n(fill color)' if '67' in label else 'Zn-64 Enrichment\n(fill color)'
        # Legends in right margin (bbox_to_anchor in axes coords: 1.02 = just right of axes)
        leg1 = ax.legend(handles=enrich_handles, title=enrich_title, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, title_fontsize=10, framealpha=0.95)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=irrad_handles, title='Irradiation\n(marker)', loc='upper left', bbox_to_anchor=(1.02, 0.55), fontsize=9, title_fontsize=10, framealpha=0.95)
        ax.add_artist(leg2)
        ax.legend(handles=geom_handles, title='Geometry\n(edge color)', loc='upper left', bbox_to_anchor=(1.02, 0.1), fontsize=8, title_fontsize=10, framealpha=0.95)
        ax.text(0.02, 0.02, f"Total: {len(rows)} cases", transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # 2) NPV vs Payback scatter (risk/return) — one JPG per isotope; big, wide, tall plot with small side space for legend
    for rows, label in [(rows64, "Cu-64"), (rows67, "Cu-67")]:
        if not rows:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        x_vals = [r['payback_yr'] for r in rows]
        y_vals = [r['npv']/1e6 for r in rows]
        _scatter_legend_encoding(ax, rows, label, x_vals, y_vals)
        ax.set_xlabel("Payback period (yr)")
        ax.set_ylabel("NPV (USD, millions)")
        ax.set_title(f"{label}: NPV vs Payback{_sub}")
        ax.axhline(0, ls="--", color="gray")
        ax.grid(True, alpha=0.25)
        fig.subplots_adjust(left=0.06, right=0.82, top=0.92, bottom=0.07)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        if output_dir:
            suffix = "Cu64" if "64" in label else "Cu67"
            plt.savefig(_plot_path(output_dir, sf, f"npv_vs_payback_scatter_{suffix}.jpeg"), dpi=500)
        plt.close(fig)

    # 3) NPV vs IRR scatter — two separate PNGs (Cu-64, Cu-67); thin margins so plot is bigger and wider
    for rows, label in [(rows64, "Cu-64"), (rows67, "Cu-67")]:
        if not rows:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        x_vals = [r['irr_pct'] if not np.isnan(r['irr_pct']) else 0 for r in rows]
        y_vals = [r['npv']/1e6 for r in rows]
        _scatter_legend_encoding(ax, rows, label, x_vals, y_vals)
        ax.set_xlabel("IRR (%)")
        ax.set_ylabel("NPV (USD, millions)")
        ax.set_title(f"{label}: NPV vs IRR{_sub}")
        ax.axhline(0, ls="--", color="gray")
        ax.grid(True, alpha=0.25)
        irrs = np.array(x_vals)
        npvs = np.array(y_vals)
        x_hi = max(irrs) * 1.05 if np.any(irrs > 0) else max(1, IRR_DISPLAY_CAP_PCT)
        ax.set_xlim(0, x_hi)
        pad = max(1.0, (float(np.max(npvs)) - float(np.min(npvs))) * 0.05)
        y_min = min(0.0, float(np.min(npvs))) - pad
        y_max = float(np.max(npvs)) + pad
        ax.set_ylim(y_min, y_max)
        fig.subplots_adjust(left=0.06, right=0.80, top=0.92, bottom=0.07)
        plt.tight_layout(rect=[0, 0, 0.80, 1])
        if output_dir:
            suffix = "Cu64" if "64" in label else "Cu67"
            plt.savefig(_plot_path(output_dir, sf, f"npv_vs_irr_scatter_{suffix}.png"), dpi=300)
        plt.close(fig)


def run_data_driven_budget_plots(output_dir=None, n_budget=200, rev_caps_64_m=(10, 20, 50, 100, 200), rev_caps_67_m=(1, 3, 5, 10, 20), tag=None, cap64=None, cap67=None, purity_cap_64=False):
    """
    Data-driven NPV vs loading budget and NPV vs thickness by revenue ceiling (all scenarios).
    n_budget: number of budget points (finer grid = cleaner step curves, no smoothing).
    When tag is set, saves to run_data/{tag}/ (e.g. sell_all, market_cap, purity_cap, market_purity_cap).
    All four scenarios get the full set of revenue ceilings (e.g. Cap $10M–$200M); scenario (market cap,
    purity threshold) is shown in folder name and plot subtitle (_sub).
    cap64, cap67, purity_cap_64: constraint for this scenario; NPV values use this scenario.
    """
    if run_data_df is None or run_data_df.empty or price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        return
    if 'zn_feedstock_cost' not in run_data_df.columns or 'outer_cm' not in run_data_df.columns:
        return
    sf = os.path.join(FOLDER_RUN_DATA, tag if tag else FOLDER_ADDITIONAL_PLOTS)
    thick_col = 'outer_cm'
    _sub = _title_suffix(tag, purity_cap_64)

    def _build_rows(is_cu67, caps_tuple):
        sub = run_data_df[run_data_df['use_zn67'] == is_cu67].copy()
        if sub.empty:
            return []
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            return []
        rows = []
        caps = caps_tuple[0] if not is_cu67 else caps_tuple[1]
        cap = cap67 if is_cu67 else cap64
        for _, row in sub.iterrows():
            try:
                load = float(row['zn_feedstock_cost'])
                thick = float(row[thick_col])
            except (TypeError, KeyError):
                continue
            if np.isnan(load) or load < 0:
                continue
            # "No cap" for budget plots still honors scenario constraints (e.g., purity threshold)
            # and only removes the revenue ceiling itself.
            npv_uncap = _npv_from_run_row(row, 1.0, None, is_cu67, purity_cap_64=purity_cap_64)
            npv_scenario = _npv_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64)
            enr = float(row.get('zn64_enrichment', 0))
            rows.append({
                'loading': load, 'thickness_cm': thick, 'enrichment': enr,
                'npv_uncap': npv_uncap,
                'npv_scenario': npv_scenario,
                'npv_caps': [_npv_from_run_row(row, 1.0, cap * 1e6, is_cu67, purity_cap_64=purity_cap_64) for cap in caps],
            })
        return rows

    def _max_npv_and_thickness_at_budget(rows, budget, cap_idx=None, use_uncap_for_nocap=False):
        """At given budget, max NPV among runs with loading <= budget; return (npv, thickness_cm, enrichment).
        When cap_idx is None: use npv_scenario (constraint scenario), unless use_uncap_for_nocap then npv_uncap."""
        best_npv, best_t, best_enr = -np.inf, np.nan, np.nan
        for r in rows:
            if r['loading'] > budget:
                continue
            if cap_idx is not None:
                npv = r['npv_caps'][cap_idx]
            elif use_uncap_for_nocap:
                npv = r['npv_uncap']
            else:
                npv = r['npv_scenario']
            if npv > best_npv:
                best_npv = npv
                best_t = r['thickness_cm']
                best_enr = r.get('enrichment', np.nan)
        return (best_npv, best_t, best_enr) if best_npv > -np.inf else (np.nan, np.nan, np.nan)

    def _max_npv_at_thickness(rows, thick, cap_idx=None, use_uncap=False):
        """At given thickness, max NPV among runs with that thickness. For NPV vs thickness by rev cap plots."""
        best = -np.inf
        for r in rows:
            if not np.isclose(r['thickness_cm'], thick, atol=0.02):
                continue
            if cap_idx is not None:
                v = r['npv_caps'][cap_idx]
            else:
                v = r['npv_uncap'] if use_uncap else r.get('npv_scenario', r['npv_uncap'])
            if np.isfinite(v) and v > best:
                best = v
        return best if best > -np.inf else np.nan

    def _max_npv_at_enrichment(rows, enr, cap_idx=None, use_uncap=False, tol=0.001):
        """At given enrichment, max NPV among runs with that enrichment (tolerance-based). For NPV vs enrichment by rev cap plots."""
        best = -np.inf
        for r in rows:
            e = r.get('enrichment', np.nan)
            if not np.isfinite(e) or abs(float(e) - float(enr)) > tol:
                continue
            if cap_idx is not None:
                v = r['npv_caps'][cap_idx]
            else:
                v = r['npv_uncap'] if use_uncap else r.get('npv_scenario', r['npv_uncap'])
            if np.isfinite(v) and v > best:
                best = v
        return best if best > -np.inf else np.nan

    # Use full set of revenue ceilings for all scenarios so every case (sell_all, market_cap,
    # purity_cap, market_purity_cap) gets the same cap curves; scenario is shown in folder + _sub title.
    caps64 = list(rev_caps_64_m)
    caps67 = list(rev_caps_67_m)
    rows64 = _build_rows(False, (caps64, caps67))
    rows67 = _build_rows(True, (caps64, caps67))
    if not rows64 and not rows67:
        return
    all_loads = []
    if rows64:
        all_loads.extend([r['loading'] for r in rows64])
    if rows67:
        all_loads.extend([r['loading'] for r in rows67])
    load_lo = max(1e4, min(all_loads) * 0.5)
    load_hi = max(all_loads) * 1.05
    budget_grid = np.logspace(np.log10(load_lo), np.log10(load_hi), n_budget)
    budget_m = budget_grid / 1e6

    def _plot_cu64_npv_vs_budget_single(rows_cu64, out_dir, subfolder, filename, title_extra=""):
        """One-panel Cu-64 NPV vs loading budget: gray dashed No cap, solid caps, linear y, smooth."""
        if not rows_cu64 or not out_dir:
            return
        cool_caps = _cool_palette_vibrant(max(len(caps64), 1))
        no_cap_color = (0.45, 0.45, 0.45)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        npvs_u = np.array([_max_npv_and_thickness_at_budget(rows_cu64, b, use_uncap_for_nocap=True)[0] for b in budget_grid])
        npvs_u = np.nan_to_num(npvs_u, nan=0.0)
        valid_u = npvs_u > 0
        if np.any(valid_u):
            ax.plot(budget_m[valid_u], npvs_u[valid_u] / 1e6, "--", color=no_cap_color, lw=2.0, alpha=1.0, label="No cap", zorder=4)
            i_peak_u = np.argmax(npvs_u)
            if npvs_u[i_peak_u] > 0:
                ax.plot(budget_m[i_peak_u], npvs_u[i_peak_u] / 1e6, "*", color=no_cap_color, ms=16, zorder=5)
        for i, cap_m in enumerate(caps64):
            npvs = np.array([_max_npv_and_thickness_at_budget(rows_cu64, b, cap_idx=i)[0] for b in budget_grid])
            npvs = np.nan_to_num(npvs, nan=0.0)
            valid = npvs > 0
            if np.any(valid):
                c = cool_caps[i % len(cool_caps)]
                ax.plot(budget_m[valid], npvs[valid] / 1e6, "-", color=c, lw=1.8, marker="o", markersize=4, markevery=max(1, valid.sum() // 15), label=f"Cap ${cap_m:.0f}M", zorder=2)
                i_peak = np.argmax(npvs)
                if npvs[i_peak] > 0:
                    ax.plot(budget_m[i_peak], npvs[i_peak] / 1e6, "*", color=c, ms=14, zorder=5)
        ax.set_xlabel("Loading budget (USD, millions)")
        ax.set_ylabel("NPV (USD, millions)")
        title_line1 = "Cu-64: NPV vs. Loading Budget by Revenue Ceiling (* = optimum)"
        title_line2 = title_extra.strip() if title_extra else ""
        ax.set_title(title_line1 + ("\n" + title_line2 if title_line2 else ""))
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=True, title="Revenue ceiling", fontsize=9)
        plt.tight_layout()
        plt.savefig(_plot_path(out_dir, subfolder, filename), dpi=500)
        plt.close(fig)

    # ---- 2) NPV vs loading budget by revenue ceiling, Cu-64 above Cu-67 (* = optimum) ----
    # Smooth plots: gray dashed "No cap", linear y-axis, solid colored caps with markers
    cool_caps = _cool_palette_vibrant(max(len(caps64), 1))
    warm_caps = _warm_palette_vibrant(max(len(caps67), 1))
    no_cap_color = (0.45, 0.45, 0.45)
    fig, (ax64, ax67) = plt.subplots(2, 1, figsize=(10, 8))
    for ax, rows, caps, label, palette in [(ax64, rows64, caps64, "Cu-64", cool_caps), (ax67, rows67, caps67, "Cu-67", warm_caps)]:
        if not rows:
            ax.set_visible(False)
            continue
        # "No cap" curve: remove revenue ceiling but keep scenario constraints (e.g., purity).
        npvs_u = np.array([_max_npv_and_thickness_at_budget(rows, b, use_uncap_for_nocap=True)[0] for b in budget_grid])
        npvs_u = np.nan_to_num(npvs_u, nan=0.0)
        valid_u = npvs_u > 0
        if np.any(valid_u):
            ax.plot(budget_m[valid_u], npvs_u[valid_u] / 1e6, "--", color=no_cap_color, lw=2.0, alpha=1.0, label="No cap", zorder=4)
            i_peak_u = np.argmax(npvs_u)
            if npvs_u[i_peak_u] > 0:
                ax.plot(budget_m[i_peak_u], npvs_u[i_peak_u] / 1e6, "*", color=no_cap_color, ms=16, zorder=5)
        for i, cap_m in enumerate(caps):
            npvs = np.array([_max_npv_and_thickness_at_budget(rows, b, cap_idx=i)[0] for b in budget_grid])
            npvs = np.nan_to_num(npvs, nan=0.0)
            valid = npvs > 0
            if np.any(valid):
                c = palette[i % len(palette)]
                ax.plot(budget_m[valid], npvs[valid] / 1e6, "-", color=c, lw=1.8, marker="o", markersize=4, markevery=max(1, valid.sum() // 15), label=f"Cap ${cap_m:.0f}M", zorder=2)
                i_peak = np.argmax(npvs)
                if npvs[i_peak] > 0:
                    ax.plot(budget_m[i_peak], npvs[i_peak] / 1e6, "*", color=c, ms=14, zorder=5)
        ax.set_xlabel("Loading budget (USD, millions)")
        ax.set_ylabel("NPV (USD, millions)")
        title_line1 = f"{label}: NPV vs. Loading Budget by Revenue Ceiling (* = optimum)"
        ax.set_title(title_line1 + (f"\n{_sub}" if _sub else ""))
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=True, title="Revenue ceiling", fontsize=9)
    plt.tight_layout()
    plt.savefig(_plot_path(output_dir, sf, "npv_vs_loading_by_revcap_run_data.jpeg"), dpi=500)
    plt.close(fig)

    # ---- 2b) Cu-64 only: single NPV vs loading budget plot (best case over all enrichment/thickness) ----
    if rows64 and output_dir:
        _plot_cu64_npv_vs_budget_single(
            rows64,
            output_dir,
            sf,
            "cu64_npv_vs_loading_budget_best_case.jpeg",
            title_extra=(f"{_sub} — best case (enrichment, thickness)" if _sub else " — best case (enrichment, thickness)"),
        )

    # ---- 2b2) Cu-64 only: NPV vs thickness by revenue ceiling (square figure, same style as budget plot) ----
    # For each condition (No cap, Cap $10M, ...): curve = max NPV over runs at each thickness; star = thickness that maximizes NPV for that condition.
    if rows64 and output_dir:
        thick_vals = np.sort(np.unique([r['thickness_cm'] for r in rows64 if np.isfinite(r.get('thickness_cm', np.nan))]))
        if len(thick_vals) >= 1:
            cool_caps = _cool_palette_vibrant(max(len(caps64), 1))
            no_cap_color = (0.45, 0.45, 0.45)
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            npvs_u = np.array([_max_npv_at_thickness(rows64, t, use_uncap=True) for t in thick_vals])
            valid_u = np.isfinite(npvs_u) & (npvs_u > 0)
            if np.any(valid_u):
                ax.plot(thick_vals[valid_u], npvs_u[valid_u] / 1e6, "--", color=no_cap_color, lw=2.0, alpha=1.0, label="No cap", zorder=4)
                # Optimum: thickness that gives highest NPV for this condition (no rev cap; scenario purity still applied)
                i_peak_u = np.nanargmax(npvs_u) if np.any(np.isfinite(npvs_u)) else None
                if i_peak_u is not None and np.isfinite(npvs_u[i_peak_u]) and npvs_u[i_peak_u] > 0:
                    ax.plot(thick_vals[i_peak_u], npvs_u[i_peak_u] / 1e6, "*", color=no_cap_color, ms=16, zorder=5)
            for i, cap_m in enumerate(caps64):
                npvs = np.array([_max_npv_at_thickness(rows64, t, cap_idx=i) for t in thick_vals])
                valid = np.isfinite(npvs) & (npvs > 0)
                if np.any(valid):
                    c = cool_caps[i % len(cool_caps)]
                    ax.plot(thick_vals[valid], npvs[valid] / 1e6, "-", color=c, lw=1.8, marker="o", markersize=4, markevery=max(1, valid.sum() // 10), label=f"Cap ${cap_m:.0f}M", zorder=2)
                    # Optimum: thickness that gives highest NPV for this revenue ceiling (under current scenario)
                    i_peak = np.nanargmax(npvs) if np.any(np.isfinite(npvs)) else None
                    if i_peak is not None and np.isfinite(npvs[i_peak]) and npvs[i_peak] > 0:
                        ax.plot(thick_vals[i_peak], npvs[i_peak] / 1e6, "*", color=c, ms=14, zorder=5)
            ax.set_xlabel("Thickness (cm)")
            ax.set_ylabel("NPV (USD, millions)")
            ax.set_title("Cu-64: NPV vs. Thickness by Revenue Ceiling (* = optimum)" + (f"\n{_sub}" if _sub else ""))
            ax.set_yscale("linear")
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(frameon=True, title="Revenue ceiling", fontsize=9)
            plt.tight_layout()
            plt.savefig(_plot_path(output_dir, sf, "cu64_npv_vs_thickness_by_revcap.jpeg"), dpi=500)
            plt.close(fig)

    # ---- 2b2b) Cu-64: NPV vs enrichment by revenue ceiling (* = optimum) ----
    # No-cap curve uses scenario production limits (purity etc.); each cap gets curve + star at enrichment that maximizes NPV.
    if rows64 and output_dir:
        enr_vals = _get_unique_enrichments(rows64, tol=0.001)
        if len(enr_vals) >= 1:
            cool_caps = _cool_palette_vibrant(max(len(caps64), 1))
            no_cap_color = (0.45, 0.45, 0.45)
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            enr_pct = np.array(enr_vals) * 100.0
            npvs_u = np.array([_max_npv_at_enrichment(rows64, e, use_uncap=True) for e in enr_vals])
            valid_u = np.isfinite(npvs_u) & (npvs_u > 0)
            if np.any(valid_u):
                ax.plot(enr_pct[valid_u], npvs_u[valid_u] / 1e6, "--", color=no_cap_color, lw=2.0, alpha=1.0, label="No cap", zorder=4)
                i_peak_u = np.nanargmax(npvs_u) if np.any(np.isfinite(npvs_u)) else None
                if i_peak_u is not None and np.isfinite(npvs_u[i_peak_u]) and npvs_u[i_peak_u] > 0:
                    ax.plot(enr_pct[i_peak_u], npvs_u[i_peak_u] / 1e6, "*", color=no_cap_color, ms=16, zorder=5)
            for i, cap_m in enumerate(caps64):
                npvs = np.array([_max_npv_at_enrichment(rows64, e, cap_idx=i) for e in enr_vals])
                valid = np.isfinite(npvs) & (npvs > 0)
                if np.any(valid):
                    c = cool_caps[i % len(cool_caps)]
                    ax.plot(enr_pct[valid], npvs[valid] / 1e6, "-", color=c, lw=1.8, marker="o", markersize=4, markevery=max(1, valid.sum() // 10), label=f"Cap ${cap_m:.0f}M", zorder=2)
                    i_peak = np.nanargmax(npvs) if np.any(np.isfinite(npvs)) else None
                    if i_peak is not None and np.isfinite(npvs[i_peak]) and npvs[i_peak] > 0:
                        ax.plot(enr_pct[i_peak], npvs[i_peak] / 1e6, "*", color=c, ms=14, zorder=5)
            ax.set_xlabel("Zn-64 Enrichment (%)")
            ax.set_ylabel("NPV (USD, millions)")
            ax.set_title("Cu-64: NPV vs. Enrichment by Revenue Ceiling (* = optimum)" + (f"\n{_sub}" if _sub else ""))
            ax.set_yscale("linear")
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(frameon=True, title="Revenue ceiling", fontsize=9)
            plt.tight_layout()
            plt.savefig(_plot_path(output_dir, sf, "cu64_npv_vs_enrichment_by_revcap.jpeg"), dpi=500)
            plt.close(fig)

    # ---- 2b3) Cu-67: NPV vs thickness by revenue ceiling (same style as Cu-64) ----
    if rows67 and output_dir:
        thick_vals_67 = np.sort(np.unique([r['thickness_cm'] for r in rows67 if np.isfinite(r.get('thickness_cm', np.nan))]))
        if len(thick_vals_67) >= 1:
            warm_caps = _warm_palette_vibrant(max(len(caps67), 1))
            no_cap_color = (0.45, 0.45, 0.45)
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            npvs_u_67 = np.array([_max_npv_at_thickness(rows67, t, use_uncap=True) for t in thick_vals_67])
            valid_u_67 = np.isfinite(npvs_u_67) & (npvs_u_67 > 0)
            if np.any(valid_u_67):
                ax.plot(thick_vals_67[valid_u_67], npvs_u_67[valid_u_67] / 1e6, "--", color=no_cap_color, lw=2.0, alpha=1.0, label="No cap", zorder=4)
                i_peak_u_67 = np.nanargmax(npvs_u_67) if np.any(np.isfinite(npvs_u_67)) else None
                if i_peak_u_67 is not None and np.isfinite(npvs_u_67[i_peak_u_67]) and npvs_u_67[i_peak_u_67] > 0:
                    ax.plot(thick_vals_67[i_peak_u_67], npvs_u_67[i_peak_u_67] / 1e6, "*", color=no_cap_color, ms=16, zorder=5)
            for i, cap_m in enumerate(caps67):
                npvs_67 = np.array([_max_npv_at_thickness(rows67, t, cap_idx=i) for t in thick_vals_67])
                valid_67 = np.isfinite(npvs_67) & (npvs_67 > 0)
                if np.any(valid_67):
                    c = warm_caps[i % len(warm_caps)]
                    ax.plot(thick_vals_67[valid_67], npvs_67[valid_67] / 1e6, "-", color=c, lw=1.8, marker="o", markersize=4, markevery=max(1, valid_67.sum() // 10), label=f"Cap ${cap_m:.0f}M", zorder=2)
                    i_peak_67 = np.nanargmax(npvs_67) if np.any(np.isfinite(npvs_67)) else None
                    if i_peak_67 is not None and np.isfinite(npvs_67[i_peak_67]) and npvs_67[i_peak_67] > 0:
                        ax.plot(thick_vals_67[i_peak_67], npvs_67[i_peak_67] / 1e6, "*", color=c, ms=14, zorder=5)
            ax.set_xlabel("Thickness (cm)")
            ax.set_ylabel("NPV (USD, millions)")
            ax.set_title("Cu-67: NPV vs. Thickness by Revenue Ceiling (* = optimum)" + (f"\n{_sub}" if _sub else ""))
            ax.set_yscale("linear")
            ax.grid(True, alpha=0.25, which="both")
            ax.legend(frameon=True, title="Revenue ceiling", fontsize=9)
            plt.tight_layout()
            plt.savefig(_plot_path(output_dir, sf, "cu67_npv_vs_thickness_by_revcap.jpeg"), dpi=500)
            plt.close(fig)

    # ---- 2c) Cu-64: NPV vs loading budget by Zn cost multiplier (0.7×–3× expected cost) ----
    zncost_multipliers = [0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    sub64 = run_data_df[run_data_df['use_zn67'] == False] if 'use_zn67' in run_data_df.columns else run_data_df
    if _contingency_mode() and not sub64.empty:
        sub64 = _run_data_cu64_only_contingency_enrichments(sub64)
    if not sub64.empty and output_dir:
        npvs_zncost = np.zeros((len(zncost_multipliers), len(budget_grid)))
        for mi, m in enumerate(zncost_multipliers):
            for bi, b in enumerate(budget_grid):
                valid = sub64[sub64['zn_feedstock_cost'].astype(float) * m <= b]
                if valid.empty:
                    npvs_zncost[mi, bi] = np.nan
                    continue
                vals = []
                for _, row in valid.iterrows():
                    try:
                        v = _npv_from_run_row(row, 1.0, None, False, purity_cap_64=purity_cap_64, loading_multiplier=m)
                        if np.isfinite(v):
                            vals.append(v)
                    except Exception:
                        pass
                npvs_zncost[mi, bi] = max(vals) if vals else np.nan
        no_cap_color = (0.45, 0.45, 0.45)
        cool_pal = _cool_palette_vibrant(max(len(zncost_multipliers), 1))
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for mi, m in enumerate(zncost_multipliers):
            npvs = np.nan_to_num(npvs_zncost[mi], nan=0.0)
            valid = npvs > 0
            if not np.any(valid):
                continue
            label = "1× expected" if m == 1.0 else f"{m:.1f}× Zn cost"
            if m == 1.0:
                ax.plot(budget_m[valid], npvs[valid] / 1e6, "--", color=no_cap_color, lw=2.0, alpha=1.0, label=label, zorder=4)
            else:
                c = cool_pal[mi % len(cool_pal)]
                ax.plot(budget_m[valid], npvs[valid] / 1e6, "-", color=c, lw=1.8, marker="o", markersize=4, markevery=max(1, valid.sum() // 15), label=label, zorder=2)
            i_peak = np.argmax(npvs)
            if npvs[i_peak] > 0:
                ax.plot(budget_m[i_peak], npvs[i_peak] / 1e6, "*", color=no_cap_color if m == 1.0 else cool_pal[mi % len(cool_pal)], ms=14, zorder=5)
        ax.set_xlabel("Loading budget (USD, millions)")
        ax.set_ylabel("NPV (USD, millions)")
        ax.set_title("Cu-64: NPV vs. Loading Budget by Zn Cost Multiplier (* = optimum)" + (f"\n{_sub}" if _sub else ""))
        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=True, title="Zn cost", fontsize=9)
        plt.tight_layout()
        plt.savefig(_plot_path(output_dir, sf, "cu64_npv_vs_loading_budget_by_zncost_multiplier.jpeg"), dpi=500)
        plt.close(fig)

    # ---- 3) Optimal thickness vs loading budget: one point per budget, colored by optimal enrichment ----
    fig, (ax64, ax67) = plt.subplots(2, 1, figsize=(14, 8))
    for ax, rows, label, is_cu67 in [(ax64, rows64, "Cu-64", False), (ax67, rows67, "Cu-67", True)]:
        if not rows:
            ax.set_visible(False)
            continue
        res = [_max_npv_and_thickness_at_budget(rows, b) for b in budget_grid]
        opt_t = np.array([r[1] for r in res])
        opt_enr = np.array([r[2] for r in res])
        # Unique enrichments (finite) for legend; assign color per enrichment
        u_enr = sorted(set(e for e in opt_enr if np.isfinite(e)))
        enr_to_color = {e: _get_enrichment_color(e, is_cu67) for e in u_enr}

        def _color_for(e):
            if not np.isfinite(e):
                return (0.85, 0.85, 0.85)
            best = min(u_enr, key=lambda u: abs(u - e))
            return enr_to_color[best]

        point_colors = [_color_for(opt_enr[i]) for i in range(len(budget_m))]
        # Step line in light gray (trend), then points colored by enrichment
        valid = np.isfinite(opt_t)
        if np.any(valid):
            ax.plot(budget_m[valid], opt_t[valid], "-", color="gray", lw=1, alpha=0.6, zorder=1)
        ax.scatter(budget_m, opt_t, c=point_colors, s=60, edgecolors="k", linewidths=0.5, zorder=3)
        # Legend: enrichment -> color, off to the right
        legend_handles = [
            Patch(facecolor=enr_to_color[e], edgecolor="k", label=f"{e*100:.1f}%" if e >= 0.1 else f"{e*100:.2f}%")
            for e in u_enr
        ]
        ax.legend(handles=legend_handles, frameon=True, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), title="Opt. enrichment")
        ax.set_xlabel("Loading budget (USD, millions)")
        ax.set_ylabel("Optimal thickness (cm)")
        ax.set_title(f"{label}: Opt. thick. vs budget{_sub}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25, which="both")
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(_plot_path(output_dir, sf, "opt_thickness_vs_budget_run_data.jpeg"), dpi=500)
    plt.close(fig)

    # ---- 4) Optimal thickness vs market size (revenue cap), Cu-64 above Cu-67 ----
    color_cu64_mkt, color_cu67_mkt = _cool_palette_cu64(5)[2], _warm_palette_cu67(5)[2]
    fig, (ax64, ax67) = plt.subplots(2, 1, figsize=(10, 8))
    for ax, rows, caps, label, color in [(ax64, rows64, caps64, "Cu-64", color_cu64_mkt), (ax67, rows67, caps67, "Cu-67", color_cu67_mkt)]:
        if not rows:
            ax.set_visible(False)
            continue
        opt_thick = []
        for cap_idx in range(len(caps)):
            best_npv, best_t = -np.inf, np.nan
            for r in rows:
                if r['npv_caps'][cap_idx] > best_npv:
                    best_npv, best_t = r['npv_caps'][cap_idx], r['thickness_cm']
            opt_thick.append(best_t if best_npv > -np.inf else np.nan)
        cap_vals = np.array(caps, dtype=float)
        valid = ~np.isnan(opt_thick)
        if np.any(valid):
            ax.plot(cap_vals[valid], np.array(opt_thick)[valid], "s-", color=color, ms=8, label="Opt. thickness")
        ax.set_xlabel("Revenue ceiling / market size (USD millions/yr)")
        ax.set_ylabel("Optimal thickness (cm)")
        ax.set_title(f"{label}: Opt. thickness vs market{_sub}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(_plot_path(output_dir, sf, "opt_thickness_vs_market_run_data.jpeg"), dpi=500)
    plt.close(fig)

    # ---- 5) NPV vs loading budget by price (2 rows × 4 cols: Cu-64 then Cu-67; cool / warm tones) ----
    run_data_driven_npv_vs_price_figure(output_dir=output_dir, sf=sf, n_budget=n_budget, cap64=cap64, cap67=cap67, purity_cap_64=purity_cap_64, tag=tag)

    # ---- 6) NPV considerations: compare cases (geometry + enrichment) at a few loading budgets ----
    def _constraint_label(tag, _c64, _c67, purity_cap_64):
        if tag == RUN_DATA_SELL_ALL:
            return "Sell all"
        if tag == RUN_DATA_MARKET_CAP:
            return "Market cap"
        if tag == RUN_DATA_PURITY_CAP:
            return "Purity ≥99.9%"
        if tag == RUN_DATA_MARKET_PURITY_CAP:
            return "Cap + purity"
        return tag or "default"
    run_data_driven_npv_considerations(output_dir=output_dir, sf=sf, rows64=rows64, rows67=rows67,
                                      cap64=cap64, cap67=cap67, purity_cap_64=purity_cap_64,
                                      constraint_label=_constraint_label(tag, cap64, cap67, purity_cap_64))

    # CSVs: NPV vs loading budget (interpolated) and optimal thickness vs market size
    if output_dir:
        for rows, caps, iso_name in [(rows64, caps64, "cu64"), (rows67, caps67, "cu67")]:
            if not rows:
                continue
            path_budget = _csv_path(output_dir, f"npv_vs_loading_budget_{iso_name}_interpolated.csv")
            with open(path_budget, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    ["loading_budget_millions", "npv_uncap_millions", "opt_thickness_cm"]
                    + [f"npv_cap_{int(c)}M_millions" for c in caps]
                )
                for i, b in enumerate(budget_grid):
                    npv_u, opt_t, _ = _max_npv_and_thickness_at_budget(rows, b)
                    npvs_cap = [_max_npv_and_thickness_at_budget(rows, b, cap_idx=j)[0] for j in range(len(caps))]
                    w.writerow(
                        [round(budget_m[i], 6), round((npv_u if not np.isnan(npv_u) else 0) / 1e6, 4), round(opt_t, 4)]
                        + [round((x if not np.isnan(x) else 0) / 1e6, 4) for x in npvs_cap]
                    )
            path_market = _csv_path(output_dir, f"opt_thickness_vs_market_{iso_name}.csv")
            opt_thick_list = []
            for cap_idx in range(len(caps)):
                best_npv, best_t = -np.inf, np.nan
                for r in rows:
                    if r['npv_caps'][cap_idx] > best_npv:
                        best_npv, best_t = r['npv_caps'][cap_idx], r['thickness_cm']
                opt_thick_list.append(best_t if best_npv > -np.inf else np.nan)
            with open(path_market, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["revenue_cap_millions_yr", "opt_thickness_cm"])
                for cap_m, ot in zip(caps, opt_thick_list):
                    w.writerow([int(cap_m), round(float(ot), 4) if not np.isnan(ot) else ""])


def run_data_driven_npv_considerations(output_dir=None, sf=None, rows64=None, rows67=None, cap64=None, cap67=None, purity_cap_64=False, constraint_label=None):
    """
    NPV considerations: (1) cases at loading budgets; (2) thickness vs enrichment with NPV heatmap and IRR contours.
    Subplot 1: scatter of NPV vs loading budget, legend for geometry + enrichment.
    Subplot 2: heatmap of NPV (interpolated), labeled contour lines of IRR, key points scattered.
    cap64, cap67, purity_cap_64: constraint for this scenario (used in heatmap and for scenario NPV in scatter).
    constraint_label: short string for figure title (e.g. "Sell all (no cap)").
    """
    if not rows64 and not rows67:
        return
    if run_data_df is None or run_data_df.empty:
        return
    # Loading budgets (USD millions) at which to compare cases
    budget_m_few = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
    budget_few = budget_m_few * 1e6

    def _geometry_bucket(thickness_cm, t_unique):
        """Classify thickness into thinnest / medium / thickest."""
        if len(t_unique) == 0:
            return "medium"
        if len(t_unique) == 1:
            return "medium"
        t_lo, t_hi = float(t_unique[0]), float(t_unique[-1])
        t_mid = float(t_unique[len(t_unique) // 2]) if len(t_unique) >= 2 else (t_lo + t_hi) / 2
        if thickness_cm <= (t_lo + t_mid) / 2 + 0.01:
            return "thinnest"
        if thickness_cm >= (t_mid + t_hi) / 2 - 0.01:
            return "thickest"
        return "medium"

    def _npv_for_row(r, is_cu67):
        """NPV to use for this scenario (scatter): scenario NPV if present else uncap."""
        return r.get("npv_scenario", r.get("npv_uncap"))

    def _max_npv_at_budget_for_geom_enr(rows, budget, geom_bucket, enr, t_unique, tol=0.001, is_cu67=False):
        best = -np.inf
        for r in rows:
            if r["loading"] > budget:
                continue
            if _geometry_bucket(r["thickness_cm"], t_unique) != geom_bucket:
                continue
            if abs(r.get("enrichment", 0) - enr) > tol:
                continue
            npv = _npv_for_row(r, is_cu67)
            if npv > best:
                best = npv
        return best if best > -np.inf else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for row_idx, (rows, label, is_cu67) in enumerate([
        (rows64 or [], "Cu-64", False),
        (rows67 or [], "Cu-67", True),
    ]):
        ax_budget = axes[row_idx, 0]
        ax_heat = axes[row_idx, 1]
        if not rows:
            ax_budget.set_visible(False)
            ax_heat.set_visible(False)
            continue

        # ---- Subplot 1: NPV vs loading budget (cases at selected budgets) ----
        t_unique = sorted(set(r["thickness_cm"] for r in rows))
        geom_order = ["thinnest", "medium", "thickest"]
        geoms = [g for g in geom_order if any(_geometry_bucket(r["thickness_cm"], t_unique) == g for r in rows)]
        enrichments = sorted(set(r["enrichment"] for r in rows if np.isfinite(r.get("enrichment", np.nan))))
        markers = {"thinnest": "o", "medium": "s", "thickest": "^"}
        geom_labels = {"thinnest": "Thinnest Zn", "medium": "Medium", "thickest": "Thickest"}
        for geom in geoms:
            for enr in enrichments:
                xs, ys = [], []
                for b in budget_few:
                    npv = _max_npv_at_budget_for_geom_enr(rows, b, geom, enr, t_unique, is_cu67=is_cu67)
                    if np.isfinite(npv) and npv > 0:
                        xs.append(b / 1e6)
                        ys.append(npv / 1e6)
                if not xs:
                    continue
                c = _get_enrichment_color(enr, is_cu67)
                enr_str = f"{enr*100:.1f}%" if enr >= 0.1 else f"{enr*100:.2f}%"
                ax_budget.scatter(xs, ys, marker=markers.get(geom, "o"), s=70, c=[c], edgecolors="k", linewidths=0.5,
                                 label=f"{geom_labels.get(geom, geom)} | {enr_str}", alpha=0.9)
        ax_budget.set_xlabel("Loading budget (USD, millions)")
        ax_budget.set_ylabel("NPV (USD, millions)")
        ax_budget.set_title(f"{label}: NPV vs loading budget")
        ax_budget.set_xscale("log")
        ax_budget.set_yscale("log")
        ax_budget.grid(True, alpha=0.25, which="both")
        ax_budget.legend(frameon=True, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)

        # ---- Subplot 2: Thickness vs enrichment — NPV heatmap + IRR contours + key points ----
        sub = run_data_df[run_data_df["use_zn67"] == is_cu67] if "use_zn67" in run_data_df.columns else run_data_df
        if sub.empty:
            ax_heat.set_visible(False)
            continue
        if not is_cu67 and _contingency_mode():
            sub = _run_data_cu64_only_contingency_enrichments(sub)
        if sub.empty:
            ax_heat.set_visible(False)
            continue
        thick_col = "outer_cm"
        cap = cap67 if is_cu67 else cap64
        pts_thick, pts_enr, pts_npv, pts_irr = [], [], [], []
        for _, row in sub.iterrows():
            try:
                t = float(row[thick_col])
                e = float(row.get("zn64_enrichment", 0))
                npv = _npv_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64)
                ir = _irr_from_run_row(row, 1.0, cap, is_cu67, purity_cap_64=purity_cap_64)
                irr_pct = ir * 100 if np.isfinite(ir) else np.nan
            except (TypeError, KeyError, ValueError):
                continue
            if not np.isfinite(t) or not np.isfinite(e):
                continue
            pts_thick.append(t)
            pts_enr.append(e)
            pts_npv.append(npv / 1e6)
            pts_irr.append(irr_pct if np.isfinite(irr_pct) else np.nan)
        pts_thick = np.array(pts_thick)
        pts_enr = np.array(pts_enr) * 100  # percent for axis
        pts_npv = np.array(pts_npv)
        pts_irr = np.array(pts_irr)
        valid_npv = np.isfinite(pts_npv) & (pts_npv > -1e9)
        valid_irr = np.isfinite(pts_irr) & (pts_irr >= 0)
        if np.sum(valid_npv) < 3:
            ax_heat.set_visible(False)
            continue
        # NPV heatmap: use tricontourf when data allow Delaunay; else scatter only (avoids qhull "singular input data" with 3 enrichments)
        n_unique_xy = len(set(zip(pts_thick[valid_npv], pts_enr[valid_npv])))
        use_tri = n_unique_xy >= 4
        try:
            if use_tri:
                ax_heat.tricontourf(pts_thick[valid_npv], pts_enr[valid_npv], pts_npv[valid_npv], levels=12, cmap="viridis", alpha=0.85)
            # IRR contour lines (labeled) — only if triangulation is used and we have enough points
            if use_tri and np.sum(valid_irr) >= 3:
                irr_vals = pts_irr[valid_irr]
                irr_min, irr_max = float(np.min(irr_vals)), float(np.max(irr_vals))
                irr_max = min(irr_max, 1000.0)
                n_lev = 8
                irr_levels = np.linspace(max(0, irr_min), irr_max, n_lev)
                if len(irr_levels) >= 2:
                    cs = ax_heat.tricontour(pts_thick[valid_irr], pts_enr[valid_irr], pts_irr[valid_irr],
                                            levels=irr_levels, colors="white", linewidths=1.2, alpha=0.95)
                    ax_heat.clabel(cs, cs.levels, inline=True, fontsize=8, fmt="%.0f%%")
                else:
                    single_lev = irr_max if irr_max > 0 else irr_min
                    ax_heat.tricontour(pts_thick[valid_irr], pts_enr[valid_irr], pts_irr[valid_irr],
                                       levels=[single_lev], colors="white", linewidths=1.2, alpha=0.95)
                    ax_heat.text(0.5, 0.95, f"IRR = {single_lev:.1f}%", transform=ax_heat.transAxes, fontsize=9, color="white", ha="center", va="top")
        except (RuntimeError, ValueError):
            use_tri = False
        if not use_tri:
            # Fallback: scatter by NPV (no contour) when triangulation fails or too few unique points
            sc = ax_heat.scatter(pts_thick, pts_enr, c=pts_npv, s=80, cmap="viridis", edgecolors="k", linewidths=0.6, zorder=5, alpha=0.9)
        else:
            ax_heat.scatter(pts_thick, pts_enr, s=25, c="white", edgecolors="k", linewidths=0.6, zorder=5, alpha=0.9)
        ax_heat.set_xlabel("Thickness (cm)")
        ax_heat.set_ylabel("Enrichment (%)")
        ax_heat.set_title(f"{label}: Thickness vs Enrichment — NPV, IRR")
        ax_heat.grid(True, alpha=0.25)
        vmin, vmax = np.nanmin(pts_npv[valid_npv]), np.nanmax(pts_npv[valid_npv])
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_heat, shrink=0.7, label="NPV (USD, millions)")

    if constraint_label:
        fig.suptitle(f"Constraint: {constraint_label}", fontsize=12, y=1.00)
    plt.tight_layout(rect=[0, 0, 0.96, 0.98])
    if output_dir and sf:
        plt.savefig(_plot_path(output_dir, sf, "npv_considerations.jpeg"), dpi=500)
    plt.close(fig)


def run_data_driven_analyses(output_dir=None):
    """
    Run data-driven NPV analyses. Output per constraint in four folders:
    1) run_data/sell_all           — no cap; full revenue
    2) run_data/market_cap         — revenue cap $30M Cu-64, $6M Cu-67
    3) run_data/purity_cap        — Cu-64 revenue only if radionuclide purity ≥99.9%
    4) run_data/market_purity_cap — both market cap and purity threshold
    """
    plt.rcParams.update({"text.usetex": False})
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Saving plots to {output_dir}")

    # Scenario tables (printed) and per-scenario enrichment + thickness plots — all 4 cases
    run_data_driven_scenario(RUN_DATA_SELL_ALL, sell_fraction=1.0, cap64=None, cap67=None, purity_cap_64=False, output_dir=output_dir)
    run_data_driven_scenario(RUN_DATA_MARKET_CAP, sell_fraction=1.0, cap64=30e6, cap67=6e6, purity_cap_64=False, output_dir=output_dir)
    run_data_driven_scenario(RUN_DATA_PURITY_CAP, sell_fraction=1.0, cap64=None, cap67=None, purity_cap_64=True, output_dir=output_dir)
    run_data_driven_scenario(RUN_DATA_MARKET_PURITY_CAP, sell_fraction=1.0, cap64=30e6, cap67=6e6, purity_cap_64=True, output_dir=output_dir)

    for tag, cap64, cap67, purity in [
        (RUN_DATA_SELL_ALL, None, None, False),
        (RUN_DATA_MARKET_CAP, 30e6, 6e6, False),
        (RUN_DATA_PURITY_CAP, None, None, True),
        (RUN_DATA_MARKET_PURITY_CAP, 30e6, 6e6, True),
    ]:
        run_data_driven_enrichment_plots_combined(output_dir=output_dir, tag=tag, cap64=cap64, cap67=cap67, purity_cap_64=purity)
        run_data_driven_thickness_plots_for_scenario(output_dir=output_dir, tag=tag, cap64=cap64, cap67=cap67, purity_cap_64=purity)
        run_data_driven_payback_plots(output_dir=output_dir, tag=tag, cap64=cap64, cap67=cap67, purity_cap_64=purity)
        run_data_driven_irr_plots(output_dir=output_dir, tag=tag, cap64=cap64, cap67=cap67, purity_cap_64=purity)
        run_data_driven_budget_plots(output_dir=output_dir, tag=tag, cap64=cap64, cap67=cap67, purity_cap_64=purity)
        run_data_driven_investor_plots(output_dir=output_dir, tag=tag, cap64=cap64, cap67=cap67, purity_cap_64=purity)

    # Legacy thickness (revenue vs thickness, no scenario) — save to sell_all for consistency
    run_data_driven_thickness_plots(output_dir=output_dir)


def _update_thicknesses_from_run_data():
    """Set THICKNESSES_CM/M from run data (unique outer_cm, or outer minus inner). Call after set_run_data_from_csv or run_flare_combined."""
    global THICKNESSES_CM, THICKNESSES_M
    if run_data_df is not None and not run_data_df.empty and 'outer_cm' in run_data_df.columns:
        if 'inner_cm' in run_data_df.columns:
            t = (run_data_df['outer_cm'].astype(float) - run_data_df['inner_cm'].astype(float).fillna(0)).dropna().unique()
        else:
            t = run_data_df['outer_cm'].dropna().unique()
        t = np.sort(np.asarray(t, dtype=float))
        if len(t) > 0:
            THICKNESSES_CM = t
            THICKNESSES_M = t / 100.0
            return
    THICKNESSES_CM = np.array([5.0, 10.0, 15.0, 20.0])
    THICKNESSES_M = THICKNESSES_CM / 100.0





def run_all_analyses(output_dir=None):
    """Run data-driven NPV analyses only (from summary CSV: thickness, enrichment, loading, production, NPV)."""
    run_data_driven_analyses(output_dir=output_dir)


def run_data_variable_analyses(output_dir=None):
    """
    Run a restricted set of data-driven NPV analyses and save all plots
    into a separate folder named run_data_variable (mirrors irrad/cooldown
    handling from the main scenarios).

    This assumes run_data_df and pricing have already been initialized
    (e.g. via run_flare_from_openmc, run_flare_combined, or set_run_data_from_csv).

    Also prints a simple market-size trajectory:
    - Starts at $20M in year 0
    - Increases toward an asymptote of $300M
    - Reaches 99% of the asymptote by year 10
    """
    def _market_size_usd(t_years, m0=20e6, m_inf=300e6, frac_at_10=0.99):
        """
        Saturating exponential market size:

            M(t) = M_inf - (M_inf - M0) * exp(-k t)
            where k is chosen so that M(10) = frac_at_10 * M_inf.

        This gives:

            k = - (1/10) * ln((M_inf - frac_at_10*M_inf) / (M_inf - M0))
        """
        # Compute k from boundary condition at t=10
        if frac_at_10 <= 0 or frac_at_10 >= 1:
            frac_at_10 = 0.99
        num = m_inf - frac_at_10 * m_inf
        den = m_inf - m0
        if den <= 0 or num <= 0:
            return m0
        k = - (1.0 / 10.0) * np.log(num / den)
        return m_inf - (m_inf - m0) * np.exp(-k * float(t_years))

    def _build_cu64_market_matching_table(years):
        """
        For each time t in years, pick an "ideal" Cu-64 configuration that:
        - Has radionuclide purity >= 99.9%
        - Provides production to meet (or get as close as possible to) the Cu-64
          market size M(t) from _market_size_usd.

        Returns a list of dict rows with:
          time_years, market_cap_usd, cu64_price_per_mci_usd,
          thickness_cm, production_mci_yr_999, loading_usd,
          incr_loading_usd, cum_loading_usd, discounted_cum_loading_usd,
          and npv_usd (NPV at that time's market cap and purity gate).
        """
        if run_data_df is None or run_data_df.empty:
            return []
        if 'use_zn67' not in run_data_df.columns or 'zn_feedstock_cost' not in run_data_df.columns:
            return []

        df = run_data_df[run_data_df['use_zn67'] == False].copy()
        if df.empty:
            return []
        # Apply contingency filter if needed (still Cu-64 only)
        if _contingency_mode():
            df = _run_data_cu64_only_contingency_enrichments(df)
            if df.empty:
                return []

        # Keep only rows with explicit Cu-64 purity >= 99.9%
        if 'cu64_radionuclide_purity' not in df.columns:
            return []
        df = df[df['cu64_radionuclide_purity'].astype(float) >= 0.999]
        if df.empty:
            return []

        # Basic quantities per row
        df = df.copy()
        df['prod_g_yr'] = df['cu64_g_yr'].astype(float)
        df['thickness_cm'] = df['outer_cm'].astype(float)
        df['loading_usd'] = df['zn_feedstock_cost'].astype(float)

        # Revenue per year at current Cu-64 price (no cap, purity already enforced via filter)
        df['rev_usd_yr'] = df['prod_g_yr'] * float(price_cu64_usd_per_g)

        # Reconstruct $/mCi for Cu-64 from price_per_g and specific activity
        sa_ci_per_g_64 = _specific_activity_ci_per_g("64")
        price_per_mci_64 = float(price_cu64_usd_per_g) / (1000.0 * sa_ci_per_g_64)

        rows = []
        prev_loading = None
        cum_loading = 0.0

        for t in years:
            market_cap = _market_size_usd(t)

            # Choose configuration: among rows with rev >= market_cap, pick the one
            # with the smallest revenue (closest just-above match); if none, pick
            # the row with max revenue.
            above = df[df['rev_usd_yr'] >= market_cap]
            if not above.empty:
                idx = (above['rev_usd_yr'] - market_cap).abs().idxmin()
            else:
                idx = df['rev_usd_yr'].idxmax()
            sel = df.loc[idx]

            thickness = float(sel['thickness_cm'])
            loading = float(sel['loading_usd'])
            prod_g_yr = float(sel['prod_g_yr'])

            # Annual production in mCi (all at >=99.9% purity by construction)
            prod_ci_yr = prod_g_yr * sa_ci_per_g_64
            prod_mci_yr = prod_ci_yr * 1000.0

            # Incremental and cumulative loading cost as we step through time
            if prev_loading is None:
                incr = loading
            else:
                incr = max(0.0, loading - prev_loading)
            cum_loading += incr
            prev_loading = loading

            # Discount cumulative loading cost back to present using simple 1/(1+r)^t
            discount_factor = 1.0 / ((1.0 + r) ** float(t))
            discounted_cum = cum_loading * discount_factor

            # NPV for this configuration under a revenue ceiling equal to M(t),
            # with Cu-64 purity constraint applied.
            npv_t = _npv_from_run_row(
                sel,
                sell_fraction=1.0,
                cap_usd_per_yr=float(market_cap),
                is_cu67=False,
                purity_cap_64=True,
            )

            rows.append({
                "time_years": float(t),
                "market_cap_usd": float(market_cap),
                "cu64_price_per_mci_usd": float(price_per_mci_64),
                "thickness_cm": thickness,
                "production_mci_yr_999": float(prod_mci_yr),
                "loading_usd": loading,
                "incremental_loading_usd": float(incr),
                "cumulative_loading_usd": float(cum_loading),
                "discounted_cumulative_loading_usd": float(discounted_cum),
                "npv_usd": float(npv_t),
            })

        return rows
    if run_data_df is None or run_data_df.empty:
        print("  No run data; skipping run_data_variable analyses")
        return
    if price_cu64_usd_per_g is None or price_cu67_usd_per_g is None:
        raise ValueError(
            "Pricing not set. Run set_pricing_from_run_config() first "
            "(e.g. via run_flare_from_openmc or run_flare_combined)."
        )

    base_output = None
    if output_dir:
        base_output = os.path.join(output_dir, FOLDER_RUN_DATA_VARIABLE)
    else:
        base_output = FOLDER_RUN_DATA_VARIABLE
    os.makedirs(base_output, exist_ok=True)
    print(f"  Saving run_data_variable plots to {base_output}")

    # Print market size formula and table (years 1, 3, 5, 10, 15, 20)
    print("\n" + "=" * 80)
    print("  MARKET SIZE TRAJECTORY (run_data_variable scenario)")
    print("  M(t) = M_inf - (M_inf - M0) * exp(-k t)")
    print("    with M0 = $20M, M_inf = $300M, and k chosen so that M(10) ≈ 0.99 * M_inf")
    years = [1, 3, 5, 10, 15, 20]
    header = f"  {'Year':>4}  {'Market Size (USD)':>22}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for y in years:
        m = _market_size_usd(y)
        print(f"  {y:4d}  {m:22,.0f}")
    print("=" * 80 + "\n")

    # Build and print Cu-64 thickness/production/loading trajectory to track market demand
    years_full = list(range(1, T_years + 1))
    traj_rows = _build_cu64_market_matching_table(years_full)
    if traj_rows:
        print("  IDEAL Cu-64 THICKNESS / LOADING TRAJECTORY TO MEET VARIABLE MARKET")
        header = (
            f"  {'Year':>4}  {'Market Cap ($)':>15}  {'Price $/mCi':>12}  "
            f"{'Thick (cm)':>10}  {'Prod (mCi/yr)':>15}  {'ΔZn cost ($)':>13}  "
            f"{'Cum Zn ($)':>13}  {'Disc. Cum Zn ($)':>17}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in traj_rows:
            print(
                f"  {int(row['time_years']):4d}  "
                f"{row['market_cap_usd']:15,.0f}  "
                f"{row['cu64_price_per_mci_usd']:12.4f}  "
                f"{row['thickness_cm']:10.2f}  "
                f"{row['production_mci_yr_999']:15,.0f}  "
                f"{row['incremental_loading_usd']:13,.0f}  "
                f"{row['cumulative_loading_usd']:13,.0f}  "
                f"{row['discounted_cumulative_loading_usd']:17,.0f}"
            )
        print()
        print("  Column definitions (formula list):")
        print("    1) Year t: discrete time index t = 1..T_years.")
        print("    2) Market Cap ($): M(t) = M_inf - (M_inf - M0) * exp(-k t), with M0=$20M, M_inf=$300M,")
        print("       and k chosen so that M(10) ≈ 0.99 * M_inf.")
        print("    3) Price $/mCi: p_64_mCi = price_cu64_usd_per_g / (1000 * SA_64), where")
        print("       SA_64 = specific activity of Cu-64 in Ci/g.")
        print("    4) Thick (cm): outer_cm of the Cu-64 configuration that best matches M(t)")
        print("       (purity ≥ 99.9%, chosen to have revenue ≥ M(t) with minimal excess;")
        print("       if none reach M(t), use the configuration with maximum revenue).")
        print("    5) Prod (mCi/yr): prod_mCi(t) = prod_g_yr * SA_64 * 1000, using Cu-64 runs")
        print("       at ≥99.9% purity and the chosen thickness.")
        print("    6) ΔZn cost ($): incremental Zn loading cost added at time t,")
        print("       Δload(t) = max(0, load(t) - load(t-1)), where load(t) is the Zn")
        print("       feedstock cost for the chosen configuration.")
        print("    7) Cum Zn ($): cumulative Zn loading cost, Cum_load(t) = Σ_{τ=1..t} Δload(τ).")
        print("    8) Disc. Cum Zn ($): discounted cumulative loading cost,")
        print("       PV_load(t) = Cum_load(t) / (1 + r)^t using the global discount rate r.")
        print()

        # Save to CSV for downstream analysis
        csv_path = os.path.join(base_output, "cu64_market_matching_thickness_trajectory.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_years",
                "market_cap_usd",
                "cu64_price_per_mci_usd",
                "thickness_cm",
                "production_mci_yr_999",
                "loading_usd",
                "incremental_loading_usd",
                "cumulative_loading_usd",
                "discounted_cumulative_loading_usd",
                "npv_usd",
            ])
            for row in traj_rows:
                writer.writerow([
                    row["time_years"],
                    row["market_cap_usd"],
                    row["cu64_price_per_mci_usd"],
                    row["thickness_cm"],
                    row["production_mci_yr_999"],
                    row["loading_usd"],
                    row["incremental_loading_usd"],
                    row["cumulative_loading_usd"],
                    row["discounted_cumulative_loading_usd"],
                    row["npv_usd"],
                ])

        # Simple 1D plot: NPV (left y) vs time with market demand curve,
        # and blanket thickness (right y) vs time.
        times = [row["time_years"] for row in traj_rows]
        market_m = [row["market_cap_usd"] / 1e6 for row in traj_rows]
        npv_m = [row["npv_usd"] / 1e6 for row in traj_rows]
        thickness = [row["thickness_cm"] for row in traj_rows]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        # Left axis: NPV and market demand (both in $M)
        ax1.plot(times, npv_m, color="tab:blue", label="NPV (scenario)")
        ax1.plot(times, market_m, color="gray", linestyle="--", label="Market demand")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("NPV and market (USD, millions)")

        # Label the market demand curve near its end
        if times:
            ax1.text(
                times[-1],
                market_m[-1],
                "Market demand curve",
                color="gray",
                fontsize=9,
                ha="right",
                va="bottom",
            )

        # Right axis: blanket thickness
        ax2.plot(times, thickness, color="tab:red", label="Blanket thickness")
        ax2.set_ylabel("Blanket thickness (cm)")

        # Clean, simple title and legend
        fig.suptitle("Cu-64 NPV and blanket thickness vs time\n(variable market, purity ≥99.9%)")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        fig.tight_layout()
        plot_path = os.path.join(base_output, "npv_and_thickness_vs_time_variable.jpeg")
        plt.savefig(plot_path, dpi=500)
        plt.close(fig)

    # For the variable folder, run a purity-threshold scenario for Cu-64 with
    # a market cap that evolves over time according to _market_size_usd. To
    # keep the existing constant-cap machinery, we approximate this with an
    # effective annual cap equal to the average market size over the project
    # lifetime (1..T_years), while still enforcing the 99.9% purity gate.
    tag = RUN_DATA_PURITY_CAP
    cap67 = None
    purity = True

    caps = [_market_size_usd(t) for t in range(1, T_years + 1)]
    cap64 = float(np.mean(caps)) if len(caps) > 0 else None

    run_data_driven_scenario(
        tag,
        sell_fraction=1.0,
        cap64=cap64,
        cap67=cap67,
        purity_cap_64=purity,
        output_dir=base_output,
    )
    run_data_driven_enrichment_plots_combined(
        output_dir=base_output,
        tag=tag,
        cap64=cap64,
        cap67=cap67,
        purity_cap_64=purity,
    )
    run_data_driven_thickness_plots_for_scenario(
        output_dir=base_output,
        tag=tag,
        cap64=cap64,
        cap67=cap67,
        purity_cap_64=purity,
    )
    run_data_driven_payback_plots(
        output_dir=base_output,
        tag=tag,
        cap64=cap64,
        cap67=cap67,
        purity_cap_64=purity,
    )
    run_data_driven_irr_plots(
        output_dir=base_output,
        tag=tag,
        cap64=cap64,
        cap67=cap67,
        purity_cap_64=purity,
    )
    run_data_driven_budget_plots(
        output_dir=base_output,
        tag=tag,
        cap64=cap64,
        cap67=cap67,
        purity_cap_64=purity,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        arg1 = sys.argv[1]
        econ_h = int(sys.argv[2]) if len(sys.argv) >= 3 else _default_npv_irrad_hours()
        if os.path.isfile(arg1):
            set_run_data_from_csv(arg1, irrad_hours=econ_h)
            run_data_driven_analyses()
        elif os.path.isdir(arg1):
            run_flare_from_openmc(arg1, econ_irrad_hours=econ_h)
        else:
            run_data_driven_analyses()
    else:
        run_data_driven_analyses()