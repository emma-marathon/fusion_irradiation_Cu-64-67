'''
Helper functions for neutronics analysis:
- Reaction rate extraction from tallies
- Bateman equation for isotope evolution (production + decay)
- Geometry volume calculations
- Zn-64/Zn-67 enrichment: anchor-based piecewise linear interpolation and plotting.
'''
import os
import numpy as np
import openmc

# ---------------------------------------------------------------------------
# Zn-64 enrichment: fraction anchors natural (0.4917), 0.53, 0.71, 99.18% (0.9918).
# Cost anchors from Cu64 enrichment estimates (median of low/high) for 71%–100%;
# natural Zn = $1000/kg. Enrichments without data: interpolated estimate.
# ---------------------------------------------------------------------------
ZN64_ANCHOR_X = [0.4917, 0.53, 0.71, 0.9918]
ZN64_ANCHOR_FRACTIONS = [
    {'Zn66': 0.2773, 'Zn67': 0.0404, 'Zn68': 0.1845, 'Zn70': 0.0061},
    {'Zn66': 0.294, 'Zn67': 0.031, 'Zn68': 0.140, 'Zn70': 0.005},
    {'Zn66': 0.247, 'Zn67': 0.016, 'Zn68': 0.03, 'Zn70': 0.001},
    {'Zn66': 0.004, 'Zn67': 0.001, 'Zn68': 0.003, 'Zn70': 0.0002},
]
# Cost: Enrichment Target 71%–100%; median of Cost Estimate Low/High ($/kg). Natural = $1000.
# Low:  2696, 3040, 2921, 2705, 2974, 5172, 6308, 8076, 11228, 18204, 30504
# High: 3606, 4065, 3906, 3617, 3977, 6916, 8435, 10800, 15015, 24344, 40794
# Enrichments 49.17%–99.9%; cost anchors aligned (no 100%).
ZN64_COST_ANCHOR_X = [
    0.4917,  # natural
    0.53,    # estimate between natural and 71%
    0.71, 0.76, 0.81, 0.86, 0.91, 0.96, 0.97, 0.98, 0.99, 0.999  # 99.9% max
]
ZN64_ANCHOR_COST = [
    1000.0,   # natural
    2100.0,   # 53% estimate (between 1000 and 71% cost)
    3151.0,   # 71%  median(2696, 3606)
    3552.5,   # 76%  median(3040, 4065)
    3413.5,   # 81%  median(2921, 3906)
    3161.0,   # 86%  median(2705, 3617)
    3475.5,   # 91%  median(2974, 3977)
    6044.0,   # 96%  median(5172, 6916)
    7371.5,   # 97%  median(6308, 8435)
    9438.0,   # 98%  median(8076, 10800)
    13121.5,  # 99%  median(11228, 15015)
    21274.0,  # 99.9% (interp from 99%–100% medians)
]
ZN64_RANGE = (0.4917, 1.0)

# Contingency: fixed $/kg for 3 enrichments only (no interpolation). NPV runs only for these.
# Natural $1000/kg; 71% = mean(2696, 3606) = 3151; 99% = $12121/kg (not 99.9%)
ZN64_CONTINGENCY_ENRICHMENTS = (0.4917, 0.71, 0.99)
ZN64_CONTINGENCY_COST_PER_KG = {
    0.4917: 1000.0,
    0.71: 3151.0,    # mean(2696, 3606)
    0.99: 12121.0,   # 99% enrichment $/kg
}

# ---------------------------------------------------------------------------
# Zn-67 enrichment: fraction anchors 0.0404 (natural), 0.073, then interpolated to 17.7%.
# Cost anchors from Cu67 enrichment cost estimates (median of low/high) for 7.3%–17.7%.
# For e > 7.3%, other isotopes (Zn64, Zn66, Zn68, Zn70) interpolate toward a high-enrichment
# composition so fractions vary smoothly with Zn67 enrichment.
# ---------------------------------------------------------------------------
ZN67_ANCHOR_X = [0.0404, 0.073, 0.10, 0.14, 0.177]
ZN67_ANCHOR_FRACTIONS = [
    {'Zn64': 0.486, 'Zn66': 0.279, 'Zn68': 0.188, 'Zn70': 0.0066},
    {'Zn64': 0.005, 'Zn66': 0.342, 'Zn68': 0.557, 'Zn70': 0.023},
    {'Zn64': 0.005, 'Zn66': 0.333, 'Zn68': 0.539, 'Zn70': 0.023},   # 10% Zn67: others sum 0.90
    {'Zn64': 0.004, 'Zn66': 0.319, 'Zn68': 0.514, 'Zn70': 0.023},   # 14% Zn67: others sum 0.86
    {'Zn64': 0.004, 'Zn66': 0.302, 'Zn68': 0.491, 'Zn70': 0.026},   # 17.7% Zn67: others sum 0.823
]
# Cost: Enrichment Target 7.30%–17.70%; median of Cost Estimate Low/High ($/kg)
# Low:  6014, 6868, 10857, 17841, 27638, 40151, 55344, 73232, 93868, 117342, 143776, 164121
# High: 8042, 9184, 14519, 23859, 36961, 53695, 74013, 97935, 125532, 156925, 192275, 219483
ZN67_COST_ANCHOR_X = [0.0404, 0.073, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.177]
ZN67_ANCHOR_COST = [
    1000.0,   # natural 4.04%
    7028.0,   # 7.30%  median(6014, 8042)
    8026.0,   # 8.00%  median(6868, 9184)
    12688.0,  # 9.00%  median(10857, 14519)
    20850.0,  # 10.00% median(17841, 23859)
    32299.5,  # 11.00% median(27638, 36961)
    46923.0,  # 12.00% median(40151, 53695) 
    64678.5,  # 13.00% median(55344, 74013)
    85583.5,  # 14.00% median(73232, 97935)
    109700.0, # 15.00% median(93868, 125532)
    137133.5, # 16.00% median(117342, 156925)
    168025.5, # 17.00% median(143776, 192275)
    191802.0, # 17.70% median(164121, 219483)
]
ZN67_RANGE = (0.0404, 0.177)

_ZN64_GRID = list(ZN64_COST_ANCHOR_X)
ZN64_LOG_ANCHOR_X, ZN64_LOG_ANCHOR_FRACTIONS = [0.4917, 0.9918], [ZN64_ANCHOR_FRACTIONS[0], ZN64_ANCHOR_FRACTIONS[3]]

def _interp(x, xs, ys, r):
    """Linear interpolate; clamp x to range r."""
    return float(np.interp(max(r[0], min(r[1], float(x))), xs, ys))

def _fractions_interp(e, anchor_x, anchor_dicts, main_key, other_keys, r):
    """{main_key: e, others: linear interp from anchor_dicts}."""
    e = max(r[0], min(r[1], float(e)))
    return {main_key: e, **{k: _interp(e, anchor_x, [d[k] for d in anchor_dicts], r) for k in other_keys}}

def _zn64_fractions_interp(zn64_enrichment):
    return _fractions_interp(zn64_enrichment, ZN64_ANCHOR_X, ZN64_ANCHOR_FRACTIONS, 'Zn64', ['Zn66', 'Zn67', 'Zn68', 'Zn70'], ZN64_RANGE)

def _zn64_fractions_interp_log(zn64_enrichment):
    e = max(ZN64_RANGE[0], min(ZN64_RANGE[1], float(zn64_enrichment)))
    if e <= 0:
        return {'Zn64': e, **{k: ZN64_LOG_ANCHOR_FRACTIONS[0][k] for k in ['Zn66', 'Zn67', 'Zn68', 'Zn70']}}
    t = float(np.clip((np.log(e) - np.log(ZN64_LOG_ANCHOR_X[0])) / (np.log(ZN64_LOG_ANCHOR_X[1]) - np.log(ZN64_LOG_ANCHOR_X[0])), 0, 1))
    return {'Zn64': e, **{k: float((1-t)*ZN64_LOG_ANCHOR_FRACTIONS[0][k] + t*ZN64_LOG_ANCHOR_FRACTIONS[1][k]) for k in ['Zn66', 'Zn67', 'Zn68', 'Zn70']}}

def get_zn_fractions_log(zn64_enrichment):
    return _zn64_fractions_interp_log(float(zn64_enrichment))

def _zn67_fractions_interp(zn67_enrichment):
    return _fractions_interp(zn67_enrichment, ZN67_ANCHOR_X, ZN67_ANCHOR_FRACTIONS, 'Zn67', ['Zn64', 'Zn66', 'Zn68', 'Zn70'], ZN67_RANGE)

def _build_enrichment_lookup(ev, grid, r, frac_fn=None, keys=None, cost_x=None, cost_y=None):
    es = sorted(set(e for e in (ev or grid) if r[0] <= e <= r[1]))
    if frac_fn is not None and keys is not None:
        return {e: {k: frac_fn(e)[k] for k in keys} for e in es}
    return {e: _interp(e, cost_x, cost_y, r) for e in es}

def build_zn64_enrichment_map(enrichment_values=None):
    return _build_enrichment_lookup(enrichment_values, ZN64_COST_ANCHOR_X, ZN64_RANGE, _zn64_fractions_interp, ['Zn66', 'Zn67', 'Zn68', 'Zn70'])

def build_zn64_enrichment_cost(enrichment_values=None):
    return _build_enrichment_lookup(enrichment_values, ZN64_COST_ANCHOR_X, ZN64_RANGE, cost_x=ZN64_COST_ANCHOR_X, cost_y=ZN64_ANCHOR_COST)

def build_zn67_enrichment_map(enrichment_values=None):
    return _build_enrichment_lookup(enrichment_values, ZN67_ANCHOR_X, ZN67_RANGE, _zn67_fractions_interp, ['Zn64', 'Zn66', 'Zn68', 'Zn70'])

def build_zn67_enrichment_cost(enrichment_values=None):
    return _build_enrichment_lookup(enrichment_values, ZN67_COST_ANCHOR_X, ZN67_RANGE, cost_x=ZN67_COST_ANCHOR_X, cost_y=ZN67_ANCHOR_COST)

def _enrichment_interp(x, xs, ys, x_range):
    """Legacy alias: piecewise linear; clamp to x_range."""
    return _interp(x, xs, ys, x_range)


class NeutronicsConfig:
    CHANNELS = [
        ("Zn63", "(n,gamma)", "Zn64"), ("Zn65", "(n,2n)", "Zn64"),
        ("Zn64", "(n,2n)", "Zn63"),   # Zn-63 production from Zn-64(n,2n); t1/2 38.47 min
        ("Zn63", "(n,2n)", "Zn62"),   # Zn-62 (t1/2 ~9.2 h) -> feeds Cu-62
        ("Zn62", "(n,2n)", "Zn61"),   # Zn-61 (t1/2 ~89 s) -> feeds Cu-61
        ("Zn64", "(n,gamma)", "Zn65"), ("Zn66", "(n,2n)", "Zn65"),
        ("Zn65", "(n,gamma)", "Zn66"), ("Zn67", "(n,2n)", "Zn66"),
        ("Zn66", "(n,gamma)", "Zn67"), ("Zn68", "(n,2n)", "Zn67"),
        ("Zn67", "(n,gamma)", "Zn68"), ("Zn69", "(n,2n)", "Zn68"),
        ("Zn68", "(n,gamma)", "Zn69"), ("Zn70", "(n,2n)", "Zn69"),
        ("Zn68", "(n,gamma)", "Zn69m"),  # Isomer: branch from Zn68(n,gamma); rate set in build_channel_rr_per_s
        ("Zn69", "(n,gamma)", "Zn70"),
        # Cu production: stable + short-lived (all incorporated in cavity; short-lived decay fast)
        ("Zn64", "(n,p)", "Cu64"), ("Zn67", "(n,p)", "Cu67"), ("Zn68", "(n,d)", "Cu67"),
        ("Zn63", "(n,p)", "Cu63"), ("Zn65", "(n,p)", "Cu65"),  # stable
        ("Zn66", "(n,p)", "Cu66"),   # Cu-66 t1/2 ~5.1 min
        ("Zn62", "(n,p)", "Cu62"),   # Cu-62 t1/2 ~9.7 min
        ("Zn61", "(n,p)", "Cu61"),   # Cu-61 t1/2 ~3.3 h
        ("Zn69", "(n,p)", "Cu69"),   # Cu-69 t1/2 ~5.1 min (β⁻)
        ("Zn70", "(n,p)", "Cu70"),   # Cu-70 t1/2 ~44.5 s (β⁻ → Zn-70)
        # Zn (n,alpha) -> Ni (stable Ni61,63,65,67; Ni64 short-lived -> Cu64)
        ("Zn64", "(n,a)", "Ni61"), ("Zn66", "(n,a)", "Ni63"), ("Zn67", "(n,a)", "Ni64"),
        ("Zn68", "(n,a)", "Ni65"), ("Zn70", "(n,a)", "Ni67"),
    ]
    # Source strength fixed at 5e13 n/s. Fusion power (neutron power) from 14.1 MeV neutron.
    SOURCE_STRENGTH = 5.0e13  # n/s
    NEUTRON_ENERGY_MEV = 14.1
    FUSION_POWER_W = SOURCE_STRENGTH * (NEUTRON_ENERGY_MEV * 1e6 * 1.602e-19)  # W (14.1 MeV neutron)


# Built from anchor-based piecewise linear interpolation
ZN64_ENRICHMENT_MAP = build_zn64_enrichment_map(_ZN64_GRID)
ZN64_ENRICHMENT_COST = build_zn64_enrichment_cost(_ZN64_GRID)
ZN67_ENRICHMENT_MAP = build_zn67_enrichment_map(ZN67_COST_ANCHOR_X)
ZN67_ENRICHMENT_COST = build_zn67_enrichment_cost()
_nat_fracs = _zn64_fractions_interp(0.4917)
NATURAL_ZN_FRACTIONS = {k: _nat_fracs[k] for k in ['Zn66', 'Zn67', 'Zn68', 'Zn70']}
CHANNELS = NeutronicsConfig.CHANNELS
FUSION_POWER_W = NeutronicsConfig.FUSION_POWER_W
SOURCE_STRENGTH = NeutronicsConfig.SOURCE_STRENGTH
# Zn68(n,gamma) populates Zn69 (ground) and Zn69m (isomer). OpenMC tally is total Zn69; fraction to isomer (literature/ENDF).
ZN69M_BRANCH_FRACTION = 0.5
NEUTRON_ENERGY_MEV = NeutronicsConfig.NEUTRON_ENERGY_MEV


def get_zn64_enrichment_cost_per_kg(zn64_enrichment):
    """Return $/kg for Zn-64 enriched material (piecewise linear 49.17%–100%, median cost anchors)."""
    return _enrichment_interp(float(zn64_enrichment), ZN64_COST_ANCHOR_X, ZN64_ANCHOR_COST, ZN64_RANGE)


def get_zn64_enrichment_cost_per_kg_contingency(zn64_enrichment, tol=0.005):
    """Return $/kg for Zn-64 contingency (natural, 71%, 99% only). No interpolation.
    If enrichment does not match one of ZN64_CONTINGENCY_ENRICHMENTS within tol, returns None."""
    e = float(zn64_enrichment)
    for anchor in ZN64_CONTINGENCY_ENRICHMENTS:
        if abs(e - anchor) <= tol:
            return ZN64_CONTINGENCY_COST_PER_KG[anchor]
    return None


def get_zn67_enrichment_cost_per_kg(zn67_enrichment):
    """Return $/kg for Zn-67 enriched material (piecewise linear 4.04%–17.7%, median cost anchors)."""
    return _enrichment_interp(float(zn67_enrichment), ZN67_COST_ANCHOR_X, ZN67_ANCHOR_COST, ZN67_RANGE)


def parse_dir_name(dir_name):
    """Parse directory name to extract simulation parameters.
    Naming: layer thicknesses + chamber (single vs dual) + enrichment.
    Single outer: Zn64 enrichment → Cu-64 production; Zn67 enrichment → Cu-67 production.
    """
    params = {'inner': 0, 'outer': 20, 'struct': 0, 'boron': 0, 'multi': 0, 'moderator': 0,
              'zn_enrichment': 0.486, 'use_zn67': False, 'zn67_enrichment_inner': None,
              'chamber': 'single_cu64'}  # single_cu64 | single_cu67 | dual
    try:
        if '_inner' in dir_name:
            params['inner'] = float(dir_name.split('_inner')[1].split('_')[0])
        if '_outer' in dir_name:
            params['outer'] = float(dir_name.split('_outer')[1].split('_')[0])
        if '_struct' in dir_name:
            params['struct'] = float(dir_name.split('_struct')[1].split('_')[0])
        if '_boron' in dir_name:
            params['boron'] = float(dir_name.split('_boron')[1].split('_')[0])
        if '_multi' in dir_name:
            params['multi'] = float(dir_name.split('_multi')[1].split('_')[0])
        if '_moderator' in dir_name:
            params['moderator'] = float(dir_name.split('_moderator')[1].split('_')[0])
        # New naming: explicit single_cu64 / single_cu67 / dual
        if '_single_cu64_' in dir_name or dir_name.startswith('single_cu64_'):
            params['chamber'] = 'single_cu64'
            if '_zn64_' in dir_name:
                zn_str = dir_name.split('_zn64_')[1].replace('%', '').split('_')[0]
                params['zn_enrichment'] = float(zn_str) / 100.0
            elif '_zn' in dir_name:
                zn_str = dir_name.split('_zn')[1].replace('%', '').split('_')[0]
                params['zn_enrichment'] = float(zn_str) / 100.0
            params['use_zn67'] = False
        elif '_single_cu67_' in dir_name or dir_name.startswith('single_cu67_'):
            params['chamber'] = 'single_cu67'
            if '_zn67_' in dir_name:
                zn_str = dir_name.split('_zn67_')[1].replace('%', '').split('_')[0]
                params['zn_enrichment'] = float(zn_str) / 100.0
            elif '_zn67' in dir_name:
                zn_str = dir_name.split('_zn67')[1].replace('%', '').split('_')[0]
                params['zn_enrichment'] = float(zn_str) / 100.0
            params['use_zn67'] = True
        elif '_dual_' in dir_name or dir_name.startswith('dual_'):
            params['chamber'] = 'dual'
            if '_zn64' in dir_name and '_inner_zn67' in dir_name:
                zn64_str = dir_name.split('_zn64')[1].replace('%', '').split('_')[0]
                params['zn_enrichment'] = float(zn64_str) / 100.0
                zn67_str = dir_name.split('_inner_zn67')[1].replace('%', '').split('_')[0]
                params['zn67_enrichment_inner'] = float(zn67_str) / 100.0
            params['use_zn67'] = False
        # Legacy naming (no chamber tag)
        elif '_zn64' in dir_name and '_inner_zn67' in dir_name:
            zn64_str = dir_name.split('_zn64')[1].replace('%', '').split('_')[0]
            params['zn_enrichment'] = float(zn64_str) / 100.0
            params['zn67_enrichment_inner'] = float(dir_name.split('_inner_zn67')[1].replace('%', '').split('_')[0]) / 100.0
            params['chamber'] = 'dual'
        elif '_zn67' in dir_name:
            zn_str = dir_name.split('_zn67')[1].replace('%', '').split('_')[0]
            params['zn_enrichment'] = float(zn_str) / 100.0
            params['use_zn67'] = True
            params['chamber'] = 'single_cu67'
        elif '_zn' in dir_name:
            zn_str = dir_name.split('_zn')[1].replace('%', '').split('_')[0]
            params['zn_enrichment'] = float(zn_str) / 100.0
            params['use_zn67'] = False
            params['chamber'] = 'single_cu64'
    except (ValueError, IndexError):
        pass
    return params
    
def compute_volumes_from_params(z_inner_thickness, z_outer_thickness, struct_thickness, 
                                 multi_thickness, moderator_thickness, target_height=100.0):
    """
    Compute material volumes analytically 
    
    Return 
    dict[int, float]
        material_id -> volume (cm³) for materials 0 (inner) and 1 (outer).
    """
    inner_radius = 5.0  # cm away from neutron source
    z_inner_radius = inner_radius + struct_thickness
    z_inner_multi_radius = z_inner_radius + z_inner_thickness
    
    if multi_thickness > 0:
        inner_moderator_radius = z_inner_multi_radius + multi_thickness
    else:
        inner_moderator_radius = z_inner_multi_radius
    
    if moderator_thickness > 0:
        z_outer_moderator_radius = inner_moderator_radius + moderator_thickness
    else:
        z_outer_moderator_radius = inner_moderator_radius
    z_outer_radius = z_outer_moderator_radius + z_outer_thickness

    # V_annulus = π * (R_outer² - R_inner²) * height
    inner_target_volume = np.pi * (z_inner_multi_radius**2 - z_inner_radius**2) * target_height
    outer_target_volume = np.pi * (z_outer_radius**2 - z_outer_moderator_radius**2) * target_height
    
    return {
        0: inner_target_volume,
        1: outer_target_volume
    }


def compute_volumes_from_dir_name(dir_name, target_height=100.0):
    """
    Parse geometry parameters from directory name and compute volumes.
    
    Parameters
    ----------
    dir_name : str
        Directory name like 'irrad_output_inner5_outer30_struct2_multi0_moderator0_zn48.6%'.
    target_height : float
        Height of cylindrical geometry (cm), default 100.
    
    Returns
    -------
    dict[int, float]
        material_id -> volume (cm³) for materials 0 (inner) and 1 (outer).
    """
    # Parse parameters from directory name (same as analyze_depletion.py lines 1318-1323)
    z_inner = float(dir_name.split("_inner")[1].split("_")[0])
    z_outer = float(dir_name.split("_outer")[1].split("_")[0])
    struct = float(dir_name.split("_struct")[1].split("_")[0])
    multi = float(dir_name.split("_multi")[1].split("_")[0])
    moderator = float(dir_name.split("_moderator")[1].split("_")[0])
    
    return compute_volumes_from_params(
        z_inner_thickness=z_inner,
        z_outer_thickness=z_outer,
        struct_thickness=struct,
        multi_thickness=multi,
        moderator_thickness=moderator,
        target_height=target_height
    )

def get_zn_fractions(zn64_enrichment):
    """Zn64+others fractions; piecewise linear (anchors: 0.4917, 0.53, 0.71, 0.9918)."""
    return _zn64_fractions_interp(float(zn64_enrichment))


def get_zn67_fractions(zn67_enrichment):
    """Zn67+others fractions; piecewise linear (anchors 4.04%, 7.3%, 10%, 14%, 17.7%)."""
    return _zn67_fractions_interp(float(zn67_enrichment))


def calculate_enriched_zn_density(zn64_enrichment):
    """Density (g/cm³) for Zn-64 enriched target: 7.14 * (M_avg / 65.38). Returns 7.14 for natural."""
    if zn64_enrichment == 0.4917:
        return 7.14
    fracs = get_zn_fractions(zn64_enrichment)
    total = sum(fracs.values())
    fracs = {k: v / total for k, v in fracs.items()}
    M_avg = sum(fracs[iso] * openmc.data.atomic_mass(iso) for iso in fracs)
    return 7.14 * (M_avg / 65.38)


def calculate_enriched_zn67_density(zn67_enrichment):
    """Density (g/cm³) for Zn-67 enriched target: 7.14 * (M_avg / 65.38)."""
    fracs = get_zn67_fractions(zn67_enrichment)
    total = sum(fracs.values())
    fracs = {k: v / total for k, v in fracs.items()}
    M_avg = sum(fracs[iso] * openmc.data.atomic_mass(iso) for iso in fracs)
    return 7.14 * (M_avg / 65.38)


def plot_zn64_enrichment(output_path=None, max_enrichment=0.999):
    """Plot Zn-64 isotope fractions and cost vs enrichment (piecewise linear). Use max_enrichment to cap x-axis (e.g. 0.999 for 99.9%; no 100%, 99.5%)."""
    import matplotlib.pyplot as plt
    x_max = min(float(max_enrichment), ZN64_RANGE[1])
    e = np.linspace(ZN64_RANGE[0], x_max, 200)
    f = _zn64_fractions_interp
    fracs_zn66 = [f(x)['Zn66'] for x in e]
    fracs_zn67 = [f(x)['Zn67'] for x in e]
    fracs_zn68 = [f(x)['Zn68'] for x in e]
    fracs_zn70 = [f(x)['Zn70'] for x in e]
    costs = [_enrichment_interp(x, ZN64_COST_ANCHOR_X, ZN64_ANCHOR_COST, ZN64_RANGE) for x in e]
    # Only show anchors <= max_enrichment (exclude 99.5%, 100%)
    anchor_x = [x for x in ZN64_ANCHOR_X if x <= x_max]
    cost_anchor_x = [x for x in ZN64_COST_ANCHOR_X if x <= x_max]
    cost_anchor_vals = [ZN64_ANCHOR_COST[i] for i in range(len(ZN64_COST_ANCHOR_X)) if ZN64_COST_ANCHOR_X[i] <= x_max]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # Zn64 fraction = enrichment (identity); then other isotopes
    ax1.plot(e * 100, e, label='Zn64', lw=2)
    ax1.plot(e * 100, fracs_zn66, label='Zn66')
    ax1.plot(e * 100, fracs_zn67, label='Zn67')
    ax1.plot(e * 100, fracs_zn68, label='Zn68')
    ax1.plot(e * 100, fracs_zn70, label='Zn70')
    if anchor_x:
        ax1.scatter([x * 100 for x in anchor_x], [f(x)['Zn64'] for x in anchor_x],
                    color='black', s=40, zorder=5, label='frac anchors')
    ax1.set_ylabel('Isotope fraction')
    ax1.set_xlabel('Zn-64 enrichment (%)')
    ax1.set_title('Zn-64 enrichment: isotope fractions (piecewise linear)')
    ax1.legend(loc='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(e * 100, costs, 'C0', lw=2, label='Cost ($/kg)')
    if cost_anchor_x and cost_anchor_vals:
        ax2.scatter([x * 100 for x in cost_anchor_x], cost_anchor_vals, color='black', s=40, zorder=5, label='cost anchors')
    ax2.set_ylabel('Cost ($/kg)')
    ax2.set_xlabel('Zn-64 enrichment (%)')
    ax2.set_title('Zn-64 enrichment cost (piecewise linear)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_zn64_enrichment_log(output_path=None, max_enrichment=0.999):
    """Plot Zn-64 isotope fractions vs enrichment for log interpolation (natural to 99.18%). Use max_enrichment to cap x-axis (e.g. 99.9%; no 100%)."""
    import matplotlib.pyplot as plt
    x_max = min(float(max_enrichment), ZN64_LOG_ANCHOR_X[1])
    e = np.linspace(ZN64_LOG_ANCHOR_X[0], x_max, 200)
    f = _zn64_fractions_interp_log
    fig, ax = plt.subplots(figsize=(8, 5))
    # Zn64 fraction = enrichment (identity); then other isotopes
    ax.plot(e * 100, e, label='Zn64', lw=2)
    ax.plot(e * 100, [f(x)['Zn66'] for x in e], label='Zn66')
    ax.plot(e * 100, [f(x)['Zn67'] for x in e], label='Zn67')
    ax.plot(e * 100, [f(x)['Zn68'] for x in e], label='Zn68')
    ax.plot(e * 100, [f(x)['Zn70'] for x in e], label='Zn70')
    anchor_x = [x for x in ZN64_LOG_ANCHOR_X if x <= x_max]
    if anchor_x:
        ax.scatter([x * 100 for x in anchor_x], [f(x)['Zn64'] for x in anchor_x],
                   color='black', s=40, zorder=5, label='anchors (natural, 99.18%)')
    ax.set_ylabel('Isotope fraction')
    ax.set_xlabel('Zn-64 enrichment (%)')
    ax.set_title('Zn-64 enrichment: isotope fractions (log interpolation)')
    ax.legend(loc='right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_zn67_enrichment(output_path=None):
    """Plot Zn-67 isotope fractions and cost vs enrichment (linear between anchors)."""
    import matplotlib.pyplot as plt
    e = np.linspace(ZN67_RANGE[0], ZN67_RANGE[1], 200)
    f = _zn67_fractions_interp
    fracs_zn64 = [f(x)['Zn64'] for x in e]
    fracs_zn66 = [f(x)['Zn66'] for x in e]
    fracs_zn67 = [f(x)['Zn67'] for x in e]
    fracs_zn68 = [f(x)['Zn68'] for x in e]
    fracs_zn70 = [f(x)['Zn70'] for x in e]
    costs = [_enrichment_interp(x, ZN67_ANCHOR_X, ZN67_ANCHOR_COST, ZN67_RANGE) for x in e]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(e * 100, fracs_zn64, label='Zn64')
    ax1.plot(e * 100, fracs_zn66, label='Zn66')
    ax1.plot(e * 100, fracs_zn67, label='Zn67', lw=2)
    ax1.plot(e * 100, fracs_zn68, label='Zn68')
    ax1.plot(e * 100, fracs_zn70, label='Zn70')
    ax1.scatter([x * 100 for x in ZN67_ANCHOR_X], [f(x)['Zn67'] for x in ZN67_ANCHOR_X],
                color='black', s=40, zorder=5, label='anchors')
    ax1.set_ylabel('Isotope fraction')
    ax1.set_xlabel('Zn-67 enrichment (%)')
    ax1.set_title('Zn-67 enrichment: isotope fractions (linear interp)')
    ax1.legend(loc='right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(e * 100, costs, 'C0', lw=2, label='Cost ($/kg)')
    ax2.scatter([x * 100 for x in ZN67_ANCHOR_X], ZN67_ANCHOR_COST, color='black', s=40, zorder=5, label='anchors')
    ax2.set_ylabel('Cost ($/kg)')
    ax2.set_xlabel('Zn-67 enrichment (%)')
    ax2.set_title('Zn-67 enrichment cost (linear interp)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_enrichment_both(output_dir='.'):
    """Plot Zn-64 and Zn-67 enrichment curves (fractions + cost) to output_dir."""
    plot_zn64_enrichment(os.path.join(output_dir, 'zn64_enrichment_interp.png'))
    plot_zn67_enrichment(os.path.join(output_dir, 'zn67_enrichment_interp.png'))


def get_initial_zn_atoms_fallback(volume_cm3, zn64_enrichment, density_g_cm3):
    """Fallback when statepoint unavailable. Uses ZN64_ENRICHMENT_MAP."""
    fracs = get_zn_fractions(zn64_enrichment)
    total = sum(fracs.values())
    fracs = {k: v / total for k, v in fracs.items()}
    avg_mass = sum(fracs[iso] * openmc.data.atomic_mass(iso) for iso in fracs)
    total_atoms = (volume_cm3 * density_g_cm3 / avg_mass) * 6.022e23
    initial_atoms = {iso: total_atoms * f for iso, f in fracs.items()}
    for nuc in ('Cu64', 'Cu67', 'Zn65', 'Zn69', 'Zn69m'):
        initial_atoms[nuc] = 0.0
    return initial_atoms


def get_initial_atoms_from_statepoint(path, material_id, volume_cm3):
    """
    Get initial atom counts from the material in summary.h5.
    Uses nuclide atom densities from the simulation (no fraction guessing).
    
    Parameters
    ----------
    path : str
        Path to statepoint file or directory containing summary.h5
    material_id : int
        Material ID (e.g., 1 for outer Zn target)
    volume_cm3 : float
        Material volume in cm³
    
    Returns
    -------
    dict
        Nuclide name -> initial atom count (float)
    """
    import os
    run_dir = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else os.path.abspath(path)
    summary_path = os.path.join(run_dir, 'summary.h5')
    if not os.path.exists(summary_path):
        return None
    try:
        summary = openmc.Summary(summary_path)
        # summary.materials is list-like (index by position); look up by material id
        mat = next((m for m in summary.materials if getattr(m, 'id', None) == material_id), None)
        if mat is None:
            return None
        nuc_densities = mat.get_nuclide_atom_densities()  # atom/b-cm
        initial_atoms = {}
        for nuc, dens_atom_b_cm in nuc_densities.items():
            # atoms = volume_cm3 * (atoms/cm³) = volume_cm3 * dens_atom_b_cm * 1e24
            initial_atoms[nuc] = float(volume_cm3 * dens_atom_b_cm * 1e24)
        # Ensure all Zn/Cu isotopes from Bateman chain are present (0 if not in material)
        for parent, _, daughter in CHANNELS:
            if parent not in initial_atoms:
                initial_atoms[parent] = 0.0
            if daughter not in initial_atoms:
                initial_atoms[daughter] = 0.0
        return initial_atoms
    except (IndexError, KeyError, AttributeError, FileNotFoundError) as e:
        print(f"  Warning: Could not get initial atoms for material {material_id}: {e}")
        return None


def get_material_density_from_statepoint(path, material_id):
    """
    Extract material density (g/cm3) from summary.h5.
    Uses OpenMC's get_mass_density() which returns g/cm3 regardless of storage units.
    
    Parameters
    ----------
    path : str
        Path to statepoint file (e.g. statepoint.100.h5) or directory containing summary.h5
    material_id : int
        Material ID (e.g., 1 for outer Zn target)
    
    Returns
    -------
    float or None
        Density in g/cm3, or None if not found
    """
    import os
    run_dir = os.path.dirname(os.path.abspath(path)) if os.path.isfile(path) else os.path.abspath(path)
    summary_path = os.path.join(run_dir, 'summary.h5')
    if not os.path.exists(summary_path):
        return None
    try:
        summary = openmc.Summary(summary_path)
        # summary.materials is list-like (index by position); look up by material id
        mat = next((m for m in summary.materials if getattr(m, 'id', None) == material_id), None)
        if mat is None:
            return None
        if hasattr(mat, 'get_mass_density'):
            return float(mat.get_mass_density())
        # Fallback: manual conversion from atom/b-cm
        d = mat.density
        d_val = float(d[0]) if isinstance(d, (list, tuple)) else float(d)
        units = getattr(mat, 'density_units', 'g/cm3')
        if isinstance(units, (list, tuple)):
            units = units[0] if units else 'g/cm3'
        if units in ('g/cm3', 'g/cc'):
            return d_val
        if units == 'kg/m3':
            return d_val / 1000.0
        if units in ('atom/b-cm', 'atom/cm3'):
            from scipy.constants import Avogadro
            nuclides = getattr(mat, 'nuclides', []) or []
            if nuclides:
                nuc_tuples = [(n.name, n.percent) if hasattr(n, 'name') else (n[0], n[1]) for n in nuclides]
                M_avg = sum(frac * openmc.data.atomic_mass(nuc) for nuc, frac in nuc_tuples)
            else:
                M_avg = 65.38
            N_per_cm3 = d_val * 1e24 if units == 'atom/b-cm' else d_val
            return N_per_cm3 * M_avg / Avogadro
        return d_val
    except (KeyError, IndexError, AttributeError, FileNotFoundError) as e:
        print(f"  Warning: Could not get density for material {material_id}: {e}")
        return None


def channel_rate_per_s(sp, tally_name, cell_id, score, parent_nuclide, source_strength):
    """
    Get reaction rate [atoms/s] for one (cell, score, nuclide).
    
    cell_id is the target cell ID in the geometry:
      - 0: inner_target (material_id=0)
      - 1: outer_target (material_id=1) - this is cell_id=6 in geometry but we want material_id=1
    
    The function finds the correct bin by looking at the CellFilter.
    """
    
    try:
        t = sp.get_tally(name=tally_name)
    except LookupError as e:
        print(f"Tally '{tally_name}' not found: {e}")
        return 0.0
    if parent_nuclide not in t.nuclides:
        return 0.0
    if score not in t.scores:
        return 0.0
    
    tally_mean = np.asarray(t.mean)
    n_cell_bins = tally_mean.shape[0]
    
    bin_idx = None
    all_cells = sp.summary.geometry.get_all_cells() if (sp.summary and sp.summary.geometry) else {}
    for filt in t.filters:
        if isinstance(filt, openmc.CellFilter):
            cell_bins = filt.bins
            for i, cell_bin in enumerate(cell_bins):
                # cell_bin can be cell id (int) or Cell object; match by id first (works without summary)
                if isinstance(cell_bin, int):
                    if cell_bin == cell_id:
                        bin_idx = i
                        break
                    cell_obj = all_cells.get(cell_bin)
                else:
                    cell_obj = cell_bin
                if cell_obj is None:
                    continue
                obj_id = getattr(cell_obj, 'id', None)
                name = (getattr(cell_obj, 'name', '') or '').lower()
                if obj_id is not None and cell_id == obj_id:
                    bin_idx = i
                    break
                if cell_id == 0 and ('inner' in name or 'zn_sphere' in name):
                    bin_idx = i
                    break
                if cell_id == 1 and ('outer' in name or 'zn_sphere' in name):
                    bin_idx = i
                    break
            if bin_idx is not None:
                break
    # Fallback only when single bin or known 0/1 convention
    if bin_idx is None:
        if n_cell_bins == 1:
            bin_idx = 0
        elif cell_id == 1 and n_cell_bins > 0:
            bin_idx = 0
        else:
            bin_idx = min(cell_id, n_cell_bins - 1) if n_cell_bins > 0 else 0
    
    nuc_idx = t.get_nuclide_index(parent_nuclide)
    score_idx = t.get_score_index(score)
    
    try:
        if tally_mean.ndim == 2:
            # Single filter bin: shape (n_nuclides, n_scores) — e.g. sphere geometry
            val = float(tally_mean[nuc_idx, score_idx])
        else:
            val = float(tally_mean[bin_idx, nuc_idx, score_idx])
    except IndexError:
        print(f"  Warning: index out of range for tally '{tally_name}' (shape={tally_mean.shape}, bin={bin_idx}, nuc={nuc_idx}, score={score_idx})")
        val = 0.0
    
    return val * source_strength

def build_channel_rr_per_s(sp, cell_id, source_strength=None):
    if source_strength is None:
        source_strength = SOURCE_STRENGTH
    ch = {}

    # Zn chain (each is a separate channel)
    ch["Zn63 (n,gamma) Zn64"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn63", source_strength)
    ch["Zn64 (n,2n) Zn63"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn64", source_strength)
    ch["Zn63 (n,2n) Zn62"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn63", source_strength)
    ch["Zn62 (n,2n) Zn61"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn62", source_strength)
    ch["Zn65 (n,2n) Zn64"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn65", source_strength)

    ch["Zn64 (n,gamma) Zn65"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn64", source_strength)
    ch["Zn66 (n,2n) Zn65"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn66", source_strength)

    ch["Zn65 (n,gamma) Zn66"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn65", source_strength)
    ch["Zn67 (n,2n) Zn66"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn67", source_strength)

    ch["Zn66 (n,gamma) Zn67"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn66", source_strength)
    ch["Zn68 (n,2n) Zn67"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn68", source_strength)

    ch["Zn67 (n,gamma) Zn68"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn67", source_strength)
    ch["Zn69 (n,2n) Zn68"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn69", source_strength)

    R_zn68_ng = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn68", source_strength)
    R_zn68_ng = float(R_zn68_ng) if R_zn68_ng is not None else 0.0
    # Zn68(n,gamma) populates both Zn69 and Zn69m; OpenMC tally is total Zn69. Use branch fraction for Zn69m.
    ch["Zn68 (n,gamma) Zn69m"] = ZN69M_BRANCH_FRACTION * R_zn68_ng
    ch["Zn68 (n,gamma) Zn69"] = (1.0 - ZN69M_BRANCH_FRACTION) * R_zn68_ng
    ch["Zn70 (n,2n) Zn69"]    = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,2n)",    "Zn70", source_strength)

    ch["Zn69 (n,gamma) Zn70"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,gamma)", "Zn69", source_strength)

    # Zn (n,alpha) -> Ni production (tallied in Zn_rxn_rates when (n,a) score present)
    for _parent, _daughter in [('Zn64', 'Ni61'), ('Zn66', 'Ni63'), ('Zn67', 'Ni64'), ('Zn68', 'Ni65'), ('Zn70', 'Ni67')]:
        ch[f"{_parent} (n,a) {_daughter}"] = channel_rate_per_s(sp, "Zn_rxn_rates", cell_id, "(n,a)", _parent, source_strength)

    # Cu production (radioisotopes and stable; short-lived Cu61, Cu62, Cu66 incorporated in cavity, decay fast)
    ch["Zn64 (n,p) Cu64"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn64", source_strength)
    ch["Zn67 (n,p) Cu67"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn67", source_strength)
    ch["Zn68 (n,d) Cu67"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,d)", "Zn68", source_strength)
    ch["Zn63 (n,p) Cu63"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn63", source_strength)
    ch["Zn65 (n,p) Cu65"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn65", source_strength)
    ch["Zn66 (n,p) Cu66"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn66", source_strength)
    ch["Zn62 (n,p) Cu62"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn62", source_strength)
    # Zn61 (n,p) Cu61 omitted: Zn61 not in tally (not in typical nuclear data); Bateman uses CHANNELS.get(key, 0)
    ch["Zn69 (n,p) Cu69"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn69", source_strength)
    ch["Zn70 (n,p) Cu70"] = channel_rate_per_s(sp, "Cu_Production_rxn_rates", cell_id, "(n,p)", "Zn70", source_strength)
    n_zero = sum(1 for v in ch.values() if (v is None or (hasattr(v, '__float__') and float(v) == 0)))
    n_nonzero = len(ch) - n_zero
    print(f"  [build_channel_rr_per_s] cell_id={cell_id}: {n_nonzero} non-zero, {n_zero} zero rates")
    if n_nonzero > 0:
        for k, v in ch.items():
            vf = float(v) if v is not None else 0.0
            if vf > 0:
                print(f"    {k}: {vf:.4e}")
    return ch

def _cell_filter_bin_index(sp, tally, cell_id):
    """Return the cell filter bin index. cell_id: 0=inner, 1=outer (fusion), or actual cell id (e.g. 2 for sphere zn_sphere)."""
    tally_mean = np.asarray(tally.mean)
    n_cell_bins = tally_mean.shape[0]
    bin_idx = None
    for filt in tally.filters:
        if isinstance(filt, openmc.CellFilter):
            cell_bins = filt.bins
            for i, cell_bin in enumerate(cell_bins):
                cell_obj = sp.summary.geometry.get_all_cells().get(cell_bin)
                if cell_obj is not None:
                    obj_id = getattr(cell_obj, 'id', None)
                    if obj_id is not None and cell_id == obj_id:
                        bin_idx = i
                        break
                    if cell_id == 0 and 'inner' in getattr(cell_obj, 'name', '').lower():
                        bin_idx = i
                        break
                    if cell_id == 1 and 'outer' in getattr(cell_obj, 'name', '').lower():
                        bin_idx = i
                        break
                    if cell_id == 0 and 'zn_sphere' in getattr(cell_obj, 'name', '').lower():
                        bin_idx = i
                        break
                    if cell_id == 1 and 'zn_sphere' in getattr(cell_obj, 'name', '').lower():
                        bin_idx = i
                        break
            break
    if bin_idx is None:
        if n_cell_bins == 1:
            bin_idx = 0
        else:
            bin_idx = min(cell_id, n_cell_bins - 1)
    return bin_idx

def get_volumetric_heating_w_cm3(sp, cell_id, source_strength, volume_cm3):
    """
    Get volumetric heating [W/cm³] for one cell from the 'volumetric_heating' tally.
    OpenMC heating-local is eV per source particle (per cell); convert to W/cm³.
    """
    try:
        t = sp.get_tally(name='volumetric_heating')
    except LookupError:
        return 0.0
    mean = np.asarray(t.mean).flatten()
    bin_idx = _cell_filter_bin_index(sp, t, cell_id)
    if bin_idx >= len(mean):
        return 0.0
    ev_per_s = float(mean[bin_idx]) * source_strength
    power_W = ev_per_s * 1.602e-19
    if volume_cm3 <= 0:
        return 0.0
    return power_W / volume_cm3

def compute_outer_surface_area_cm2_from_params(z_inner_thickness, z_outer_thickness, struct_thickness,
                                                multi_thickness, moderator_thickness, target_height=100.0):
    """Lateral + end annulus surface area (cm²) for the outer Zn cell (for convection)."""
    inner_radius = 5.0
    z_inner_radius = inner_radius + struct_thickness
    z_inner_multi_radius = z_inner_radius + z_inner_thickness
    if multi_thickness > 0:
        inner_moderator_radius = z_inner_multi_radius + multi_thickness
    else:
        inner_moderator_radius = z_inner_multi_radius
    if moderator_thickness > 0:
        z_outer_moderator_radius = inner_moderator_radius + moderator_thickness
    else:
        z_outer_moderator_radius = inner_moderator_radius
    z_outer_radius = z_outer_moderator_radius + z_outer_thickness
    lateral = 2.0 * np.pi * z_outer_radius * target_height
    end_annulus = 2.0 * np.pi * (z_outer_radius**2 - z_outer_moderator_radius**2)
    return lateral + end_annulus


def compute_inner_surface_area_cm2_from_params(z_inner_thickness, z_outer_thickness, struct_thickness,
                                                multi_thickness, moderator_thickness, target_height=100.0):
    """Lateral + end annulus surface area (cm²) for the inner Zn cell (for convection)."""
    inner_radius = 5.0
    z_inner_radius = inner_radius + struct_thickness
    z_inner_multi_radius = z_inner_radius + z_inner_thickness
    lateral = 2.0 * np.pi * z_inner_multi_radius * target_height
    end_annulus = 2.0 * np.pi * (z_inner_multi_radius**2 - z_inner_radius**2)
    return lateral + end_annulus


def _half_life_seconds(nuclide: str):
    try:
        hl = openmc.data.half_life(nuclide)
        if hl is None or not np.isfinite(hl) or hl <= 0:
            if nuclide == 'Zn63':
                return 38.47 * 60.0  # t1/2 38.47 min (β+/EC → Cu-63)
            if nuclide == 'Zn69m':
                return 13.756 * 3600.0  # t1/2 13.756 h (IT → 0.439 MeV γ)
            if nuclide == 'Ni64':
                return 0.00036  # t1/2 ~0.36 ms (β- → Cu-64)
            if nuclide == 'Cu61':
                return 3.343 * 3600.0  # t1/2 ~3.34 h
            if nuclide == 'Cu62':
                return 9.672 * 60.0  # t1/2 ~9.67 min
            if nuclide == 'Cu66':
                return 5.12 * 60.0  # t1/2 ~5.1 min
            if nuclide == 'Cu69':
                return 5.1 * 60.0   # t1/2 ~5.1 min (β⁻)
            if nuclide == 'Cu70':
                return 44.5        # t1/2 ~44.5 s (β⁻ → Zn-70)
            if nuclide == 'Zn62':
                return 9.26 * 3600.0  # t1/2 ~9.26 h (β+ → Cu-62)
            if nuclide == 'Zn61':
                return 89.1  # t1/2 ~89 s (β+ → Cu-61)
            return None
        return float(hl)
    except Exception:
        if nuclide == 'Zn63':
            return 38.47 * 60.0
        if nuclide == 'Zn69m':
            return 13.756 * 3600.0
        if nuclide == 'Ni64':
            return 0.00036
        if nuclide == 'Cu61':
            return 3.343 * 3600.0
        if nuclide == 'Cu62':
            return 9.672 * 60.0
        if nuclide == 'Cu66':
            return 5.12 * 60.0
        if nuclide == 'Cu69':
            return 5.1 * 60.0
        if nuclide == 'Cu70':
            return 44.5
        if nuclide == 'Zn62':
            return 9.26 * 3600.0
        if nuclide == 'Zn61':
            return 89.1
        return None


def get_decay_constant(nuclide: str) -> float:
    """Decay constant (1/s) from half-life lookup with a few manual fallbacks."""
    hl = _half_life_seconds(nuclide)
    if hl is not None and hl > 0:
        return float(np.log(2.0) / hl)
    # Fallbacks for nuclides not in OpenMC decay data
    if nuclide == 'Pb209':
        return float(np.log(2.0) / (3.25 * 3600.0))  # t1/2 3.25 h
    if nuclide == 'Pb203':
        return float(np.log(2.0) / (52.0 * 3600.0))  # t1/2 ~52 h
    if nuclide == 'Pb205':
        return float(np.log(2.0) / (1.73e7 * 365.25 * 86400.0))  # t1/2 ~1.73e7 y (EC)
    if nuclide == 'Bi210':
        return float(np.log(2.0) / (5.012 * 86400.0))  # t1/2 5.012 d
    if nuclide == 'Po210':
        return float(np.log(2.0) / (138.376 * 86400.0))  # t1/2 138.376 d
    if nuclide == 'Zn63':
        # Mirror _half_life_seconds Zn63 fallback explicitly (38.47 min)
        return float(np.log(2.0) / (38.47 * 60.0))
    if nuclide == 'Zn69m':
        return float(np.log(2.0) / (13.756 * 3600.0))  # t1/2 13.756 h
    if nuclide == 'Cu61':
        return float(np.log(2.0) / (3.343 * 3600.0))
    if nuclide == 'Cu62':
        return float(np.log(2.0) / (9.672 * 60.0))
    if nuclide == 'Cu66':
        return float(np.log(2.0) / (5.12 * 60.0))
    if nuclide == 'Cu69':
        return float(np.log(2.0) / (5.1 * 60.0))
    if nuclide == 'Cu70':
        return float(np.log(2.0) / 44.5)
    if nuclide == 'Zn62':
        return float(np.log(2.0) / (9.26 * 3600.0))
    if nuclide == 'Zn61':
        return float(np.log(2.0) / 89.1)
    return 0.0


def activity_Bq_after_irrad_cooldown(production_rate_per_s: float, nuclide: str, irrad_s: float, cooldown_s: float) -> float:
    """Activity [Bq] at end of cooldown for one irrad + cooldown. Any tallied radioisotope (Pb209, Pb203, etc.).
    N_eoi = R*(1 - exp(-lam*irrad_s))/lam; activity = N_eoi * exp(-lam*cooldown_s) * lam."""
    if production_rate_per_s <= 0:
        return 0.0
    lam = get_decay_constant(nuclide)
    if lam <= 0:
        return 0.0
    n_eoi = production_rate_per_s * (1.0 - np.exp(-lam * irrad_s)) / lam
    return float(n_eoi * np.exp(-lam * cooldown_s) * lam)


def activity_Bq_after_cyclic(
    production_rate_per_s: float,
    nuclide: str,
    n_cycles: int,
    irrad_s_per_cycle: float,
    cooldown_s_between: float,
) -> float:
    """Activity [Bq] at EOI after n_cycles of (irrad + cooldown). For any tallied radioisotope (Pb209, Pb203, Bi210, etc.).
    Each cycle: N = N*exp(-lam*irrad_s) + R*(1-exp(-lam*irrad_s))/lam; then N = N*exp(-lam*cooldown_s). Return lam*N."""
    if production_rate_per_s <= 0 or n_cycles <= 0:
        return 0.0
    lam = get_decay_constant(nuclide)
    if lam <= 0:
        return 0.0
    R = production_rate_per_s
    n_atoms = 0.0
    for _ in range(n_cycles):
        n_atoms = n_atoms * np.exp(-lam * irrad_s_per_cycle) + R * (1.0 - np.exp(-lam * irrad_s_per_cycle)) / lam
        n_atoms = n_atoms * np.exp(-lam * cooldown_s_between)
    return float(lam * n_atoms)


def apply_single_decay_step(atoms_by_nuclide: dict, dt_s: float) -> dict:
    """Applies N <- N * exp(-lambda*dt) for all nuclides with finite half-life."""
    out = dict(atoms_by_nuclide)
    for nuc, N in out.items():
        hl = _half_life_seconds(nuc)
        if hl is None:
            continue
        lam = np.log(2.0) / hl
        out[nuc] = float(N) * np.exp(-lam * dt_s)
    return out


def evolve_bateman_irradiation(initial_atoms: dict, channel_rr_per_s: dict, dt_s: float) -> dict:
    """
    Proper Bateman equation for production + decay during irradiation.
    
    For each daughter isotope being produced at rate R and decaying with λ:
        N(t) = N₀ × e^(-λt) + (R/λ) × (1 - e^(-λt))
    
    This correctly reaches saturation N_sat = R/λ for t >> half-life.
    
    For parent isotopes being consumed (but not decaying significantly):
        N_parent(t) = N₀ - R × t  (clamped to not go negative)
    
    Parameters:
    -----------
    initial_atoms : dict
        Initial atom counts for each nuclide
    channel_rr_per_s : dict
        Reaction rates per second for each channel
    dt_s : float
        Irradiation time in seconds
    
    Returns:
    --------
    dict : Final atom counts after irradiation (with proper decay accounting)
    """
    N = {k: float(v) for k, v in initial_atoms.items()}
    
    # Track production rates for each daughter
    daughter_production_rates = {}
    # Total requested consumption per parent (sum of R*dt_s over all channels using that parent)
    parent_requested = {}
    # List of (parent, daughter, R) for channels with R > 0
    channel_list = []

    # First pass: collect requested consumption per parent and (parent, daughter, R) for each channel
    for parent, rxn, daughter in CHANNELS:
        key = f"{parent} {rxn} {daughter}"
        R = channel_rr_per_s.get(key, 0.0)
        if R is None:
            continue
        R = float(np.asarray(R).flat[0])
        if R <= 0:
            continue
        requested = R * dt_s
        parent_requested[parent] = parent_requested.get(parent, 0.0) + requested
        channel_list.append((parent, daughter, R))

    # Scale factor per parent so total consumption does not exceed available (same parent can feed multiple channels, e.g. Zn64 -> Zn65 and Zn64 -> Cu64)
    parent_scale = {}
    for parent, total in parent_requested.items():
        avail = N.get(parent, 0.0)
        if total <= 0 or avail <= 0:
            parent_scale[parent] = 0.0
        elif total > avail:
            parent_scale[parent] = avail / total
        else:
            parent_scale[parent] = 1.0

    # Apply scaled rates to daughter production and consume parents
    for parent, daughter, R in channel_list:
        scale = parent_scale.get(parent, 0.0)
        R_effective = R * scale
        if daughter not in daughter_production_rates:
            daughter_production_rates[daughter] = 0.0
        daughter_production_rates[daughter] += R_effective

    for parent, total in parent_requested.items():
        avail = N.get(parent, 0.0)
        scale = parent_scale.get(parent, 0.0)
        consumed = min(avail, total * scale) if scale > 0 else 0.0
        N[parent] = avail - consumed

    # Second pass: Apply Bateman equation for daughters (production + decay)
    for daughter, R_total in daughter_production_rates.items():
        if R_total <= 0:
            continue
        
        N0 = N.get(daughter, 0.0)
        hl = _half_life_seconds(daughter)
        
        if hl is not None and hl > 0:
            lam = np.log(2.0) / hl
            exp_term = np.exp(-lam * dt_s)
            
            # Bateman equation: N(t) = N₀ × e^(-λt) + (R/λ) × (1 - e^(-λt))
            N_final = N0 * exp_term + (R_total / lam) * (1.0 - exp_term)
        else:
            # Stable isotope: just add production linearly
            N_final = N0 + R_total * dt_s
        
        N[daughter] = N_final
    
    # Third pass: Decay all nuclides that weren't produced (just decaying)
    for nuc in list(N.keys()):
        if nuc in daughter_production_rates:
            continue  # Already handled with Bateman
        
        hl = _half_life_seconds(nuc)
        if hl is not None and hl > 0:
            lam = np.log(2.0) / hl
            N[nuc] = float(N[nuc]) * np.exp(-lam * dt_s)
    
    return N


def scale_channel_rr_by_parents(channel_rr_per_s: dict, current_atoms: dict, initial_atoms: dict) -> dict:
    """Scale reaction rates by current/initial parent atom ratio for depletion.
    Returns a new dict of channel -> R_scaled (atoms/s). Channels whose parent is not in initial_atoms are unchanged."""
    out = {}
    for key, R in channel_rr_per_s.items():
        R = float(np.asarray(R).flat[0]) if R is not None else 0.0
        if R <= 0:
            out[key] = 0.0
            continue
        parts = key.split()
        if len(parts) < 3:
            out[key] = R
            continue
        parent = parts[0]
        n_init = float(initial_atoms.get(parent, 0) or 0)
        n_curr = float(current_atoms.get(parent, 0) or 0)
        if n_init <= 0:
            out[key] = 0.0
        else:
            out[key] = R * (n_curr / n_init)
    return out


def evolve_bateman_irradiation_with_history(
    initial_atoms: dict,
    channel_rr_per_s: dict,
    total_time_s: float,
    n_steps: int = None,
) -> list:
    """
    Run Bateman depletion in timesteps with parent depletion (RR scaled by current/initial parent).
    Returns list of (time_s, atoms_dict) at each step so masses and activities can be computed per timestep.
    Uses all CHANNELS; reaction rates are scaled each step by current parent inventory.
    """
    if n_steps is None or n_steps < 1:
        n_steps = max(10, int(total_time_s / 3600))  # at least 10 steps or ~hourly
    dt_s = total_time_s / n_steps
    history = [(0.0, {k: float(v) for k, v in initial_atoms.items()})]
    atoms = {k: float(v) for k, v in initial_atoms.items()}
    for i in range(n_steps):
        rr_scaled = scale_channel_rr_by_parents(channel_rr_per_s, atoms, initial_atoms)
        atoms = evolve_bateman_irradiation(atoms, rr_scaled, dt_s)
        history.append(((i + 1) * dt_s, dict(atoms)))
    return history


# Natural Zn isotopic fractions (for production scaling: x/zn_x_nat)
ZN64_X_NAT = 0.4917
ZN67_X_NAT = NATURAL_ZN_FRACTIONS.get('Zn67', 0.041)
ZN68_X_NAT = NATURAL_ZN_FRACTIONS.get('Zn68', 0.188)

SECONDS_PER_YEAR = 365.25 * 24 * 3600
N_A = 6.02214076e23
A_CU64_G_MOL = 63.930
A_CU67_G_MOL = 66.928
# Decay constants (1/s) for average production rate over finite irradiation
T_HALF_CU64_S = 12.701 * 3600
T_HALF_CU67_S = 61.83 * 3600
LAMBDA_CU64_S = np.log(2) / T_HALF_CU64_S
LAMBDA_CU67_S = np.log(2) / T_HALF_CU67_S


MCI_TO_BQ = 3.7e7  # 1 mCi = 3.7e7 Bq
HOURS_PER_YEAR = 24 * 365  # for production scale-up: mCi_yr = mCi_run * (HOURS_PER_YEAR / irrad_hours)

# Economic parameters for NPV. r=0.1 (10%) used for present value; simple_analyze passes T=8, FLARE, OPEX for 8-year model.
NPV_DISCOUNT_RATE = 0.1   # discount rate for NPV (r = 10%)
NPV_T_YEARS = 20
NPV_CAPEX_USD = 2.0e7
NPV_OPEX_FIXED_USD_PER_YR = 2.0e5
NPV_RELOAD_FRACTION_PER_YEAR = 0.0


def annuity_factor(rate, n_years):
    """Present value annuity factor: sum of 1/(1+r)^t for t = 1..n.
    Used for NPV of annual net cash flows."""
    if rate == 0:
        return float(n_years)
    return (1.0 - (1.0 + rate)**(-n_years)) / rate


def specific_activity_ci_per_g(isotope):
    """Specific activity in Ci/g (physical constant). isotope is '64' or '67'."""
    if isotope == "64":
        return N_A * np.log(2) / (A_CU64_G_MOL * T_HALF_CU64_S * 3.7e10)
    return N_A * np.log(2) / (A_CU67_G_MOL * T_HALF_CU67_S * 3.7e10)


def print_specific_activities():
    """Print Cu-64 and Cu-67 specific activities (Ci/g and Bq/g). Also written to cu_summary CSV columns."""
    ci_per_g_64 = specific_activity_ci_per_g("64")
    ci_per_g_67 = specific_activity_ci_per_g("67")
    bq_per_g_64 = ci_per_g_64 * 3.7e10
    bq_per_g_67 = ci_per_g_67 * 3.7e10
    mci_per_g_64 = ci_per_g_64 * 1000.0
    mci_per_g_67 = ci_per_g_67 * 1000.0
    print("  Specific activities (physical constants):")
    print(f"    Cu-64: {ci_per_g_64:.6g} Ci/g  ({bq_per_g_64:.4e} Bq/g, {mci_per_g_64:.2f} mCi/g)")
    print(f"    Cu-67: {ci_per_g_67:.6g} Ci/g  ({bq_per_g_67:.4e} Bq/g, {mci_per_g_67:.2f} mCi/g)")
    print("  (These values are also in cu_summary CSV columns cu64_specific_activity_Ci_per_g, cu67_specific_activity_Ci_per_g.)")


def npv_from_cu_summary_row(
    row,
    price_cu64_usd_per_g,
    price_cu67_usd_per_g,
    sell_fraction=1.0,
    cap_usd_per_yr=None,
    purity_cap_64=False,
    r=NPV_DISCOUNT_RATE,
    T_years=NPV_T_YEARS,
    capex_usd=NPV_CAPEX_USD,
    opex_fixed_usd_per_yr=NPV_OPEX_FIXED_USD_PER_YR,
    reload_fraction_per_year=NPV_RELOAD_FRACTION_PER_YEAR,
):
    """
    NPV for one cu_summary row using production (g/yr), loading cost, and economic params.
    Row must have: zn_feedstock_cost, cu64_g_yr, cu67_g_yr, use_zn67; optionally cu64_radionuclide_purity.
    Returns NPV in USD.
    """
    is_cu67 = bool(row.get("use_zn67", False))
    prod_g_yr = float(row["cu67_g_yr"] if is_cu67 else row["cu64_g_yr"])
    price = price_cu67_usd_per_g if is_cu67 else price_cu64_usd_per_g
    purity = float(row.get("cu64_radionuclide_purity", 0) or 0) if not is_cu67 else 1.0
    loading = float(row.get("zn_feedstock_cost", 0) or 0)
    if np.isnan(loading):
        return np.nan
    rev = sell_fraction * prod_g_yr * price
    if not is_cu67 and purity_cap_64:
        rev = rev if purity >= 0.999 else 0.0
    if cap_usd_per_yr is not None:
        rev = min(rev, float(cap_usd_per_yr))
    af = annuity_factor(r, T_years)
    annual_net = rev - opex_fixed_usd_per_yr - reload_fraction_per_year * loading
    return -capex_usd - loading + af * annual_net


def find_cu_summary_csv(analyze_dir):
    """Return path to a cu_summary CSV under analyze_dir, or None if not found.
    Prefers analyze/simple/outer (then complex/outer) to match statepoints layout."""
    import glob
    import os
    candidates = [
        os.path.join(analyze_dir, 'simple', 'outer', 'cu_summary_all.csv'),
        os.path.join(analyze_dir, 'complex', 'outer', 'cu_summary_all.csv'),
        os.path.join(analyze_dir, 'cu_summary_all.csv'),
        os.path.join(analyze_dir, 'outer', 'cu_summary_all.csv'),
        os.path.join(analyze_dir, 'cu_summary.csv'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    matches = glob.glob(os.path.join(analyze_dir, '**', 'cu_summary*.csv'), recursive=True)
    return matches[0] if matches else None


def load_run_data_from_cu_summary(csv_path, irrad_hours=8760, cooldown_days=0, tol=0.01):
    """
    Load full run data from cu_summary for data-driven NPV.
    Returns DataFrame with one row per (geometry, enrichment) case.
    Call with irrad_hours from run_config.NPV_IRRAD_HOURS (e.g. 1 or 8).

    g/yr = (mCi at EOI from CSV) * (HOURS_PER_YEAR / irrad_hours) / (mCi/g).
    No depletion or cooldown correction — simple scale-up from run length to annual.
    """
    import os
    import pandas as pd
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    irrad_lo = irrad_hours * (1 - tol)
    irrad_hi = irrad_hours * (1 + tol)
    mask_irrad = (df['irrad_hours'] >= irrad_lo) & (df['irrad_hours'] <= irrad_hi)
    mask_cool = np.isclose(df['cooldown_days'].astype(float), float(cooldown_days), atol=0.01)
    mask = mask_irrad & mask_cool
    if mask.sum() == 0:
        max_irrad = df['irrad_hours'].max()
        mask_irrad = np.abs(df['irrad_hours'] - max_irrad) < 0.1
        mask = mask_irrad & mask_cool
    if mask.sum() == 0:
        cool_vals = df['cooldown_days'].dropna().unique()
        closest_cool = min(cool_vals, key=lambda c: abs(float(c) - float(cooldown_days)))
        mask_cool = np.isclose(df['cooldown_days'].astype(float), float(closest_cool), atol=0.01)
        mask = mask_irrad & mask_cool
    sub = df.loc[mask].copy()
    if sub.empty:
        return None
    # mCi at EOI: from CSV column or Bq / MCI_TO_BQ
    if 'cu64_mCi' in sub.columns:
        cu64_mCi = sub['cu64_mCi'].fillna(0).values.astype(float)
    else:
        cu64_mCi = sub['cu64_Bq'].fillna(0).values.astype(float) / MCI_TO_BQ
    if 'cu67_mCi' in sub.columns:
        cu67_mCi = sub['cu67_mCi'].fillna(0).values.astype(float)
    else:
        cu67_mCi = sub['cu67_Bq'].fillna(0).values.astype(float) / MCI_TO_BQ
    irrad_h = np.maximum(sub['irrad_hours'].values.astype(float), 1e-6)
    # mCi/yr = mCi_run * (HOURS_PER_YEAR / irrad_hours); g/yr = mCi_yr / (mCi/g)
    mCi_per_g_64 = 1000.0 * specific_activity_ci_per_g("64")
    mCi_per_g_67 = 1000.0 * specific_activity_ci_per_g("67")
    mCi_yr_64 = cu64_mCi * (HOURS_PER_YEAR / irrad_h)
    mCi_yr_67 = cu67_mCi * (HOURS_PER_YEAR / irrad_h)
    sub['cu64_g_yr'] = (mCi_yr_64 / mCi_per_g_64).astype(float)
    sub['cu67_g_yr'] = (mCi_yr_67 / mCi_per_g_67).astype(float)
    return sub


def load_cu64_purity_lookup_from_cu_summary(csv_path, irrad_hours=8760, cooldown_days=0, tol=0.01):
    """
    Load Cu-64 radionuclide purity vs Zn-64 enrichment from cu_summary CSV.
    Uses cu64_radionuclide_purity at given irrad_hours and cooldown_days (EOI).

    Returns
    -------
    dict
        zn64_enrichment -> purity (float 0-1). Used to filter sellable production (purity >= 0.999).
    """
    import pandas as pd
    import os
    if not os.path.isfile(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if 'cu64_radionuclide_purity' not in df.columns:
        return {}
    irrad_lo, irrad_hi = irrad_hours * (1 - tol), irrad_hours * (1 + tol)
    mask_irrad = (df['irrad_hours'] >= irrad_lo) & (df['irrad_hours'] <= irrad_hi)
    mask_cool = np.isclose(df['cooldown_days'].astype(float), float(cooldown_days), atol=0.01)
    mask = mask_irrad & mask_cool
    if mask.sum() == 0:
        max_irrad = df['irrad_hours'].max()
        mask_irrad = np.abs(df['irrad_hours'] - max_irrad) < 0.1
        mask = mask_irrad & mask_cool
    sub = df.loc[mask]
    if sub.empty:
        return {}
    lookup = {}
    for _, row in sub.iterrows():
        e = float(row.get('zn64_enrichment', 0.4917))
        p = float(row.get('cu64_radionuclide_purity', 0) or 0)
        lookup[e] = p
    return lookup

