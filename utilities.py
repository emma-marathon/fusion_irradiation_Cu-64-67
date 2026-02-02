'''
Helper functions for neutronics analysis:
- Reaction rate extraction from tallies
- Bateman equation for isotope evolution (production + decay)
- Geometry volume calculations
'''
import numpy as np
import openmc

# -------------------------------
# Zn-64 enrichment map (single source of truth for fusion_irradiation & analysis)
# -------------------------------
ZN64_ENRICHMENT_MAP = {
    0.50: {'Zn66': 0.277, 'Zn67': 0.040, 'Zn68': 0.177, 'Zn70': 0.002},
    0.53: {'Zn66': 0.274, 'Zn67': 0.0398, 'Zn68': 0.155, 'Zn70': 0.001},
    0.60: {'Zn66': 0.162, 'Zn67': 0.0277, 'Zn68': 0.1038, 'Zn70': 0.0008},
    0.70: {'Zn66': 0.047, 'Zn67': 0.016, 'Zn68': 0.03, 'Zn70': 0.0007},
    0.80: {'Zn66': 0.012, 'Zn67': 0.0043, 'Zn68': 0.014, 'Zn70': 0.0006},
    0.90: {'Zn66': 0.007, 'Zn67': 0.0023, 'Zn68': 0.0091, 'Zn70': 0.0005},
    0.99: {'Zn66': 0.004, 'Zn67': 0.0011, 'Zn68': 0.0031, 'Zn70': 0.0004},
}
NATURAL_ZN_FRACTIONS = {'Zn66': 0.279, 'Zn67': 0.041, 'Zn68': 0.188, 'Zn70': 0.0062}  # for 0.486

# -------------------------------
# Reaction channels for Zn/Cu system
# -------------------------------
CHANNELS = [
    ("Zn63", "(n,gamma)", "Zn64"),
    ("Zn65", "(n,2n)",    "Zn64"),

    ("Zn64", "(n,gamma)", "Zn65"),
    ("Zn66", "(n,2n)",    "Zn65"),

    ("Zn65", "(n,gamma)", "Zn66"),
    ("Zn67", "(n,2n)",    "Zn66"),

    ("Zn66", "(n,gamma)", "Zn67"),
    ("Zn68", "(n,2n)",    "Zn67"),

    ("Zn67", "(n,gamma)", "Zn68"),
    ("Zn69", "(n,2n)",    "Zn68"),

    ("Zn68", "(n,gamma)", "Zn69"),
    ("Zn70", "(n,2n)",    "Zn69"),

    ("Zn69", "(n,gamma)", "Zn70"),

    ("Zn64", "(n,p)",     "Cu64"),
    ("Zn67", "(n,p)",     "Cu67"),
    ("Zn68", "(n,d)",     "Cu67"),
]


def compute_volumes_from_params(z_inner_thickness, z_outer_thickness, struct_thickness, 
                                 multi_thickness, moderator_thickness, target_height=100.0):
    """
    Compute material volumes analytically using geometry parameters.
    Uses the same formulas as fusion_irradiation.py create_geometry().
    
    Parameters
    ----------
    z_inner_thickness : float
        Inner target thickness (cm).
    z_outer_thickness : float
        Outer target thickness (cm).
    struct_thickness : float
        Structure thickness (cm).
    multi_thickness : float
        Multiplier thickness (cm), 0 if none.
    moderator_thickness : float
        Moderator thickness (cm), 0 if none.
    target_height : float
        Height of cylindrical geometry (cm), default 100.
    
    Returns
    -------
    dict[int, float]
        material_id -> volume (cm³) for materials 0 (inner) and 1 (outer).
    """
    import numpy as np
    
    # Radii (same logic as fusion_irradiation.py lines 151-163)
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
    """Return Zn64+others fractions from enrichment map. For natural (0.486/0.4917) use NATURAL_ZN_FRACTIONS."""
    if zn64_enrichment in ZN64_ENRICHMENT_MAP:
        return {'Zn64': zn64_enrichment, **ZN64_ENRICHMENT_MAP[zn64_enrichment]}
    if zn64_enrichment in (0.486, 0.4917):
        other_sum = sum(NATURAL_ZN_FRACTIONS.values())
        scale = (1 - zn64_enrichment) / other_sum if other_sum > 0 else 1
        return {'Zn64': zn64_enrichment, **{k: v * scale for k, v in NATURAL_ZN_FRACTIONS.items()}}
    keys = sorted(ZN64_ENRICHMENT_MAP.keys(), key=lambda k: abs(k - zn64_enrichment))
    return {'Zn64': zn64_enrichment, **ZN64_ENRICHMENT_MAP[keys[0]]}


def calculate_enriched_zn_density(zn64_enrichment):
    """density = 7.14 * (M_avg / 65.38). Returns 7.14 for natural."""
    if zn64_enrichment in (0.486, 0.4917):
        return 7.14
    fracs = get_zn_fractions(zn64_enrichment)
    total = sum(fracs.values())
    fracs = {k: v / total for k, v in fracs.items()}
    M_avg = sum(fracs[iso] * openmc.data.atomic_mass(iso) for iso in fracs)
    return 7.14 * (M_avg / 65.38)


def get_initial_zn_atoms_fallback(volume_cm3, zn64_enrichment, density_g_cm3):
    """Fallback when statepoint unavailable. Uses ZN64_ENRICHMENT_MAP."""
    fracs = get_zn_fractions(zn64_enrichment)
    total = sum(fracs.values())
    fracs = {k: v / total for k, v in fracs.items()}
    avg_mass = sum(fracs[iso] * openmc.data.atomic_mass(iso) for iso in fracs)
    total_atoms = (volume_cm3 * density_g_cm3 / avg_mass) * 6.022e23
    initial_atoms = {iso: total_atoms * f for iso, f in fracs.items()}
    for nuc in ('Cu64', 'Cu67', 'Zn65', 'Zn69'):
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
        mat = summary.materials[material_id]
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
    except (KeyError, AttributeError, FileNotFoundError) as e:
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
        mat = summary.materials[material_id]
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
    import openmc
    
    try:
        t = sp.get_tally(name=tally_name)
    except LookupError as e:
        print(f"Tally '{tally_name}' not found: {e}")
        return 0.0
    if parent_nuclide not in t.nuclides:
        return 0.0
    
    # Get tally shape to determine number of cell bins
    tally_mean = np.asarray(t.mean)
    n_cell_bins = tally_mean.shape[0]
    
    # Find the correct bin index by looking at CellFilter
    # We want the outer target which has name 'outer_target' (cell_id=6 in geometry)
    bin_idx = None
    for filt in t.filters:
        if isinstance(filt, openmc.CellFilter):
            cell_bins = filt.bins
            for i, cell_bin in enumerate(cell_bins):
                # cell_bin is the cell ID
                cell_obj = sp.summary.geometry.get_all_cells().get(cell_bin)
                if cell_obj is not None:
                    # Map requested cell_id (0=inner, 1=outer) to actual cell names
                    if cell_id == 0 and 'inner' in cell_obj.name.lower():
                        bin_idx = i
                        break
                    elif cell_id == 1 and 'outer' in cell_obj.name.lower():
                        bin_idx = i
                        break
            break
    
    # Fallback: if we couldn't find by name, use old logic
    if bin_idx is None:
        if n_cell_bins == 1:
            bin_idx = 0
        elif cell_id == 1:
            # Outer target is typically first in output_cells order
            bin_idx = 0
        else:
            bin_idx = min(cell_id, n_cell_bins - 1)
    
    nuc_idx = t.get_nuclide_index(parent_nuclide)
    score_idx = t.get_score_index(score)
    
    try:
        val = float(tally_mean[bin_idx, nuc_idx, score_idx])
    except IndexError:
        print(f"  Warning: bin_idx={bin_idx} out of range for tally '{tally_name}' (shape={tally_mean.shape})")
        val = 0.0
    
    return val * source_strength

def build_channel_rr_per_s(sp, cell_id, source_strength=5e13):
    ch = {}

    # Zn chain (each is a separate channel)
    ch["Zn63 (n,gamma) Zn64"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn63", source_strength)
    ch["Zn65 (n,2n) Zn64"]    = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,2n)",    "Zn65", source_strength)

    ch["Zn64 (n,gamma) Zn65"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn64", source_strength)
    ch["Zn66 (n,2n) Zn65"]    = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,2n)",    "Zn66", source_strength)

    ch["Zn65 (n,gamma) Zn66"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn65", source_strength)
    ch["Zn67 (n,2n) Zn66"]    = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,2n)",    "Zn67", source_strength)

    ch["Zn66 (n,gamma) Zn67"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn66", source_strength)
    ch["Zn68 (n,2n) Zn67"]    = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,2n)",    "Zn68", source_strength)

    ch["Zn67 (n,gamma) Zn68"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn67", source_strength)
    ch["Zn69 (n,2n) Zn68"]    = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,2n)",    "Zn69", source_strength)

    ch["Zn68 (n,gamma) Zn69"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn68", source_strength)
    ch["Zn70 (n,2n) Zn69"]    = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,2n)",    "Zn70", source_strength)

    ch["Zn69 (n,gamma) Zn70"] = channel_rate_per_s(sp, "Zn rxn rates", cell_id, "(n,gamma)", "Zn69", source_strength)

    # Cu production
    ch["Zn64 (n,p) Cu64"] = channel_rate_per_s(sp, "Cu Production rxn rates", cell_id, "(n,p)", "Zn64", source_strength)
    ch["Zn67 (n,p) Cu67"] = channel_rate_per_s(sp, "Cu Production rxn rates", cell_id, "(n,p)", "Zn67", source_strength)
    ch["Zn68 (n,d) Cu67"] = channel_rate_per_s(sp, "Cu Production rxn rates", cell_id, "(n,d)", "Zn68", source_strength)
    n_zero = sum(1 for v in ch.values() if (v is None or (hasattr(v, '__float__') and float(v) == 0)))
    n_nonzero = len(ch) - n_zero
    print(f"  [build_channel_rr_per_s] cell_id={cell_id}: {n_nonzero} non-zero, {n_zero} zero rates")
    if n_nonzero > 0:
        for k, v in ch.items():
            vf = float(v) if v is not None else 0.0
            if vf > 0:
                print(f"    {k}: {vf:.4e}")
    return ch

def _half_life_seconds(nuclide: str):
    try:
        hl = openmc.data.half_life(nuclide)
        if hl is None or not np.isfinite(hl) or hl <= 0:
            return None
        return float(hl)
    except Exception:
        return None

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
    
    # First pass: calculate consumption of parents and production rates
    for parent, rxn, daughter in CHANNELS:
        key = f"{parent} {rxn} {daughter}"
        R = channel_rr_per_s.get(key, 0.0)
        if R is None:
            continue
        R = float(np.asarray(R).flat[0])
        if R <= 0:
            continue
        
        # Check if we have enough parent atoms
        avail = N.get(parent, 0.0)
        consumed_total = R * dt_s
        
        # Clamp consumption to available atoms
        if consumed_total > avail:
            # Scale down the effective rate
            R_effective = avail / dt_s if dt_s > 0 else 0
        else:
            R_effective = R
        
        # Track production rate for this daughter
        if daughter not in daughter_production_rates:
            daughter_production_rates[daughter] = 0.0
        daughter_production_rates[daughter] += R_effective
        
        # Consume parent (simple linear consumption - valid if parent doesn't decay significantly)
        N[parent] = avail - R_effective * dt_s
    
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

