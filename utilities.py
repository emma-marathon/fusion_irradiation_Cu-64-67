'''
Helper functions for neutronics analysis:
- Reaction rate extraction from tallies
- Bateman equation for isotope evolution (production + decay)
- Geometry volume calculations
'''
import numpy as np
import openmc

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

def get_material_density_from_statepoint(sp, material_id):
    """
    Extract material density (g/cm3) from statepoint summary.
    
    The density is set during material definition in fusion_irradiation.py
    and stored in the statepoint summary.
    
    Parameters
    ----------
    sp : openmc.StatePoint
        Opened statepoint file
    material_id : int
        Material ID (e.g., 1 for outer Zn target)
    
    Returns
    -------
    float
        Density in g/cm3, or None if not found
    """
    def extract_density(mat):
        """Helper to extract density from a material object."""
        if mat is None:
            return None
        # Try different ways to get density
        if hasattr(mat, 'density'):
            density = mat.density
            if isinstance(density, (list, tuple)):
                return float(density[0])
            if density is not None:
                return float(density)
        # Some OpenMC versions store as get_mass_density()
        if hasattr(mat, 'get_mass_density'):
            try:
                return float(mat.get_mass_density())
            except:
                pass
        return None
    
    try:
        # Method 1: sp.summary.materials dict (keyed by ID)
        if hasattr(sp, 'summary') and hasattr(sp.summary, 'materials'):
            materials = sp.summary.materials
            
            # Direct dict access
            if isinstance(materials, dict):
                if material_id in materials:
                    d = extract_density(materials[material_id])
                    if d is not None:
                        return d
            
            # Iterate if it has values()
            if hasattr(materials, 'values'):
                for mat in materials.values():
                    if hasattr(mat, 'id') and mat.id == material_id:
                        d = extract_density(mat)
                        if d is not None:
                            return d
            
            # Iterate if it's a list
            if hasattr(materials, '__iter__') and not isinstance(materials, dict):
                for mat in materials:
                    if hasattr(mat, 'id') and mat.id == material_id:
                        d = extract_density(mat)
                        if d is not None:
                            return d
        
        # Method 2: sp.summary.geometry -> get_all_materials()
        if hasattr(sp, 'summary') and hasattr(sp.summary, 'geometry'):
            geom = sp.summary.geometry
            if hasattr(geom, 'get_all_materials'):
                all_mats = geom.get_all_materials()
                if material_id in all_mats:
                    d = extract_density(all_mats[material_id])
                    if d is not None:
                        return d
        
        # Method 3: Check cells for material fill
        if hasattr(sp, 'summary') and hasattr(sp.summary, 'geometry'):
            geom = sp.summary.geometry
            if hasattr(geom, 'get_all_cells'):
                cells = geom.get_all_cells()
                for cell in cells.values():
                    if hasattr(cell, 'fill') and hasattr(cell.fill, 'id'):
                        if cell.fill.id == material_id:
                            d = extract_density(cell.fill)
                            if d is not None:
                                return d
        
        print(f"  Warning: Could not find material {material_id} in statepoint summary")
        return None
        
    except Exception as e:
        print(f"  Warning: Error extracting density for material {material_id}: {e}")
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

