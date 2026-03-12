"""
OpenMC point-source D-T test: Zn-filled lead or bismuth pig (6 in from source, 1 cm boron).
Default run: 8 h irradiation (EOI, no cooldown); Cu retained in cavity.

Outputs:
- Dose at 1 m and vs distance: cavity (Zn, Cu, Ni) unshielded by element; total = cavity shielded + wall.
  Shielding: per-isotope HVL (zn_waste GAMMA_ENERGIES); wall (Pb or Bi) + optional quartz (bismuth_quartz) + boron.
  Distance: inverse-square from pig center; reference lines (1 m, 50 mSv/yr occupational, Chest CT).
- Activity and dose from build_all_products_table / zn_waste get_dose_coeff for all isotopes (cavity + wall).
- CSVs: cu_summary, zn_summary, test_all_products (activities, dose rates, HVL), dose_vs_time_1m.
"""

import glob
import openmc
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import zipfile
import pandas as pd

from utilities import (
    get_zn_fractions,
    calculate_enriched_zn_density,
    build_channel_rr_per_s,
    get_initial_atoms_from_statepoint,
    get_material_density_from_statepoint,
    get_initial_zn_atoms_fallback,
    evolve_bateman_irradiation,
    apply_single_decay_step,
    get_zn64_enrichment_cost_per_kg,
    ZN64_ENRICHMENT_MAP,
    get_decay_constant,
    activity_Bq_after_irrad_cooldown,
    activity_Bq_after_cyclic,
)

# Test-specific: reduced source strength and power
SOURCE_STRENGTH = 3.6e13  
SOURCE_ENERGY_MEV = 14.1
FUSION_POWER_W = SOURCE_STRENGTH * (SOURCE_ENERGY_MEV * 1e6 * 1.602e-19)  # W from 14.1 MeV neutron

# Geometry: 16A-308-LC Thick-Walled Lead Pig (exterior 4 in x 5.875 in, interior 1.5 in x 3.25 in, wall 1.25 in, 28 lbs)
IN_TO_CM = 2.54
PIG_OUTER_DIAMETER_IN = 4.0       # exterior diameter (in)
PIG_OUTER_HEIGHT_IN = 5.875      # exterior height (in)
PIG_INNER_DIAMETER_IN = 1.5      # interior diameter (in)
PIG_INTERIOR_HEIGHT_IN = 3.25    # interior height (in)
PIG_WALL_THICKNESS_IN = 1.25     # radial wall (in); (4 - 1.5)/2 = 1.25
PIG_OUTER_RADIUS_CM = (PIG_OUTER_DIAMETER_IN / 2.0) * IN_TO_CM
PIG_INNER_RADIUS_CM = (PIG_INNER_DIAMETER_IN / 2.0) * IN_TO_CM
PIG_WALL_THICKNESS_CM = PIG_WALL_THICKNESS_IN * IN_TO_CM
PIG_OUTER_HALF_HEIGHT_CM = (PIG_OUTER_HEIGHT_IN / 2.0) * IN_TO_CM
PIG_INTERIOR_HEIGHT_CM = PIG_INTERIOR_HEIGHT_IN * IN_TO_CM
# Cavity centered in pig: symmetric top and bottom lead
PIG_Z_LO_CM = -PIG_OUTER_HALF_HEIGHT_CM
PIG_Z_HI_CM = PIG_OUTER_HALF_HEIGHT_CM
PIG_Z_CAVITY_LO_CM = -PIG_INTERIOR_HEIGHT_CM / 2.0
PIG_Z_CAVITY_HI_CM = PIG_INTERIOR_HEIGHT_CM / 2.0

BORON_THICKNESS_CM = 1.0   # 1 cm boron outside pig
# HVL for Zn-65 gamma (~1.1 MeV) in natural boron. Formula: HVL = ln(2)/mu, mu = (mu/rho)*rho;
# (mu/rho) from NIST X-Ray Mass Attenuation (B, ~1 MeV): ~0.06 cm^2/g; rho_B ~ 2.34 g/cm^3 -> HVL ~ 5 cm.
# NIST: https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z05.html
HVL_BORON_CM = 5.0
ZN_TARGET_MASS_G = 1000.0  # 1 kg Zn target
SOURCE_TO_CYLINDER_CM = 15.24  # 6 inches: distance from source to nearest pig surface
SPHERE_RADIUS_CM = 100.0
# Water bath + vacuum bubble geometry
VACUUM_BUBBLE_RADIUS_CM = 40.0
WATER_RADIUS_CM = 100.0

# Pig center: source at origin; nearest surface = front of pig => center = SOURCE_TO_CYLINDER_CM + outer_radius
CYLINDER_X_CENTER = SOURCE_TO_CYLINDER_CM + PIG_OUTER_RADIUS_CM + BORON_THICKNESS_CM

# All tests use this Zn fill mass; cavity is partially filled to this mass, rest is void.
TARGET_ZN_MASS_G = 670.0  # 0.67 kg Zn in every configuration (thin/thick lead, bismuth, bismuth_quartz, no_wall)

# Optional: bismuth_quartz geometry (Zn fill = TARGET_ZN_MASS_G in quartz tube + bismuth wrapper)
PIG_GEOMETRY_TYPE = None       # 'bismuth_quartz' or None
PIG_ZN_DENSITY_OVERRIDE_G_CM3 = None  # set for bismuth_quartz
PIG_QUARTZ_OUTER_RADIUS_CM = None     # quartz outer radius (bismuth_quartz only)
PIG_QUARTZ_THICKNESS_CM = 0.0   # quartz tube thickness when bismuth_quartz; used in dose shielding
HVL_QUARTZ_CM = 5.0            # ~1 MeV gamma in fused silica (density 2.2); used if quartz present

# Four shielded cases: thick lead, thin lead, thick bismuth, thin bismuth; then unshielded (no wall).
# Bismuth: lead-free alternative; all configs 0.67 kg Zn fill (same as TARGET_ZN_MASS_G).
PIG_CONFIGS = [
    {
        'folder_name': '16A_308_LC_thick',
        'label': '16A-308-LC Thick-Walled Lead (1.25 cm wall)',
        'outer_diameter_in': 4.0,
        'outer_height_in': 5.875,
        'inner_diameter_in': 1.5,
        'interior_height_in': 3.25,
        'wall_thickness_in': 1.25,
        'wall_material': 'lead',
    },
    {
        'folder_name': '16A_305_LC_thin',
        'label': '16A-305-LC Thin Walled Lead (0.5 cm wall)',
        'outer_diameter_in': 2.37,
        'outer_height_in': 5.0,
        'inner_diameter_in': 1.37,
        'interior_height_in': 4.0,
        'wall_thickness_in': 0.5,
        'wall_material': 'lead',
    },
    {
        'folder_name': 'bismuth_pig_1kg_zn',
        'label': 'Bismuth Pig Thick (1.25 cm wall)',
        'outer_diameter_in': 4.0,
        'outer_height_in': 7.5,
        'inner_diameter_in': 1.5,
        'interior_height_in': 5.0,
        'wall_thickness_in': 1.25,
        'wall_material': 'bismuth',
    },
    {
        'folder_name': 'bismuth_pig_thin_1p03kg_zn',
        'label': 'Bismuth Pig Thin (0.5 cm wall)',
        'outer_diameter_in': 2.5,
        'outer_height_in': 5.0,
        'inner_diameter_in': 1.5,
        'interior_height_in': 5.0,
        'wall_thickness_in': 0.5,
        'wall_material': 'bismuth',
    },
    {
        'folder_name': 'bismuth_quartz',
        'label': 'Bismuth + quartz (0.25 cm quartz, 0.5 cm Bi)',
        'geometry_type': 'bismuth_quartz',
        'quartz_inner_radius_cm': 2.5,   # 50 mm inner diameter (Zn cavity)
        'quartz_thickness_cm': 0.25,     # quartz tube wall thickness (plotted and legended)
        'tube_length_cm': 13.0,         # 13 cm chamber length
        'bismuth_thickness_cm': 1.27,  # 1.27 cm bismuth outside quartz 0.5 in
        'wall_material': 'bismuth',
    },
    # No wall: single unshielded baseline (0.67 kg Zn)
    {
        'folder_name': 'no_wall_0p67kg_zn',
        'label': 'No wall, 0.67 kg Zn (unshielded)',
        'outer_diameter_in': 1.5,
        'outer_height_in': 3.25,
        'inner_diameter_in': 1.5,
        'interior_height_in': 3.25,
        'wall_thickness_in': 0.0,
        'wall_material': 'lead',
        'no_wall': True,
    },
]


PIG_WALL_MATERIAL = 'lead'
PIG_NO_WALL = False

# --- Geometry: one class, three options ---
class PigGeometry:
    """Cylinder geometry for Zn target: (r_inner, r_outer, z extents, wall/quartz/bismuth)."""
    __slots__ = ('r_inner', 'r_outer', 'z_lo', 'z_hi', 'z_cavity_lo', 'z_cavity_hi', 'wall_thickness',
                 'wall_material', 'no_wall', 'kind', 'quartz_r_outer', 'zn_fill_z_hi_cm')
    def __init__(self, r_inner, r_outer, z_lo, z_hi, z_cavity_lo, z_cavity_hi, wall_thickness=0,
                 wall_material='lead', no_wall=False, kind='cylinder', quartz_r_outer=None, zn_fill_z_hi_cm=None):
        self.r_inner, self.r_outer = float(r_inner), float(r_outer)
        self.z_lo, self.z_hi = float(z_lo), float(z_hi)
        self.z_cavity_lo, self.z_cavity_hi = float(z_cavity_lo), float(z_cavity_hi)
        self.wall_thickness = float(wall_thickness)
        self.wall_material = wall_material
        self.no_wall = bool(no_wall)
        self.kind = kind  # 'cylinder' | 'cylinder_mass' | 'quartz_bismuth'
        self.quartz_r_outer = float(quartz_r_outer) if quartz_r_outer is not None else None
        self.zn_fill_z_hi_cm = float(zn_fill_z_hi_cm) if zn_fill_z_hi_cm is not None else z_cavity_hi

def cylinder_fixed(r_inner_cm, outer_radius_cm, cavity_half_height_cm, wall_thickness_cm, wall_material='lead'):
    """Set cylinder: cavity (r_inner, cavity height) + wall; Zn fills full cavity."""
    z_c = 0.0
    z_lo = -(cavity_half_height_cm + wall_thickness_cm)
    z_hi = cavity_half_height_cm + wall_thickness_cm
    return PigGeometry(r_inner_cm, outer_radius_cm, z_lo, z_hi, -cavity_half_height_cm, cavity_half_height_cm,
                       wall_thickness_cm, wall_material, no_wall=(wall_thickness_cm <= 0))

def cylinder_for_mass(mass_kg, rho_zn_g_cm3, wall_thickness_cm, aspect_ratio=1.0):
    """Set wall thickness; cavity volume = mass_kg/rho so given mass Zn fits. aspect_ratio = height/(2*r)."""
    vol_cm3 = (mass_kg * 1000.0) / rho_zn_g_cm3
    r_inner = (vol_cm3 / (np.pi * aspect_ratio)) ** (1.0 / 3.0)
    h_half = r_inner * aspect_ratio
    r_outer = r_inner + wall_thickness_cm
    return PigGeometry(r_inner, r_outer, -(h_half + wall_thickness_cm), h_half + wall_thickness_cm,
                       -h_half, h_half, wall_thickness_cm, 'bismuth', no_wall=False, kind='cylinder_mass')

def quartz_bismuth(r_zn_cm, zn_height_cm, t_quartz_cm, t_bismuth_cm):
    """Quartz cylinder + caps (bottom: zn_lo-t_q..zn_lo, top: zn_hi..zn_hi+t_q); bismuth top disk (zn_hi+t_q .. zn_hi+t_q+t_bi) R=r_zn+t_q+t_bi; bismuth annulus around all."""
    h = zn_height_cm / 2.0
    z_zn_lo, z_zn_hi = -h, h
    r_q = r_zn_cm + t_quartz_cm
    r_outer = r_q + t_bismuth_cm
    z_lo = z_zn_lo - t_quartz_cm
    z_hi = z_zn_hi + t_quartz_cm + t_bismuth_cm
    return PigGeometry(r_zn_cm, r_outer, z_lo, z_hi, z_zn_lo, z_zn_hi, t_bismuth_cm, 'bismuth',
                       no_wall=False, kind='quartz_bismuth', quartz_r_outer=r_q)

PIG_GEOM = None  # set by apply_pig_config from PIG_CONFIGS


def _geom_from_globals():
    """Build PigGeometry from current PIG_* globals (legacy)."""
    return PigGeometry(PIG_INNER_RADIUS_CM, PIG_OUTER_RADIUS_CM, PIG_Z_LO_CM, PIG_Z_HI_CM,
                       PIG_Z_CAVITY_LO_CM, PIG_Z_CAVITY_HI_CM, PIG_WALL_THICKNESS_CM, PIG_WALL_MATERIAL,
                       PIG_NO_WALL, 'bismuth_quartz' if PIG_GEOMETRY_TYPE == 'bismuth_quartz' else 'cylinder',
                       PIG_QUARTZ_OUTER_RADIUS_CM)


def apply_pig_config(config):
    """Set PIG_GEOM and global PIG_* from config (cylinder, or bismuth_quartz)."""
    global PIG_OUTER_DIAMETER_IN, PIG_OUTER_HEIGHT_IN, PIG_INNER_DIAMETER_IN, PIG_INTERIOR_HEIGHT_IN
    global PIG_WALL_THICKNESS_IN, PIG_OUTER_RADIUS_CM, PIG_INNER_RADIUS_CM, PIG_WALL_THICKNESS_CM
    global PIG_OUTER_HALF_HEIGHT_CM, PIG_INTERIOR_HEIGHT_CM, PIG_Z_LO_CM, PIG_Z_HI_CM
    global PIG_Z_CAVITY_LO_CM, PIG_Z_CAVITY_HI_CM, CYLINDER_X_CENTER, PIG_WALL_MATERIAL, PIG_NO_WALL
    global PIG_GEOMETRY_TYPE, PIG_ZN_DENSITY_OVERRIDE_G_CM3, PIG_QUARTZ_OUTER_RADIUS_CM, PIG_QUARTZ_THICKNESS_CM, PIG_GEOM

    if config.get('geometry_type') == 'bismuth_quartz':
        r_in, t_q, L, t_bi = config['quartz_inner_radius_cm'], config['quartz_thickness_cm'], config['tube_length_cm'], config['bismuth_thickness_cm']
        PIG_GEOM = quartz_bismuth(r_in, L, t_q, t_bi)
        PIG_GEOMETRY_TYPE, PIG_ZN_DENSITY_OVERRIDE_G_CM3 = 'bismuth_quartz', None
        PIG_INNER_RADIUS_CM, PIG_QUARTZ_OUTER_RADIUS_CM, PIG_QUARTZ_THICKNESS_CM = r_in, r_in + t_q, t_q
        PIG_OUTER_RADIUS_CM, PIG_INTERIOR_HEIGHT_CM = r_in + t_q + t_bi, L
        PIG_OUTER_HALF_HEIGHT_CM, PIG_WALL_THICKNESS_CM, PIG_WALL_MATERIAL, PIG_NO_WALL = L/2 + t_bi, t_bi, 'bismuth', False
    else:
        PIG_GEOMETRY_TYPE, PIG_ZN_DENSITY_OVERRIDE_G_CM3, PIG_QUARTZ_OUTER_RADIUS_CM, PIG_QUARTZ_THICKNESS_CM = None, None, None, 0.0
        od, oh, id_, ih = config['outer_diameter_in'], config['outer_height_in'], config['inner_diameter_in'], config['interior_height_in']
        wt, wm, no_wall = config['wall_thickness_in'], config.get('wall_material', 'lead'), config.get('no_wall', False)
        r_in, r_out = (id_/2)*IN_TO_CM, (od/2)*IN_TO_CM
        h_out, h_cav = (oh/2)*IN_TO_CM, (ih/2)*IN_TO_CM
        PIG_GEOM = cylinder_fixed(r_in, r_out, h_cav, wt*IN_TO_CM, wm)
        PIG_GEOM.no_wall = no_wall
        PIG_OUTER_RADIUS_CM, PIG_INNER_RADIUS_CM = r_out, r_in
        PIG_WALL_THICKNESS_CM, PIG_OUTER_HALF_HEIGHT_CM, PIG_INTERIOR_HEIGHT_CM = wt*IN_TO_CM, h_out, ih*IN_TO_CM
        PIG_WALL_MATERIAL, PIG_NO_WALL = wm, no_wall

    PIG_Z_LO_CM, PIG_Z_HI_CM = PIG_GEOM.z_lo, PIG_GEOM.z_hi
    PIG_Z_CAVITY_LO_CM, PIG_Z_CAVITY_HI_CM = PIG_GEOM.z_cavity_lo, PIG_GEOM.z_cavity_hi
    PIG_OUTER_DIAMETER_IN, PIG_OUTER_HEIGHT_IN = (PIG_OUTER_RADIUS_CM*2)/IN_TO_CM, (PIG_OUTER_HALF_HEIGHT_CM*2)/IN_TO_CM
    PIG_INNER_DIAMETER_IN, PIG_INTERIOR_HEIGHT_IN = (PIG_INNER_RADIUS_CM*2)/IN_TO_CM, PIG_INTERIOR_HEIGHT_CM/IN_TO_CM
    PIG_WALL_THICKNESS_IN = PIG_WALL_THICKNESS_CM / IN_TO_CM
    CYLINDER_X_CENTER = SOURCE_TO_CYLINDER_CM + PIG_OUTER_RADIUS_CM + BORON_THICKNESS_CM


# Run parameters (quick test)
PARTICLES = int(1e5)
BATCHES = 20

# Material colors (match fusion_irradiation; lead = wall; bismuth = distinct)
MATERIAL_COLORS = {
    'vacuum': '#9467bd',
    'zn_target': '#8c564b',
    'wall': '#ff7f0e',   # lead pig
    'lead': '#ff7f0e',
    'bismuth': '#c0392b',  # vivid red-brown so bismuth layer is clearly visible
    'quartz': '#b0c4de',   # quartz glass
    'boron': '#2ca02c',
    'water': '#6495ed',
    'void': '#cccccc',
}


def _hex_to_rgb(hex_str):
    """Convert '#rrggbb' to (r, g, b) 0-255 for OpenMC plot.colors."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


# Copper isotope atomic masses [g/mol] (incl. short-lived Cu61, Cu62, Cu66, Cu69, Cu70 — incorporated in cavity)
CU_ATOMIC_MASS_G_MOL = {
    'Cu61': 60.966, 'Cu62': 61.963, 'Cu63': 62.9296, 'Cu64': 63.9298,
    'Cu65': 64.9278, 'Cu66': 65.9289, 'Cu67': 66.9277, 'Cu69': 68.9256, 'Cu70': 69.9254,
}
AVOGADRO = 6.02214076e23


def copper_mass_from_atoms(atoms):
    """From an atoms dict (nuclide -> count), return masses [g] per Cu isotope and total copper mass [g]."""
    out = {f'{iso.lower()}_g': 0.0 for iso in CU_ATOMIC_MASS_G_MOL}
    for iso, mass_g_mol in CU_ATOMIC_MASS_G_MOL.items():
        n = float(atoms.get(iso, 0) or 0)
        out[f'{iso.lower()}_g'] = n * mass_g_mol / AVOGADRO
    out['total_cu_g'] = sum(out[k] for k in out if k.endswith('_g'))
    return out


def create_materials(zn64_enrichment=0.4917, include_water=False, wall_material=None):
    """Create materials: Zn target, pig wall (lead or bismuth), vacuum (None). Optionally water for bath geometry.
    For geometry_type bismuth_quartz: returns [zn, quartz, bismuth, boron]; Zn at real density, TARGET_ZN_MASS_G in cavity (rest void)."""
    if wall_material is None:
        wall_material = PIG_WALL_MATERIAL
    zn_mat = openmc.Material(material_id=1, name='zn_target')
    density = PIG_ZN_DENSITY_OVERRIDE_G_CM3 if PIG_ZN_DENSITY_OVERRIDE_G_CM3 is not None else calculate_enriched_zn_density(zn64_enrichment)
    zn_mat.set_density('g/cm3', density)
    if zn64_enrichment == 0.4917:
        zn_mat.add_element('Zn', 1.0)
    else:
        fracs = get_zn_fractions(zn64_enrichment)
        for iso, f in fracs.items():
            zn_mat.add_nuclide(iso, f)
    zn_mat.temperature = 294

    if PIG_GEOMETRY_TYPE == 'bismuth_quartz':
        # Zn + quartz tube + bismuth wrapper
        quartz_mat = openmc.Material(material_id=2, name='quartz')
        quartz_mat.set_density('g/cm3', 2.2)  # fused silica
        quartz_mat.add_element('Si', 1.0)
        quartz_mat.add_element('O', 2.0)
        quartz_mat.temperature = 294
        bismuth_mat = openmc.Material(material_id=3, name='bismuth')
        bismuth_mat.set_density('g/cm3', 9.78)
        bismuth_mat.add_nuclide('Bi209', 1.0)
        bismuth_mat.temperature = 294
        boron_mat = openmc.Material(material_id=4, name='boron')
        boron_mat.set_density('g/cm3', 2.34)
        boron_mat.add_element('B', 1.0)
        boron_mat.temperature = 294
        mats = [zn_mat, quartz_mat, bismuth_mat, boron_mat]
        if include_water:
            water_mat = openmc.Material(material_id=5, name='water')
            water_mat.set_density('g/cm3', 1.0)
            water_mat.add_nuclide('H1', 2.0)
            water_mat.add_nuclide('O16', 1.0)
            water_mat.temperature = 294
            mats.append(water_mat)
        return mats

    # Pig wall: lead (11.34 g/cm³) or bismuth (9.78 g/cm³, lead-free shielding)
    wall_mat = openmc.Material(material_id=2, name=wall_material)
    if wall_material == 'bismuth':
        wall_mat.set_density('g/cm3', 9.78)
        wall_mat.add_nuclide('Bi209', 1.0)
    else:
        wall_mat.set_density('g/cm3', 11.34)
        wall_mat.add_element('Pb', 1.0)
    wall_mat.temperature = 294

    boron_mat = openmc.Material(material_id=4, name='boron')
    boron_mat.set_density('g/cm3', 2.34)  # Natural boron
    boron_mat.add_element('B', 1.0)
    boron_mat.temperature = 294

    mats = [zn_mat, wall_mat, boron_mat]
    if include_water:
        water_mat = openmc.Material(material_id=5, name='water')
        water_mat.set_density('g/cm3', 1.0)
        water_mat.add_nuclide('H1', 2.0)
        water_mat.add_nuclide('O16', 1.0)
        water_mat.temperature = 294
        mats.append(water_mat)
    return mats


def _build_pig_cells(materials, zn64_enrichment=0.4917):
    """Build pig cells from PigGeometry: (1) cylinder + lead thick/thin, (2) Zn fill = TARGET_ZN_MASS_G (rest void), (3) quartz + bismuth if bismuth_quartz. Returns (pig_region, pig_cells, cells_with_fill, plot_colors)."""
    g = PIG_GEOM if PIG_GEOM is not None else _geom_from_globals()
    x0 = CYLINDER_X_CENTER
    z_lo = openmc.ZPlane(z0=g.z_lo)
    z_hi = openmc.ZPlane(z0=g.z_hi)
    z_cavity_lo = openmc.ZPlane(z0=g.z_cavity_lo)
    z_cavity_hi = openmc.ZPlane(z0=g.z_cavity_hi)
    inner_cyl = openmc.ZCylinder(x0=x0, y0=0.0, r=g.r_inner)
    outer_cyl = openmc.ZCylinder(x0=x0, y0=0.0, r=g.r_outer)

    # Zn: partial fill to TARGET_ZN_MASS_G (0.67 kg); rest of cavity is void
    rho = calculate_enriched_zn_density(zn64_enrichment)
    vol_zn = TARGET_ZN_MASS_G / rho
    cavity_height = g.z_cavity_hi - g.z_cavity_lo
    h_zn = min(vol_zn / (np.pi * g.r_inner**2), cavity_height)
    z_fill_hi = openmc.ZPlane(z0=g.z_cavity_lo + h_zn)
    zn_region = -inner_cyl & +z_cavity_lo & -z_fill_hi
    has_partial_fill = h_zn < cavity_height - 1e-6
    if has_partial_fill:
        cavity_void_region = -inner_cyl & +z_fill_hi & -z_cavity_hi

    pig_cells, cells_with_fill, plot_colors = [], [], {}

    if len(materials) >= 4 and getattr(materials[1], 'name', '') == 'quartz':
        zn_mat, quartz_mat, bismuth_mat, boron_mat = materials[0], materials[1], materials[2], materials[3]
        r_q = g.quartz_r_outer
        t_q = r_q - g.r_inner
        z_quartz_top = openmc.ZPlane(z0=g.z_cavity_hi + t_q)
        z_quartz_bottom = openmc.ZPlane(z0=g.z_cavity_lo - t_q)
        quartz_outer_cyl = openmc.ZCylinder(x0=x0, y0=0.0, r=r_q)
        quartz_side = (+inner_cyl & -quartz_outer_cyl) & +z_cavity_lo & -z_cavity_hi
        quartz_bottom_cap = (-quartz_outer_cyl) & +z_lo & -z_cavity_lo
        quartz_top_cap = (-quartz_outer_cyl) & +z_cavity_hi & -z_quartz_top
        quartz_region = quartz_side | quartz_bottom_cap | quartz_top_cap
        bismuth_shell = (+quartz_outer_cyl & -outer_cyl) & +(z_quartz_bottom) & -(z_quartz_top)
        bismuth_bottom_cap = (-outer_cyl) & +z_lo & -z_quartz_bottom
        bismuth_top_cap = (-outer_cyl) & +z_quartz_top & -z_hi
        bismuth_region = bismuth_shell | bismuth_bottom_cap | bismuth_top_cap
        pig_region = zn_region | bismuth_region | quartz_region
        if has_partial_fill:
            pig_region = pig_region | cavity_void_region
        zn_cell = openmc.Cell(cell_id=1, name='zn_target', fill=zn_mat, region=zn_region)
        quartz_cell = openmc.Cell(cell_id=2, name='quartz', fill=quartz_mat, region=quartz_region)
        bismuth_cell = openmc.Cell(cell_id=3, name='bismuth', fill=bismuth_mat, region=bismuth_region)
        pig_cells, cells_with_fill = [zn_cell, quartz_cell, bismuth_cell], [zn_cell, quartz_cell, bismuth_cell]
        if has_partial_fill:
            cavity_void_cell = openmc.Cell(cell_id=6, name='cavity_void', fill=None, region=cavity_void_region)
            pig_cells.insert(1, cavity_void_cell)
            plot_colors = {zn_cell: _hex_to_rgb(MATERIAL_COLORS['zn_target']), cavity_void_cell: _hex_to_rgb(MATERIAL_COLORS['void']), quartz_cell: _hex_to_rgb(MATERIAL_COLORS['quartz']), bismuth_cell: _hex_to_rgb(MATERIAL_COLORS['bismuth'])}
        else:
            plot_colors = {zn_cell: _hex_to_rgb(MATERIAL_COLORS['zn_target']), quartz_cell: _hex_to_rgb(MATERIAL_COLORS['quartz']), bismuth_cell: _hex_to_rgb(MATERIAL_COLORS['bismuth'])}
        if BORON_THICKNESS_CM > 0:
            boron_outer_cyl = openmc.ZCylinder(x0=x0, y0=0.0, r=g.r_outer + BORON_THICKNESS_CM)
            boron_region = (+outer_cyl & -boron_outer_cyl) & +z_lo & -z_hi
            pig_region = pig_region | boron_region
            pig_cells.append(openmc.Cell(cell_id=5, name='boron', fill=boron_mat, region=boron_region))
            cells_with_fill.append(pig_cells[-1])
            plot_colors[pig_cells[-1]] = _hex_to_rgb(MATERIAL_COLORS['boron'])
        return pig_region, pig_cells, cells_with_fill, plot_colors

    zn_mat, wall_mat, boron_mat = materials[0], materials[1], materials[2]
    if g.no_wall or g.wall_thickness <= 0:
        zn_cell = openmc.Cell(cell_id=1, name='zn_target', fill=zn_mat, region=zn_region)
        if has_partial_fill:
            cavity_void_cell = openmc.Cell(cell_id=2, name='cavity_void', fill=None, region=cavity_void_region)
            pig_region = zn_region | cavity_void_region
            return pig_region, [zn_cell, cavity_void_cell], [zn_cell], {zn_cell: _hex_to_rgb(MATERIAL_COLORS['zn_target']), cavity_void_cell: _hex_to_rgb(MATERIAL_COLORS['void'])}
        return zn_region, [zn_cell], [zn_cell], {zn_cell: _hex_to_rgb(MATERIAL_COLORS['zn_target'])}

    wall_shell = (+inner_cyl & -outer_cyl) & +z_cavity_lo & -z_cavity_hi
    wall_top_cap = (-outer_cyl) & +z_cavity_hi & -z_hi
    wall_bottom_cap = (-outer_cyl) & +z_lo & -z_cavity_lo
    wall_region = wall_shell | wall_top_cap | wall_bottom_cap
    pig_region = zn_region | wall_region
    if has_partial_fill:
        pig_region = pig_region | cavity_void_region
    if BORON_THICKNESS_CM > 0:
        boron_outer_cyl = openmc.ZCylinder(x0=x0, y0=0.0, r=g.r_outer + BORON_THICKNESS_CM)
        pig_region = pig_region | ((+outer_cyl & -boron_outer_cyl) & +z_lo & -z_hi)
    wall_name = getattr(wall_mat, 'name', g.wall_material)
    zn_cell = openmc.Cell(cell_id=1, name='zn_target', fill=zn_mat, region=zn_region)
    wall_cell = openmc.Cell(cell_id=2, name=wall_name, fill=wall_mat, region=wall_region)
    pig_cells, cells_with_fill = [zn_cell, wall_cell], [zn_cell, wall_cell]
    plot_colors = {zn_cell: _hex_to_rgb(MATERIAL_COLORS['zn_target']), wall_cell: _hex_to_rgb(MATERIAL_COLORS.get(wall_name, MATERIAL_COLORS['wall']))}
    if has_partial_fill:
        cavity_void_cell = openmc.Cell(cell_id=3, name='cavity_void', fill=None, region=cavity_void_region)
        pig_cells.insert(1, cavity_void_cell)
        plot_colors[cavity_void_cell] = _hex_to_rgb(MATERIAL_COLORS['void'])
    if BORON_THICKNESS_CM > 0:
        boron_region = (+outer_cyl & -boron_outer_cyl) & +z_lo & -z_hi
        pig_cells.append(openmc.Cell(cell_id=4, name='boron', fill=boron_mat, region=boron_region))
        cells_with_fill.append(pig_cells[-1])
        plot_colors[pig_cells[-1]] = _hex_to_rgb(MATERIAL_COLORS['boron'])
    return pig_region, pig_cells, cells_with_fill, plot_colors


def create_geometry(materials, zn64_enrichment=0.4917):
    """
    Vacuum sphere; point source at origin. Pig (or Zn-only cavity if no wall) at (CYLINDER_X_CENTER, 0, 0).
    With wall: exterior/interior/wall; Zn fills cavity; 1 cm boron outside pig if BORON_THICKNESS_CM > 0.
    No wall (PIG_NO_WALL): Zn cavity only, no shielding.
    bismuth_quartz: Zn cavity + quartz tube + bismuth cylinder and end caps (materials = [zn, quartz, bismuth, boron]).
    """
    sphere = openmc.Sphere(r=SPHERE_RADIUS_CM, boundary_type='vacuum')
    pig_region, pig_cells, cells_with_fill, plot_colors = _build_pig_cells(materials, zn64_enrichment)
    vacuum_region = -sphere & ~pig_region
    vacuum_cell = openmc.Cell(cell_id=10, name='vacuum', fill=None, region=vacuum_region)
    cells_list = pig_cells + [vacuum_cell]
    plot_colors[vacuum_cell] = _hex_to_rgb(MATERIAL_COLORS['vacuum'])
    is_bismuth_quartz = len(materials) >= 4 and getattr(materials[1], 'name', '') == 'quartz'
    plot = openmc.Plot()
    plot.basis = 'xz'
    plot.origin = (0.0, 0.0, 0.0)
    plot.width = (200.0, 200.0)
    plot.pixels = (2400, 2400) if is_bismuth_quartz else (1600, 1600)
    plot.color_by = 'cell'
    plot.colors = plot_colors
    plot.filename = 'images/geometry_xz'
    plots = openmc.Plots([plot])
    universe = openmc.Universe(cells=cells_list)
    geometry = openmc.Geometry(universe)
    return geometry, plots, cells_with_fill


def create_geometry_water_bath(materials, zn64_enrichment=0.4917):
    """
    Vacuum bubble (transmissive) 40 cm radius at center, surrounded by water 40–100 cm,
    void outside 100 cm. D-T source at center. Pig from _build_pig_cells (same Zn/wall/boron logic as vacuum-sphere case).
    """
    # Resolve water material: bismuth_quartz has water at index 4, standard at index 3
    water_mat = None
    for m in materials:
        if getattr(m, 'name', '') == 'water':
            water_mat = m
            break
    if water_mat is None:
        water_mat = openmc.Material(material_id=5, name='water')
        water_mat.set_density('g/cm3', 1.0)
        water_mat.add_nuclide('H1', 2.0)
        water_mat.add_nuclide('O16', 1.0)
        water_mat.temperature = 294

    inner_sphere = openmc.Sphere(r=VACUUM_BUBBLE_RADIUS_CM, boundary_type='transmission')
    outer_sphere = openmc.Sphere(r=WATER_RADIUS_CM, boundary_type='vacuum')
    pig_region, pig_cells, cells_with_fill, plot_colors = _build_pig_cells(materials, zn64_enrichment)

    vacuum_region = -inner_sphere & ~pig_region
    water_region = +inner_sphere & -outer_sphere & ~pig_region
    void_region = +outer_sphere

    vacuum_cell = openmc.Cell(cell_id=10, name='vacuum_bubble', fill=None, region=vacuum_region)
    water_cell = openmc.Cell(cell_id=11, name='water', fill=water_mat, region=water_region)
    void_cell = openmc.Cell(cell_id=12, name='void', fill=None, region=void_region)
    cells_list = pig_cells + [vacuum_cell, water_cell, void_cell]
    plot_colors[vacuum_cell] = _hex_to_rgb(MATERIAL_COLORS['vacuum'])
    plot_colors[water_cell] = _hex_to_rgb(MATERIAL_COLORS['water'])
    plot_colors[void_cell] = _hex_to_rgb(MATERIAL_COLORS['void'])
    cells_with_fill.append(water_cell)

    is_bismuth_quartz = len(materials) >= 4 and getattr(materials[1], 'name', '') == 'quartz'
    plot = openmc.Plot()
    plot.basis = 'xz'
    plot.origin = (0.0, 0.0, 0.0)
    plot.width = (250.0, 250.0)
    plot.pixels = (2400, 2400) if is_bismuth_quartz else (1600, 1600)
    plot.color_by = 'cell'
    plot.colors = plot_colors
    plot.filename = 'images/geometry_xz'
    plots = openmc.Plots([plot])
    universe = openmc.Universe(cells=cells_list)
    geometry = openmc.Geometry(universe)
    return geometry, plots, cells_with_fill


# Set True for 40 cm vacuum bubble + 100 cm water bath; False for simple vacuum sphere
USE_WATER_BATH_GEOMETRY = True


def create_source():
    """Point source D-T at origin, 14.1 MeV, isotropic. Strength 5e13 n/s (match utilities)."""
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0.0, 0.0, 0.0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([SOURCE_ENERGY_MEV * 1e6], [1.0])
    return source


def create_tallies(cells_with_fill):
    """Flux per cell with CCFE-709 energy groups; Zn_rxn_rates and Cu_Production_rxn_rates for zn_target."""
    tallies = openmc.Tallies()
    neutron_filter = openmc.ParticleFilter(['neutron'])
    energy_filter = openmc.EnergyFilter.from_group_structure('CCFE-709')
    zn_nuclides = ['Zn62', 'Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']  # Zn61 not in JENDL

    # Flux spectra per cell
    for c in cells_with_fill:
        cell_filter = openmc.CellFilter([c])
        t = openmc.Tally(name=f'{c.name}_spectra')
        t.filters = [neutron_filter, energy_filter, cell_filter]
        t.scores = ['flux']
        tallies.append(t)

    # Zn and Cu tallies for zn_target only (1 bin)
    zn_cell = next((c for c in cells_with_fill if c.name == 'zn_target'), cells_with_fill[0])
    zn_cell_filter = openmc.CellFilter([zn_cell])

    # Zn: all potential reactions (n,gamma), (n,2n), (n,p), (n,d), (n,a)->Ni
    t_zn = openmc.Tally(name='Zn_rxn_rates')
    t_zn.filters = [zn_cell_filter]
    t_zn.scores = ['(n,gamma)', '(n,2n)', '(n,a)']
    t_zn.nuclides = zn_nuclides
    tallies.append(t_zn)

    t_cu = openmc.Tally(name='Cu_Production_rxn_rates')
    t_cu.filters = [zn_cell_filter]
    t_cu.scores = ['(n,p)', '(n,d)']
    t_cu.nuclides = zn_nuclides
    tallies.append(t_cu)

    # Lead pig: all potential reactions (n,gamma), (n,2n), (n,p), (n,d), (n,a)
    # Pb-208 (n,gamma) -> Pb-209; Pb-204 (n,2n) -> Pb-203; (n,p)/(n,d)/(n,a) -> Tl, Hg, etc.
    lead_cell = next((c for c in cells_with_fill if c.name == 'lead'), None)
    if lead_cell is not None:
        pb_nuclides = ['Pb204', 'Pb206', 'Pb207', 'Pb208']
        lead_cell_filter = openmc.CellFilter([lead_cell])
        t_lead = openmc.Tally(name='Lead rxn rates')
        t_lead.filters = [lead_cell_filter]
        t_lead.scores = ['(n,gamma)', '(n,2n)', '(n,p)', '(n,d)', '(n,a)']
        t_lead.nuclides = pb_nuclides
        tallies.append(t_lead)

    # Bismuth pig: all potential reactions (n,gamma), (n,2n), (n,p), (n,d), (n,a)
    # Bi-209 (n,gamma) -> Bi-210; Bi-209 (n,2n) -> Bi-208
    bismuth_cells = [c for c in cells_with_fill if getattr(c, 'name', '') == 'bismuth']
    if bismuth_cells:
        bismuth_cell_filter = openmc.CellFilter(bismuth_cells)
        t_bismuth = openmc.Tally(name='Bismuth rxn rates')
        t_bismuth.filters = [bismuth_cell_filter]
        t_bismuth.scores = ['(n,gamma)', '(n,2n)', '(n,p)', '(n,d)', '(n,a)']
        t_bismuth.nuclides = ['Bi209']
        tallies.append(t_bismuth)

    return tallies


# Analysis parameters (same as simple_analyze)
IRRADIATION_HOURS = [1, 2, 4, 8, 24]
COOLDOWN_DAYS = [0, 1, 2]
IRRAD_MULTI_STEP_THRESHOLD_H = 72
ZN_MATERIAL_ID = 1

_ZN64_KNOWN = sorted(ZN64_ENRICHMENT_MAP.keys())
_TOL = 0.0005


def _norm_enrich(e):
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return e
    e = float(e)
    for k in _ZN64_KNOWN:
        if abs(e - k) <= _TOL:
            return k
    best = min(_ZN64_KNOWN, key=lambda k: abs(e - k))
    return best if abs(e - best) <= 0.01 else e


def _enrich_label(e):
    """Format enrichment for plot labels: 0.999 and 1.0→'99.9%', 0.99→'99%'; one decimal (99.9 not 99.90)."""
    if e is None or (isinstance(e, float) and np.isnan(e)):
        return '?'
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


def _get_zn_material_id_from_summary(summary_path):
    """Find material_id of the material filling the zn_target cell (fixes OpenMC material ID ordering)."""
    try:
        summary = openmc.Summary(summary_path)
        for cell in summary.geometry.get_all_cells().values():
            if getattr(cell, 'name', '') == 'zn_target':
                fill = cell.fill
                if fill is not None and hasattr(fill, 'id'):
                    return fill.id
    except Exception:
        pass
    return ZN_MATERIAL_ID


def get_lead_reaction_rates(sp, source_strength=None):
    """Get lead pig activation reaction rates [atoms/s] from 'Lead rxn rates' tally.
    Returns dict with keys like 'Pb208 (n,gamma) Pb209'. Primary concern: Pb-209 from Pb-208(n,gamma), t1/2 ~3.25 h."""
    if source_strength is None:
        source_strength = SOURCE_STRENGTH
    out = {}
    try:
        t = sp.get_tally(name='Lead rxn rates')
    except LookupError:
        return out
    tally_mean = np.asarray(t.mean)
    # Find cell bin for 'lead' cell; support 2D (single filter bin) like get_bismuth_reaction_rates
    bin_idx = 0
    n_cell_bins = 1 if tally_mean.ndim == 2 else tally_mean.shape[0]
    for filt in t.filters:
        if isinstance(filt, openmc.CellFilter):
            for i, cell_bin in enumerate(filt.bins):
                cell_obj = sp.summary.geometry.get_all_cells().get(cell_bin)
                if cell_obj is None:
                    continue
                name = getattr(cell_obj, 'name', '') or ''
                cid = getattr(cell_obj, 'id', None)
                if name == 'lead' or cid == 2:  # wall cell is id=2 in test geometry
                    bin_idx = i
                    break
            # Fallback: single bin -> use 0 (lead is the only wall cell in this tally)
            if n_cell_bins == 1:
                bin_idx = 0
            break
    # (n,gamma), (n,2n), (n,p), (n,d), (n,a) with product nuclides
    for parent, score, product in [
        ('Pb208', '(n,gamma)', 'Pb209'),
        ('Pb207', '(n,gamma)', 'Pb208'),
        ('Pb206', '(n,gamma)', 'Pb207'),
        ('Pb204', '(n,gamma)', 'Pb205'),
        ('Pb208', '(n,2n)', 'Pb207'),
        ('Pb207', '(n,2n)', 'Pb206'),
        ('Pb206', '(n,2n)', 'Pb205'),
        ('Pb204', '(n,2n)', 'Pb203'),
        ('Pb208', '(n,p)', 'Tl208'),
        ('Pb207', '(n,p)', 'Tl207'),
        ('Pb206', '(n,p)', 'Tl206'),
        ('Pb204', '(n,p)', 'Tl204'),
        ('Pb208', '(n,d)', 'Tl207'),
        ('Pb207', '(n,d)', 'Tl206'),
        ('Pb206', '(n,d)', 'Tl205'),
        ('Pb204', '(n,d)', 'Tl203'),
        ('Pb208', '(n,a)', 'Pt205'),
        ('Pb207', '(n,a)', 'Pt204'),
        ('Pb206', '(n,a)', 'Pt203'),
        ('Pb204', '(n,a)', 'Pt201'),
    ]:
        if parent not in t.nuclides or score not in t.scores:
            continue
        nuc_idx = t.get_nuclide_index(parent)
        score_idx = t.get_score_index(score)
        try:
            if tally_mean.ndim == 2:
                val = float(tally_mean[nuc_idx, score_idx]) * source_strength
            else:
                val = float(tally_mean[bin_idx, nuc_idx, score_idx]) * source_strength
        except IndexError:
            val = 0.0
        out[f'{parent} {score} {product}'] = val
    return out


def get_bismuth_reaction_rates(sp, source_strength=None):
    """Get bismuth pig activation reaction rates [atoms/s] from 'Bismuth rxn rates' tally.
    Sums over all bismuth cells (cylinder + top plate + bottom plate).
    Bi-209 (n,gamma) -> Bi-210 (t1/2 5.01 d); Bi-209 (n,2n) -> Bi-208 (3.68e5 y).
    Returns dict with keys like 'Bi209 (n,gamma) Bi210', 'Bi209 (n,2n) Bi208'."""
    if source_strength is None:
        source_strength = SOURCE_STRENGTH
    out = {}
    try:
        t = sp.get_tally(name='Bismuth rxn rates')
    except LookupError:
        return out
    tally_mean = np.asarray(t.mean)
    if tally_mean.ndim == 2:
        n_cell_bins = 1
    else:
        n_cell_bins = tally_mean.shape[0]
    # (n,gamma), (n,2n), (n,p), (n,d), (n,a)
    for parent, score, product in [
        ('Bi209', '(n,gamma)', 'Bi210'),
        ('Bi209', '(n,2n)', 'Bi208'),
        ('Bi209', '(n,p)', 'Pb209'),
        ('Bi209', '(n,d)', 'Pb208'),
        ('Bi209', '(n,a)', 'Tl205'),
    ]:
        if parent not in t.nuclides or score not in t.scores:
            continue
        nuc_idx = t.get_nuclide_index(parent)
        score_idx = t.get_score_index(score)
        val = 0.0
        for bin_idx in range(n_cell_bins):
            try:
                if tally_mean.ndim == 2:
                    v = float(tally_mean[nuc_idx, score_idx])
                else:
                    v = float(tally_mean[bin_idx, nuc_idx, score_idx])
                val += v * source_strength
            except IndexError:
                pass
        out[f'{parent} {score} {product}'] = val
    return out


def analyze_case_test(sp_file, volume_cm3, zn64_enrichment=0.4917):
    """Analyze test geometry: single zn_target cell. Resolves Zn material from cell fill."""
    dir_name = os.path.basename(os.path.dirname(sp_file))
    run_dir = os.path.dirname(os.path.abspath(sp_file))
    summary_path = os.path.join(run_dir, 'summary.h5')
    material_id = _get_zn_material_id_from_summary(summary_path)

    sp = openmc.StatePoint(sp_file)
    initial_atoms = get_initial_atoms_from_statepoint(sp_file, material_id, volume_cm3)
    zn_density = get_material_density_from_statepoint(sp_file, material_id)
    rr = build_channel_rr_per_s(sp, cell_id=1, source_strength=SOURCE_STRENGTH)
    lead_rr = get_lead_reaction_rates(sp, SOURCE_STRENGTH)
    pb209_rate = float(lead_rr.get('Pb208 (n,gamma) Pb209', 0.0) or 0.0)
    pb203_rate = float(lead_rr.get('Pb204 (n,2n) Pb203', 0.0) or 0.0)
    pb205_rate = float(lead_rr.get('Pb204 (n,gamma) Pb205', 0.0) or 0.0)
    bismuth_rr = get_bismuth_reaction_rates(sp, SOURCE_STRENGTH)
    bi210_rate = float(bismuth_rr.get('Bi209 (n,gamma) Bi210', 0.0) or 0.0)
    bi208_rate = float(bismuth_rr.get('Bi209 (n,2n) Bi208', 0.0) or 0.0)
    zn63_rate = float(rr.get('Zn64 (n,2n) Zn63', 0.0) or 0.0)
    sp.close()

    zn_parents = ['Zn64', 'Zn66', 'Zn67', 'Zn68', 'Zn70']
    zn_total = sum(float(initial_atoms.get(iso, 0) or 0) for iso in zn_parents) if initial_atoms else 0
    if initial_atoms is None or zn_total <= 0:
        zn_density = calculate_enriched_zn_density(zn64_enrichment)
        initial_atoms = get_initial_zn_atoms_fallback(volume_cm3, zn64_enrichment, zn_density)
        print("  [analyze_case_test] Using fallback initial atoms (statepoint had no Zn)")
    elif zn_density is None or zn_density < 5.5:
        zn_density = calculate_enriched_zn_density(zn64_enrichment)
        print(f"  [analyze_case_test] Using Zn density {zn_density:.4f} g/cm³ (statepoint gave wrong material)")

    mass_g = volume_cm3 * zn_density
    return {
        'dir_name': dir_name,
        'sp_file': sp_file,
        'material_id': material_id,
        'zn64_enrichment': zn64_enrichment,
        'use_zn67': False,
        'outer_volume_cm3': volume_cm3,
        'zn_density_g_cm3': zn_density,
        'zn_mass_g': mass_g,
        'initial_atoms': initial_atoms,
        'reaction_rates': rr,
        'lead_reaction_rates': lead_rr,
        'pb209_production_rate_per_s': pb209_rate,
        'pb203_production_rate_per_s': pb203_rate,
        'pb205_production_rate_per_s': pb205_rate,
        'bismuth_reaction_rates': bismuth_rr,
        'bi210_production_rate_per_s': bi210_rate,
        'bi208_production_rate_per_s': bi208_rate,
        'zn63_production_rate_per_s': zn63_rate,
    }


def compute_activities_cyclic(case, n_cycles, irrad_hours_per_cycle, cooldown_hours_between, final_cooldown_days=0, remove_cu_after_each_irrad=False):
    """Bateman for n cycles: irrad + cooldown between cycles. EOI after last irrad. RR scaled by parent depletion.
    Tracks copper mass (Cu63+64+65+67) at each irradiation and cooldown step.

    If remove_cu_after_each_irrad is True, Cu63/Cu64/Cu65/Cu67 are set to zero after each irradiation
    (before cooldown), so each cycle starts with no copper in the target. Final returned activities
    and masses then reflect only the last cycle's Cu production."""
    init = case['initial_atoms']
    rr0 = case['reaction_rates']
    irrad_s = irrad_hours_per_cycle * 3600
    cooldown_s = cooldown_hours_between * 3600
    atoms = {k: float(v) for k, v in init.items()}
    copper_mass_steps = []
    cu_isotopes = list(CU_ATOMIC_MASS_G_MOL.keys())  # Cu63, Cu64, Cu65, Cu67

    for cycle in range(n_cycles):
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
        atoms = evolve_bateman_irradiation(atoms, rr, irrad_s)
        step = {'cycle': cycle + 1, 'phase': 'irrad', 'hours': irrad_hours_per_cycle}
        step.update(copper_mass_from_atoms(atoms))
        copper_mass_steps.append(step)
        if remove_cu_after_each_irrad:
            for iso in cu_isotopes:
                atoms[iso] = 0.0
        if cycle < n_cycles - 1:
            atoms = apply_single_decay_step(atoms, cooldown_s)
            step_cool = {'cycle': cycle + 1, 'phase': 'cooldown', 'hours': cooldown_hours_between / 24.0}
            step_cool.update(copper_mass_from_atoms(atoms))
            copper_mass_steps.append(step_cool)

    if final_cooldown_days > 0:
        atoms = apply_single_decay_step(atoms, final_cooldown_days * 86400)
        step_final = {'cycle': n_cycles, 'phase': 'final_cooldown', 'days': final_cooldown_days}
        step_final.update(copper_mass_from_atoms(atoms))
        copper_mass_steps.append(step_final)

    lam_cu64 = get_decay_constant('Cu64')
    lam_cu67 = get_decay_constant('Cu67')
    lam_cu61 = get_decay_constant('Cu61')
    lam_cu62 = get_decay_constant('Cu62')
    lam_cu66 = get_decay_constant('Cu66')
    lam_cu69 = get_decay_constant('Cu69')
    lam_cu70 = get_decay_constant('Cu70')
    lam_zn65 = get_decay_constant('Zn65')
    lam_zn63 = get_decay_constant('Zn63')
    lam_zn69m = get_decay_constant('Zn69m')
    cu64_atoms = atoms.get('Cu64', 0)
    cu67_atoms = atoms.get('Cu67', 0)
    cu61_atoms = atoms.get('Cu61', 0)
    cu62_atoms = atoms.get('Cu62', 0)
    cu66_atoms = atoms.get('Cu66', 0)
    cu69_atoms = atoms.get('Cu69', 0)
    cu70_atoms = atoms.get('Cu70', 0)
    zn65_atoms = atoms.get('Zn65', 0)
    zn63_atoms = atoms.get('Zn63', 0)
    zn69m_atoms = atoms.get('Zn69m', 0)
    total_cu = cu64_atoms + cu67_atoms + cu61_atoms + cu62_atoms + cu66_atoms + cu69_atoms + cu70_atoms
    cu64_activity = cu64_atoms * lam_cu64
    cu67_activity = cu67_atoms * lam_cu67
    cu61_activity = cu61_atoms * lam_cu61
    cu62_activity = cu62_atoms * lam_cu62
    cu66_activity = cu66_atoms * lam_cu66
    cu69_activity = cu69_atoms * lam_cu69
    cu70_activity = cu70_atoms * lam_cu70
    total_cu_activity = cu64_activity + cu67_activity + cu61_activity + cu62_activity + cu66_activity + cu69_activity + cu70_activity
    cu64_radionuclide_purity = cu64_activity / total_cu_activity if total_cu_activity > 0 else 0
    cu67_radionuclide_purity = cu67_activity / total_cu_activity if total_cu_activity > 0 else 0
    cu64_atomic_purity = cu64_atoms / total_cu if total_cu > 0 else 0
    cu67_atomic_purity = cu67_atoms / total_cu if total_cu > 0 else 0
    copper_final = copper_mass_from_atoms(atoms)

    return {
        'cu64_mCi': cu64_atoms * lam_cu64 / 3.7e7,
        'cu67_mCi': cu67_atoms * lam_cu67 / 3.7e7,
        'zn65_mCi': zn65_atoms * lam_zn65 / 3.7e7,
        'zn63_Bq': zn63_atoms * lam_zn63,
        'zn69m_Bq': zn69m_atoms * lam_zn69m,
        'cu64_Bq': cu64_activity,
        'cu67_Bq': cu67_activity,
        'cu61_Bq': cu61_activity,
        'cu62_Bq': cu62_activity,
        'cu66_Bq': cu66_activity,
        'cu69_Bq': cu69_activity,
        'cu70_Bq': cu70_activity,
        'zn65_Bq': zn65_atoms * lam_zn65,
        'cu64_atomic_purity': cu64_atomic_purity,
        'cu67_atomic_purity': cu67_atomic_purity,
        'cu64_radionuclide_purity': cu64_radionuclide_purity,
        'cu67_radionuclide_purity': cu67_radionuclide_purity,
        'copper_mass_steps': copper_mass_steps,
        'copper_mass_final_g': copper_final['total_cu_g'],
        'cu63_g': copper_final['cu63_g'],
        'cu64_g': copper_final['cu64_g'],
        'cu65_g': copper_final['cu65_g'],
        'cu67_g': copper_final['cu67_g'],
    }


def _get_final_atoms_cyclic(case, n_cycles, irrad_h_per_cycle, cooldown_h_between, final_cooldown_days=0):
    """Return final atoms dict (Zn, Cu, Ni, and Bateman-tracked nuclides) after n cycles of irrad + cooldown.
    Cooldown is applied between cycles and after the last cycle (so activity decays with time after EOI)."""
    init = case['initial_atoms']
    rr0 = case['reaction_rates']
    irrad_s = irrad_h_per_cycle * 3600
    cooldown_s = cooldown_h_between * 3600
    atoms = {k: float(v) for k, v in init.items()}
    for cycle in range(n_cycles):
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
        atoms = evolve_bateman_irradiation(atoms, rr, irrad_s)
        if cycle < n_cycles - 1:
            atoms = apply_single_decay_step(atoms, cooldown_s)
    # Apply cooldown after last cycle so state = EOI + decay(cooldown_h_between); needed for dose vs time
    atoms = apply_single_decay_step(atoms, cooldown_s)
    if final_cooldown_days > 0:
        atoms = apply_single_decay_step(atoms, final_cooldown_days * 86400)
    return atoms


def _get_depletion_history_cyclic(case, n_cycles, irrad_h_per_cycle, cooldown_h_between, final_cooldown_days=0):
    """Return list of (step, cycle, phase, time_s, atoms) at each irrad and cooldown step (Bateman with parent depletion)."""
    init = case['initial_atoms']
    rr0 = case['reaction_rates']
    irrad_s = irrad_h_per_cycle * 3600
    cooldown_s = cooldown_h_between * 3600
    atoms = {k: float(v) for k, v in init.items()}
    history = []
    step = 0
    time_s = 0.0
    for cycle in range(n_cycles):
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
        atoms = evolve_bateman_irradiation(atoms, rr, irrad_s)
        time_s += irrad_s
        step += 1
        history.append((step, cycle, 'irrad', time_s, dict(atoms)))
        if cycle < n_cycles - 1:
            atoms = apply_single_decay_step(atoms, cooldown_s)
            time_s += cooldown_s
            step += 1
            history.append((step, cycle, 'cooldown', time_s, dict(atoms)))
    if final_cooldown_days > 0:
        atoms = apply_single_decay_step(atoms, final_cooldown_days * 86400)
        time_s += final_cooldown_days * 86400
        step += 1
        history.append((step, n_cycles - 1, 'final_cooldown', time_s, dict(atoms)))
    return history


def build_all_products_table(case, n_cycles, irrad_h_per_cycle, cooldown_h_between, final_cooldown_days=0):
    """Build a table of all product nuclides: Cu, Zn, Ni (from Zn n,a), Bi, Bi daughters, Pb, lead daughters, Tl, Pt.
    Returns list of dicts: nuclide, production_rate_per_s, final_atoms, mass_g, activity_Bq."""
    irrad_s = irrad_h_per_cycle * 3600
    total_irrad_s = n_cycles * irrad_s
    cooldown_s = cooldown_h_between * 3600

    atoms = _get_final_atoms_cyclic(case, n_cycles, irrad_h_per_cycle, cooldown_h_between, final_cooldown_days)

    # Aggregate production rates by product from Zn/Cu/Ni, lead, and bismuth (Ni from Bateman CHANNELS)
    rr = case.get('reaction_rates') or {}
    lead_rr = case.get('lead_reaction_rates') or {}
    bismuth_rr = case.get('bismuth_reaction_rates') or {}
    prod_rate = {}
    for ch_key, rate in list(rr.items()) + list(lead_rr.items()) + list(bismuth_rr.items()):
        rate = float(rate or 0)
        if rate <= 0:
            continue
        parts = ch_key.split()
        if len(parts) >= 3:
            product = parts[2]
            prod_rate[product] = prod_rate.get(product, 0) + rate

    # Pb-209, Pb-203, Pb-205: activity at EOI then atoms = A/lam (lead cases)
    for nuclide in ('Pb209', 'Pb203', 'Pb205'):
        R = float(prod_rate.get(nuclide, 0) or case.get(f'pb{nuclide[2:]}_production_rate_per_s', 0) or 0)
        if R <= 0:
            continue
        act_Bq = activity_Bq_after_cyclic(R, nuclide, n_cycles, irrad_h_per_cycle * 3600, cooldown_h_between * 3600)
        lam = get_decay_constant(nuclide)
        if lam and lam > 0:
            atoms[nuclide] = atoms.get(nuclide, 0) + act_Bq / lam

    # Bi-210 -> Po-210 chain
    bi210_rate = float(case.get('bi210_production_rate_per_s', 0) or prod_rate.get('Bi210', 0) or 0)
    if bi210_rate > 0:
        bi210_Bq, po210_Bq = _wall_bi210_po210_Bq(bi210_rate, irrad_h_per_cycle * 3600, cooldown_h_between * 3600, n_cycles)
        lam_bi = get_decay_constant('Bi210')
        lam_po = get_decay_constant('Po210')
        if lam_bi > 0:
            atoms['Bi210'] = atoms.get('Bi210', 0) + bi210_Bq / lam_bi
        if lam_po > 0:
            atoms['Po210'] = atoms.get('Po210', 0) + po210_Bq / lam_po
    # Bi-208 (long-lived)
    bi208_rate = float(case.get('bi208_production_rate_per_s', 0) or prod_rate.get('Bi208', 0) or 0)
    if bi208_rate > 0:
        atoms['Bi208'] = atoms.get('Bi208', 0) + bi208_rate * total_irrad_s

    # Tl, Pt (and any other products from lead/bismuth): simple buildup N = R*t for stable; decay for radioactive
    for product, R in prod_rate.items():
        R = float(R)
        if R <= 0 or product in atoms and atoms[product] > 0:
            continue
        if product in ('Pb209', 'Pb203', 'Pb205') or product.startswith('Bi') or product == 'Po210':
            continue  # already handled above
        lam = get_decay_constant(product)
        if lam and lam > 0:
            n_eoi = R * (1.0 - np.exp(-lam * irrad_s)) / lam
            for _ in range(n_cycles - 1):
                n_eoi = n_eoi * np.exp(-lam * cooldown_s) + R * (1.0 - np.exp(-lam * irrad_s)) / lam
            if final_cooldown_days > 0:
                n_eoi = n_eoi * np.exp(-lam * final_cooldown_days * 86400)
            atoms[product] = atoms.get(product, 0) + n_eoi
        else:
            atoms[product] = atoms.get(product, 0) + R * total_irrad_s

    try:
        from zn_waste import get_dose_coeff, GAMMA_ENERGIES, get_gamma_hvl, print_hvl_calc_table
        # Precompute HVLs for all isotopes in GAMMA_ENERGIES (μ, B, HVL formulas).
        print_hvl_calc_table()
        HVL_BY_ISOTOPE = {iso: get_gamma_hvl(iso) for iso in GAMMA_ENERGIES}
    except ImportError:
        def get_dose_coeff(_):
            return 0.0
        GAMMA_ENERGIES = {}
        HVL_BY_ISOTOPE = {}
        def get_gamma_hvl(_):
            return {'HVL_Pb_cm': 2.0, 'HVL_Bi_cm': 2.4, 'HVL_concrete_cm': 8.0}
    rows = []
    for nuc in sorted(atoms.keys()):
        n_atoms = float(atoms.get(nuc, 0) or 0)
        if n_atoms <= 0:
            continue
        R = prod_rate.get(nuc, 0)
        try:
            mass_g = n_atoms * openmc.data.atomic_mass(nuc) / AVOGADRO
        except (KeyError, AttributeError, TypeError):
            mass_g = 0.0
        lam = get_decay_constant(nuc)
        activity_Bq = n_atoms * lam if lam and lam > 0 else 0.0
        activity_MBq = activity_Bq / 1e6
        coeff = get_dose_coeff(nuc)
        dose_rate_uSv_hr_1m = activity_MBq * (coeff if coeff is not None else 0.0)
        half_life_s = (np.log(2) / lam) if lam and lam > 0 else 0.0
        half_life_str = _format_halflife(nuc)
        gamma_data = GAMMA_ENERGIES.get(nuc, {})
        hvl = HVL_BY_ISOTOPE.get(nuc, get_gamma_hvl(nuc))
        HVL_Pb_cm = hvl['HVL_Pb_cm']
        HVL_Bi_cm = hvl['HVL_Bi_cm']
        HVL_concrete_cm = hvl['HVL_concrete_cm']
        HVL_boron_cm = hvl.get('HVL_boron_cm', HVL_BORON_CM)
        HVL_quartz_cm = hvl.get('HVL_quartz_cm', HVL_QUARTZ_CM)
        if PIG_NO_WALL or PIG_WALL_THICKNESS_CM <= 0:
            shield_factor = 1.0
        else:
            hvl_wall = HVL_Bi_cm if PIG_WALL_MATERIAL == 'bismuth' else HVL_Pb_cm
            shield_factor = 2.0 ** (PIG_WALL_THICKNESS_CM / hvl_wall)
            if PIG_QUARTZ_THICKNESS_CM > 0 and HVL_quartz_cm > 0:
                shield_factor *= 2.0 ** (PIG_QUARTZ_THICKNESS_CM / HVL_quartz_cm)
            if BORON_THICKNESS_CM > 0:
                shield_factor *= 2.0 ** (BORON_THICKNESS_CM / HVL_boron_cm)
        dose_rate_uSv_hr_1m_shielded = dose_rate_uSv_hr_1m / shield_factor if shield_factor > 0 else dose_rate_uSv_hr_1m
        rows.append({
            'nuclide': nuc,
            'production_rate_per_s': R,
            'final_atoms': n_atoms,
            'mass_g': mass_g,
            'activity_Bq': activity_Bq,
            'activity_MBq': activity_MBq,
            'half_life_s': half_life_s,
            'half_life': half_life_str,
            'E_MeV': gamma_data.get('E_MeV'),
            'HVL_Pb_cm': HVL_Pb_cm,
            'HVL_Bi_cm': HVL_Bi_cm,
            'HVL_concrete_cm': HVL_concrete_cm,
            'HVL_boron_cm': HVL_boron_cm,
            'HVL_quartz_cm': HVL_quartz_cm,
            'shield_factor': shield_factor,
            'MIRD_uSv_hr_per_MBq_1m': coeff,
            'dose_rate_uSv_hr_1m': dose_rate_uSv_hr_1m,
            'dose_rate_uSv_hr_1m_shielded': dose_rate_uSv_hr_1m_shielded,
        })
    return rows


def print_isotopes_dose_table(rows, title='Isotopes present after irradiation — dose rate at 1 m'):
    """Print table of isotopes with activity, mass, half-life, HVLs (Pb, Bi, concrete, boron), and dose rate (unshielded and shielded)."""
    if not rows:
        return
    try:
        from zn_waste import get_dose_coeff
    except ImportError:
        def get_dose_coeff(_):
            return None
    print(f"\n  {title}")
    print("  " + "-" * 140)
    h = (f"  {'Nuclide':<8} {'mass_g':>10} {'activity_Bq':>12} {'half_life':>10} "
         f"{'HVL_Pb_cm':>10} {'HVL_Bi_cm':>10} {'HVL_B_cm':>8} {'HVL_Q_cm':>8} {'dose_µSv/h@1m':>14} {'dose_shielded':>14}")
    print(h)
    print("  " + "-" * 148)
    for r in rows:
        nuc = r.get('nuclide', '')
        mass_g = r.get('mass_g', 0) or 0
        act_Bq = r.get('activity_Bq', 0) or 0
        half_life = r.get('half_life', '—')
        hvl_pb = r.get('HVL_Pb_cm', 0)
        hvl_bi = r.get('HVL_Bi_cm', 0)
        hvl_b = r.get('HVL_boron_cm', 0)
        hvl_q = r.get('HVL_quartz_cm', 0)
        dose = r.get('dose_rate_uSv_hr_1m', 0) or 0
        dose_sh = r.get('dose_rate_uSv_hr_1m_shielded', 0) or dose
        print(f"  {nuc:<8} {mass_g:>10.3e} {act_Bq:>12.3e} {str(half_life):>10} {hvl_pb:>10.2f} {hvl_bi:>10.2f} {hvl_b:>8.1f} {hvl_q:>8.1f} {dose:>14.3e} {dose_sh:>14.3e}")
    total_dose = sum((r.get('dose_rate_uSv_hr_1m') or 0) for r in rows)
    total_shielded = sum((r.get('dose_rate_uSv_hr_1m_shielded') or 0) for r in rows)
    print("  " + "-" * 148)
    print(f"  {'Total':<8} {'':<10} {'':<12} {'':<10} {'':<10} {'':<10} {'':<8} {'':<8} {total_dose:>14.3e} {total_shielded:>14.3e}  µSv/hr at 1 m")
    print()


def _format_halflife(nuclide):
    """Return half-life string (e.g. '12.7 h', '5.01 d') for labeling; 'stable' if no decay."""
    lam = get_decay_constant(nuclide)
    if not lam or lam <= 0:
        return "stable"
    hl_s = np.log(2) / lam
    if hl_s < 60:
        return f"{hl_s:.2g} s"
    if hl_s < 3600:
        return f"{hl_s / 60:.2g} m"
    if hl_s < 86400:
        return f"{hl_s / 3600:.2g} h"
    if hl_s < 86400 * 365.25:
        return f"{hl_s / 86400:.2g} d"
    return f"{hl_s / (86400 * 365.25):.2g} y"


def plot_radioisotopes_bar(case, output_dir, irrad_hours=8, cooldown_hours=0):
    """Bar plot of radioactive isotopes with dose > 0: MBq and dose/hr at 1 m, with half-life just above each bar."""
    rows = build_all_products_table(case, 1, irrad_hours, cooldown_hours, final_cooldown_days=0)
    if not rows:
        return
    radioactive = [r for r in rows if (r.get('activity_Bq') or 0) > 0]
    is_bismuth = (PIG_WALL_MATERIAL == 'bismuth')

    def allowed(n):
        if n.startswith('Zn') or n.startswith('Cu') or n.startswith('Ni'):
            return True
        if is_bismuth:
            return n.startswith('Bi') or n == 'Po210'
        return n.startswith('Pb') or n.startswith('Tl') or n.startswith('Pt')

    filtered = [r for r in radioactive if allowed(r.get('nuclide', ''))]
    # Exclude isotopes with zero dose (only show those that contribute to dose)
    filtered = [r for r in filtered if (r.get('dose_rate_uSv_hr_1m') or 0) > 0]
    if not filtered:
        return
    filtered.sort(key=lambda r: -(r.get('activity_MBq') or 0))
    isotopes = [r['nuclide'] for r in filtered]
    mbq = np.array([r.get('activity_MBq') or 0 for r in filtered])
    dose = np.array([r.get('dose_rate_uSv_hr_1m') or 0 for r in filtered])
    halflife_str = [_format_halflife(n) for n in isotopes]

    fig, ax1 = plt.subplots(figsize=(max(10, len(isotopes) * 0.7), 6))
    x = np.arange(len(isotopes))
    width = 0.35
    ax2 = ax1.twinx()
    bars_dose = ax2.bar(x - width / 2, dose, width, label='Dose rate (µSv/hr @ 1 m)', color='tab:orange', alpha=0.85, zorder=1)
    bars_mbq = ax1.bar(x + width / 2, mbq, width, label='Activity (MBq)', color='tab:blue', alpha=0.9, zorder=2)
    ax1.set_ylabel('Activity (MBq)', color='tab:blue', fontsize=11)
    ax2.set_ylabel('Dose rate (µSv/hr at 1 m)', color='tab:orange', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(isotopes, rotation=45, ha='right')
    ax1.set_xlabel('Isotope')
    ax1.set_title(f'Radioisotopes after {irrad_hours} h irrad, {cooldown_hours} h cooldown')
    ax1.set_ylim(0, max(mbq) * 1.2 if mbq.max() > 0 else 1)
    ax2.set_ylim(0, max(dose) * 1.25 if dose.max() > 0 else 1)
    # Time to decay (half-life) just above each bar, inside the plot
    y1_max = ax1.get_ylim()[1]
    for i, hl in enumerate(halflife_str):
        bar_top = mbq[i] + (y1_max * 0.02)
        ax1.text(i, bar_top, hl, ha='center', va='bottom', fontsize=7)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    path = os.path.join(output_dir, 'radioisotopes_bar_MBq_dose.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def write_all_products_csv(case, output_dir, n_cycles=10, irrad_h_per_cycle=8, cooldown_h_between=0, final_cooldown_days=0):
    """Write one CSV of all isotopes after irrad: mass_g, activity_Bq, dose_rate_uSv_hr_1m, HVL_cm (Pb or Bi from wall), half_life; used for shielded/unshielded dose at 1 m and surface."""
    rows = build_all_products_table(case, n_cycles, irrad_h_per_cycle, cooldown_h_between, final_cooldown_days)
    if not rows:
        return
    print_isotopes_dose_table(rows, title='Isotopes present after irradiation — dose rate at 1 m')
    df = pd.DataFrame(rows)
    df['HVL_cm'] = df.apply(lambda r: r['HVL_Bi_cm'] if PIG_WALL_MATERIAL == 'bismuth' else r['HVL_Pb_cm'], axis=1)
    path = os.path.join(output_dir, 'test_all_products.csv')
    df.to_csv(path, index=False, float_format='%.6g')
    print(f"  Saved: {output_dir}/test_all_products.csv ({len(rows)} product nuclides, all produced isotopes)")


def write_depletion_timesteps_csv(case, output_dir, n_cycles=10, irrad_h_per_cycle=8, cooldown_h_between=0, final_cooldown_days=0):
    """Write CSV of masses and activities of Bateman-tracked nuclides (Zn, Cu, Ni) at each timestep.
    Nickel is produced in large amounts by Zn (n,alpha) reactions: Zn64->Ni61, Zn66->Ni63, Zn67->Ni64,
    Zn68->Ni65, Zn70->Ni67. With ~1 kg Zn and 5e13 n/s over 8 h, (n,a) rates accumulate to 1e11–1e13 atoms.
    All Cu isotopes (Cu61–Cu70) are included at every step, even when zero."""
    history = _get_depletion_history_cyclic(case, n_cycles, irrad_h_per_cycle, cooldown_h_between, final_cooldown_days)
    all_cu_isotopes = list(CU_ATOMIC_MASS_G_MOL.keys())  # Cu61, Cu62, Cu63, Cu64, Cu65, Cu66, Cu67, Cu69, Cu70
    timestep_rows = []
    for step, cycle, phase, time_s, atoms in history:
        # Include all Cu isotopes at every step (use 0 when not present)
        for nuc in all_cu_isotopes:
            n_atoms = float(atoms.get(nuc, 0) or 0)
            try:
                mass_g = n_atoms * openmc.data.atomic_mass(nuc) / AVOGADRO
            except (KeyError, AttributeError, TypeError):
                mass_g = 0.0
            lam = get_decay_constant(nuc)
            activity_Bq = n_atoms * lam if lam and lam > 0 else 0.0
            timestep_rows.append({
                'step': step, 'cycle': cycle, 'phase': phase, 'time_s': time_s,
                'nuclide': nuc, 'atoms': n_atoms, 'mass_g': mass_g, 'activity_Bq': activity_Bq,
            })
        # Other nuclides (Zn, Ni, etc.) when present
        for nuc, n_atoms in atoms.items():
            n_atoms = float(n_atoms or 0)
            if nuc in all_cu_isotopes or n_atoms <= 0:
                continue
            try:
                mass_g = n_atoms * openmc.data.atomic_mass(nuc) / AVOGADRO
            except (KeyError, AttributeError, TypeError):
                mass_g = 0.0
            lam = get_decay_constant(nuc)
            activity_Bq = n_atoms * lam if lam and lam > 0 else 0.0
            timestep_rows.append({
                'step': step, 'cycle': cycle, 'phase': phase, 'time_s': time_s,
                'nuclide': nuc, 'atoms': n_atoms, 'mass_g': mass_g, 'activity_Bq': activity_Bq,
            })
    if not timestep_rows:
        return
    path = os.path.join(output_dir, 'test_depletion_timesteps.csv')
    pd.DataFrame(timestep_rows).to_csv(path, index=False, float_format='%.6g')
    print(f"  Saved: {path} ({len(timestep_rows)} rows, all Cu isotopes + Zn/Ni at each Bateman step)")


def print_copper_mass_steps(act, title='Copper mass (Cu63+64+65+67) at each step'):
    """Print copper mass [g] at each irradiation and cooldown step from act (from compute_activities_cyclic or compute_activities)."""
    steps = act.get('copper_mass_steps') or []
    if not steps:
        return
    print(f"  {title}")
    print(f"    {'cycle':<6} {'phase':<14} {'cu63_g':>10} {'cu64_g':>10} {'cu65_g':>10} {'cu67_g':>10} {'total_cu_g':>12}")
    for s in steps:
        cycle = s.get('cycle', '')
        phase = s.get('phase', '')
        print(f"    {str(cycle):<6} {str(phase):<14} {s.get('cu63_g', 0):10.4e} {s.get('cu64_g', 0):10.4e} {s.get('cu65_g', 0):10.4e} {s.get('cu67_g', 0):10.4e} {s.get('total_cu_g', 0):12.4e}")
    total = act.get('copper_mass_final_g')
    if total is not None:
        print(f"    Final total copper: {total:.4e} g (Cu63: {act.get('cu63_g', 0):.4e}, Cu64: {act.get('cu64_g', 0):.4e}, Cu65: {act.get('cu65_g', 0):.4e}, Cu67: {act.get('cu67_g', 0):.4e} g)")


def compute_activities(case, irrad_hours, cooldown_days):
    """Bateman irradiation + decay with RR scaling for long irradiations.
    Tracks copper mass (Cu63+64+65+67) at end of irradiation and after cooldown."""
    irrad_s = irrad_hours * 3600
    cooldown_s = cooldown_days * 86400
    init = case['initial_atoms']
    rr0 = case['reaction_rates']
    copper_mass_steps = []

    if irrad_hours >= IRRAD_MULTI_STEP_THRESHOLD_H:
        n_steps = max(52, min(365, int(irrad_s / 86400)))
        dt_s = irrad_s / n_steps
        atoms = {k: float(v) for k, v in init.items()}
        for _ in range(n_steps):
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
    step_eoi = {'phase': 'irrad', 'hours': irrad_hours, 'days': cooldown_days}
    step_eoi.update(copper_mass_from_atoms(atoms_eoi))
    copper_mass_steps.append(step_eoi)

    atoms_final = apply_single_decay_step(atoms_eoi, cooldown_s)
    step_final = {'phase': 'cooldown', 'hours': irrad_hours, 'days': cooldown_days}
    step_final.update(copper_mass_from_atoms(atoms_final))
    copper_mass_steps.append(step_final)

    lam_cu64 = get_decay_constant('Cu64')
    lam_cu67 = get_decay_constant('Cu67')
    lam_cu61 = get_decay_constant('Cu61')
    lam_cu62 = get_decay_constant('Cu62')
    lam_cu66 = get_decay_constant('Cu66')
    lam_cu69 = get_decay_constant('Cu69')
    lam_cu70 = get_decay_constant('Cu70')
    lam_zn65 = get_decay_constant('Zn65')
    lam_zn63 = get_decay_constant('Zn63')
    lam_zn69m = get_decay_constant('Zn69m')
    cu64_atoms = atoms_final.get('Cu64', 0)
    cu67_atoms = atoms_final.get('Cu67', 0)
    cu61_atoms = atoms_final.get('Cu61', 0)
    cu62_atoms = atoms_final.get('Cu62', 0)
    cu66_atoms = atoms_final.get('Cu66', 0)
    cu69_atoms = atoms_final.get('Cu69', 0)
    cu70_atoms = atoms_final.get('Cu70', 0)
    zn65_atoms = atoms_final.get('Zn65', 0)
    zn63_atoms = atoms_final.get('Zn63', 0)
    zn69m_atoms = atoms_final.get('Zn69m', 0)
    total_cu = cu64_atoms + cu67_atoms + cu61_atoms + cu62_atoms + cu66_atoms + cu69_atoms + cu70_atoms
    cu64_activity = cu64_atoms * lam_cu64
    cu67_activity = cu67_atoms * lam_cu67
    cu61_activity = cu61_atoms * lam_cu61
    cu62_activity = cu62_atoms * lam_cu62
    cu66_activity = cu66_atoms * lam_cu66
    cu69_activity = cu69_atoms * lam_cu69
    cu70_activity = cu70_atoms * lam_cu70
    total_cu_activity = cu64_activity + cu67_activity + cu61_activity + cu62_activity + cu66_activity + cu69_activity + cu70_activity
    cu64_atomic_purity = cu64_atoms / total_cu if total_cu > 0 else 0
    cu67_atomic_purity = cu67_atoms / total_cu if total_cu > 0 else 0
    cu64_radionuclide_purity = cu64_activity / total_cu_activity if total_cu_activity > 0 else 0
    cu67_radionuclide_purity = cu67_activity / total_cu_activity if total_cu_activity > 0 else 0
    copper_final = copper_mass_from_atoms(atoms_final)

    return {
        'cu64_mCi': cu64_atoms * lam_cu64 / 3.7e7,
        'cu67_mCi': cu67_atoms * lam_cu67 / 3.7e7,
        'zn65_mCi': zn65_atoms * lam_zn65 / 3.7e7,
        'zn63_Bq': zn63_atoms * lam_zn63,
        'zn69m_Bq': zn69m_atoms * lam_zn69m,
        'cu64_Bq': cu64_activity,
        'cu67_Bq': cu67_activity,
        'cu61_Bq': cu61_activity,
        'cu62_Bq': cu62_activity,
        'cu66_Bq': cu66_activity,
        'cu69_Bq': cu69_activity,
        'cu70_Bq': cu70_activity,
        'zn65_Bq': zn65_atoms * lam_zn65,
        'cu64_atomic_purity': cu64_atomic_purity,
        'cu67_atomic_purity': cu67_atomic_purity,
        'cu64_radionuclide_purity': cu64_radionuclide_purity,
        'cu67_radionuclide_purity': cu67_radionuclide_purity,
        'copper_mass_steps': copper_mass_steps,
        'copper_mass_final_g': copper_final['total_cu_g'],
        'cu63_g': copper_final['cu63_g'],
        'cu64_g': copper_final['cu64_g'],
        'cu65_g': copper_final['cu65_g'],
        'cu67_g': copper_final['cu67_g'],
    }


def _wall_bi210_po210_Bq(R_bi210_per_s, irrad_s, cooldown_s, n_cycles=1):
    """Wall Bi-210 -> Po-210 chain: return (Bi-210 Bq, Po-210 Bq) after irrad_s + cooldown_s, repeated n_cycles. Single-nuclide wall (Pb, Bi208) use activity_Bq_after_irrad_cooldown instead."""
    if R_bi210_per_s <= 0:
        return 0.0, 0.0
    lam_bi = get_decay_constant('Bi210')
    lam_po = get_decay_constant('Po210')
    N_bi, N_po = 0.0, 0.0
    dt = 60.0
    for _ in range(n_cycles):
        for duration_s, R in [(irrad_s, R_bi210_per_s), (cooldown_s, 0.0)]:
            t = 0.0
            while t < duration_s:
                step = min(dt, duration_s - t)
                N_bi = max(0.0, N_bi + (R - lam_bi * N_bi) * step)
                N_po = max(0.0, N_po + (lam_bi * N_bi - lam_po * N_po) * step)
                t += step
    return lam_bi * N_bi, lam_po * N_po


def build_summary_dataframes(case, zn64_enrichment=0.4917):
    """Build Cu and Zn summary DataFrames for test case. Includes lead (Pb-209/Pb-203) or bismuth (Bi-210) activation.
    Numerator for dose ratio = Zn+Cu inside shielding + wall (lead/bismuth) activity; denominator = Zn+Cu unshielded."""
    print_dose_constants_uSv_hr_per_MBq_1m()
    cu_rows = []
    zn_rows = []
    volume_cm3 = case['outer_volume_cm3']
    zn_density = case['zn_density_g_cm3']
    mass_g = case['zn_mass_g']
    mass_kg = mass_g / 1000.0
    zn_cost_per_kg = get_zn64_enrichment_cost_per_kg(zn64_enrichment)
    pb209_rate = case.get('pb209_production_rate_per_s', 0.0) or 0.0
    pb203_rate = case.get('pb203_production_rate_per_s', 0.0) or 0.0
    pb205_rate = case.get('pb205_production_rate_per_s', 0.0) or 0.0
    bi210_rate = case.get('bi210_production_rate_per_s', 0.0) or 0.0
    bi208_rate = case.get('bi208_production_rate_per_s', 0.0) or 0.0

    # If shielded run but wall production rates are zero, re-read from statepoint (fixes tally indexing issues)
    if not PIG_NO_WALL and (pb209_rate + pb203_rate + pb205_rate + bi210_rate + bi208_rate) <= 0.0:
        sp_path = case.get('sp_file')
        if sp_path and os.path.isfile(sp_path):
            try:
                sp = openmc.StatePoint(sp_path)
                lead_rr = get_lead_reaction_rates(sp, SOURCE_STRENGTH)
                pb209_rate = float(lead_rr.get('Pb208 (n,gamma) Pb209', 0.0) or 0.0)
                pb203_rate = float(lead_rr.get('Pb204 (n,2n) Pb203', 0.0) or 0.0)
                pb205_rate = float(lead_rr.get('Pb204 (n,gamma) Pb205', 0.0) or 0.0)
                bismuth_rr = get_bismuth_reaction_rates(sp, SOURCE_STRENGTH)
                bi210_rate = float(bismuth_rr.get('Bi209 (n,gamma) Bi210', 0.0) or 0.0)
                bi208_rate = float(bismuth_rr.get('Bi209 (n,2n) Bi208', 0.0) or 0.0)
                sp.close()
            except Exception:
                pass

    from zn_waste import get_dose_coeff, CAVITY_ISOTOPES

    for irrad_h in IRRADIATION_HOURS:
        for cool_d in COOLDOWN_DAYS:
            act = compute_activities(case, irrad_h, cool_d)
            # Cavity dose from all CAVITY_ISOTOPES
            _activity_Bq = {nuc: act.get(_act_key(nuc)) or 0 for nuc in CAVITY_ISOTOPES}
            _dose_summary = cavity_dose_at_1m_and_surface(_activity_Bq, lead_wall_cm=0 if PIG_NO_WALL else PIG_WALL_THICKNESS_CM)
            wall_Bq = {}
            if PIG_NO_WALL:
                _total_dose_1m = _dose_summary['dose_at_1m_uSv_hr']
            else:
                # Wall dose: all wall isotopes we have rates for, using zn_waste get_dose_coeff
                if PIG_WALL_MATERIAL == 'bismuth':
                    bi210_Bq, po210_Bq = _wall_bi210_po210_Bq(bi210_rate, irrad_h * 3600, cool_d * 86400, n_cycles=1)
                    wall_Bq = {
                        'Bi208': activity_Bq_after_irrad_cooldown(bi208_rate, 'Bi208', irrad_h * 3600, cool_d * 86400),
                        'Bi210': bi210_Bq,
                        'Po210': po210_Bq,
                    }
                else:
                    wall_Bq = {
                        nuc: activity_Bq_after_irrad_cooldown(rate, nuc, irrad_h * 3600, cool_d * 86400)
                        for nuc, rate in [('Pb203', pb203_rate), ('Pb205', pb205_rate), ('Pb209', pb209_rate)]
                    }
                _wall_dose_1m = sum((get_dose_coeff(nuc) or 0) * (bq / 1e6) for nuc, bq in wall_Bq.items())
                _total_dose_1m = _dose_summary['dose_at_1m_uSv_hr'] + _wall_dose_1m

            total_lead_Bq = wall_Bq.get('Pb203', 0) + wall_Bq.get('Pb205', 0) + wall_Bq.get('Pb209', 0)
            pb209_Bq = wall_Bq.get('Pb209', 0)
            pb203_Bq = wall_Bq.get('Pb203', 0)
            bi210_Bq = wall_Bq.get('Bi210', 0)
            po210_Bq = wall_Bq.get('Po210', 0)
            total_bismuth_wall_Bq = bi210_Bq + po210_Bq
            total_activity_wall_zn65_Bq = total_lead_Bq + total_bismuth_wall_Bq + act['zn65_Bq']
            cu_rows.append({
                'dir_name': case['dir_name'],
                'zn64_enrichment': zn64_enrichment,
                'zn_feedstock_cost': zn_cost_per_kg * mass_kg,
                'use_zn67': False,
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
                'cu64_atomic_purity': act['cu64_atomic_purity'],
                'cu67_atomic_purity': act['cu67_atomic_purity'],
                'cu64_radionuclide_purity': act['cu64_radionuclide_purity'],
                'cu67_radionuclide_purity': act['cu67_radionuclide_purity'],
                'copper_mass_final_g': act.get('copper_mass_final_g'),
                'cu63_g': act.get('cu63_g'),
                'cu64_g': act.get('cu64_g'),
                'cu65_g': act.get('cu65_g'),
                'cu67_g': act.get('cu67_g'),
                'pb209_production_rate_per_s': pb209_rate,
                'pb203_production_rate_per_s': pb203_rate,
                'pb209_activity_Bq': pb209_Bq,
                'pb203_activity_Bq': pb203_Bq,
                'total_lead_activity_Bq': total_lead_Bq,
                'bi210_production_rate_per_s': bi210_rate,
                'bi208_production_rate_per_s': bi208_rate,
                'bi210_activity_Bq': bi210_Bq,
                'po210_activity_Bq': po210_Bq,
                'total_bismuth_wall_activity_Bq': total_bismuth_wall_Bq,
                'total_activity_wall_zn65_Bq': total_activity_wall_zn65_Bq,
                'total_dose_1m_uSv_hr': _total_dose_1m,
            })
            zn_rows.append({
                'dir_name': case['dir_name'],
                'zn64_enrichment': zn64_enrichment,
                'zn_feedstock_cost': zn_cost_per_kg * mass_kg,
                'use_zn67': False,
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
                'pb209_production_rate_per_s': pb209_rate,
                'pb203_production_rate_per_s': pb203_rate,
                'pb209_activity_Bq': pb209_Bq,
                'pb203_activity_Bq': pb203_Bq,
                'total_lead_activity_Bq': total_lead_Bq,
                'bi210_production_rate_per_s': bi210_rate,
                'bi208_production_rate_per_s': bi208_rate,
                'bi210_activity_Bq': bi210_Bq,
                'po210_activity_Bq': po210_Bq,
                'total_bismuth_wall_activity_Bq': total_bismuth_wall_Bq,
            })

    return pd.DataFrame(cu_rows), pd.DataFrame(zn_rows)


def plot_activity_vs_variables(cu_df, output_dir):
    """Activity vs irradiation, cooldown, enrichment."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    df = cu_df.copy()
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    enrichments = sorted(df['zn64_enrichment'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))

    ax = axes[0]
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['cooldown_days'] == 1) & (df['irrad_hours'] <= 72)]
        sub = sub.groupby('irrad_hours').mean(numeric_only=True).reset_index().sort_values('irrad_hours')
        if not sub.empty:
            ax.semilogy(sub['irrad_hours'], sub['cu64_mCi'], 'o-', color=colors[i], label=f'Zn64 {_enrich_label(enrich)}')
    ax.set_xlabel('Irradiation [h]')
    ax.set_ylabel('Cu-64 Activity [mCi]')
    ax.set_title('Activity vs Irradiation Time (cooldown: 1 d)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['irrad_hours'] == 8)]
        sub = sub.sort_values('cooldown_days')
        if not sub.empty:
            ax.semilogy(sub['cooldown_days'], sub['cu64_mCi'], 'o-', color=colors[i], label=f'Zn64 {_enrich_label(enrich)}')
    ax.set_xlabel('Cooldown [days]')
    ax.set_ylabel('Cu-64 Activity [mCi]')
    ax.set_title('Activity vs Cooldown Time (irradiation: 8 h)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sub = df[(df['cooldown_days'] == 1) & (df['irrad_hours'] == 8)]
    sub = sub.groupby('zn64_enrichment').mean(numeric_only=True).reset_index().sort_values('zn64_enrichment')
    if not sub.empty:
        ax.semilogy(sub['zn64_enrichment'] * 100, sub['cu64_mCi'], 'o-', color='steelblue')
    ax.set_xlabel('Zn-64 Enrichment [%]')
    ax.set_ylabel('Cu-64 Activity [mCi]')
    ax.set_title('Activity vs Enrichment (irradiation: 8 h, cooldown: 1 d)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_vs_variables.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/activity_vs_variables.png")


def plot_purity_vs_variables(cu_df, output_dir):
    """Purity vs irradiation, cooldown, enrichment."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    df = cu_df.copy()
    df['zn64_enrichment'] = df['zn64_enrichment'].apply(_norm_enrich)
    enrichments = sorted(df['zn64_enrichment'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(enrichments)))

    ax = axes[0]
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['cooldown_days'] == 1) & (df['irrad_hours'] <= 72)]
        sub = sub.groupby('irrad_hours').mean(numeric_only=True).reset_index().sort_values('irrad_hours')
        if not sub.empty:
            ax.plot(sub['irrad_hours'], sub['cu64_radionuclide_purity'] * 100, 'o-', color=colors[i], label=f'Zn64 {_enrich_label(enrich)}')
    ax.set_xlabel('Irradiation [h]')
    ax.set_ylabel('Cu-64 Radionuclide Purity [%]')
    ax.set_title('(a) vs Irradiation (1d cooldown)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, enrich in enumerate(enrichments):
        sub = df[(df['zn64_enrichment'] == enrich) & (df['irrad_hours'] == 8)]
        sub = sub.sort_values('cooldown_days')
        if not sub.empty:
            ax.plot(sub['cooldown_days'], sub['cu64_radionuclide_purity'] * 100, 'o-', color=colors[i], label=f'Zn64 {_enrich_label(enrich)}')
    ax.set_xlabel('Cooldown [days]')
    ax.set_ylabel('Cu-64 Radionuclide Purity [%]')
    ax.set_title('(b) vs Cooldown (8h irrad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sub = df[(df['cooldown_days'] == 1) & (df['irrad_hours'] == 8)]
    sub = sub.groupby('zn64_enrichment').mean(numeric_only=True).reset_index().sort_values('zn64_enrichment')
    if not sub.empty:
        ax.plot(sub['zn64_enrichment'] * 100, sub['cu64_radionuclide_purity'] * 100, 'o-', color='steelblue')
    ax.set_xlabel('Zn-64 Enrichment [%]')
    ax.set_ylabel('Cu-64 Radionuclide Purity [%]')
    ax.set_title('(c) vs Enrichment (8h irrad, 1d cool)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'purity_vs_variables.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/purity_vs_variables.png")


def plot_cu64_activity_vs_irradiation_8h(cu_df, output_dir, dose_mCi=4.0):
    """Cu-64 activity [mCi] vs irradiation time (0–8 h, EOI). Uses same cu_df/CSV as other activity plots."""
    if cu_df is None or cu_df.empty or 'cu64_mCi' not in cu_df.columns:
        return
    df = cu_df[(cu_df['cooldown_days'] == 0) & (cu_df['irrad_hours'] <= 8)].sort_values('irrad_hours').drop_duplicates(subset=['irrad_hours'], keep='first')
    if df.empty:
        return
    irrad_h = np.asarray(df['irrad_hours'], dtype=float)
    cu64_mCi = np.asarray(df['cu64_mCi'], dtype=float)
    if irrad_h.size and irrad_h[0] > 0:
        irrad_h = np.concatenate([[0.0], irrad_h])
        cu64_mCi = np.concatenate([[0.0], cu64_mCi])

    lam = get_decay_constant('Cu64') or (np.log(2.0) / (12.701 * 3600.0))
    if not (np.isfinite(lam) and lam > 0):
        lam = np.log(2.0) / (12.701 * 3600.0)
    # Bateman saturation: A(t) = A_sat * (1 - exp(-lam*t)); fit A_sat from last point
    i_last = min(np.searchsorted(irrad_h, 8.0), len(irrad_h) - 1) if irrad_h.size else 0
    t_last_s = irrad_h[i_last] * 3600.0
    exp_last = np.exp(-lam * t_last_s) if t_last_s > 0 else 1.0
    A_sat = (float(cu64_mCi[i_last]) / (1.0 - exp_last)) if exp_last < 1.0 else (float(cu64_mCi[i_last]) * 1.2)
    t_smooth = np.linspace(0.0, 8.0, 200)
    activity_smooth = A_sat * (1.0 - np.exp(-lam * t_smooth * 3600.0))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t_smooth, activity_smooth, '-', lw=2, color='steelblue')
    ax.scatter(irrad_h, cu64_mCi, s=70, zorder=3, color='darkblue', edgecolors='white', linewidths=1.5)
    peak_mCi = float(np.max(cu64_mCi)) if cu64_mCi.size else 0.0
    if peak_mCi > 0 and dose_mCi > 0:
        ax.axhline(peak_mCi, color='black', linestyle='--', lw=1, alpha=0.8, zorder=2)
        ax.text(8.0, peak_mCi, f'  {peak_mCi / dose_mCi:.1f} doses ({dose_mCi:.0f} mCi/dose)', fontsize=10, va='bottom', ha='right')

    # Total Cu after 8 h from same row (copper_mass_final_g or sum of cu*_g)
    row_8 = df[df['irrad_hours'] == df['irrad_hours'].max()]
    total_cu_g = None
    if not row_8.empty:
        r = row_8.iloc[0]
        total_cu_g = r.get('copper_mass_final_g')
        if total_cu_g is None or not np.isfinite(total_cu_g):
            total_cu_g = sum((r.get(k) or 0) for k in r.index if isinstance(k, str) and k.startswith('cu') and k.endswith('_g') and k[2:-2].isdigit())
    if total_cu_g is not None and np.isfinite(total_cu_g) and total_cu_g >= 1e-12:
        ug = total_cu_g * 1e6
        mass_str = f'{ug:.2f} µg' if ug >= 1.0 else (f'{ug:.2e} µg' if ug >= 1e-3 else f'{total_cu_g * 1e9:.2e} ng')
    else:
        mass_str = '0 µg'
    ax.text(0.5, 0.98, f'Total Cu after 8 h: {mass_str}', transform=ax.transAxes, fontsize=11, va='top', ha='center')

    ax.set_xlabel('Irradiation time [h]', fontsize=12)
    ax.set_ylabel('Cu-64 activity [mCi]', fontsize=12)
    ax.set_title('Cu-64 activity vs irradiation time (0–8 h, EOI)')
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, max(np.max(activity_smooth) * 1.08, peak_mCi * 1.08, 0.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'cu64_activity_vs_irradiation_8h.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


DOSE_LIMIT_OCCUPATIONAL_ANNUAL_MSV = 50.0      # mSv/yr (5 rem, 10 CFR 20.1201)
DOSE_LIMIT_OCCUPATIONAL_ANNUAL_UVSV = 50_000.0 # µSv/yr

# Reference lines for dose-vs-distance / dose-vs-time plots (comparison only).
CHEST_CT_DOSE_UVSV = 8000.0          # 8 mSv, Chest CT scan
OCCUPATIONAL_ANNUAL_DOSE_UVSV = 50_000.0  # 50 mSv/yr occupational limit (10 CFR 20.1201)


def _act_key(nuclide):
    """Act dict key for Bq, e.g. Zn65 -> zn65_Bq."""
    return nuclide[0].lower() + nuclide[1:] + '_Bq'


def activity_mbq_from_act(act):
    """Dict nuclide -> activity_MBq for cavity isotopes (from compute_activities* return)."""
    from zn_waste import CAVITY_ISOTOPES
    return {nuc: (act.get(_act_key(nuc)) or 0) / 1e6 for nuc in CAVITY_ISOTOPES}


def unshielded_dose_1m_uSv_hr_from_mbq(activity_mbq_dict):
    """Dose rate (µSv/hr) at 1 m: Σ get_dose_coeff(nuc)*MBq (untabled nuclides = 0)."""
    from zn_waste import get_dose_coeff
    total = 0.0
    for nuc, mbq in (activity_mbq_dict or {}).items():
        c = get_dose_coeff(nuc)
        if c is not None:
            total += c * mbq
    return total


def _get_zn63_dose_const():
    from zn_waste import get_dose_coeff
    return get_dose_coeff('Zn63')


_dose_constants_printed = False


def print_dose_constants_uSv_hr_per_MBq_1m():
    """Print dose constants (µSv/hr at 1 m per 1 MBq) once per run."""
    global _dose_constants_printed
    if _dose_constants_printed:
        return
    _dose_constants_printed = True
    from zn_waste import get_dose_coeff, CAVITY_ISOTOPES
    print("  Dose constants (µSv/hr per 1 MBq @ 1 m; zn_waste get_dose_coeff):")
    for nuc in CAVITY_ISOTOPES:
        c = get_dose_coeff(nuc)
        if c is not None:
            print(f"    {nuc}: {c:.4f}")
    for key in ('Pb203', 'Pb209', 'Bi210', 'Po210'):
        c = get_dose_coeff(key)
        if c is not None and c != 0:
            print(f"    Wall {key}: {c:.4e}")


def print_final_dose_constants_uSv_hr_per_MBq_1m():
    """Print dose constants at end of run."""
    from zn_waste import get_dose_coeff, CAVITY_ISOTOPES
    print("\n" + "=" * 70)
    print("  FINAL: Dose constants (µSv/hr per 1 MBq @ 1 m)")
    print("=" * 70)
    print("  Cavity (Zn + Cu + Ni):")
    for nuc in CAVITY_ISOTOPES:
        c = get_dose_coeff(nuc)
        if c is not None:
            print(f"    {nuc}: {c:.4f}")
    print("  Wall (lead/bismuth); zn_waste get_dose_coeff:")
    for key in ('Pb203', 'Pb209', 'Bi210', 'Po210'):
        c = get_dose_coeff(key)
        if c is not None:
            print(f"    {key}: {c:.4e} µSv/hr per 1 MBq @ 1 m")
    print("=" * 70 + "\n")


def cavity_dose_at_1m_and_surface(activity_Bq_dict, lead_wall_cm=None, pig_outer_radius_cm=None, boron_thickness_cm=None, wall_material=None):
    """Cavity (Zn+Cu+…) dose at 1 m and at pig surface from activity dict. Uses GAMMA_ENERGIES per-isotope HVL (Pb or Bi). Returns dict with dose_at_1m_uSv_hr, dose_at_surface_uSv_hr, unshielded_dose_at_1m_uSv_hr, unshielded_dose_at_surface_uSv_hr, shield_factor."""
    from zn_waste import GAMMA_ENERGIES, get_dose_coeff, get_gamma_hvl
    HVL_BY_ISOTOPE = {iso: get_gamma_hvl(iso) for iso in GAMMA_ENERGIES}
    lead_wall_cm = PIG_WALL_THICKNESS_CM if lead_wall_cm is None else lead_wall_cm
    pig_outer_radius_cm = PIG_OUTER_RADIUS_CM if pig_outer_radius_cm is None else pig_outer_radius_cm
    boron_thickness_cm = BORON_THICKNESS_CM if boron_thickness_cm is None else boron_thickness_cm
    wall_material = PIG_WALL_MATERIAL if wall_material is None else wall_material
    activity_mbq = {nuc: (bq or 0) / 1e6 for nuc, bq in (activity_Bq_dict or {}).items()}
    unshielded_1m = unshielded_dose_1m_uSv_hr_from_mbq(activity_mbq)
    if unshielded_1m <= 0:
        r_cm = max(0.1, pig_outer_radius_cm)
        return {'dose_at_1m_uSv_hr': 0.0, 'dose_at_surface_uSv_hr': 0.0, 'unshielded_dose_at_1m_uSv_hr': 0.0, 'unshielded_dose_at_surface_uSv_hr': 0.0, 'shield_factor': 1.0}
    dose_at_1m = 0.0
    for nuc, mbq in activity_mbq.items():
        if mbq <= 0:
            continue
        coeff = get_dose_coeff(nuc) or 0.0
        if coeff <= 0:
            continue
        hvl = HVL_BY_ISOTOPE.get(nuc, get_gamma_hvl(nuc))
        hvl_wall = hvl['HVL_Bi_cm'] if wall_material == 'bismuth' else hvl['HVL_Pb_cm']
        hvl_boron = hvl.get('HVL_boron_cm', HVL_BORON_CM)
        hvl_quartz = hvl.get('HVL_quartz_cm', HVL_QUARTZ_CM)
        sf = 2.0 ** (lead_wall_cm / hvl_wall)
        if PIG_QUARTZ_THICKNESS_CM > 0 and hvl_quartz > 0:
            sf *= 2.0 ** (PIG_QUARTZ_THICKNESS_CM / hvl_quartz)
        if boron_thickness_cm > 0:
            sf *= 2.0 ** (boron_thickness_cm / hvl_boron)
        dose_at_1m += (coeff * mbq) / sf
    shield_factor = unshielded_1m / dose_at_1m if dose_at_1m > 0 else 1.0
    r_cm = max(0.1, pig_outer_radius_cm)
    inv_r2 = (100.0 / r_cm) ** 2
    return {
        'dose_at_1m_uSv_hr': dose_at_1m,
        'dose_at_surface_uSv_hr': dose_at_1m * inv_r2,
        'unshielded_dose_at_1m_uSv_hr': unshielded_1m,
        'unshielded_dose_at_surface_uSv_hr': unshielded_1m * inv_r2,
        'shield_factor': shield_factor,
    }



def add_axis_labels(plot_path, output_dir='.', extent=(-100, 100, -100, 100), use_water_bath=False):
    """Add axis labels, legend, and plasma source marker to geometry plot."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    full_path = os.path.join(output_dir, f'{plot_path}.png')
    if not os.path.exists(full_path):
        return
    img = plt.imread(full_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    xlo, xhi, ylo, yhi = extent
    ax.imshow(img, extent=[xlo, xhi, ylo, yhi], origin='upper')
    ax.scatter(0, 0, c='hotpink', s=120, marker='o', edgecolors='black', linewidths=2, zorder=10, label='D-T source (0,0,0)')
    step = 20 if abs(xhi - xlo) <= 220 else 50
    ax.set_xticks(np.arange(xlo, xhi + 1, step))
    ax.set_yticks(np.arange(ylo, yhi + 1, step))
    ax.set_xlabel('X Position [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position [cm]', fontsize=12, fontweight='bold')
    ax.grid(True, which='major', alpha=0.5, linestyle='-')
    ax.grid(True, which='minor', alpha=0.3, linestyle='--')
    wall_color = MATERIAL_COLORS.get(PIG_WALL_MATERIAL, MATERIAL_COLORS['wall'])
    if PIG_GEOMETRY_TYPE == 'bismuth_quartz':
        # Bismuth and quartz: cylinder + top and bottom lids (end caps)
        has_quartz_tube = PIG_QUARTZ_OUTER_RADIUS_CM is not None and PIG_QUARTZ_OUTER_RADIUS_CM > PIG_INNER_RADIUS_CM + 1e-6
        quartz_legend = f'Quartz tube + lids (r={PIG_INNER_RADIUS_CM:.2f}–{PIG_QUARTZ_OUTER_RADIUS_CM:.2f} cm)' if has_quartz_tube else 'Quartz (tube + lids)'
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='hotpink', markeredgecolor='black', markersize=10, label='D-T source (0,0,0)'),
            Patch(facecolor=MATERIAL_COLORS['vacuum'], edgecolor='black', label='Vacuum (purple: in chamber + outside)'),
            Patch(facecolor=MATERIAL_COLORS['zn_target'], edgecolor='black', label=f'Zn ({TARGET_ZN_MASS_G/1000:.2f} kg, r={PIG_INNER_RADIUS_CM:.2f} cm)'),
            Patch(facecolor=MATERIAL_COLORS['quartz'], edgecolor='black', label=quartz_legend),
            Patch(facecolor=wall_color, edgecolor='black', label=f'Bismuth cylinder + top/bottom lids (r≤{PIG_OUTER_RADIUS_CM:.2f} cm)'),
        ]
    else:
        wall_legend = f'Bismuth cylinder + top/bottom lids (r={PIG_OUTER_RADIUS_CM:.2f} cm)' if PIG_WALL_MATERIAL == 'bismuth' else f'Lead cylinder + top/bottom lids (r={PIG_OUTER_RADIUS_CM:.2f} cm)'
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='hotpink', markeredgecolor='black', markersize=10, label='D-T source (0,0,0)'),
            Patch(facecolor=MATERIAL_COLORS['vacuum'], edgecolor='black', label='Vacuum'),
            Patch(facecolor=wall_color, edgecolor='black', label=wall_legend),
            Patch(facecolor=MATERIAL_COLORS['zn_target'], edgecolor='black', label=f'Zn (cavity r={PIG_INNER_RADIUS_CM:.2f} cm)'),
        ]
    if BORON_THICKNESS_CM > 0:
        legend_elements.append(Patch(facecolor=MATERIAL_COLORS['boron'], edgecolor='black', label=f'Boron ({BORON_THICKNESS_CM:.0f} cm outside pig)'))
    if use_water_bath:
        legend_elements.append(Patch(facecolor=MATERIAL_COLORS['water'], edgecolor='black', label=f'Water (r={WATER_RADIUS_CM:.0f} cm)'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    fig.tight_layout()
    labeled_path = os.path.join(output_dir, 'geometry_xz_labeled.png')
    fig.savefig(labeled_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {labeled_path}")


def plot_dose_vs_distance(output_dir, case, irrad_hours=8, cooldown_hours=0, exposure_hours=1.0):
    """Dose vs distance from pig center: unshielded Zn, Cu, Ni; wall (lead or bismuth + daughters); total (cavity shielded by isotope + wall; wall + quartz + boron if present)."""
    is_bismuth = (PIG_WALL_MATERIAL == 'bismuth')
    has_wall = not (PIG_NO_WALL or PIG_WALL_THICKNESS_CM <= 0)
    wall_name = 'bismuth' if is_bismuth else ('lead' if has_wall else 'none')
    if PIG_QUARTZ_THICKNESS_CM > 0:
        wall_name = wall_name + '+quartz'

    rows = build_all_products_table(case, 1, irrad_hours, cooldown_hours, final_cooldown_days=0)
    # Classify by isotope: cavity (Zn, Cu, Ni) vs wall (Pb/Bi + daughters)
    cavity_prefixes = ('Zn', 'Cu', 'Ni')
    if is_bismuth:
        wall_nuclides = ('Bi', 'Po210')
    else:
        wall_nuclides = ('Pb', 'Tl', 'Pt')

    def is_cavity(r):
        n = r.get('nuclide', '') or ''
        return any(n.startswith(p) for p in cavity_prefixes)

    def is_wall(r):
        n = r.get('nuclide', '') or ''
        if is_bismuth:
            return n.startswith('Bi') or n == 'Po210'
        return any(n.startswith(p) for p in wall_nuclides)

    cavity_rows = [r for r in rows if is_cavity(r)]
    wall_rows = [r for r in rows if is_wall(r)]

    def sum_dose_1m(grp):
        return sum((r.get('dose_rate_uSv_hr_1m') or 0) for r in grp)

    def sum_shielded_1m(grp):
        return sum((r.get('dose_rate_uSv_hr_1m_shielded') or r.get('dose_rate_uSv_hr_1m') or 0) for r in grp)

    # Per-element unshielded (for plot curves)
    dose_zn_1m = sum_dose_1m([r for r in cavity_rows if r.get('nuclide', '').startswith('Zn')])
    dose_cu_1m = sum_dose_1m([r for r in cavity_rows if r.get('nuclide', '').startswith('Cu')])
    dose_ni_1m = sum_dose_1m([r for r in cavity_rows if r.get('nuclide', '').startswith('Ni')])
    dose_wall_1m = sum_dose_1m(wall_rows)
    unshielded_cavity_1m = sum_dose_1m(cavity_rows)
    # Cavity shielded: per-isotope (each row has its own HVL/shield_factor in build_all_products_table)
    cavity_shielded_1m = sum_shielded_1m(cavity_rows)

    r_cm = np.linspace(max(10.0, PIG_OUTER_RADIUS_CM + BORON_THICKNESS_CM + 1), 150.0, 200)
    r_cm = np.maximum(r_cm, 0.1)
    inv_r2 = (100.0 / r_cm) ** 2

    rate_zn = dose_zn_1m * inv_r2
    rate_cu = dose_cu_1m * inv_r2
    rate_ni = dose_ni_1m * inv_r2
    rate_wall = dose_wall_1m * inv_r2
    rate_cavity_sh = cavity_shielded_1m * inv_r2
    rate_total = rate_cavity_sh + rate_wall

    EXPOSURE_HOURS = exposure_hours
    d_zn = rate_zn * EXPOSURE_HOURS
    d_cu = rate_cu * EXPOSURE_HOURS
    d_ni = rate_ni * EXPOSURE_HOURS
    d_wall = rate_wall * EXPOSURE_HOURS
    d_cavity_sh = rate_cavity_sh * EXPOSURE_HOURS
    d_total = rate_total * EXPOSURE_HOURS

    DOSE_YMIN, DOSE_YMAX = 0.5, 100_000.0
    WORKER_1M_RADIUS_CM = 100.0

    fig, ax = plt.subplots(figsize=(10, 6))
    zn_mass_kg = case.get('zn_mass_g', 0) / 1000.0
    ax.semilogy(r_cm, np.maximum(d_zn, DOSE_YMIN), 'C0-', lw=2, label=f'Zn unshielded ({zn_mass_kg:.2f} kg Zn)')
    ax.semilogy(r_cm, np.maximum(d_cu, DOSE_YMIN), 'C3-', lw=2, label='Cu unshielded')
    if dose_ni_1m > 0:
        ax.semilogy(r_cm, np.maximum(d_ni, DOSE_YMIN), 'C4-', lw=1.5, label='Ni unshielded')
    if dose_wall_1m > 0:
        ax.semilogy(r_cm, np.maximum(d_wall, DOSE_YMIN), 'C1-', lw=2, label=f'Wall ({wall_name} + daughters)')
    ax.semilogy(r_cm, np.maximum(d_cavity_sh, DOSE_YMIN), 'C2--', lw=1.2, label='Chamber (Zn+Cu+Ni shielded)')
    ax.semilogy(r_cm, np.maximum(d_total, DOSE_YMIN), 'C2-', lw=2.5, label='Total (Zn+Cu+Ni' + (' behind shield + wall' if has_wall else '') + ')')
    ax.axvline(WORKER_1M_RADIUS_CM, color='#2d2d2d', linestyle='-', lw=1.2, label='1 m')
    ax.axhline(CHEST_CT_DOSE_UVSV, color='#555555', linestyle=':', lw=1.2, label='Chest CT (8 mSv)')
    ax.axhline(OCCUPATIONAL_ANNUAL_DOSE_UVSV, color='black', linestyle=':', lw=1.0)
    x_center = 0.5 * (r_cm.min() + r_cm.max())
    ax.text(x_center, OCCUPATIONAL_ANNUAL_DOSE_UVSV, '50 mSv/yr occupational', fontsize=8, color='black', va='bottom', ha='center')
    ax.text(x_center, CHEST_CT_DOSE_UVSV, 'Chest CT (8 mSv)', fontsize=8, color='#555555', va='bottom', ha='center')

    ax.set_ylim(DOSE_YMIN, DOSE_YMAX)
    ax.set_xlabel('Distance from pig center [cm]', fontsize=12)
    ax.set_ylabel(f'Dose (µSv) for {EXPOSURE_HOURS} h exposure', fontsize=12)
    cool_str = f'{cooldown_hours} h cool' if cooldown_hours else 'EOI'
    ax.set_title(f'Dose vs distance — {irrad_hours} h irrad ({cool_str}); Cu retained; {wall_name}; {EXPOSURE_HOURS} h exposure')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'dose_vs_distance_from_pig.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_dose_vs_time(output_dir, case, irrad_hours=8, time_max_hours=24 * 30):
    """Dose rate (µSv/hr) at 1 m vs time after EOI: Zn, Cu, Ni (unshielded), Wall, Cavity (shielded), Total.
    Math: at each time t, cavity atoms = EOI state decayed by t (apply_single_decay_step only; no production).
    Dose = sum over isotopes of (N_i * lambda_i * dose_coeff_i / 1e6), so cavity dose must be non-increasing.
    We enforce that with cumulative minimum. Writes dose_vs_time_1m.csv."""
    is_bismuth = (PIG_WALL_MATERIAL == 'bismuth')
    has_wall = not (PIG_NO_WALL or PIG_WALL_THICKNESS_CM <= 0)
    wall_name = 'bismuth' if is_bismuth else ('lead' if has_wall else 'none')
    if PIG_QUARTZ_THICKNESS_CM > 0:
        wall_name = wall_name + '+quartz'

    cavity_prefixes = ('Zn', 'Cu', 'Ni')
    if is_bismuth:
        wall_nuclides = ('Bi', 'Po210')
    else:
        wall_nuclides = ('Pb', 'Tl', 'Pt')

    def is_cavity(r):
        n = r.get('nuclide', '') or ''
        return any(n.startswith(p) for p in cavity_prefixes)

    def is_wall(r):
        n = r.get('nuclide', '') or ''
        if is_bismuth:
            return n.startswith('Bi') or n == 'Po210'
        return any(n.startswith(p) for p in wall_nuclides)

    def sum_dose_1m(grp):
        return sum((r.get('dose_rate_uSv_hr_1m') or 0) for r in grp)

    def sum_shielded_1m(grp):
        return sum((r.get('dose_rate_uSv_hr_1m_shielded') or r.get('dose_rate_uSv_hr_1m') or 0) for r in grp)

    # Time grid: many points from 50 s out to 72 h (or time_max_hours) for smooth decay curve (log x-axis)
    # Cu-64 decays by ~70 h; Cu-67 (61.8 h) dominates after that — dose is non-increasing (decay only)
    t_min_h = 50.0 / 3600
    t_max_h = min(72.0, time_max_hours)
    n_points = 120
    time_hours = np.unique(np.logspace(np.log10(t_min_h), np.log10(max(t_min_h + 1e-6, t_max_h)), n_points))
    time_hours = time_hours[time_hours <= time_max_hours]

    dose_zn = []
    dose_cu = []
    dose_ni = []
    dose_wall = []
    cavity_shielded_list = []
    for t_h in time_hours:
        rows = build_all_products_table(case, 1, irrad_hours, t_h, final_cooldown_days=0)
        cavity_rows = [r for r in rows if is_cavity(r)]
        wall_rows = [r for r in rows if is_wall(r)]
        dose_zn.append(sum_dose_1m([r for r in cavity_rows if r.get('nuclide', '').startswith('Zn')]))
        dose_cu.append(sum_dose_1m([r for r in cavity_rows if r.get('nuclide', '').startswith('Cu')]))
        dose_ni.append(sum_dose_1m([r for r in cavity_rows if r.get('nuclide', '').startswith('Ni')]))
        dose_wall.append(sum_dose_1m(wall_rows))
        cavity_shielded_list.append(sum_shielded_1m(cavity_rows))

    dose_zn = np.array(dose_zn)
    dose_cu = np.array(dose_cu)
    dose_ni = np.array(dose_ni)
    dose_wall = np.array(dose_wall)
    cavity_shielded = np.array(cavity_shielded_list)
    # No production after EOI: cavity doses must be non-increasing in time (decay only)
    dose_zn = np.minimum.accumulate(dose_zn)
    dose_cu = np.minimum.accumulate(dose_cu)
    dose_ni = np.minimum.accumulate(dose_ni)
    cavity_shielded = np.minimum.accumulate(cavity_shielded)
    cavity_unshielded = dose_zn + dose_cu + dose_ni
    dose_total = cavity_shielded + dose_wall

    # CSV: many points
    csv_path = os.path.join(output_dir, 'dose_vs_time_1m.csv')
    time_days = time_hours / 24.0
    df = pd.DataFrame({
        'time_h': time_hours,
        'time_days': time_days,
        'dose_zn_1m_uSv_hr': dose_zn,
        'dose_cu_1m_uSv_hr': dose_cu,
        'dose_ni_1m_uSv_hr': dose_ni,
        'dose_wall_1m_uSv_hr': dose_wall,
        'dose_cavity_unshielded_1m_uSv_hr': cavity_unshielded,
        'dose_cavity_shielded_1m_uSv_hr': cavity_shielded,
        'dose_total_1m_uSv_hr': dose_total,
    })
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path} ({len(df)} points)")

    # Plot: dose rate at 1 m vs time (hours)
    fig, ax = plt.subplots(figsize=(10, 6))
    DOSE_YMIN = 1e-3
    DOSE_YMAX = 1e6
    ax.semilogy(time_hours, np.maximum(dose_zn, DOSE_YMIN), 'C0-', lw=2, label='Zn (unshielded)')
    ax.semilogy(time_hours, np.maximum(dose_cu, DOSE_YMIN), 'C3-', lw=2, label='Cu (unshielded)')
    if np.any(dose_ni > 0):
        ax.semilogy(time_hours, np.maximum(dose_ni, DOSE_YMIN), 'C4-', lw=1.5, label='Ni (unshielded)')
    if np.any(dose_wall > 0):
        ax.semilogy(time_hours, np.maximum(dose_wall, DOSE_YMIN), 'C1-', lw=2, label=f'Wall ({wall_name} + daughters)')
    ax.semilogy(time_hours, np.maximum(cavity_shielded, DOSE_YMIN), 'C5--', lw=1.5, label='Cavity (Zn+Cu+Ni shielded)')
    ax.semilogy(time_hours, np.maximum(dose_total, DOSE_YMIN), 'C2-', lw=2.5, label='Total')
    ax.set_ylim(DOSE_YMIN, DOSE_YMAX)
    ax.set_xscale('log')
    ax.set_xlim(time_hours.min(), time_hours.max())
    # Tick labels at nice times (subset so axis is readable)
    tick_hours = np.array([50/3600, 1/60, 5/60, 15/60, 30/60, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0])
    tick_hours = tick_hours[(tick_hours >= time_hours.min()) & (tick_hours <= time_hours.max())]
    def _time_label(h):
        if h < 1/60:
            return f'{int(round(h * 3600))} s'
        if h < 1:
            return f'{int(round(h * 60))} min'
        return f'{h:g} h'
    ax.xaxis.set_major_locator(plt.FixedLocator(tick_hours))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _time_label(x)))
    ax.set_xlabel('Time after end of irradiation', fontsize=12)
    ax.set_ylabel('Dose rate at 1 m (µSv/hr)', fontsize=12)
    ax.set_title(f'Dose rate at 1 m vs time — 1×({irrad_hours} h irrad + cooldown); Cu retained; {wall_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'dose_vs_time_1m.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def run_single_case(output_dir, config):
    """Run one pig case: apply config, build model, run OpenMC, analyze, and write all results to output_dir."""
    apply_pig_config(config)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    if PIG_NO_WALL:
        wall_label = 'No wall (Zn only, unshielded)'
    else:
        wall_label = 'Bismuth pig' if PIG_WALL_MATERIAL == 'bismuth' else 'Lead pig'
    print("=" * 60)
    print("OpenMC Point Source + Zn Pig Test")
    print("=" * 60)
    print(f"  Case: {config['label']}")
    print(f"  Source: D-T point at origin, {SOURCE_ENERGY_MEV} MeV, {FUSION_POWER_W} W")
    print(f"  Source strength: {SOURCE_STRENGTH:.2e} n/s")
    zn_dens = calculate_enriched_zn_density(0.4917)
    cavity_vol_cm3 = np.pi * PIG_INNER_RADIUS_CM**2 * PIG_INTERIOR_HEIGHT_CM
    zn_mass_cavity_g = TARGET_ZN_MASS_G  # 0.67 kg in every config; Zn fills only part of cavity, rest void
    h_zn_cm = zn_mass_cavity_g / zn_dens / (np.pi * PIG_INNER_RADIUS_CM**2)
    if PIG_GEOMETRY_TYPE == 'bismuth_quartz':
        print(f"  {wall_label}: quartz cavity r={PIG_INNER_RADIUS_CM:.2f} cm, tube h={PIG_INTERIOR_HEIGHT_CM:.2f} cm; Zn fill h={h_zn_cm:.2f} cm ({TARGET_ZN_MASS_G/1000:.2f} kg), rest void")
    else:
        if PIG_NO_WALL:
            print(f"  {wall_label}: cavity {PIG_INNER_DIAMETER_IN} x {PIG_INTERIOR_HEIGHT_IN} in (r={PIG_INNER_RADIUS_CM:.2f} cm)")
        else:
            print(f"  {wall_label}: exterior {PIG_OUTER_DIAMETER_IN} x {PIG_OUTER_HEIGHT_IN} in, interior {PIG_INNER_DIAMETER_IN} x {PIG_INTERIOR_HEIGHT_IN} in, wall {PIG_WALL_THICKNESS_IN} in")
        print(f"  Zn (cavity fill): {zn_mass_cavity_g:.0f} g ({TARGET_ZN_MASS_G/1000:.2f} kg), fill h={h_zn_cm:.2f} cm (r={PIG_INNER_RADIUS_CM:.2f} cm), rest void")
    print(f"  Distance source to pig: {SOURCE_TO_CYLINDER_CM} cm (6 in)")
    print(f"  Particles: {PARTICLES}, Batches: {BATCHES}")
    print()

    use_water_bath = USE_WATER_BATH_GEOMETRY
    materials = create_materials(include_water=use_water_bath)
    if use_water_bath:
        geometry, plots, cells_with_fill = create_geometry_water_bath(materials)
        print(f"  Geometry: 40 cm vacuum bubble + 100 cm water + void, vacuum BC")
    else:
        geometry, plots, cells_with_fill = create_geometry(materials)
    source = create_source()
    tallies = create_tallies(cells_with_fill)

    model = openmc.Model()
    model.materials = openmc.Materials(materials)
    model.geometry = geometry
    model.settings = openmc.Settings()
    model.settings.run_mode = 'fixed source'
    model.settings.particles = PARTICLES
    model.settings.batches = BATCHES
    model.settings.source = source
    model.settings.statepoint = {'batches': [BATCHES]}
    model.settings.output = {'summary': True, 'tallies': True}
    model.tallies = tallies
    model.plots = plots

    # Volume calculation (use finite box; void region gives infinite bounding_box)
    all_cells = list(model.geometry.get_all_cells().values())
    bb = model.geometry.root_universe.bounding_box
    try:
        ll, ur = bb.lower_left, bb.upper_right
        if not np.all(np.isfinite(ll)) or not np.all(np.isfinite(ur)):
            raise ValueError("infinite bounds")
    except (ValueError, TypeError):
        extent = WATER_RADIUS_CM + 10.0
        ll = [-extent, -extent, -extent]
        ur = [extent, extent, extent]
    else:
        ll, ur = list(ll), list(ur)
    cell_vc = openmc.VolumeCalculation(
        domains=all_cells,
        samples=1_000_000,
        lower_left=ll,
        upper_right=ur
    )
    model.settings.volume_calculations = [cell_vc]

    cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        model.export_to_xml()
        os.makedirs('images', exist_ok=True)  # OpenMC plot_geometry expects images/ in cwd
        openmc.plot_geometry()
        model.calculate_volumes()
        vol_path = os.path.join(output_dir, 'volume_1.h5')
        if os.path.isfile(vol_path):
            cell_vc.load_results(vol_path)
            model.geometry.add_volume_information(cell_vc)
            print("Cell volumes [cm³]:")
            material_masses_g = {}
            for c in cells_with_fill:
                v = getattr(c, 'volume', None)
                print(f"  {c.name}: {v:.2f}" if v and v > 0 else f"  {c.name}: (not set)")
                mat = getattr(c, 'fill', None)
                rho = getattr(mat, 'density', None) if mat is not None else None
                if v and v > 0 and rho is not None:
                    m_g = float(v) * float(rho)
                    key = getattr(mat, 'name', getattr(c, 'name', 'unknown'))
                    material_masses_g[key] = material_masses_g.get(key, 0.0) + m_g
            if material_masses_g:
                print("Material masses in pig [g]:")
                for key in ['zn_target', 'quartz', 'bismuth', 'lead', 'boron']:
                    if key in material_masses_g:
                        print(f"  {key}: {material_masses_g[key]:.2f} g")
        model.run()
    finally:
        os.chdir(cwd)

    statepoint_path = os.path.join(output_dir, f'statepoint.{BATCHES}.h5')
    if not os.path.exists(statepoint_path):
        print("Statepoint not found; skipping flux plots and analysis")
        return

    # Use design-based volume for Zn (cavity fill = TARGET_ZN_MASS_G); OpenMC MC volume is unreliable for tiny cell in large domain
    zn_dens = calculate_enriched_zn_density(0.4917)
    volume_cm3 = TARGET_ZN_MASS_G / zn_dens  # 0.67 kg Zn in every config; rest of cavity is void

    # Tally analysis: reaction rates, Bateman, CSVs, plots, Energy Solutions quote (incl. waste/dose)
    print("\n" + "=" * 60)
    print("Tally analysis: Cu-64/Cu-67/Zn-65 production")
    print("=" * 60)
    try:
        case = analyze_case_test(statepoint_path, volume_cm3, zn64_enrichment=0.4917)
        cu_df, zn_df = build_summary_dataframes(case, zn64_enrichment=0.4917)
        cu_df.to_csv(os.path.join(output_dir, 'cu_summary.csv'), index=False)
        zn_df.to_csv(os.path.join(output_dir, 'zn_summary.csv'), index=False)
        print(f"  Saved: {output_dir}/cu_summary.csv ({len(cu_df)} rows)")
        print(f"  Saved: {output_dir}/zn_summary.csv ({len(zn_df)} rows)")
        # Copper mass summary from CSV at 8 h irrad, 0 d cooldown (all Cu isotopes present in CSV)
        cu_row_8 = cu_df[(cu_df['irrad_hours'] == 8) & (cu_df['cooldown_days'] == 0)]
        if not cu_row_8.empty:
            r8 = cu_row_8.iloc[0]
            total_cu_g = r8.get('copper_mass_final_g')
            if total_cu_g is not None:
                total_cu_g = float(np.nan_to_num(total_cu_g, nan=0.0, posinf=0.0, neginf=0.0))
                cu_cols = sorted([k for k in r8.index if isinstance(k, str) and k.startswith('cu') and k.endswith('_g') and k[2:-2].isdigit()], key=lambda k: int(k[2:-2]))
                def _safe_mg(r, key):
                    v = r.get(key, 0)
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        v = 0.0
                    return float(v) * 1000.0
                parts = [f"Cu-{k[2:-2]}: {_safe_mg(r8, k):.3f} mg" for k in cu_cols]
                print("Copper mass after 8 h irradiation (EOI):")
                print(f"  Total Cu: {total_cu_g*1000:.3f} mg" + (f" ({', '.join(parts)})" if parts else ""))
        # Store no-shielding Zn-65 activity at EOI (10×8 h irrad) for reference (0.67 kg Zn)
        if config.get('folder_name') == 'no_wall_0p67kg_zn':
            act_ref = compute_activities_cyclic(case, 10, 8, 0, final_cooldown_days=0, remove_cu_after_each_irrad=True)
            ref_MBq = act_ref['zn65_Bq'] / 1e6
            with open(os.path.join(output_dir, f'zn65_activity_{TARGET_ZN_MASS_G/1000:.2f}kg_reference_MBq.txt'), 'w', encoding='utf-8') as f:
                f.write(str(ref_MBq))
        write_all_products_csv(case, output_dir, n_cycles=1, irrad_h_per_cycle=8, cooldown_h_between=0, final_cooldown_days=0)
        write_depletion_timesteps_csv(case, output_dir, n_cycles=1, irrad_h_per_cycle=8, cooldown_h_between=0, final_cooldown_days=0)
        plot_activity_vs_variables(cu_df, output_dir)
        plot_purity_vs_variables(cu_df, output_dir)
        plot_cu64_activity_vs_irradiation_8h(cu_df, output_dir, dose_mCi=4.0)
        plot_dose_vs_distance(output_dir, case, irrad_hours=8, cooldown_hours=0, exposure_hours=1.0)
        plot_dose_vs_time(output_dir, case, irrad_hours=8, time_max_hours=24 * 30)
        # Bar plot: all radioactive isotopes (MBq + dose/hr @ 1 m), 8 h irrad, no cooldown
        plot_radioisotopes_bar(case, output_dir, irrad_hours=8, cooldown_hours=0)
    except Exception as e:
        print(f"  Analysis error: {e}")
        import traceback
        traceback.print_exc()

    sp = openmc.StatePoint(statepoint_path)
    first_tally = sp.get_tally(name=f'{cells_with_fill[0].name}_spectra')
    energy_filt = first_tally.filters[1]
    energy_bins = np.asarray(energy_filt.bins)
    if energy_bins.ndim == 2:
        energy_edges_eV = np.concatenate([energy_bins[:, 0], [energy_bins[-1, 1]]])
    else:
        energy_edges_eV = energy_bins
    lethargy_bin_width = np.log(energy_edges_eV[1:] / energy_edges_eV[:-1])

    flux_spectra_by_cell = {}
    cell_volumes = {}
    for c in cells_with_fill:
        try:
            cell_tally = sp.get_tally(name=f'{c.name}_spectra')
        except LookupError:
            continue
        openmc_flux = cell_tally.mean.flatten()
        vol = c.volume
        if vol is None or vol <= 0:
            continue
        flux = (openmc_flux / vol) * SOURCE_STRENGTH
        flux_spectra_by_cell[c.name] = flux
        cell_volumes[c.name] = float(vol)
    sp.close()

    # Aggregate by material: one curve per material (e.g. total lead or bismuth, not top/bottom/sides)
    by_material = {}
    for name, flux in flux_spectra_by_cell.items():
        vol = cell_volumes[name]
        if name not in by_material:
            by_material[name] = []
        by_material[name].append((flux, vol))
    norm_flux_by_material = {}
    for mat, pairs in by_material.items():
        total_vol = sum(v for _, v in pairs)
        if total_vol <= 0:
            continue
        flux_agg = sum(f * v for f, v in pairs) / total_vol
        norm_flux_by_material[mat] = flux_agg / lethargy_bin_width

    # Flux per lethargy: one line per material (lead, bismuth, zn_target, water, etc.)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for mat, norm_flux in norm_flux_by_material.items():
        color = MATERIAL_COLORS.get(mat, MATERIAL_COLORS.get('zn_target' if 'zn' in mat else 'wall', '#888'))
        ax1.step(energy_edges_eV[:-1], norm_flux, where='post', lw=2, color=color, label=mat)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Energy [eV]", fontsize=12)
    ax1.set_ylabel("Flux per unit lethargy [n/cm²-s]", fontsize=12)
    if PIG_NO_WALL:
        flux_pig_label = 'no shield (Zn only)'
    else:
        flux_pig_label = 'bismuth pig' if PIG_WALL_MATERIAL == 'bismuth' else 'lead pig'
    ax1.set_title(f"Flux per lethargy: point source + Zn {flux_pig_label}\n{SOURCE_ENERGY_MEV} MeV, {SOURCE_STRENGTH:.1e} n/s")
    ax1.legend()
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.set_xlim(1e-5, 2e7)
    ax1.set_ylim(1e1, 1e13)
    fig1.savefig(os.path.join(output_dir, 'flux_per_lethargy.png'), dpi=300)
    plt.close(fig1)
    print(f"  Saved {output_dir}/flux_per_lethargy.png")

    # Axis labels on geometry plot (hot pink source at 0,0)
    ext = (-125, 125, -125, 125) if USE_WATER_BATH_GEOMETRY else (-100, 100, -100, 100)
    add_axis_labels('images/geometry_xz', output_dir, extent=ext, use_water_bath=USE_WATER_BATH_GEOMETRY)

    print(f"\nCase complete. Outputs in {output_dir}/")


def plot_dose_rate_ratio_vs_time(output_dirs, configs, base_dir):
    """
    Plot dose rate ratio (total shielded / no-wall) vs time after EOI.
    5 wall cases vs 1 no-wall 0.67 kg Zn baseline (no_wall_0p67kg_zn). Data from dose_vs_time_1m.csv.
    """
    dir_by_folder = {c['folder_name']: d for d, c in zip(output_dirs, configs)}
    label_by_folder = {c['folder_name']: c.get('label', c['folder_name']) for c in configs}
    data = {}
    for folder_name, output_dir in dir_by_folder.items():
        csv_path = os.path.join(output_dir, 'dose_vs_time_1m.csv')
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            data[folder_name] = {'df': df, 'label': label_by_folder.get(folder_name, folder_name)}
        except Exception as e:
            print(f"  Warning: could not load {csv_path}: {e}")
            continue

    # Series: (label, denominator_baseline_key, numerator_case_key) for the 5 wall cases; all 0.67 kg Zn
    baseline_key = 'no_wall_0p67kg_zn'
    series = []
    for folder in COMBINED_SHIELDED_FOLDERS:
        if folder in data and data.get(baseline_key):
            series.append((data[folder]['label'] + f' / {TARGET_ZN_MASS_G/1000:.2f} kg no wall', baseline_key, folder))

    if not series:
        print("  Skip dose rate ratio vs time: no dose_vs_time_1m data or baselines")
        return

    out_dir = os.path.join(base_dir, 'comparison_shielding_effect')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, denom_key, num_key in series:
        denom_df = data[denom_key]['df']
        num_df = data[num_key]['df']
        # Merge on time_h (inner: keep only common times)
        m = pd.merge(
            num_df[['time_h', 'dose_total_1m_uSv_hr']].rename(columns={'dose_total_1m_uSv_hr': 'total_case'}),
            denom_df[['time_h', 'dose_total_1m_uSv_hr']].rename(columns={'dose_total_1m_uSv_hr': 'total_base'}),
            on='time_h',
            how='inner'
        )
        m = m[m['total_base'] > 0]
        m['ratio'] = m['total_case'] / m['total_base']
        # Y = % dose reduced vs no-wall (higher = better shielding); clip so 0–100%
        m['pct_reduced'] = np.clip((1.0 - m['ratio']) * 100.0, 0.0, 100.0)
        if not m.empty:
            ax.plot(m['time_h'], m['pct_reduced'], '-', lw=2, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Time after end of irradiation [h]', fontsize=12)
    ax.set_ylabel('Dose reduction (%)', fontsize=12)
    ax.set_title(f'Shielding effectiveness vs time — 5 wall cases vs 1 no-wall {TARGET_ZN_MASS_G/1000:.2f} kg Zn\nHigher % = more dose reduced; 0% = same as no wall', fontsize=11)
    ax.axhline(0.0, color='k', linestyle='--', alpha=0.6, label='Same as no wall')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    # Tick labels at nice times
    tick_hours = np.array([50/3600, 1/60, 5/60, 15/60, 30/60, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0])
    tick_hours = tick_hours[(tick_hours >= 50/3600) & (tick_hours <= 72.0)]
    def _time_label(h):
        if h < 1/60:
            return f'{int(round(h * 3600))} s'
        if h < 1:
            return f'{int(round(h * 60))} min'
        return f'{h:g} h'
    ax.xaxis.set_major_locator(plt.FixedLocator(tick_hours))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _time_label(x)))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dose_rate_ratio_vs_time.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_dir}/dose_rate_ratio_vs_time.png")


def plot_cu64_yield_ratio_comparison(output_dirs, configs, base_dir):
    """
    Plot Cu-64 activity ratios: 5 wall cases vs 1 no-wall 0.67 kg Zn baseline (no_wall_0p67kg_zn).
    Two plots: ratio vs irradiation time (fixed cooldown); ratio vs cooldown time (fixed irrad). Bar: Cu-64 doses at 8 h.
    """
    # Load cu_summary from each output_dir; key by folder_name
    data = {}
    for output_dir, config in zip(output_dirs, configs):
        folder_name = config['folder_name']
        csv_path = os.path.join(output_dir, 'cu_summary.csv')
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df['folder_name'] = folder_name
            data[folder_name] = {'df': df, 'label': config.get('label', folder_name)}
        except Exception as e:
            print(f"  Warning: could not load {csv_path}: {e}")
            continue

    baseline_067 = data.get('no_wall_0p67kg_zn')
    if baseline_067 is None:
        print("  Skip Cu-64 ratio comparison: no_wall_0p67kg_zn cu_summary not found")
        return

    # Series to plot: (numerator_label, denominator_key, numerator_key); all 0.67 kg Zn
    series = []
    for folder in COMBINED_SHIELDED_FOLDERS:
        if folder in data and folder != 'no_wall_0p67kg_zn':
            series.append((data[folder]['label'] + f' / {TARGET_ZN_MASS_G/1000:.2f} kg unshielded', 'no_wall_0p67kg_zn', folder))

    if not series:
        print("  Skip Cu-64 ratio comparison: no comparison series")
        return

    out_dir = os.path.join(base_dir, 'comparison_shielding_effect')
    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: Ratio vs irradiation time (fixed cooldown = 0)
    cooldown_fixed = 0
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for label, denom_key, num_key in series:
        denom = data[denom_key]['df']
        num = data[num_key]['df']
        sub_d = denom[denom['cooldown_days'] == cooldown_fixed][['irrad_hours', 'cu64_Bq']].drop_duplicates('irrad_hours').sort_values('irrad_hours')
        sub_n = num[num['cooldown_days'] == cooldown_fixed][['irrad_hours', 'cu64_Bq']].drop_duplicates('irrad_hours').sort_values('irrad_hours')
        m = sub_d.merge(sub_n, on='irrad_hours', suffixes=('_base', '_case'))
        m = m[m['cu64_Bq_base'] > 0]
        m['ratio'] = m['cu64_Bq_case'] / m['cu64_Bq_base']
        if not m.empty:
            ax1.plot(m['irrad_hours'], m['ratio'], 'o-', lw=2, label=label)
    ax1.set_xlabel('Irradiation time [h]', fontsize=12)
    ax1.set_ylabel('Cu-64 activity ratio (case / baseline)', fontsize=12)
    ax1.set_title(f'Cu-64 yield ratio vs irradiation time (cooldown = {cooldown_fixed} d)\nEffect of shielding on production', fontsize=12)
    ax1.axhline(1.0, color='k', linestyle='--', alpha=0.6)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, 'cu64_ratio_vs_irradiation_time.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Saved {out_dir}/cu64_ratio_vs_irradiation_time.png")

    # Plot 2: Ratio vs cooldown time (fixed irrad = 8 h)
    irrad_fixed = 8
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for label, denom_key, num_key in series:
        denom = data[denom_key]['df']
        num = data[num_key]['df']
        sub_d = denom[denom['irrad_hours'] == irrad_fixed][['cooldown_days', 'cu64_Bq']].drop_duplicates('cooldown_days').sort_values('cooldown_days')
        sub_n = num[num['irrad_hours'] == irrad_fixed][['cooldown_days', 'cu64_Bq']].drop_duplicates('cooldown_days').sort_values('cooldown_days')
        m = sub_d.merge(sub_n, on='cooldown_days', suffixes=('_base', '_case'))
        m = m[m['cu64_Bq_base'] > 0]
        m['ratio'] = m['cu64_Bq_case'] / m['cu64_Bq_base']
        if not m.empty:
            ax2.plot(m['cooldown_days'], m['ratio'], 'o-', lw=2, label=label)
    ax2.set_xlabel('Cooldown time [days]', fontsize=12)
    ax2.set_ylabel('Cu-64 activity ratio (case / baseline)', fontsize=12)
    ax2.set_title(f'Cu-64 yield ratio vs cooldown time (irradiation = {irrad_fixed} h)\nEffect of shielding on production', fontsize=12)
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.6)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, 'cu64_ratio_vs_cooldown_time.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved {out_dir}/cu64_ratio_vs_cooldown_time.png")

    # Bar chart: Cu-64 production (doses) after 8 h irradiation — 5 shielded cases + unshielded baseline (all 0.67 kg Zn)
    DOSE_MCI_PER_DOSE = 4.0  # mCi per dose
    irrad_bar, cooldown_bar = 8, 0
    bar_categories = [
        ('Thin lead', 'no_wall_0p67kg_zn', '16A_305_LC_thin'),
        ('Thick lead', 'no_wall_0p67kg_zn', '16A_308_LC_thick'),
        ('Thin bismuth', 'no_wall_0p67kg_zn', 'bismuth_pig_thin_1p03kg_zn'),
        ('Thick bismuth', 'no_wall_0p67kg_zn', 'bismuth_pig_1kg_zn'),
        ('Bismuth quartz', 'no_wall_0p67kg_zn', 'bismuth_quartz'),
    ]
    labels_short = [t[0] for t in bar_categories]
    unshielded_doses = []
    shielded_doses = []
    for _, denom_key, num_key in bar_categories:
        if denom_key not in data or num_key not in data:
            unshielded_doses.append(0.0)
            shielded_doses.append(0.0)
            continue
        denom_df = data[denom_key]['df']
        num_df = data[num_key]['df']
        sub_u = denom_df[(denom_df['irrad_hours'] == irrad_bar) & (denom_df['cooldown_days'] == cooldown_bar)]
        sub_s = num_df[(num_df['irrad_hours'] == irrad_bar) & (num_df['cooldown_days'] == cooldown_bar)]
        cu64_mCi_u = float(sub_u['cu64_mCi'].iloc[0]) if not sub_u.empty and 'cu64_mCi' in sub_u.columns else 0.0
        cu64_mCi_s = float(sub_s['cu64_mCi'].iloc[0]) if not sub_s.empty and 'cu64_mCi' in sub_s.columns else 0.0
        unshielded_doses.append(cu64_mCi_u / DOSE_MCI_PER_DOSE if DOSE_MCI_PER_DOSE > 0 else 0.0)
        shielded_doses.append(cu64_mCi_s / DOSE_MCI_PER_DOSE if DOSE_MCI_PER_DOSE > 0 else 0.0)
    x = np.arange(len(labels_short))
    width_back = 0.7   # unshielded bar behind (wider)
    width_front = 0.45 # shielded bar in front
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    bars_u = ax_bar.bar(x, unshielded_doses, width_back, label='Unshielded Zn (same mass)', color='lightgray', edgecolor='gray', zorder=1)
    bars_s = ax_bar.bar(x, shielded_doses, width_front, label='Shielded (lead/bismuth pig)', color='steelblue', edgecolor='navy', zorder=2)
    ax_bar.set_ylabel(f'Cu-64 production [doses] ({DOSE_MCI_PER_DOSE:.0f} mCi/dose)', fontsize=12)
    ax_bar.set_xlabel('Configuration', fontsize=12)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_short)
    ax_bar.set_title(f'Cu-64 production after 8 h (0 d cooldown) — 5 wall cases vs 1 no-wall {TARGET_ZN_MASS_G/1000:.2f} kg Zn', fontsize=12)
    ax_bar.legend(loc='upper right', fontsize=10)
    ax_bar.grid(True, axis='y', alpha=0.3)
    fig_bar.tight_layout()
    fig_bar.savefig(os.path.join(out_dir, 'cu64_doses_8h_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_bar)
    print(f"  Saved {out_dir}/cu64_doses_8h_bar.png")

# 5 wall cases (all 0.67 kg Zn); compared to 1 no-wall 0.67 kg Zn baseline
COMBINED_SHIELDED_FOLDERS = ['16A_305_LC_thin', '16A_308_LC_thick', 'bismuth_pig_1kg_zn', 'bismuth_pig_thin_1p03kg_zn', 'bismuth_quartz']
COMBINED_HEATMAP_LABELS = {
    '16A_305_LC_thin': 'lead_thin',
    '16A_308_LC_thick': 'lead_thick',
    'bismuth_pig_1kg_zn': 'bismuth_thick',
    'bismuth_pig_thin_1p03kg_zn': 'bismuth_thin',
    'bismuth_quartz': 'bismuth_quartz',
}
COMBINED_NONSHIELDED_FOLDERS = ['no_wall_0p67kg_zn']  # single baseline no-wall test

# Single no-wall baseline for ratio comparison (5 wall cases vs 1 no-wall 0.67 kg Zn)
SHIELDED_TO_BASELINE = {
    '16A_308_LC_thick': 'no_wall_0p67kg_zn',
    '16A_305_LC_thin': 'no_wall_0p67kg_zn',
    'bismuth_pig_1kg_zn': 'no_wall_0p67kg_zn',
    'bismuth_pig_thin_1p03kg_zn': 'no_wall_0p67kg_zn',
    'bismuth_quartz': 'no_wall_0p67kg_zn',
}

TEST_OUTPUT_ZIP_NAME = 'test_output.zip'
NO_SHIELDING_FOLDER = 'no_shielding'


def _is_cavity_nuclide(nuclide):
    n = nuclide or ''
    return n.startswith('Zn') or n.startswith('Cu') or n.startswith('Ni')


def _is_wall_nuclide(nuclide):
    n = nuclide or ''
    return n.startswith('Pb') or n.startswith('Bi') or n == 'Po210' or n.startswith('Tl') or n.startswith('Pt')


def _dose_sums_from_products_df(df):
    """From test_all_products DataFrame: cavity_shielded = sum dose_rate_uSv_hr_1m_shielded (Zn,Cu,Ni); wall = sum dose_rate_uSv_hr_1m (Pb,Bi,Po,Tl,Pt)."""
    cavity_shielded = 0.0
    wall_dose = 0.0
    for _, r in df.iterrows():
        nuc = r.get('nuclide', '') or ''
        d = r.get('dose_rate_uSv_hr_1m') or 0
        d_sh = r.get('dose_rate_uSv_hr_1m_shielded') or d
        if _is_cavity_nuclide(nuc):
            cavity_shielded += float(d_sh)
        elif _is_wall_nuclide(nuc):
            wall_dose += float(d)
    return cavity_shielded, wall_dose


def print_final_dose_comparison_table(output_dirs, configs):
    """Print table: 5 wall cases vs 1 no-wall 0.67 kg Zn baseline. After 8 h irrad (EOI, no cooldown): Cavity_sh, Wall, Total_sh, No-wall dose, ratio, % reduced."""
    dir_by_folder = {c['folder_name']: d for d, c in zip(output_dirs, configs)}
    label_by_folder = {c['folder_name']: c.get('label', c['folder_name']) for c in configs}

    # Load single no-wall baseline (0.67 kg Zn): total dose at 1 m from test_all_products.csv
    baselines = {}
    for no_wall in COMBINED_NONSHIELDED_FOLDERS:
        path = os.path.join(dir_by_folder.get(no_wall, ''), 'test_all_products.csv')
        if not os.path.isfile(path):
            baselines[no_wall] = None
            continue
        try:
            df = pd.read_csv(path)
            baselines[no_wall] = float(df['dose_rate_uSv_hr_1m'].fillna(0).sum())
        except Exception:
            baselines[no_wall] = None

    # Build rows for the 5 shielded cases
    rows = []
    for folder in COMBINED_SHIELDED_FOLDERS:
        path = os.path.join(dir_by_folder.get(folder, ''), 'test_all_products.csv')
        if not os.path.isfile(path):
            rows.append({'folder': folder, 'label': label_by_folder.get(folder, folder), 'cavity_shielded': None, 'wall_dose': None, 'total_shielded': None, 'baseline_folder': SHIELDED_TO_BASELINE.get(folder), 'baseline_dose': None, 'ratio': None})
            continue
        try:
            df = pd.read_csv(path)
            cavity_shielded, wall_dose = _dose_sums_from_products_df(df)
            total_shielded = cavity_shielded + wall_dose
        except Exception:
            cavity_shielded = wall_dose = total_shielded = None
        baseline_folder = SHIELDED_TO_BASELINE.get(folder)
        baseline_dose = baselines.get(baseline_folder) if baseline_folder else None
        ratio = (total_shielded / baseline_dose) if (total_shielded is not None and baseline_dose is not None and baseline_dose > 0) else None
        rows.append({
            'folder': folder,
            'label': label_by_folder.get(folder, folder),
            'cavity_shielded': cavity_shielded,
            'wall_dose': wall_dose,
            'total_shielded': total_shielded,
            'baseline_folder': baseline_folder,
            'baseline_dose': baseline_dose,
            'ratio': ratio,
        })

    # Print table (all values = dose rate at 1 m from coeff × activity at EOI, 8 h irrad, no cooldown)
    print("\n" + "=" * 108)
    print(f"  FINAL: Dose rate at 1 m — 5 wall cases vs 1 no-wall baseline ({TARGET_ZN_MASS_G/1000:.2f} kg Zn, 8 h irrad EOI)")
    print("=" * 108)
    print(f"  {'Case':<42} {'Cavity_sh':>12} {'Wall':>12} {'Total_sh':>12} {'No-wall':>12} {'Total/No-wall':>12} {'% reduced':>10}")
    print(f"  {'(label)':<42} {'(µSv/h@1m)':>12} {'(µSv/h@1m)':>12} {'(µSv/h@1m)':>12} {'(µSv/h@1m)':>12} {'(ratio)':>12} {'(shield)':>10}")
    print("  " + "-" * 108)
    for r in rows:
        label = (r['label'][:40] + '..') if len(r['label']) > 42 else r['label']
        cav = r['cavity_shielded']
        wall = r['wall_dose']
        tot = r['total_shielded']
        base = r['baseline_dose']
        ratio = r['ratio']
        base_name = (r['baseline_folder'] or '')[:10]
        pct_reduced = (1.0 - ratio) * 100.0 if (ratio is not None and base is not None and base > 0) else None
        if cav is not None and wall is not None and tot is not None and base is not None and ratio is not None:
            pct_str = f"{pct_reduced:>9.1f}%" if pct_reduced is not None else "—"
            print(f"  {label:<42} {cav:>12.4f} {wall:>12.4f} {tot:>12.4f} {base:>12.4f} {ratio:>12.4f} {pct_str:>10}  [{base_name}]")
        else:
            print(f"  {label:<42} {'—':>12} {'—':>12} {'—':>12} {'—':>12} {'—':>12} {'—':>10}")
    print("  " + "-" * 108)
    print("  Formulas (dose rate = coeff × activity_MBq at 1 m; activities at EOI, 8 h irrad):")
    print("    Cavity_sh = Σ (coeff_i × activity_MBq_i / shield_factor_i)  for Zn, Cu, Ni")
    print("    Wall      = Σ (coeff_j × activity_MBq_j)                    for Pb, Bi, Po, Tl, Pt (unshielded)")
    print("    Total_sh  = Cavity_sh + Wall")
    print("    No-wall   = Σ (coeff_k × activity_MBq_k)  from single no-wall run (0.67 kg Zn)")
    print("    Ratio     = Total_sh / No-wall")
    print("    % reduced = (1 - Ratio) × 100  (shielding effectiveness vs no-wall)")
    print(f"    Baseline  no_wall_0p67kg_zn ({TARGET_ZN_MASS_G/1000:.2f} kg Zn)")
    print("=" * 108 + "\n")


# Short names for scatter-plot labels (same order as COMBINED_SHIELDED_FOLDERS)
SHIELDED_SHORT_LABELS = {
    '16A_305_LC_thin': 'Thin lead',
    '16A_308_LC_thick': 'Thick lead',
    'bismuth_pig_1kg_zn': 'Thick bismuth',
    'bismuth_pig_thin_1p03kg_zn': 'Thin bismuth',
    'bismuth_quartz': 'Bismuth quartz',
}

# Acceptable dose rate for 1 h handling: 50 mSv/h = 50,000 µSv/h
ACCEPTABLE_DOSE_RATE_UVSV_HR_HANDLING = 50_000.0


def plot_dose_vs_cu64_scatter(output_dirs, configs, base_dir):
    """
    Scatter: X = Total_sh (µSv/h at 1 m), Y = Cu-64 (mCi) at EOI. One point per of the 5 wall cases (all 0.67 kg Zn).
    Best = top-left. Vertical line at acceptable dose for 1 h handling.
    """
    dir_by_folder = {c['folder_name']: d for d, c in zip(output_dirs, configs)}
    out_dir = os.path.join(base_dir, 'comparison_shielding_effect')
    os.makedirs(out_dir, exist_ok=True)

    irrad_eoi, cooldown_eoi = 8, 0
    total_sh_list = []
    cu64_mCi_list = []
    labels_list = []

    for folder in COMBINED_SHIELDED_FOLDERS:
        output_dir = dir_by_folder.get(folder)
        if not output_dir:
            continue
        # Total_sh from test_all_products.csv
        products_path = os.path.join(output_dir, 'test_all_products.csv')
        if not os.path.isfile(products_path):
            continue
        try:
            df = pd.read_csv(products_path)
            cavity_sh, wall_dose = _dose_sums_from_products_df(df)
            total_sh = (cavity_sh or 0) + (wall_dose or 0)
        except Exception:
            continue
        # Cu-64 at EOI from cu_summary.csv
        cu_path = os.path.join(output_dir, 'cu_summary.csv')
        if not os.path.isfile(cu_path):
            continue
        try:
            cudf = pd.read_csv(cu_path)
            sub = cudf[(cudf['irrad_hours'] == irrad_eoi) & (cudf['cooldown_days'] == cooldown_eoi)]
            if sub.empty or 'cu64_mCi' not in sub.columns:
                continue
            cu64_mCi = float(sub['cu64_mCi'].iloc[0])
        except Exception:
            continue
        total_sh_list.append(total_sh)
        cu64_mCi_list.append(cu64_mCi)
        labels_list.append(SHIELDED_SHORT_LABELS.get(folder, folder))

    if not total_sh_list:
        print("  Skip dose vs Cu-64 scatter: no shielded case data")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(total_sh_list, cu64_mCi_list, s=120, zorder=3, color='steelblue', edgecolor='navy', linewidths=1.5)
    for i, lbl in enumerate(labels_list):
        ax.annotate(lbl, (total_sh_list[i], cu64_mCi_list[i]), xytext=(6, 6), textcoords='offset points',
                    fontsize=9, fontweight='bold', ha='left', va='bottom')
    ax.axvline(ACCEPTABLE_DOSE_RATE_UVSV_HR_HANDLING, color='green', linestyle='--', linewidth=2, alpha=0.8,
               label=f'OK for 1 h handling (<{ACCEPTABLE_DOSE_RATE_UVSV_HR_HANDLING/1000:.0f} mSv/h)')
    ax.set_xlabel('Total dose rate at 1 m (µSv/h) — shielded', fontsize=12)
    ax.set_ylabel('Cu-64 at EOI (mCi)', fontsize=12)
    ax.set_title('Shielding trade-off: dose rate vs Cu-64 production (8 h irrad, 0 d cooldown)\nBest = top-left (low dose, high Cu-64); left of line OK for 1 h handling', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'dose_vs_cu64_scatter.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_path}")


def build_test_output_zip(output_dirs, configs, base_dir):
    """
    Build a single test_output.zip containing all test outputs. No-shielding case outputs
    no_wall_0p67kg_zn is placed under no_shielding/.
    """
    zip_path = os.path.join(base_dir, TEST_OUTPUT_ZIP_NAME)
    dir_by_folder = {config['folder_name']: output_dir for output_dir, config in zip(output_dirs, configs)}

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # No-shielding cases: all files under no_shielding/<folder_name>/
        for folder in COMBINED_NONSHIELDED_FOLDERS:
            src_dir = dir_by_folder.get(folder)
            if not src_dir or not os.path.isdir(src_dir):
                continue
            prefix = f"{NO_SHIELDING_FOLDER}/{folder}"
            for root, _, files in os.walk(src_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arc = os.path.join(prefix, os.path.relpath(full, src_dir))
                    zf.write(full, arc)
            print(f"  Added {NO_SHIELDING_FOLDER}/{folder}/ -> {TEST_OUTPUT_ZIP_NAME}")

        # All other cases: <folder_name>/ at top level of zip
        for config in configs:
            folder = config['folder_name']
            if folder in COMBINED_NONSHIELDED_FOLDERS:
                continue
            src_dir = dir_by_folder.get(folder)
            if not src_dir or not os.path.isdir(src_dir):
                continue
            for root, _, files in os.walk(src_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arc = os.path.join(folder, os.path.relpath(full, src_dir))
                    zf.write(full, arc)
            print(f"  Added {folder}/ -> {TEST_OUTPUT_ZIP_NAME}")

        # comparison_shielding_effect at top level of zip
        comp_src = os.path.join(base_dir, 'comparison_shielding_effect')
        if os.path.isdir(comp_src):
            for root, _, files in os.walk(comp_src):
                for f in files:
                    full = os.path.join(root, f)
                    arc = os.path.relpath(full, base_dir)
                    zf.write(full, arc)
            print(f"  Added comparison_shielding_effect/ -> {TEST_OUTPUT_ZIP_NAME}")

    print(f"Zipped: {zip_path}")


def main():
    """Run all pig cases (thick/thin lead, thick/thin bismuth, no wall); all 0.67 kg Zn fill, rest void. Then comparison plot."""
    base = os.getcwd()
    output_dirs = []
    for config in PIG_CONFIGS:
        output_dir = os.path.abspath(f"test_output_{config['folder_name']}")
        output_dirs.append(output_dir)
        run_single_case(output_dir, config)

    # Comparison plot: Cu-64 yield ratio vs irrad and vs cooldown
    plot_cu64_yield_ratio_comparison(output_dirs, PIG_CONFIGS, base)
    # Dose rate ratio (total / no-wall) vs time after EOI for the 5 wall cases
    plot_dose_rate_ratio_vs_time(output_dirs, PIG_CONFIGS, base)
    # Scatter: Total_sh vs Cu-64 at EOI — best = top-left; vertical line at acceptable dose for 1 h handling
    plot_dose_vs_cu64_scatter(output_dirs, PIG_CONFIGS, base)

    # Single zip: test_output.zip with no_shielding/ subfolder for no-wall cases
    build_test_output_zip(output_dirs, PIG_CONFIGS, base)

    # Table: 5 wall cases vs 1 no-wall 0.67 kg Zn — Cavity_sh, Wall, Total_sh, No-wall, ratio, % reduced
    print_final_dose_comparison_table(output_dirs, PIG_CONFIGS)

    print("\nDone. All outputs in:")
    print(f"  {os.path.join(base, TEST_OUTPUT_ZIP_NAME)}")
    for d in output_dirs:
        print(f"  {d}")

    print_final_dose_constants_uSv_hr_per_MBq_1m()


if __name__ == '__main__':
    main()
