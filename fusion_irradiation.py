"""
OpenMC fusion neutron irradiation: cylindrical target with configurable geometry.
"""

#urabot

import openmc
import numpy as np
import pandas as pd
import os
import shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
from utilities import (
    ZN64_ENRICHMENT_MAP,
    ZN67_ENRICHMENT_MAP,
    ZN64_RANGE,
    ZN67_RANGE,
    get_zn_fractions,
    get_zn67_fractions,
    calculate_enriched_zn_density,
    calculate_enriched_zn67_density,
    SOURCE_STRENGTH,
    FUSION_POWER_W,
    get_initial_atoms_from_statepoint,
    build_channel_rr_per_s,
    evolve_bateman_irradiation,
    evolve_bateman_irradiation_with_history,
    apply_single_decay_step,
    get_decay_constant,
    CHANNELS,
    get_initial_zn_atoms_fallback,
    get_material_density_from_statepoint,
)

# Material colors for geometry/flux plots
MATERIAL_COLORS = {
    'water': '#1f77b4', 'moderator': '#d62728', 'multi': '#2ca02c',
    'struct': '#ff7f0e', 'vacuum': '#9467bd', 'boron': '#e377c2',
    'inner_target': '#7f7f7f', 'outer_target': '#8c564b',
    'aluminum': '#c0c0c0', 'aluminum_cylinder': '#a0a0a0', 'concrete': '#808080',
}
# Cell name 'vacuum_cylinder' uses same color as 'vacuum' (flux-per-lethargy, radial inset)
MATERIAL_COLORS['vacuum_cylinder'] = MATERIAL_COLORS['vacuum']
# Flux plot: one curve per logical region (structure = one cell; vacuum = all vacuum cells merged)
STRUCT_CELL_NAMES = frozenset(('structure',))
VACUUM_CELL_NAMES = frozenset(('vacuum_cylinder', 'vacuum_box_void'))  # all vacuum regions -> one "vacuum" flux
VACUUM_RADIAL_LABELS = frozenset(('vacuum', 'vacuum_box_void'))  # radial plot: merge into one "vacuum" bar
# Structure material: HASTELLOY® C-276 alloy (UNS N10276); plot key and legend label
PLOT_MATERIAL_HASTELLOY = 'hastelloy_c276'
PLOT_STRUCT_LABEL = 'HASTELLOY C-276'
PLOT_VACUUM_LABEL = 'vacuum'
MATERIAL_COLORS[PLOT_MATERIAL_HASTELLOY] = MATERIAL_COLORS['struct']

# Complex geometry plot: distinct colors (hot pink, lime green, cyan for Al/steel/concrete)
COMPLEX_PLOT_COLORS = {
    'vacuum_box_void': MATERIAL_COLORS['vacuum'],   # purple: vacuum inside aluminum box, outside cylinder
    'structure': MATERIAL_COLORS['struct'],          # orange: HASTELLOY C-276 (one cell)
    'cylinder_core': '#e6f2ff',                      # very light blue (inner assembly container)
    'aluminum': '#ff69b4',                           # hot pink: Al 6061-T6 box
    'steel_cylinder': '#32cd32',                     # lime green: stainless steel sleeve
    'concrete': '#00bcd4',                           # cyan: concrete
}

# Complex geometry (vacuum box + Al + water pool; no concrete): all dimensions in cm
COMPLEX_VACUUM_BOX_HALF = 3.5 * 30.48 / 2       # 3.5 ft box, half side = 53.34
COMPLEX_AL_THICKNESS = 4 * 2.54                 # 4 in Al = 10.16
COMPLEX_AL_OUTER_HALF = COMPLEX_VACUUM_BOX_HALF + COMPLEX_AL_THICKNESS  # 55.88
COMPLEX_POOL_HEIGHT_CM = 15 * 30.48             # 15 ft = 457.2
COMPLEX_POOL_Z_HALF = COMPLEX_POOL_HEIGHT_CM / 2  # 228.6
COMPLEX_WATER_VOLUME_GAL = 19000
# Water box outer: 19,000 gal at 15 ft height -> base area ≈ 157366 cm², square half ≈ 198.5
COMPLEX_WATER_XY_HALF = (COMPLEX_WATER_VOLUME_GAL * 3785.41 / COMPLEX_POOL_HEIGHT_CM) ** 0.5 / 2  # ~198.5
COMPLEX_WATER_Z_HALF = COMPLEX_POOL_Z_HALF
# Steel sleeve (cylinder between vacuum chamber and water): 20 cm OD, 2 cm thick; from top of Zn to top of water
COMPLEX_AL_CYLINDER_OUTER_R = 10              # cm (20 cm OD)
COMPLEX_AL_CYLINDER_THICKNESS = 2            # cm
COMPLEX_AL_CYLINDER_INNER_R = COMPLEX_AL_CYLINDER_OUTER_R - COMPLEX_AL_CYLINDER_THICKNESS  # 8 cm (20 cm OD, 2 cm thick)


class FusionIrradiation:
    """OpenMC fusion neutron irradiation runner. Holds default config; run_simulation merges with job kwargs."""

    def __init__(self, inner_radius=5.0, target_height=100.0, source_neutron_energy=14.1,
                 particles=10000, batches=10, output_prefix='irrad_output', root_dir=None):
        self.inner_radius = inner_radius
        self.target_height = target_height
        self.source_neutron_energy = source_neutron_energy
        self.particles = particles
        self.batches = batches
        self.output_prefix = output_prefix
        self.root_dir = os.path.abspath(root_dir or os.getcwd())
        self.target_flux = SOURCE_STRENGTH / (2.0 * np.pi * inner_radius * target_height)

    def run_simulation(self, **kwargs):
        """Run one irradiation; kwargs override instance defaults. For parallel, pass full job dict."""
        merged = dict(
            inner_radius=self.inner_radius, target_height=self.target_height,
            source_neutron_energy=self.source_neutron_energy, particles=self.particles,
            batches=self.batches, original_dir=self.root_dir,
            target_flux=self.target_flux, neutron_energy=self.source_neutron_energy,
        )
        merged.update(kwargs)
        return run_irradiation_simulation(**merged)


def _set_zn64_enriched_material(mat, zn64_enrichment, label='outer'):
    """Set nuclide fractions and density for Zn-64 enriched material (density from utilities)."""
    e = float(zn64_enrichment)
    if not (ZN64_RANGE[0] <= e <= ZN64_RANGE[1]):
        raise ValueError(f"zn64_enrichment {zn64_enrichment} must be in [{ZN64_RANGE[0]}, {ZN64_RANGE[1]}]")
    density = calculate_enriched_zn_density(zn64_enrichment)
    if zn64_enrichment == 0.4917:
        mat.set_density('g/cm3', density)
        mat.add_element('Zn', 1.0)
        return
    fracs = get_zn_fractions(zn64_enrichment)
    density = calculate_enriched_zn_density(zn64_enrichment)
    mat.set_density('g/cm3', density)
    for iso, f in fracs.items():
        mat.add_nuclide(iso, f)
    print(f"  {label} Zn-64 {zn64_enrichment*100:.1f}%: density = {density:.4f} g/cm³")


def _set_zn67_enriched_material(mat, zn67_enrichment, label='inner'):
    """Set nuclide fractions and density for Zn-67 enriched material (density from utilities)."""
    e = float(zn67_enrichment)
    if not (ZN67_RANGE[0] <= e <= ZN67_RANGE[1]):
        raise ValueError(f"zn67_enrichment {zn67_enrichment} must be in [{ZN67_RANGE[0]}, {ZN67_RANGE[1]}]")
    fracs = get_zn67_fractions(zn67_enrichment)
    density = calculate_enriched_zn67_density(zn67_enrichment)
    mat.set_density('g/cm3', density)
    for iso, f in fracs.items():
        mat.add_nuclide(iso, f)
    print(f"  {label} Zn-67 {zn67_enrichment*100:.1f}%: density = {density:.4f} g/cm³")


def create_target_material(zn64_enrichment=0.4917, zn67_enrichment_inner=None, zn67_enrichment_outer=None):
    """Create target materials (inner/outer Zn, struct, multi, moderator, boron (not boron carbide), water).
    zn64_enrichment: outer layer when single_zn64 or dual. zn67_enrichment_outer: outer when single_zn67.
    zn67_enrichment_inner: inner layer when dual.
    """
    mat_inner = openmc.Material(material_id=0, name='target_material_inner')
    mat_outer = openmc.Material(material_id=1, name='target_material_outer')
    if zn67_enrichment_inner is not None:
        _set_zn67_enriched_material(mat_inner, zn67_enrichment_inner, 'inner')
    else:
        mat_inner.set_density('g/cm3', calculate_enriched_zn_density(0.4917))
        mat_inner.add_element('Zn', 1.0)
    mat_inner.temperature = 294
    if zn67_enrichment_outer is not None:
        _set_zn67_enriched_material(mat_outer, zn67_enrichment_outer, 'outer')
    else:
        _set_zn64_enriched_material(mat_outer, zn64_enrichment, 'outer')
    mat_outer.temperature = 294

    # HASTELLOY® C-276 alloy (UNS N10276). Weight %: Ni balance 57, Co 2.5 max, Cr 16, Mo 16, Fe 5, W 4, Mn 1 max, V 0.35 max, Si 0.08 max, C 0.01 max, Cu 0.5 max. Normalized to sum 1.
    struct_material = openmc.Material(name='hastelloy_c276', material_id=2)
    struct_material.temperature = 294
    struct_material.set_density('g/cm3', 8.89)  # typical C-276
    _c276_sum = 57 + 2.5 + 16 + 16 + 5 + 4 + 1 + 0.35 + 0.08 + 0.01 + 0.5  # 102.44
    struct_material.add_element('Ni', 57 / _c276_sum, 'wo')
    struct_material.add_element('Co', 2.5 / _c276_sum, 'wo')
    struct_material.add_element('Cr', 16 / _c276_sum, 'wo')
    struct_material.add_element('Mo', 16 / _c276_sum, 'wo')
    struct_material.add_element('Fe', 5 / _c276_sum, 'wo')
    struct_material.add_element('W', 4 / _c276_sum, 'wo')
    struct_material.add_element('Mn', 1 / _c276_sum, 'wo')
    struct_material.add_element('V', 0.35 / _c276_sum, 'wo')
    struct_material.add_element('Si', 0.08 / _c276_sum, 'wo')
    struct_material.add_element('C', 0.01 / _c276_sum, 'wo')
    struct_material.add_element('Cu', 0.5 / _c276_sum, 'wo')

    multi_material = openmc.Material(name='multi', material_id=3)
    multi_material.temperature = 294
    multi_material.set_density('g/cm3', 1.84)
    multi_material.add_element('Be', 1.0)

    moderator_material = openmc.Material(name='moderator', material_id=4)
    moderator_material.temperature = 294
    moderator_material.set_density('g/cm3', 2.2)
    moderator_material.add_element('C', 1.0)

    boron_mat = openmc.Material(name='boron', material_id=5)
    boron_mat.temperature = 294
    boron_mat.set_density('g/cm3', 2.34)  # Natural boron
    boron_mat.add_element('B', 1.0)

    water_mat = openmc.Material(name='water', material_id=6)
    water_mat.temperature = 294
    water_mat.set_density('g/cm3', 1.0)
    water_mat.add_element('H', 2.0, percent_type='ao')
    water_mat.add_element('O', 1.0, percent_type='ao')

    materials_list = [mat_inner, mat_outer, struct_material, multi_material, moderator_material, boron_mat, water_mat]
    
    return materials_list


def _box_region(xlo, xhi, ylo, yhi, zlo, zhi):
    """Axis-aligned box (all in cm): inside = xlo<x<xhi, ylo<y<yhi, zlo<z<zhi."""
    return (+openmc.XPlane(xlo) & -openmc.XPlane(xhi) &
            +openmc.YPlane(ylo) & -openmc.YPlane(yhi) &
            +openmc.ZPlane(zlo) & -openmc.ZPlane(zhi))


def create_geometry(materials, target_height=100, z_inner_thickness=15, z_outer_thickness=30, struct_thickness=2,
                    boron_thickness=1, moderator_thickness=10, multi_thickness=10, zn64_enrichment=0.4917,
                    complex_geom=False):
    """
    Cylindrical target geometry. If complex_geom=True, wrap in 3 ft vacuum box, 4 in aluminum,
    19,000 gal water pool (15 ft nominal height); otherwise water in sphere.

    Returns:
      (geometry, output_cells, surfaces_list, plots, surface_to_cell, cells_radial_for_plot)
    """
    sphere = openmc.Sphere(r=100.0, boundary_type='vacuum') if not complex_geom else None
    inner_radius = 5.0
    h2 = target_height / 2.0
    z_top = openmc.ZPlane(z0=+h2, name='target_top')
    z_bottom = openmc.ZPlane(z0=-h2, name='target_bottom')

    # Radii in order: vacuum | inner_struct | inner_target | multi | moderator | outer_target | outer_struct | boron
    r = inner_radius
    inner_struct_outer = r + struct_thickness if struct_thickness > 0 else r
    r = inner_struct_outer
    inner_target_outer = r + z_inner_thickness if z_inner_thickness > 0 else r
    r = inner_target_outer
    multi_outer = r + multi_thickness if multi_thickness > 0 else r
    r = multi_outer
    moderator_outer = r + moderator_thickness if moderator_thickness > 0 else r
    r = moderator_outer
    outer_target_outer = r + z_outer_thickness if z_outer_thickness > 0 else r
    r = outer_target_outer
    outer_struct_outer = r + struct_thickness if struct_thickness > 0 else r
    r = outer_struct_outer
    boron_outer = r + boron_thickness if boron_thickness > 0 else r
    outermost_radius = boron_outer

    # Surfaces: one ZCylinder per distinct radius
    def _cyl(name, rad):
        return openmc.ZCylinder(r=rad, name=name)
    surf_inner = _cyl('inner_surface', inner_radius)
    surf_inner_struct = _cyl('struct_outer', inner_struct_outer) if inner_struct_outer > inner_radius else surf_inner
    surf_inner_target = _cyl('inner_target_outer', inner_target_outer) if inner_target_outer > inner_struct_outer else surf_inner_struct
    surf_multi = _cyl('multi_outer', multi_outer) if multi_outer > inner_target_outer else surf_inner_target
    surf_mod = _cyl('moderator_outer', moderator_outer) if moderator_outer > multi_outer else surf_multi
    surf_outer_target = _cyl('outer_target_outer', outer_target_outer) if outer_target_outer > moderator_outer else surf_mod
    surf_outer_struct = _cyl('outer_struct_outer', outer_struct_outer) if outer_struct_outer > outer_target_outer else surf_outer_target
    surf_boron = _cyl('boron_outer', boron_outer) if boron_outer > outer_struct_outer else surf_outer_struct
    surf_outer = _cyl('outer_surface', outermost_radius)
    surfaces_list = [surf_inner]
    for s in (surf_inner_struct, surf_inner_target, surf_multi, surf_mod, surf_outer_target, surf_outer_struct, surf_boron):
        if s is not surfaces_list[-1]:
            surfaces_list.append(s)
    surfaces_list.append(surf_outer)

    struct_mat = materials[2]
    cells_list = []
    regions_to_exclude = []

    # Structure: one cell for inner annulus + outer annulus + top cap + bottom cap (same material)
    structure_cell = None
    if struct_thickness > 0:
        reg_inner = +z_bottom & -z_top & +surf_inner & -surf_inner_struct
        reg_outer = (+z_bottom & -z_top & +surf_outer_target & -surf_outer_struct) if (outer_struct_outer > outer_target_outer) else None
        z_cap_top = openmc.ZPlane(z0=h2 + struct_thickness, name='cap_top')
        z_cap_bottom = openmc.ZPlane(z0=-h2 - struct_thickness, name='cap_bottom')
        reg_top = +z_top & -z_cap_top & +surf_inner_struct & -surf_outer_struct if (outer_struct_outer > inner_struct_outer) else None
        reg_bottom = +z_cap_bottom & -z_bottom & +surf_inner_struct & -surf_outer_struct if (outer_struct_outer > inner_struct_outer) else None
        wall_region = reg_inner
        if reg_outer is not None:
            wall_region = wall_region | reg_outer
        if reg_top is not None:
            wall_region = wall_region | reg_top
        if reg_bottom is not None:
            wall_region = wall_region | reg_bottom
        structure_cell = openmc.Cell(cell_id=7, name='structure', fill=struct_mat, region=wall_region)
        cells_list.append(structure_cell)
        regions_to_exclude.append(wall_region)

    # Inner target
    inner_target_cell = None
    if z_inner_thickness > 0:
        reg = +z_bottom & -z_top & +surf_inner_struct & -surf_inner_target
        inner_target_cell = openmc.Cell(cell_id=3, name='inner_target', fill=materials[0], region=reg)
        cells_list.append(inner_target_cell)
        regions_to_exclude.append(reg)
        materials[0].volume = np.pi * (inner_target_outer**2 - inner_struct_outer**2) * target_height
    else:
        materials[0].volume = 0

    # Multi
    multi_cell = None
    if multi_thickness > 0:
        reg = +z_bottom & -z_top & +surf_inner_target & -surf_multi
        multi_cell = openmc.Cell(cell_id=4, name='multi', fill=materials[3], region=reg)
        cells_list.append(multi_cell)
        regions_to_exclude.append(reg)
        materials[3].volume = np.pi * (multi_outer**2 - inner_target_outer**2) * target_height

    # Moderator
    moderator_cell = None
    if moderator_thickness > 0:
        reg = +z_bottom & -z_top & +surf_multi & -surf_mod
        moderator_cell = openmc.Cell(cell_id=5, name='moderator', fill=materials[4], region=reg)
        cells_list.append(moderator_cell)
        regions_to_exclude.append(reg)
        materials[4].volume = np.pi * (moderator_outer**2 - multi_outer**2) * target_height

    # Outer target
    outer_target_cell = None
    if z_outer_thickness > 0:
        reg = +z_bottom & -z_top & +surf_mod & -surf_outer_target
        outer_target_cell = openmc.Cell(cell_id=6, name='outer_target', fill=materials[1], region=reg)
        cells_list.append(outer_target_cell)
        regions_to_exclude.append(reg)
        materials[1].volume = np.pi * (outer_target_outer**2 - moderator_outer**2) * target_height
    else:
        materials[1].volume = 0

    # Boron
    boron_cell = None
    if boron_thickness > 0:
        reg = +z_bottom & -z_top & +surf_outer_struct & -surf_boron
        boron_cell = openmc.Cell(cell_id=8, name='boron', fill=materials[5], region=reg)
        cells_list.append(boron_cell)
        regions_to_exclude.append(reg)
        materials[5].volume = np.pi * (boron_outer**2 - outer_struct_outer**2) * target_height
    else:
        materials[5].volume = 0

    # Struct total volume (inner + outer + top cap + bottom cap)
    if struct_thickness > 0:
        v_inner = np.pi * (inner_struct_outer**2 - inner_radius**2) * target_height
        v_outer = np.pi * (outer_struct_outer**2 - outer_target_outer**2) * target_height if (outer_struct_outer > outer_target_outer) else 0
        v_cap = np.pi * (outer_struct_outer**2 - inner_struct_outer**2) * struct_thickness * 2 if (outer_struct_outer > inner_struct_outer) else 0
        struct_mat.volume = v_inner + v_outer + v_cap
    else:
        struct_mat.volume = 0

    # Radial order: structure appears twice (inner and outer segments) for plot ordering
    radial_order = [
        (structure_cell, inner_struct_outer, 'structure'),
        (inner_target_cell, inner_target_outer, 'inner_target'),
        (multi_cell, multi_outer, 'multi'),
        (moderator_cell, moderator_outer, 'moderator'),
        (outer_target_cell, outer_target_outer, 'outer_target'),
        (structure_cell, outer_struct_outer, 'structure'),
        (boron_cell, boron_outer, 'boron'),
    ]

    # Vacuum (source region)
    vacuum_region = -surf_inner & +z_bottom & -z_top
    vacuum_cylinder_cell = openmc.Cell(cell_id=2, name='vacuum_cylinder', fill=None, region=vacuum_region)
    cells_list.insert(0, vacuum_cylinder_cell)

    if complex_geom:
        core_universe = openmc.Universe(cells=cells_list)
        cylinder_region = -surf_outer & +z_bottom & -z_top
        vb = COMPLEX_VACUUM_BOX_HALF
        ao = COMPLEX_AL_OUTER_HALF
        wx, wz = COMPLEX_WATER_XY_HALF, COMPLEX_WATER_Z_HALF
        inner_box = _box_region(-vb, vb, -vb, vb, -vb, vb)
        vacuum_void_region = inner_box & ~cylinder_region
        al_box = _box_region(-ao, ao, -ao, ao, -ao, ao)
        al_region = al_box & ~inner_box
        # Water box: create planes with boundary_type='vacuum' so OpenMC has bounded geometry
        surf_wx_lo = openmc.XPlane(x0=-wx, boundary_type='vacuum', name='water_x_lo')
        surf_wx_hi = openmc.XPlane(x0=wx, boundary_type='vacuum', name='water_x_hi')
        surf_wy_lo = openmc.YPlane(y0=-wx, boundary_type='vacuum', name='water_y_lo')
        surf_wy_hi = openmc.YPlane(y0=wx, boundary_type='vacuum', name='water_y_hi')
        surf_wz_lo = openmc.ZPlane(z0=-wz, boundary_type='vacuum', name='water_z_lo')
        surf_wz_hi = openmc.ZPlane(z0=wz, boundary_type='vacuum', name='water_z_hi')
        water_box = (+surf_wx_lo & -surf_wx_hi & +surf_wy_lo & -surf_wy_hi & +surf_wz_lo & -surf_wz_hi)
        water_region_complex = water_box & ~al_box
        outside_region = ~water_box
        # Steel cylinder (sleeve): from top of Zn to top of water
        surf_al_cyl_inner = openmc.ZCylinder(r=COMPLEX_AL_CYLINDER_INNER_R, name='al_cyl_inner')
        surf_al_cyl_outer = openmc.ZCylinder(r=COMPLEX_AL_CYLINDER_OUTER_R, name='al_cyl_outer')
        z_water_top = openmc.ZPlane(z0=wz, name='water_top')
        al_cyl_region = -surf_al_cyl_inner & +surf_al_cyl_outer & +z_top & -z_water_top
        vacuum_void_region = vacuum_void_region & ~al_cyl_region
        al_region = al_region & ~al_cyl_region
        water_region_complex = water_region_complex & ~al_cyl_region

        # Al 6061-T6: box between vacuum chamber and water
        al_mat = openmc.Material(name='Al_6061_T6', material_id=7)
        al_mat.set_density('g/cm3', 2.70)
        al_mat.add_element('Al', 0.971)
        al_mat.add_element('Mg', 0.010)
        al_mat.add_element('Si', 0.006)
        al_mat.add_element('Cu', 0.0028)
        al_mat.add_element('Cr', 0.002)
        al_mat.add_element('Fe', 0.005)
        al_mat.add_element('Mn', 0.0015)
        al_mat.add_element('Zn', 0.0025)
        al_mat.add_element('Ti', 0.0015)
        al_mat.temperature = 294

        # Stainless steel 316: cylinder (sleeve) around beam line / above target
        steel_mat = openmc.Material(name='Stainless_Steel_316', material_id=9)
        steel_mat.set_density('g/cm3', 7.99)
        steel_mat.add_element('Fe', 0.65)
        steel_mat.add_element('Cr', 0.17)
        steel_mat.add_element('Ni', 0.12)
        steel_mat.add_element('Mo', 0.025)
        steel_mat.add_element('Mn', 0.02)
        steel_mat.add_element('Si', 0.01)
        steel_mat.add_element('Cu', 0.005)
        steel_mat.temperature = 294
        materials.append(al_mat)
        materials.append(steel_mat)

        steel_cylinder_cell = openmc.Cell(cell_id=99, name='steel_cylinder', fill=steel_mat, region=al_cyl_region)
        root_cells = [
            openmc.Cell(cell_id=100, name='cylinder_core', fill=core_universe, region=cylinder_region),
            steel_cylinder_cell,
            openmc.Cell(cell_id=101, name='vacuum_box_void', fill=None, region=vacuum_void_region),
            openmc.Cell(cell_id=102, name='aluminum_box', fill=al_mat, region=al_region),
            openmc.Cell(cell_id=1, name='water', fill=materials[6], region=water_region_complex),
            openmc.Cell(cell_id=104, name='outside', fill=None, region=outside_region),
        ]
        universe = openmc.Universe(cells=root_cells)
        geometry = openmc.Geometry(universe)
        water_cell = root_cells[4]
        seen = set()
        cells_from_radial = [c for c, _, _ in radial_order if c is not None and c not in seen and not seen.add(c)]
        output_cells_for_complex = [vacuum_cylinder_cell] + cells_from_radial
        output_cells_for_complex.extend([root_cells[0], steel_cylinder_cell, root_cells[2], root_cells[3], water_cell])
    else:
        water_region = -sphere & ~vacuum_region
        for reg in regions_to_exclude:
            water_region = water_region & ~reg
        water_cell = openmc.Cell(cell_id=1, name='water', fill=materials[6], region=water_region)
        cells_list.insert(0, water_cell)
        universe = openmc.Universe(cells=cells_list)
        geometry = openmc.Geometry(universe)

    if complex_geom:
        output_cells = output_cells_for_complex
        wx_plot, wz_plot = COMPLEX_WATER_XY_HALF, COMPLEX_WATER_Z_HALF
        cells_radial_for_plot = [(vacuum_cylinder_cell, 0.0, inner_radius, 'vacuum')]
        cells_radial_for_plot.append((steel_cylinder_cell, COMPLEX_AL_CYLINDER_INNER_R, COMPLEX_AL_CYLINDER_OUTER_R, 'steel_cylinder'))
        r_prev = inner_radius
        for cell, r_outer, lbl in radial_order:
            if cell is not None:
                cells_radial_for_plot.append((cell, r_prev, r_outer, lbl))
                r_prev = r_outer
        cells_radial_for_plot.append((root_cells[0], r_prev, outermost_radius, 'cylinder_core'))
        cells_radial_for_plot.append((root_cells[2], outermost_radius, vb, 'vacuum_box_void'))
        cells_radial_for_plot.append((root_cells[3], vb, ao, 'aluminum'))
        cells_radial_for_plot.append((water_cell, ao, wx_plot, 'water'))

        # Complex geometry plot: XZ slice, full extent (vacuum box + Al + water; no concrete)
        plot_complex = openmc.Plot()
        plot_complex.width = (2 * wx_plot * 1.02, 2 * wz_plot * 1.02)
        plot_complex.origin = (0.0, 0.0, 0.0)
        plot_complex.basis = 'xz'
        plot_complex.filename = 'geometry_complex'
        plot_complex.pixels = (1200, 1200)
        plot_complex.color_by = 'cell'
        complex_colors = {}
        for cell, _r0, _r1, lbl in cells_radial_for_plot:
            hex_color = COMPLEX_PLOT_COLORS.get(lbl) or MATERIAL_COLORS.get(lbl, '#888888')
            complex_colors[cell] = _hex_to_rgb(hex_color)
        if len(root_cells) > 5:
            complex_colors[root_cells[5]] = _hex_to_rgb(COMPLEX_PLOT_COLORS.get('outside', MATERIAL_COLORS['vacuum']))
        plot_complex.colors = complex_colors
        # XY slice through beam tube (z = mid-height of steel) so stainless steel cylinder is visible as annulus
        z_steel_mid = (h2 + wz_plot) / 2.0  # between target_top and water_top
        plot_steel_xy = openmc.Plot()
        plot_steel_xy.origin = (0.0, 0.0, z_steel_mid)
        plot_steel_xy.width = (40.0, 40.0)  # 40 cm x 40 cm centered on beam (OD 20 cm)
        plot_steel_xy.basis = 'xy'
        plot_steel_xy.filename = 'geometry_complex_xy_beam_tube'
        plot_steel_xy.pixels = (600, 600)
        plot_steel_xy.color_by = 'cell'
        plot_steel_xy.colors = complex_colors
        plots = openmc.Plots([plot_complex, plot_steel_xy])
    else:
        plot = openmc.Plot()
        plot.width = (200.0, 200.0)
        plot.origin = (0.0, 0.0, 0.0)
        plot.basis = 'xz'
        plot.filename = f'geometry_inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_boron{boron_thickness}_multi{multi_thickness}_moderator{moderator_thickness}_zn{zn64_enrichment*100:.1f}%'
        plot.pixels = (1000, 1000)
        plot.color_by = 'cell'
        plot_colors = {
            vacuum_cylinder_cell: _hex_to_rgb(MATERIAL_COLORS['vacuum']),
            water_cell: _hex_to_rgb(MATERIAL_COLORS['water']),
        }
        struct_color = _hex_to_rgb(MATERIAL_COLORS['struct'])
        if structure_cell is not None:
            plot_colors[structure_cell] = struct_color
        if inner_target_cell is not None:
            plot_colors[inner_target_cell] = _hex_to_rgb(MATERIAL_COLORS['inner_target'])
        if multi_cell is not None:
            plot_colors[multi_cell] = _hex_to_rgb(MATERIAL_COLORS['multi'])
        if moderator_cell is not None:
            plot_colors[moderator_cell] = _hex_to_rgb(MATERIAL_COLORS['moderator'])
        if outer_target_cell is not None:
            plot_colors[outer_target_cell] = _hex_to_rgb(MATERIAL_COLORS['outer_target'])
        if boron_cell is not None:
            plot_colors[boron_cell] = _hex_to_rgb(MATERIAL_COLORS['boron'])
        plot.colors = plot_colors
        plots = openmc.Plots([plot])
        output_cells = [vacuum_cylinder_cell] + [c for c, _, _ in radial_order if c is not None] + [water_cell]
        # output_cells: dedupe so structure_cell (in radial_order twice) appears once
        seen = set()
        output_cells = [c for c in output_cells if c not in seen and not seen.add(c)]
        cells_radial_for_plot = [(vacuum_cylinder_cell, 0.0, inner_radius, 'vacuum')]
        r_prev = inner_radius
        for cell, r_outer, lbl in radial_order:
            if cell is not None:
                cells_radial_for_plot.append((cell, r_prev, r_outer, lbl))
                r_prev = r_outer
        cells_radial_for_plot.append((water_cell, r_prev, 100.0, 'water'))

    cells_radial = [c for c, _, _ in radial_order if c is not None]
    surface_to_cell = {
        'inner_surface': cells_radial[0] if cells_radial else None,
        'struct_outer': next((c for c in [inner_target_cell, multi_cell, moderator_cell, outer_target_cell] if c), None),
        'inner_target_outer': next((c for c in [multi_cell, moderator_cell, outer_target_cell] if c), None),
        'multi_outer': next((c for c in [moderator_cell, outer_target_cell] if c), None),
        'moderator_outer': outer_target_cell,
        'outer_target_outer': next((c for c in [structure_cell, boron_cell] if c), None),
        'outer_struct_outer': boron_cell if boron_cell else structure_cell,
        'boron_outer': boron_cell,
        'outer_surface': boron_cell if boron_cell else (structure_cell or cells_radial[-1] if cells_radial else None),
    }
    for s in surfaces_list:
        n = getattr(s, 'name', None)
        if n and n not in surface_to_cell:
            surface_to_cell[n] = cells_radial[-1] if cells_radial else None

    return geometry, output_cells, surfaces_list, plots, surface_to_cell, cells_radial_for_plot

def create_source(source_neutron_energy, target_height=100):
    """
    Create a line source along the z-axis at the origin with height of target_height cm.
    
    Parameters:
    -----------
    source_neutron_energy : float
        Neutron energy (MeV)
    target_height : float
        Height of the line source (cm)
    
    Returns:
    --------
    openmc.IndependentSource
    """
    source = openmc.IndependentSource()
    source.angle = openmc.stats.Isotropic()
    
    # Line source along z-axis: x=0, y=0, z uniform from -height/2 to +height/2
    source.space = openmc.stats.CartesianIndependent(
        openmc.stats.Discrete([0.0], [1.0]),  # x = 0
        openmc.stats.Discrete([0.0], [1.0]),  # y = 0
        openmc.stats.Uniform(-target_height/2, target_height/2)  # z from -h/2 to +h/2
    )
    source.energy = openmc.stats.Discrete([source_neutron_energy * 1e6], [1.0])  # Convert MeV to eV
    return source


def _hex_to_rgb(hex_str):
    """'#rrggbb' -> (r,g,b) 0-255 for OpenMC plot.colors."""
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def add_axis_labels(plot_filename, output_dir='.', z_inner_thickness=5, z_outer_thickness=20,
                    struct_thickness=2, boron_thickness=1, multi_thickness=0, moderator_thickness=0,
                    zn64_enrichment=0.4917, use_vacuum_outer=False):
    """Add axis labels and legend to geometry plot (base filename without .png)."""
    plot_path = os.path.join(output_dir, f'{plot_filename}.png')
    if not os.path.exists(plot_path):
        print(f"Warning: Plot file not found: {plot_path}")
        return
    img = plt.imread(plot_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, extent=[-100, 100, -100, 100], origin='upper')

    r0 = 5.0
    r_struct = r0 + struct_thickness
    r_inner_t = r_struct + z_inner_thickness
    r_multi = r_inner_t + (multi_thickness if multi_thickness > 0 else 0)
    r_mod = r_multi + (moderator_thickness if moderator_thickness > 0 else 0)
    r_outer_t = r_mod + (z_outer_thickness if z_outer_thickness > 0 else 0)
    r_outer_struct = r_outer_t + (struct_thickness if struct_thickness > 0 else 0)
    r_boron = r_outer_struct + (boron_thickness if boron_thickness > 0 else 0)

    ax.set_xticks(range(-100, 101, 20))
    ax.set_xticks(range(-100, 101, 5), minor=True)
    ax.set_yticks(range(-100, 101, 20))
    ax.set_yticks(range(-100, 101, 5), minor=True)
    ax.set_xlabel('X Position [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position [cm]', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.3, linestyle='--', linewidth=0.5)

    from matplotlib.patches import Patch
    disp = lambda v: 0.5 if v == 0 else v
    outer_label = 'Vacuum (outside chamber)' if use_vacuum_outer else 'Water (outside chamber)'
    patches = [Patch(facecolor=MATERIAL_COLORS['vacuum'], edgecolor='black', label=f'Vacuum (r < {r0:.1f} cm)')]
    r = r0
    if struct_thickness > 0:
        # One legend entry for all structure (inner, outer, caps) — same as single boron entry
        patches.append(Patch(facecolor=MATERIAL_COLORS['struct'], edgecolor='black', label=f'Structure ({disp(struct_thickness)} cm)'))
        r = r_struct
    if z_inner_thickness > 0:
        patches.append(Patch(facecolor=MATERIAL_COLORS['inner_target'], edgecolor='black', label=f'Inner target Zn {zn64_enrichment*100:.1f}% ({r:.1f}–{r_inner_t:.1f} cm)'))
        r = r_inner_t
    if multi_thickness > 0:
        patches.append(Patch(facecolor=MATERIAL_COLORS['multi'], edgecolor='black', label=f'Multiplier ({r:.1f}–{r_multi:.1f} cm)'))
        r = r_multi
    if moderator_thickness > 0:
        patches.append(Patch(facecolor=MATERIAL_COLORS['moderator'], edgecolor='black', label=f'Moderator ({r:.1f}–{r_mod:.1f} cm)'))
        r = r_mod
    if z_outer_thickness > 0:
        patches.append(Patch(facecolor=MATERIAL_COLORS['outer_target'], edgecolor='black', label=f'Outer target Zn ({r:.1f}–{r_outer_t:.1f} cm)'))
        r = r_outer_t
    if struct_thickness > 0 and r_outer_struct > r_outer_t:
        r = r_outer_struct
    if boron_thickness > 0:
        patches.append(Patch(facecolor=MATERIAL_COLORS['boron'], edgecolor='black', label=f'Boron ({r:.1f}–{r_boron:.1f} cm)'))
    patches.append(Patch(facecolor=MATERIAL_COLORS['water'], edgecolor='black', label=outer_label))
    ax.legend(handles=patches, loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black', title='Cell Layers', title_fontsize=10)
    title_parts = [f"Struct {disp(struct_thickness)} cm", f"Inner {disp(z_inner_thickness)} cm"]
    if multi_thickness > 0:
        title_parts.append(f"Multi {disp(multi_thickness)} cm")
    if moderator_thickness > 0:
        title_parts.append(f"Mod {disp(moderator_thickness)} cm")
    title_parts.extend([f"Outer {disp(z_outer_thickness)} cm", f"Boron {disp(boron_thickness)} cm"])
    ax.set_title("Geometry XZ\n" + ", ".join(title_parts), fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved labeled geometry plot: {plot_path}")


def add_axis_labels_complex(output_dir, cells_radial_for_plot):
    """Add axis labels and legend to complex geometry plot (geometry_complex.png), same workflow as simple case."""
    plot_filename = 'geometry_complex'
    plot_path = os.path.join(output_dir, f'{plot_filename}.png')
    if not os.path.exists(plot_path):
        print(f"Warning: Complex geometry plot not found: {plot_path}")
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    img = plt.imread(plot_path)
    ex = COMPLEX_WATER_XY_HALF * 1.02
    ez = COMPLEX_WATER_Z_HALF * 1.02
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, extent=[-ex, ex, -ez, ez], origin='upper')
    ax.set_xlabel('X Position [cm]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Z Position [cm]', fontsize=12, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    seen = set()
    patches = []
    for _cell, _r0, _r1, lbl in cells_radial_for_plot:
        if lbl not in seen:
            seen.add(lbl)
            color = COMPLEX_PLOT_COLORS.get(lbl) or MATERIAL_COLORS.get(
                lbl, MATERIAL_COLORS.get('struct', '#888888') if lbl in STRUCT_CELL_NAMES else '#888888'
            )
            display_name = PLOT_STRUCT_LABEL if lbl in STRUCT_CELL_NAMES else lbl
            patches.append(Patch(facecolor=color, edgecolor='black', label=display_name))
    if patches:
        ax.legend(handles=patches, loc='upper right', fontsize=9, framealpha=0.9, edgecolor='black', title='Regions', title_fontsize=10)
    ax.set_title('Geometry XZ (complex) — vacuum box, aluminum, water', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved labeled complex geometry plot: {plot_path}")


def create_tallies(cells, surfaces, surface_to_cell, verbose=True):
    """Build tallies: cell flux, flux spectra, heating; surface flux from adjacent cell. Returns openmc.Tallies."""
    tallies = openmc.Tallies()
    labels = []
    neutron_filter = openmc.ParticleFilter(['neutron'])
    energy_filter = openmc.EnergyFilter.from_group_structure('CCFE-709')
    all_cells_filter = openmc.CellFilter([c for c in cells if c.fill is not None])
    # Target cells only for Zn/Cu reaction tallies (same as test.py: 1–2 bins, predictable order)
    target_cells = [c for c in cells if c.fill is not None and getattr(c, 'name', '') in ('inner_target', 'outer_target')]
    target_cells_filter = openmc.CellFilter(target_cells) if target_cells else all_cells_filter

    t = openmc.Tally(name='flux')
    t.filters = [all_cells_filter, neutron_filter]
    t.scores = ['flux']
    tallies.append(t)
    labels.append((t.name, 'cell flux'))
    for c in cells:
        single_cell_filter = openmc.CellFilter([c])
        t = openmc.Tally(name=f'{c.name}_spectra')
        t.filters = [neutron_filter, energy_filter, single_cell_filter]
        t.scores = ['flux']
        tallies.append(t)
    t = openmc.Tally(name='volumetric_heating')
    t.filters = [all_cells_filter, neutron_filter]
    t.scores = ['heating-local']
    tallies.append(t)
    labels.append((t.name, 'cell heating (W/cm³)'))

    def make_tally(name, scores, filters: list = None, nuclides: list = None):
        tally = openmc.Tally(name=name)
        tally.scores = scores
        if filters is not None:
            tally.filters = filters
        if nuclides is not None:
            tally.nuclides = nuclides
        tallies.append(tally)
        labels.append((tally.name, name))
        return tally

    # Tally names/scores/nuclides must match utilities.build_channel_rr_per_s and channel_rate_per_s.
    # Zn: (n,gamma), (n,2n), (n,a). (n,a) on Zn64,66,67,68,70 -> Ni production (Ni61, Ni63, Ni64, Ni65, Ni67).
    # Include Zn62 for Zn62(n,2n)Zn61 chain.
    zn_nuclides = ['Zn62', 'Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']
    zn_scores = ['(n,gamma)', '(n,2n)', '(n,a)']

    # Cu production: Zn (n,p)/(n,d) -> Cu; Zn61 omitted (not in typical nuclear data), so Zn61(n,p)Cu61 stays 0.
    cu_tally_nuclides = ['Zn62', 'Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']

    Cu_tally_tot    = make_tally('Total_Cu_Production_rxn_rate',     ['(n,p)', '(n,d)'], nuclides=cu_tally_nuclides)
    Cu_tally        = make_tally('Cu_Production_rxn_rates',          ['(n,p)', '(n,d)'], filters=[target_cells_filter], nuclides=cu_tally_nuclides)
    Cu_energy_tally = make_tally('Cu_Production_rxn_rates_spectrum', ['(n,p)', '(n,d)'], filters=[all_cells_filter, energy_filter], nuclides=cu_tally_nuclides)

    Zn_tally_tot    = make_tally('Total_Zn_rxn_rate',     zn_scores, nuclides=zn_nuclides)
    Zn_tally        = make_tally('Zn_rxn_rates',          zn_scores, filters=[target_cells_filter], nuclides=zn_nuclides)
    Zn_energy_tally = make_tally('Zn_rxn_rates_spectrum', zn_scores, filters=[all_cells_filter, energy_filter], nuclides=zn_nuclides)

    # Ni production: Zn (n,a) -> Ni61, Ni63, Ni64, Ni65, Ni67
    ni_tally_nuclides = ['Zn64', 'Zn66', 'Zn67', 'Zn68', 'Zn70']
    Ni_tally_tot      = make_tally('Total_Ni_rxn_rate',   ['(n,a)'], nuclides=ni_tally_nuclides)

    if verbose:
        print("\nTallies created (label check):")
        for nm, desc in labels:
            print(f"  {nm}: {desc}")

    return tallies


# Copper isotope atomic masses [g/mol] for target-cell CSV (incl. short-lived Cu61, Cu62, Cu66, Cu69, Cu70)
_CU_ATOMIC_MASS_G_MOL = {
    'Cu61': 60.966, 'Cu62': 61.963, 'Cu63': 62.9296, 'Cu64': 63.9298,
    'Cu65': 64.9278, 'Cu66': 65.9289, 'Cu67': 66.9277, 'Cu69': 68.9256, 'Cu70': 69.9254,
}
_AVOGADRO = 6.02214076e23
_ZN_ISOTOPES = ['Zn62', 'Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn69m', 'Zn70']  # Zn61 not in JENDL


def _copper_mass_from_atoms(atoms):
    """From atoms dict return Cu masses [g] and total copper mass [g] (all Cu isotopes incl. short-lived)."""
    out = {f'{iso.lower()}_g': 0.0 for iso in _CU_ATOMIC_MASS_G_MOL}
    for iso, mass_g_mol in _CU_ATOMIC_MASS_G_MOL.items():
        n = float(atoms.get(iso, 0) or 0)
        out[f'{iso.lower()}_g'] = n * mass_g_mol / _AVOGADRO
    out['total_cu_g'] = sum(out[k] for k in out if k.endswith('_g'))
    return out


def _zinc_mass_from_atoms(atoms):
    """Total Zn mass [g] from atom counts using OpenMC atomic masses."""
    total_g = 0.0
    for nuc in _ZN_ISOTOPES:
        n = float(atoms.get(nuc, 0) or 0)
        if n <= 0:
            continue
        try:
            m_g_mol = openmc.data.atomic_mass(nuc)
            total_g += n * m_g_mol / _AVOGADRO
        except (KeyError, AttributeError):
            pass
    return total_g


def _write_fusion_irradiation_target_csv(
    sp, cells, output_dir, batches,
    irradiation_hours=8.0, cooldown_days=1.0, zn64_enrichment=0.4917,
):
    """
    Write fusion_irradiation_target.csv for the target cell(s) only: reaction tallies (counts, units),
    initial atoms, final atoms, final masses, total Zn/Cu mass, activities, specific activities (Cu64, Cu67).
    Target cells = inner_target + outer_target (summed). Uses Bateman for irradiation + cooldown.
    """
    target_cells = [c for c in cells if c.fill is not None and getattr(c, 'name', '') in ('inner_target', 'outer_target')]
    if not target_cells:
        print("  [fusion_irradiation_target CSV] No target cells (inner_target/outer_target) found; skipping.")
        return

    statepoint_path = os.path.join(output_dir, f'statepoint.{batches}.h5')
    # Combined initial atoms over target cells (by material + volume)
    initial_atoms_combined = {}
    total_volume_cm3 = 0.0
    for c in target_cells:
        vol = getattr(c, 'volume', None)
        if vol is None or vol <= 0:
            continue
        total_volume_cm3 += float(vol)
        mat = c.fill
        mid = getattr(mat, 'id', None)
        if mid is None:
            continue
        ia = get_initial_atoms_from_statepoint(statepoint_path, mid, float(vol))
        if ia is None:
            zn_density = get_material_density_from_statepoint(statepoint_path, mid) or calculate_enriched_zn_density(zn64_enrichment)
            ia = get_initial_zn_atoms_fallback(float(vol), zn64_enrichment, zn_density)
        if ia:
            for nuc, count in ia.items():
                initial_atoms_combined[nuc] = initial_atoms_combined.get(nuc, 0.0) + float(count or 0)
    if not initial_atoms_combined:
        print("  [fusion_irradiation_target CSV] No initial atoms; skipping.")
        return

    # Ensure all CHANNELS nuclides present
    for parent, _, daughter in CHANNELS:
        if parent not in initial_atoms_combined:
            initial_atoms_combined[parent] = 0.0
        if daughter not in initial_atoms_combined:
            initial_atoms_combined[daughter] = 0.0

    # Combined reaction rates (atoms/s) over target cells only (by cell id)
    rr_combined = {}
    for tc in target_cells:
        cid = getattr(tc, 'id', None)
        if cid is None:
            continue
        rr = build_channel_rr_per_s(sp, cell_id=cid, source_strength=SOURCE_STRENGTH)
        for k, v in rr.items():
            rr_combined[k] = rr_combined.get(k, 0) + float(v or 0)

    irrad_s = irradiation_hours * 3600.0
    cooldown_s = cooldown_days * 86400.0
    # Bateman depletion in timesteps (parent depletion + all CHANNELS including Zn->Ni (n,a))
    n_steps_irrad = max(10, int(irrad_s / 3600))  # hourly or at least 10 steps
    history_irrad = evolve_bateman_irradiation_with_history(
        initial_atoms_combined, rr_combined, irrad_s, n_steps=n_steps_irrad
    )
    atoms_eoi = history_irrad[-1][1]
    atoms_final = apply_single_decay_step(atoms_eoi, cooldown_s)

    cu_mass = _copper_mass_from_atoms(atoms_final)
    zn_mass_g = _zinc_mass_from_atoms(atoms_final)
    ni_mass_g = 0.0
    for nuc in ['Ni61', 'Ni63', 'Ni64', 'Ni65', 'Ni67']:
        try:
            m = openmc.data.atomic_mass(nuc)
            ni_mass_g += float(atoms_final.get(nuc, 0) or 0) * m / _AVOGADRO
        except (KeyError, AttributeError, TypeError):
            pass
    total_mass_g = cu_mass['total_cu_g'] + zn_mass_g + ni_mass_g

    # Activities [Bq] for all isotopes that have decay constant
    activities_Bq = {}
    for nuc in set(initial_atoms_combined.keys()) | set(atoms_final.keys()):
        lam = get_decay_constant(nuc)
        if lam and lam > 0:
            n_atoms = float(atoms_final.get(nuc, 0) or 0)
            activities_Bq[nuc] = n_atoms * lam

    # Specific activity = activity_iso / total_mass_g [Bq/g]
    cu64_specific_Bq_per_g = (activities_Bq.get('Cu64', 0) / total_mass_g) if total_mass_g > 0 else 0.0
    cu67_specific_Bq_per_g = (activities_Bq.get('Cu67', 0) / total_mass_g) if total_mass_g > 0 else 0.0

    # Build CSV rows with clear units
    rows = []
    # Section: Reaction tallies (target cell only)
    rows.append({'section': 'reaction_tally', 'quantity': 'reaction_rate', 'item': 'channel', 'value': '', 'units': 'atoms/s', 'description': 'Reaction rates in target cell(s) only'})
    for ch, rate in sorted(rr_combined.items()):
        rows.append({'section': 'reaction_tally', 'quantity': 'reaction_rate', 'item': ch, 'value': rate, 'units': 'atoms/s', 'description': ''})

    # Section: Initial atoms
    rows.append({'section': 'initial_atoms', 'quantity': 'atom_count', 'item': 'nuclide', 'value': '', 'units': 'atoms', 'description': 'Initial atom counts in target cell(s)'})
    for nuc in sorted(initial_atoms_combined.keys()):
        rows.append({'section': 'initial_atoms', 'quantity': 'atom_count', 'item': nuc, 'value': initial_atoms_combined[nuc], 'units': 'atoms', 'description': ''})

    # Section: Final atoms (all products)
    rows.append({'section': 'final_atoms', 'quantity': 'atom_count', 'item': 'nuclide', 'value': '', 'units': 'atoms', 'description': 'Final atom counts after irradiation + cooldown'})
    for nuc in sorted(atoms_final.keys()):
        rows.append({'section': 'final_atoms', 'quantity': 'atom_count', 'item': nuc, 'value': atoms_final[nuc], 'units': 'atoms', 'description': ''})

    # Section: Final masses per isotope
    rows.append({'section': 'final_mass_per_isotope', 'quantity': 'mass', 'item': 'nuclide', 'value': '', 'units': 'g', 'description': 'Mass of each isotope in target'})
    for nuc in sorted(atoms_final.keys()):
        n = float(atoms_final.get(nuc, 0) or 0)
        try:
            m_g_mol = openmc.data.atomic_mass(nuc)
            mass_g = n * m_g_mol / _AVOGADRO
        except (KeyError, AttributeError):
            mass_g = 0.0
        rows.append({'section': 'final_mass_per_isotope', 'quantity': 'mass', 'item': nuc, 'value': mass_g, 'units': 'g', 'description': ''})

    # Section: Total Zn, Cu, Ni and combined mass
    rows.append({'section': 'total_mass', 'quantity': 'total_Zn_mass', 'item': 'target', 'value': zn_mass_g, 'units': 'g', 'description': 'Total zinc mass in target'})
    rows.append({'section': 'total_mass', 'quantity': 'total_Cu_mass', 'item': 'target', 'value': cu_mass['total_cu_g'], 'units': 'g', 'description': 'Total copper mass in target'})
    rows.append({'section': 'total_mass', 'quantity': 'total_Ni_mass', 'item': 'target', 'value': ni_mass_g, 'units': 'g', 'description': 'Total nickel from Zn (n,a) in target'})
    rows.append({'section': 'total_mass', 'quantity': 'total_target_mass', 'item': 'target', 'value': total_mass_g, 'units': 'g', 'description': 'Zn + Cu + Ni total'})

    # Section: Activities
    rows.append({'section': 'activity', 'quantity': 'activity', 'item': 'nuclide', 'value': '', 'units': 'Bq', 'description': 'Activity of each isotope'})
    for nuc in sorted(activities_Bq.keys()):
        rows.append({'section': 'activity', 'quantity': 'activity', 'item': nuc, 'value': activities_Bq[nuc], 'units': 'Bq', 'description': ''})

    # Section: Specific activities (Cu64, Cu67) = activity_iso / total_mass_g
    rows.append({'section': 'specific_activity', 'quantity': 'Cu64_specific_activity', 'item': 'target', 'value': cu64_specific_Bq_per_g, 'units': 'Bq/g', 'description': 'Cu64 activity / total target mass'})
    rows.append({'section': 'specific_activity', 'quantity': 'Cu67_specific_activity', 'item': 'target', 'value': cu67_specific_Bq_per_g, 'units': 'Bq/g', 'description': 'Cu67 activity / total target mass'})

    # Section: Depletion timesteps — masses and activities of all particles at each step (Bateman with parent depletion)
    all_nuclides = sorted(set(initial_atoms_combined.keys()) | set(atoms_final.keys()))
    timestep_rows = []
    for step_idx, (time_s, atoms) in enumerate(history_irrad):
        for nuc in all_nuclides:
            n_atoms = float(atoms.get(nuc, 0) or 0)
            if n_atoms <= 0:
                continue
            try:
                mass_g = n_atoms * openmc.data.atomic_mass(nuc) / _AVOGADRO
            except (KeyError, AttributeError, TypeError):
                mass_g = 0.0
            lam = get_decay_constant(nuc)
            activity_Bq = n_atoms * lam if lam and lam > 0 else 0.0
            timestep_rows.append({
                'step': step_idx, 'time_s': time_s, 'phase': 'irrad',
                'nuclide': nuc, 'atoms': n_atoms, 'mass_g': mass_g, 'activity_Bq': activity_Bq,
            })
    # Cooldown endpoint
    for nuc in all_nuclides:
        n_atoms = float(atoms_final.get(nuc, 0) or 0)
        if n_atoms <= 0:
            continue
        try:
            mass_g = n_atoms * openmc.data.atomic_mass(nuc) / _AVOGADRO
        except (KeyError, AttributeError, TypeError):
            mass_g = 0.0
        lam = get_decay_constant(nuc)
        activity_Bq = n_atoms * lam if lam and lam > 0 else 0.0
        timestep_rows.append({
            'step': len(history_irrad), 'time_s': irrad_s + cooldown_s, 'phase': 'cooldown',
            'nuclide': nuc, 'atoms': n_atoms, 'mass_g': mass_g, 'activity_Bq': activity_Bq,
        })
    if timestep_rows:
        pd.DataFrame(timestep_rows).to_csv(
            os.path.join(output_dir, 'fusion_irradiation_depletion_timesteps.csv'),
            index=False, float_format='%.6g',
        )
        print(f"  Depletion timesteps (masses and activities per step) written to fusion_irradiation_depletion_timesteps.csv")
    rows.append({'section': 'depletion_timesteps', 'quantity': 'file', 'item': 'fusion_irradiation_depletion_timesteps.csv', 'value': len(timestep_rows), 'units': 'rows', 'description': 'Masses and activities of all nuclides at each Bateman step'})

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, 'fusion_irradiation_target.csv')
    df.to_csv(out_path, index=False)
    print(f"  Target-cell summary written to {out_path} (reactions, initial/final atoms, masses, activities, specific activities Cu64/Cu67; units in column 'units')")


def run_irradiation_simulation(
    target_flux,
    neutron_energy,
    source_neutron_energy,
    particles=10000,
    batches=10,
    zn64_enrichment=0.4917,
    zn67_enrichment_inner=None,
    zn67_enrichment_outer=None,
    z_inner_thickness=0,
    z_outer_thickness=0,
    struct_thickness=0,
    boron_thickness=0,
    moderator_thickness=0,
    multi_thickness=0,
    inner_radius=5,
    target_height=None,
    original_dir=None,
    complex_geom=False,
):
    """
    Run fusion neutron irradiation simulation

    Parameters:
    -----------
    target_flux : float
        Desired neutron flux at surface (n/cm²/s)
    neutron_energy : float
        Neutron energy (MeV)
    particles : int, optional
        Number of particles per batch (default: 10000)
    batches : int, optional
        Number of batches for transport calculation (default: 50)
    zn64_enrichment : float
        Fractional enrichment of Zn-64 (default: 0.4917 for natural Zn)
    original_dir : str
        Original working directory for output files

    Returns:
    --------
    None
        Simulation results are saved to output directory
    """
    if target_height is None:
        try:
            import run_config as _c
            target_height = getattr(_c, 'TARGET_HEIGHT_CM', 100.0)
        except ImportError:
            target_height = 100.0

    print("="*60)
    print("OpenMC Fusion Neutron Irradiation Simulation")
    print("="*60)
    
    root_dir = os.path.abspath(original_dir)
    output_prefix = 'irrad_output'
    layers = f'inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_boron{boron_thickness}_multi{multi_thickness}_moderator{moderator_thickness}'
    if zn67_enrichment_outer is not None:
        # Single chamber, outer Zn-67 → Cu-67 production
        subdir = f'{output_prefix}_single_cu67_{layers}_zn67_{zn67_enrichment_outer*100:.1f}%'
    elif zn67_enrichment_inner is not None:
        # Dual chamber: inner Zn-67 (Cu-67), outer Zn-64 (Cu-64)
        subdir = f'{output_prefix}_dual_{layers}_zn64_{zn64_enrichment*100:.1f}%_inner_zn67_{zn67_enrichment_inner*100:.1f}%'
    else:
        # Single chamber, outer Zn-64 → Cu-64 production
        subdir = f'{output_prefix}_single_cu64_{layers}_zn64_{zn64_enrichment*100:.1f}%'
    if complex_geom:
        subdir = subdir + '_complex'
    output_dir = os.path.join(root_dir, subdir)

    print(f"\nCreating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save original working directory

    # Geometry parameters (display 0 as 0.5 cm in results)
    def _disp_cm(v):
        return 0.5 if v == 0 else v
    print(f"\nGeometry Parameters:")
    print(f"  Target height: {target_height} cm")
    print(f"  Z inner thickness: {_disp_cm(z_inner_thickness)} cm")
    print(f"  Z outer thickness: {_disp_cm(z_outer_thickness)} cm")
    print(f"  Struct thickness: {_disp_cm(struct_thickness)} cm")
    print(f"  Boron thickness: {_disp_cm(boron_thickness)} cm")
    print(f"  Moderator thickness: {_disp_cm(moderator_thickness)} cm")
    print(f"  Multi thickness: {_disp_cm(multi_thickness)} cm")
    print(f"  Inner radius: {inner_radius} cm")
    if complex_geom:
        print(f"  Complex geometry: vacuum box + Al + water pool")

    print(f"\nInput Parameters:")
    print(f"  Target flux: {target_flux:.2e} n/cm²/s")
    print(f"  D-T Neutron energy: {neutron_energy} MeV")
    print(f"  Source Neutron energy: {source_neutron_energy} MeV")
    print(f"  Material: Zn (modify create_target_material() to change)")
    print(f"  Particles: {particles}")
    print(f"  Batches: {batches}")
    if zn67_enrichment_outer is not None:
        print(f"  Outer Zn-67 enrichment: {zn67_enrichment_outer*100:.1f}%")
    else:
        print(f"  Zn-64 enrichment (outer): {zn64_enrichment*100:.1f}%")
    if zn67_enrichment_inner is not None:
        print(f"  Inner Zn-67 enrichment: {zn67_enrichment_inner*100:.1f}%")
    print(f"  Output directory: {output_dir}")
    print(f"\nSource Parameters:")
    print(f"  Source strength: {SOURCE_STRENGTH:.2e} n/s")
    print(f"  Source power: {FUSION_POWER_W:.2e} W")
    print("\nCreating materials...")
    materials = create_target_material(
        zn64_enrichment=zn64_enrichment,
        zn67_enrichment_inner=zn67_enrichment_inner,
        zn67_enrichment_outer=zn67_enrichment_outer,
    )

    print(f"  Inner material:     {materials[0].name}")
    print(f"  Outer material:     {materials[1].name}")
    print(f"  Struct material:    {materials[2].name}")
    print(f"  Multi material:     {materials[3].name}")
    print(f"  Moderator material: {materials[4].name}")
    print(f"  Boron material:     {materials[5].name}")
    print(f"  Vacuum:             void cells (fill=None)")
    print(f"  Water material:     {materials[6].name}")

    # Create geometry
    print("Creating geometry...")
    geometry, cells, surfaces, plots, surface_to_cell, cells_radial_for_plot = create_geometry(materials, target_height, z_inner_thickness, z_outer_thickness, struct_thickness, boron_thickness, moderator_thickness, multi_thickness, zn64_enrichment, complex_geom=complex_geom)

    # Create source
    print("Creating source...")
    source = create_source(source_neutron_energy, target_height)

    # Create tallies
    print("Creating tallies...")
    tallies = create_tallies(cells, surfaces, surface_to_cell)

    # Settings
    settings = openmc.Settings()
    settings.source = source
    settings.particles = particles
    settings.batches = batches
    settings.run_mode = 'fixed source'
    
    # Create model
    model = openmc.Model(geometry=geometry, settings=settings, materials=materials, tallies=tallies, plots=plots)

    # Change to output directory for simulation
    print(f"\nChanging to output directory: {output_dir}")
    try:
        os.chdir(output_dir)

        # Export model files to output directory
        print("Exporting model files...")
        model.materials.export_to_xml()
        model.geometry.export_to_xml()
        model.plots.export_to_xml()
        model.settings.export_to_xml()
        model.tallies.export_to_xml()

        print("Plotting geometry...")
        openmc.plot_geometry()
        if not complex_geom:
            plot_filename = f'geometry_inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_boron{boron_thickness}_multi{multi_thickness}_moderator{moderator_thickness}_zn{zn64_enrichment*100:.1f}%'
            add_axis_labels(
                plot_filename,
                output_dir=output_dir,
                z_inner_thickness=z_inner_thickness,
                z_outer_thickness=z_outer_thickness,
                struct_thickness=struct_thickness,
                boron_thickness=boron_thickness,
                multi_thickness=multi_thickness,
                moderator_thickness=moderator_thickness,
                zn64_enrichment=zn64_enrichment
            )
        else:
            add_axis_labels_complex(output_dir, cells_radial_for_plot)

        print("Running simulation...")

        all_cells = list(model.geometry.get_all_cells().values())
        # Complex geometry has an unbounded "outside" cell, so root_universe.bounding_box is
        # non-finite; VolumeCalculation requires finite lower_left/upper_right.
        if complex_geom:
            ll = (-COMPLEX_WATER_XY_HALF, -COMPLEX_WATER_XY_HALF, -COMPLEX_WATER_Z_HALF)
            ur = (COMPLEX_WATER_XY_HALF, COMPLEX_WATER_XY_HALF, COMPLEX_WATER_Z_HALF)
        else:
            bb = model.geometry.root_universe.bounding_box
            ll = bb.lower_left
            ur = bb.upper_right
        cell_vc = openmc.VolumeCalculation(
            domains=all_cells,
            samples=10_000_000,
            lower_left=ll,
            upper_right=ur
        )
        model.settings.volume_calculations = [cell_vc]

        cwd = os.getcwd()
        os.chdir(output_dir)
        try:
            model.export_to_xml()
            model.calculate_volumes()
            vol_path = os.path.join(output_dir, 'volume_1.h5')
            if os.path.isfile(vol_path):
                cell_vc.load_results(vol_path)
                model.geometry.add_volume_information(cell_vc)
                print("Cell volumes [cm³]:")
                for c in all_cells:
                    v = getattr(c, 'volume', None)
                    print(f"  {c.name}: {v:.2f}" if v is not None else f"  {c.name}: (not set)")
                if hasattr(cell_vc, 'atoms_dataframe') and cell_vc.atoms_dataframe is not None:
                    cell_vc.atoms_dataframe.to_csv(os.path.join(output_dir, 'atoms_dataframe.csv'))
            model.run()
        finally:
            os.chdir(cwd)

        # Extract and display cell flux (track-length estimator)
        print("\nCell Flux Verification:")
        statepoint_file = f'statepoint.{batches}.h5'
        
        # Check if statepoint file exists before trying to extract flux (statepoint is in output_dir, we are still in it)
        if os.path.exists(statepoint_file):
            sp = openmc.StatePoint(statepoint_file)

            # Use cells from create_geometry (same list we created tallies for), not all_cells from model
            # Get energy filter from first cell's spectra tally
            # Filter order in spectra tallies: [neutron_filter, energy_filter, cell_filter]
            first_spectra = sp.get_tally(name=f'{cells[0].name}_spectra')
            energy_filt = first_spectra.filters[1]  # EnergyFilter
            energy_bins = np.asarray(energy_filt.bins)
            if energy_bins.ndim == 2:
                energy_edges_eV = np.concatenate([energy_bins[:, 0], [energy_bins[-1, 1]]])
            else:
                energy_edges_eV = energy_bins
            lethargy_bin_width = np.log(energy_edges_eV[1:] / energy_edges_eV[:-1])

            # Extract flux spectra per cell; then aggregate by plot material (struct_inner/outer/top/bottom -> one "structure")
            #   flux per bin = (tally_mean / volume) * source_strength  [n/cm²/s]
            flux_spectra_by_cell = {}
            cell_volumes = {}
            for c in cells:
                try:
                    cell_tally = sp.get_tally(name=f'{c.name}_spectra')
                except LookupError:
                    continue
                openmc_flux = cell_tally.mean.flatten()
                volume_of_cell = c.volume
                if volume_of_cell is None or volume_of_cell <= 0:
                    print(f"  Skipping {c.name}: volume={volume_of_cell}")
                    continue
                flux = (openmc_flux / volume_of_cell) * SOURCE_STRENGTH
                flux_spectra_by_cell[c.name] = flux
                cell_volumes[c.name] = float(volume_of_cell)

            def cell_name_to_plot_material(name):
                if name in STRUCT_CELL_NAMES:
                    return PLOT_MATERIAL_HASTELLOY
                if name in VACUUM_CELL_NAMES:
                    return PLOT_VACUUM_LABEL
                return name

            # Aggregate by plot material: volume-weighted average flux (one curve per material; one "structure" for all struct cells)
            by_material = {}
            for name, flux in flux_spectra_by_cell.items():
                mat = cell_name_to_plot_material(name)
                vol = cell_volumes[name]
                if mat not in by_material:
                    by_material[mat] = []
                by_material[mat].append((flux, vol))
            norm_flux_by_material = {}
            total_flux_by_material = {}
            for mat, pairs in by_material.items():
                total_vol = sum(v for _, v in pairs)
                if total_vol <= 0:
                    continue
                flux_agg = sum(f * v for f, v in pairs) / total_vol
                norm_flux_by_material[mat] = flux_agg / lethargy_bin_width
                total_flux_by_material[mat] = float(np.sum(flux_agg))
            sp.close()

            # Radial inset: merge consecutive segments with same plot material (one bar per material)
            radial_merged = []
            for cell, r_lo, r_hi, lbl in cells_radial_for_plot:
                plot_lbl = (PLOT_MATERIAL_HASTELLOY if lbl in STRUCT_CELL_NAMES else
                           PLOT_VACUUM_LABEL if lbl in VACUUM_RADIAL_LABELS else lbl)
                if radial_merged and radial_merged[-1][2] == plot_lbl:
                    radial_merged[-1] = (radial_merged[-1][0], r_hi, plot_lbl)
                else:
                    radial_merged.append((r_lo, r_hi, plot_lbl))

            material_color_map = {mat: MATERIAL_COLORS.get(mat, MATERIAL_COLORS.get('struct', '#888')) for mat in norm_flux_by_material}

            # Plot: one line per material (all struct cells merged into one "structure" line)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for mat, norm_flux in norm_flux_by_material.items():
                legend_label = PLOT_STRUCT_LABEL if mat == PLOT_MATERIAL_HASTELLOY else mat
                ax1.step(energy_edges_eV[:-1], norm_flux, where='post', lw=2,
                        color=material_color_map.get(mat, '#888'), label=legend_label)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlabel("Energy [eV]")
            ax1.set_ylabel("Flux per unit lethargy [n/cm²-s]")
            ax1.set_title(f"Flux per lethargy: {subdir}\n{source_neutron_energy} MeV, {SOURCE_STRENGTH:.1e} n/s")
            ax1.legend()
            ax1.grid(True, which="both", ls=":", alpha=0.5)
            ax1.set_xlim(1e-5, 2e7)
            ax1.set_ylim(1e1, 1e13)

            # Radial build inset: one bar per material (struct merged as HASTELLOY C-276)
            ax_rad = fig1.add_axes([0.08, 0.48, 0.14, 0.05])
            r_max = radial_merged[-1][1] if radial_merged else 40
            left = 0
            for r_lo, r_hi, plot_lbl in radial_merged:
                width = (r_hi - r_lo) / r_max
                color = MATERIAL_COLORS.get(plot_lbl, '#888')
                ax_rad.barh(0.5, width, height=0.4, left=left / r_max, color=color, edgecolor='black', linewidth=0.4)
                left = r_hi
            ax_rad.set_xlim(0, 1)
            ax_rad.set_ylim(0, 1)
            ax_rad.set_xticks(np.linspace(0, 1, 5))
            ax_rad.set_xticklabels([f'{r_max*x:.0f}' for x in np.linspace(0, 1, 5)], fontsize=5)
            ax_rad.set_xlabel('r (cm)', fontsize=6)
            ax_rad.set_yticks([])

            fig1.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
            fig1.savefig(os.path.join(output_dir, 'flux_per_lethargy.png'), dpi=300)
            plt.close(fig1)
            print(f"  Saved {os.path.join(output_dir, 'flux_per_lethargy.png')}")

            # Print scalar flux (integrated over all energies) per material
            print(f"\n  Target flux: {target_flux:.3e} n/cm²/s")
            for mat_label in sorted(total_flux_by_material.keys()):
                flux_val = total_flux_by_material[mat_label]
                ratio = flux_val / target_flux if target_flux else float('nan')
                display = PLOT_STRUCT_LABEL if mat_label == PLOT_MATERIAL_HASTELLOY else mat_label
                print(f"  {display} flux: {flux_val:.3e} n/cm²/s  (ratio: {ratio:.3f})")

            # flux_verification.csv: one row per material (structure = total struct)
            flux_rows = [
                {'Region': (PLOT_STRUCT_LABEL if mat_label == PLOT_MATERIAL_HASTELLOY else mat_label),
                 'Flux_n_per_cm2_per_s': total_flux_by_material[mat_label],
                 'Ratio_to_target': total_flux_by_material[mat_label] / target_flux if target_flux else ''}
                for mat_label in sorted(total_flux_by_material.keys())
            ]
            flux_df = pd.DataFrame(flux_rows)
            flux_df.to_csv('flux_verification.csv', index=False)
            print(f"  Cell flux verification exported to flux_verification.csv")

            # Target-cell-only CSV: reactions, initial/final atoms, masses, activities, specific activities (Cu64, Cu67)
            try:
                irrad_h = 8.0
                cool_d = 1.0
                try:
                    import run_config as _rc
                    irrad_h = getattr(_rc, 'IRRADIATION_HOURS', [8.0])
                    irrad_h = irrad_h[0] if isinstance(irrad_h, (list, tuple)) else irrad_h
                    cool_d = getattr(_rc, 'COOLDOWN_DAYS', [1.0])
                    cool_d = cool_d[0] if isinstance(cool_d, (list, tuple)) else cool_d
                except ImportError:
                    pass
                _write_fusion_irradiation_target_csv(
                    sp, cells, output_dir, batches,
                    irradiation_hours=irrad_h, cooldown_days=cool_d, zn64_enrichment=zn64_enrichment,
                )
            except Exception as e:
                print(f"  [fusion_irradiation_target CSV] Error: {e}")
          
        print(f"\nReturning to original directory: {original_dir}")
        os.chdir(original_dir)
    finally:
        os.chdir(original_dir)

    print("\n" + "="*60)
    print("Simulation complete!")
    print(f"All output files saved to: {output_dir}")
    print("="*60)

def _run_one_job(runner, job_kwargs):
    """Single job for parallel execution. Returns whatever run_irradiation_simulation returns."""
    try:
        return runner.run_simulation(**job_kwargs)
    except Exception as e:
        print(f"Job failed: {job_kwargs.get('z_inner_thickness')} {job_kwargs.get('z_outer_thickness')} ... : {e}")
        return None


def run_full_pipeline(C):
    """Run fusion + simple_analyze + zn_waste from dashboard config C."""
    import glob
    import zipfile
    import shutil
    from simple_analyze import IrradiationAnalyzer
    from zn_waste import ZnWasteAnalyzer

    # Inject config into simple_analyze (target height, economics, analysis times)
    import simple_analyze as sa
    for k in ('IRRADIATION_HOURS', 'COOLDOWN_DAYS', 'TARGET_HEIGHT_CM'):
        if hasattr(C, k):
            setattr(sa, k, getattr(C, k))
    for k in dir(C):
        if k.startswith('ECON_'):
            setattr(sa, k, getattr(C, k))

    base = os.path.abspath(getattr(C, 'RUN_BASE_DIR', os.getcwd()))
    statepoints_dir = os.path.join(base, getattr(C, 'STATEPOINTS_DIR', 'statepoints'))
    analyze_dir = os.path.join(base, getattr(C, 'ANALYZE_DIR', getattr(C, 'RESULTS_DIR', 'analyze')))
    zn_waste_dir = os.path.join(base, getattr(C, 'ZN_WASTE_DIR', 'zn_waste'))
    npv_dir = os.path.join(base, getattr(C, 'NPV_DIR', 'npv'))
    mode = getattr(C, 'RUN_MODE', 'single_zn64')
    dual = mode == 'dual'

    def _build_jobs():
        jobs = []
        z_inner_list = getattr(C, 'Z_INNER_THICKNESSES', [0])
        z_outer_list = getattr(C, 'Z_OUTER_THICKNESSES', [5])
        struct_list = getattr(C, 'STRUCT_THICKNESSES', [0.5])
        boron_list = getattr(C, 'BORON_THICKNESSES', [0])
        multi_list = getattr(C, 'MULTI_THICKNESSES', [0])
        mod_list = getattr(C, 'MODERATOR_THICKNESSES', [0])
        zn64_list = getattr(C, 'ZN64_ENRICHMENTS', [0.4917])
        zn67_list = getattr(C, 'ZN67_ENRICHMENTS', [0.0404])
        complex_geom = getattr(C, 'COMPLEX_GEOM', False)
        run_both_geom = getattr(C, 'RUN_BOTH_GEOM', False)

        def _add_job(job_dict):
            if run_both_geom:
                # Run only complex geometry (no simple); analyze writes to analyze/complex/outer/
                jobs.append({**job_dict, 'complex_geom': True})
            else:
                jobs.append({**job_dict, 'complex_geom': complex_geom})

        for z_inner in z_inner_list:
            for z_outer in z_outer_list:
                for struct in struct_list:
                    for boron in boron_list:
                        for multi in multi_list:
                            for mod in mod_list:
                                if mode == 'single_zn64':
                                    for zn64 in zn64_list:
                                        _add_job({
                                            'zn64_enrichment': zn64, 'zn67_enrichment_inner': None,
                                            'zn67_enrichment_outer': None, 'z_inner_thickness': z_inner,
                                            'z_outer_thickness': z_outer, 'struct_thickness': struct,
                                            'boron_thickness': boron, 'moderator_thickness': mod,
                                            'multi_thickness': multi,
                                        })
                                elif mode == 'single_zn67':
                                    for zn67 in zn67_list:
                                        _add_job({
                                            'zn64_enrichment': 0.4917, 'zn67_enrichment_inner': None,
                                            'zn67_enrichment_outer': zn67, 'z_inner_thickness': 0,
                                            'z_outer_thickness': z_outer, 'struct_thickness': struct,
                                            'boron_thickness': boron, 'moderator_thickness': mod,
                                            'multi_thickness': multi,
                                        })
                                else:
                                    for zn64 in zn64_list:
                                        for zn67 in zn67_list:
                                            if z_inner <= 0:
                                                continue
                                            _add_job({
                                                'zn64_enrichment': zn64, 'zn67_enrichment_inner': zn67,
                                                'zn67_enrichment_outer': None, 'z_inner_thickness': z_inner,
                                                'z_outer_thickness': z_outer, 'struct_thickness': struct,
                                                'boron_thickness': boron, 'moderator_thickness': mod,
                                                'multi_thickness': multi,
                                            })
        return jobs

    def _run_fusion():
        os.makedirs(statepoints_dir, exist_ok=True)
        runner = FusionIrradiation(
            inner_radius=getattr(C, 'INNER_RADIUS_CM', 5.0),
            target_height=getattr(C, 'TARGET_HEIGHT_CM', 100.0),
            source_neutron_energy=getattr(C, 'SOURCE_NEUTRON_ENERGY_MEV', 14.1),
            particles=getattr(C, 'PARTICLES', int(100e5)),
            batches=getattr(C, 'BATCHES', 20),
            root_dir=statepoints_dir,
        )
        jobs = _build_jobs()
        for j in jobs:
            j['original_dir'] = statepoints_dir
        if not jobs:
            print("No fusion jobs to run (check RUN_MODE and thickness/enrichment lists).")
            return []
        parallel = getattr(C, 'RUN_PARALLEL', False) and len(jobs) > 1
        max_jobs = getattr(C, 'MAX_JOBS', 4)
        if parallel:
            with mp.Pool(processes=min(max_jobs, len(jobs))) as pool:
                pool.starmap(_run_one_job, [(runner, j) for j in jobs], chunksize=1)
        else:
            for j in jobs:
                _run_one_job(runner, j)
        case_dirs = []
        prefix = getattr(C, 'OUTPUT_PREFIX', 'irrad_output')
        for d in sorted(glob.glob(os.path.join(statepoints_dir, prefix + '_*'))):
            if os.path.isdir(d) and glob.glob(os.path.join(d, 'statepoint.*.h5')):
                case_dirs.append(os.path.basename(d))
        return case_dirs

    def _run_simple_analyze(case_dirs):
        # analyze/simple/outer and analyze/complex/outer (mirror statepoints: simple vs _complex subdirs)
        # Each branch: cu_summary_all.csv, zn_summary_all.csv in separate folders for simple and complex cases
        os.makedirs(analyze_dir, exist_ok=True)
        prefix = getattr(C, 'OUTPUT_PREFIX', 'irrad_output')
        simple_case_dirs = [d for d in case_dirs if not d.endswith('_complex')]
        complex_case_dirs = [d for d in case_dirs if d.endswith('_complex')]

        for branch, allowed in [('simple', simple_case_dirs), ('complex', complex_case_dirs)]:
            if not allowed:
                continue
            out_outer = os.path.join(analyze_dir, branch, 'outer')
            os.makedirs(out_outer, exist_ok=True)
            print(f"  Saving {branch} case results to {out_outer} (cu_summary_all.csv, zn_summary_all.csv) — {len(allowed)} case(s): {allowed}")
            IrradiationAnalyzer(base_dir=statepoints_dir, output_dir=out_outer, output_prefix=prefix, pattern=None, outer_material_id=1).run(
                aggregate_to=out_outer, per_case_to=out_outer, save_aggregate=True, save_per_case=True, layout='geometry', allowed_dirs=allowed)
            if dual:
                out_inner = os.path.join(analyze_dir, branch, 'inner')
                os.makedirs(out_inner, exist_ok=True)
                IrradiationAnalyzer(base_dir=statepoints_dir, output_dir=out_inner, output_prefix=prefix, pattern=None, outer_material_id=0).run(
                    aggregate_to=out_inner, per_case_to=out_inner, save_aggregate=True, save_per_case=True, layout='geometry', allowed_dirs=allowed)
            # Combined plot for this branch
            try:
                from simple_analyze import plot_production_vs_purity_by_cooldown
                cu_outer_path = os.path.join(out_outer, 'cu_summary_all.csv')
                if os.path.exists(cu_outer_path):
                    cu_outer = pd.read_csv(cu_outer_path)
                    cu_inner = None
                    if dual:
                        cu_inner_path = os.path.join(analyze_dir, branch, 'inner', 'cu_summary_all.csv')
                        cu_inner = pd.read_csv(cu_inner_path) if os.path.exists(cu_inner_path) else None
                    plot_production_vs_purity_by_cooldown(cu_outer, out_outer, cu_inner_df=cu_inner)
            except Exception as e:
                print(f"  Warning: Could not generate production_vs_purity_by_cooldown for {branch}: {e}")

    def _run_zn_waste(case_for_waste):
        os.makedirs(zn_waste_dir, exist_ok=True)
        from_dir = os.path.join(statepoints_dir, case_for_waste)
        # Use analyze/simple/outer or analyze/complex/outer to match statepoints layout
        branch = 'complex' if case_for_waste.endswith('_complex') else 'simple'
        out_outer = os.path.join(analyze_dir, branch, 'outer')
        if not os.path.isdir(out_outer):
            # Fallback to the other branch's folder if this one wasn't created (e.g. only one geometry ran)
            out_outer = os.path.join(analyze_dir, 'simple', 'outer') if branch == 'complex' else os.path.join(analyze_dir, 'complex', 'outer')
        for f in ['zn_summary.csv', 'zn_summary_all.csv']:
            src = os.path.join(out_outer, '_by_case', case_for_waste, f) if f == 'zn_summary.csv' else os.path.join(out_outer, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(from_dir, f))
        analyzer = ZnWasteAnalyzer(output_dir=zn_waste_dir, output_prefix=getattr(C, 'OUTPUT_PREFIX', 'irrad_output'),
                                  irrad_hours=getattr(C, 'ECON_IRRAD_HOURS', 8760),
                                  cooldown_days=getattr(C, 'ECON_COOLDOWN_DAYS', 1))
        analyzer.run(from_dir=from_dir)

    def _zip_results():
        zip_path = getattr(C, 'RESULTS_ZIP_PATH', os.path.join(base, getattr(C, 'RESULTS_ZIP', 'results.zip')))
        dirs_to_zip = [statepoints_dir, analyze_dir, zn_waste_dir, npv_dir]
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for d in dirs_to_zip:
                if not os.path.isdir(d):
                    continue
                for root, _, files in os.walk(d):
                    for f in files:
                        if 'jendl' in f.lower():
                            continue
                        zf.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), base))
        print(f"Saved {zip_path}")

    print("=" * 60)
    print("RUN DASHBOARD: fusion → simple_analyze → zn_waste")
    print("=" * 60)
    print(f"Mode: {mode}  |  Statepoints: {statepoints_dir}")
    print()

    print("[1/3] Fusion irradiation...")
    case_dirs = _run_fusion()
    if not case_dirs:
        print("No case dirs produced. Exit.")
        return
    print(f"  Cases: {len(case_dirs)}")

    print("\n[2/3] Simple analyze (outer" + (" + inner)" if dual else ")") + "...")
    _run_simple_analyze(case_dirs)

    complex_geom = getattr(C, 'COMPLEX_GEOM', False)
    run_both_geom = getattr(C, 'RUN_BOTH_GEOM', False)
    case_for_waste = getattr(C, 'ZN_WASTE_CASE_DIR', None)
    if case_for_waste is None:
        idx = getattr(C, 'ZN_WASTE_CASE_INDEX', None)
        if idx is not None and 0 <= idx < len(case_dirs):
            case_for_waste = case_dirs[idx]
        elif case_dirs:
            case_for_waste = case_dirs[0]
    # When only complex was run (RUN_BOTH_GEOM), case_dirs are *_complex; use first if ZN_WASTE_CASE_DIR doesn't exist
    if run_both_geom and case_dirs and (case_for_waste not in case_dirs or not os.path.isdir(os.path.join(statepoints_dir, case_for_waste))):
        case_for_waste = case_dirs[0]
    if complex_geom:
        print("\n[3/3] Zn waste skipped (complex geometry).")
    elif case_for_waste:
        print("\n[3/3] Zn waste (selected case)...")
        _run_zn_waste(case_for_waste)
    else:
        print("\n[3/3] Zn waste skipped (set ZN_WASTE_CASE_INDEX or ZN_WASTE_CASE_DIR).")

    # FLARE NPV is run once after all cases (combined data-driven only) from run.py, not per case.

    print("\nZipping results...")
    _zip_results()
    zip_name = getattr(C, 'RESULTS_ZIP', 'results.zip')
    print(f"\nDone. Unzip {zip_name} to open statepoints/, analyze/, zn_waste/")