"""
OpenMC Planar Fusion Neutron Irradiation Model

This script simulates neutron irradiation of materials using a FLARE point source geometry.
It supports depletion calculations and outputs final material composition.

"""

import openmc
import openmc.deplete
import numpy as np
import pandas as pd
import os
import shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.constants import Avogadro


# Physical constants
MEV_TO_JOULES = 1.602176634e-13  # J/MeV

def create_target_material(zn64_enrichment=0.486):
    """
    Create target material for irradiation with specified Zn-64 enrichment.
    
    Parameters:
    -----------
    zn64_enrichment : float
        Fractional enrichment of Zn-64 (default: 0.486 for natural Zn)
        Use 0.486 for natural, or values like 0.50, 0.60, 0.90, etc. for enriched
    
    Returns:
    --------
    openmc.Material : Configured material marked as depletable
    """

    mat_inner = openmc.Material(material_id=0, name='target_material_inner')
    mat_outer = openmc.Material(material_id=1, name='target_material_outer')

    # Enriched Zn-64 - use natural composition for other isotopes
    # Natural Zn composition: Zn64: 48.6%, Zn66: 27.9%, Zn67: 4.1%, Zn68: 18.8%, Zn70: 0.6%
    # Redistribute remaining isotopes proportionally
    natural_fractions = {'Zn66': 0.279, 'Zn67': 0.041, 'Zn68': 0.188, 'Zn70': 0.0062}
    fractions_64_50 = {'Zn66': 0.277, 'Zn67': 0.040, 'Zn68': 0.177, 'Zn70': 0.002}
    fractions_64_53 = {'Zn66': 0.274, 'Zn67': 0.0398, 'Zn68': 0.155, 'Zn70': 0.001}
    fractions_64_60 = {'Zn66': 0.162, 'Zn67': 0.0277, 'Zn68': 0.1038, 'Zn70': 0.0008}
    fractions_64_70 = {'Zn66': 0.047, 'Zn67': 0.016, 'Zn68': 0.03, 'Zn70': 0.0007} # 70.6% 64Cu
    fractions_64_80 = {'Zn66': 0.012, 'Zn67': 0.0043, 'Zn68': 0.014, 'Zn70': 0.0006}
    fractions_64_90 = {'Zn66': 0.007, 'Zn67': 0.0023, 'Zn68': 0.0091, 'Zn70': 0.0005}
    fractions_64_99 = {'Zn66': 0.004, 'Zn67': 0.0011, 'Zn68': 0.0031, 'Zn70': 0.0004} # 99.9% 64Cu
  

    zn64_enrichment_map = {
        0.50: fractions_64_50,
        0.53: fractions_64_53,
        0.60: fractions_64_60,
        0.70: fractions_64_70,
        0.80: fractions_64_80,
        0.90: fractions_64_90,
        0.99: fractions_64_99
    }
    
    M_Zn64 = openmc.data.atomic_mass('Zn64')
    M_Zn66 = openmc.data.atomic_mass('Zn66')
    M_Zn67 = openmc.data.atomic_mass('Zn67')
    M_Zn68 = openmc.data.atomic_mass('Zn68')
    M_Zn70 = openmc.data.atomic_mass('Zn70')

    if zn64_enrichment != 0.486: #if Zn64 enriched
        if zn64_enrichment not in zn64_enrichment_map:
            raise ValueError(f"zn64_enrichment {zn64_enrichment} not in map. Available: {list(zn64_enrichment_map.keys())}")
        else:
            fractions = zn64_enrichment_map[zn64_enrichment]
            total_others = sum(fractions.values())
            
            # Calculate Zn66 to ensure all fractions sum to 1.0
            zn66_fraction = fractions['Zn66']
            zn67_fraction = fractions['Zn67']
            zn68_fraction = fractions['Zn68']
            zn70_fraction = fractions['Zn70']
            
            # Verify fractions sum to 1.0
            total_frac = zn64_enrichment + zn66_fraction + zn67_fraction + zn68_fraction + zn70_fraction
            if abs(total_frac - 1.0) > 0.01:
                # Adjust Zn66 to make total = 1.0
                zn66_fraction = 1.0 - zn64_enrichment - zn67_fraction - zn68_fraction - zn70_fraction
                print(f"  Adjusted Zn66 fraction to {zn66_fraction:.4f} (total now = 1.0)")
            
            mat_outer.add_nuclide('Zn64', zn64_enrichment)
            mat_outer.add_nuclide('Zn66', zn66_fraction)
            mat_outer.add_nuclide('Zn67', zn67_fraction)
            mat_outer.add_nuclide('Zn68', zn68_fraction)
            mat_outer.add_nuclide('Zn70', zn70_fraction)

            M_zn_64_enriched = (M_Zn64 * zn64_enrichment + M_Zn66 * zn66_fraction + 
                               M_Zn67 * zn67_fraction + M_Zn68 * zn68_fraction + M_Zn70 * zn70_fraction)
            zn_64_enriched_density = 7.14 * (M_zn_64_enriched / 65.38)
            mat_outer.set_density('g/cm3', zn_64_enriched_density)
            print(f"Enriched Zn{zn64_enrichment*100:.1f}%: density = {zn_64_enriched_density:.4f} g/cm³")
    elif zn64_enrichment == 0.486:
        mat_outer.set_density('g/cm3', 7.14)
        mat_outer.add_element('Zn', 1.0)
    mat_outer.depletable = True
    mat_outer.temperature = 600  # K
  
    mat_inner.set_density('g/cm3', 7.14)
    mat_inner.add_element('Zn', 1.0)
    mat_inner.depletable = True
    mat_inner.temperature = 600  # K

    struct_material = openmc.Material(name='eurofer97', material_id=2)
    struct_material.temperature = 600  # K
    struct_material.set_density('g/cm3', 7.85)
    struct_material.add_element('Fe', 0.8924, 'wo')
    struct_material.add_element('Cr', 0.09, 'wo')
    struct_material.add_element('W', 0.011, 'wo')
    struct_material.add_element('Mn', 0.004, 'wo')
    struct_material.add_element('Ta', 0.0012, 'wo')
    struct_material.add_element('C', 0.0011, 'wo')
    struct_material.add_element('N', 0.0003, 'wo')

    multi_material = openmc.Material(name='multi', material_id=3)
    multi_material.temperature = 600  # K
    multi_material.set_density('g/cm3', 3.21)
    multi_material.add_element('Be', 1.0)

    moderator_material = openmc.Material(name='moderator', material_id=4)
    moderator_material.temperature = 600  # K
    moderator_material.set_density('g/cm3', 2.0)  # Graphite density (typical: 1.7-2.2 g/cm³)
    moderator_material.add_element('C', 1.0)

    vacuum_mat = openmc.Material(name='hydrogen', material_id=5)
    vacuum_mat.temperature = 600  # K
    vacuum_mat.add_nuclide('H1', 1.0)  # 100% H-1
    vacuum_mat.set_density('g/cm3', 1e-10)

    water_mat = openmc.Material(name='water', material_id=6)
    water_mat.temperature = 293.15  # K (20°C)
    water_mat.set_density('g/cm3', 1.0)
    water_mat.add_element('H', 2.0)
    water_mat.add_element('O', 1.0)

    materials = [mat_inner, mat_outer, struct_material, multi_material, moderator_material, vacuum_mat, water_mat]
    
    return materials


def create_geometry(materials, target_height=100, z_inner_thickness=15, z_outer_thickness=30, struct_thickness=2, moderator_thickness=10, multi_thickness=10, zn64_enrichment=0.486):
    """
    Create planar slab geometry within a spherical vacuum boundary.
    Target top surface is positioned at z=0 (origin).

    Parameters:
    -----------
    target_material : openmc.Material
        Material for the target slab
    target_thickness : float
        Thickness of target in z-direction (cm)
    target_size : float
        Target x,y dimensions (cm), default 40.64 cm
    source_size : float
        Source x,y dimensions (cm), default 40.64 cm (larger than target)

    Returns:
    --------
    tuple : (openmc.Geometry, target_cell, entrance_surface)
    """
    # Create spherical vacuum boundary (100 cm radius, centered at origin)
    sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')
    inner_radius = 5.0  # cm away from neutron source
    
    # Calculate cumulative radii - only add thickness if > 0
    current_radius = inner_radius
    
    # Structure layer (between inner_radius and struct outer)
    if struct_thickness > 0:
        struct_outer_radius = current_radius + struct_thickness
    else:
        struct_outer_radius = current_radius
    current_radius = struct_outer_radius
    
    # Inner target layer
    if z_inner_thickness > 0:
        inner_target_outer_radius = current_radius + z_inner_thickness
    else:
        inner_target_outer_radius = current_radius
    current_radius = inner_target_outer_radius
    
    # Multiplier layer
    if multi_thickness > 0:
        multi_outer_radius = current_radius + multi_thickness
    else:
        multi_outer_radius = current_radius
    current_radius = multi_outer_radius
    
    # Moderator layer
    if moderator_thickness > 0:
        moderator_outer_radius = current_radius + moderator_thickness
    else:
        moderator_outer_radius = current_radius
    current_radius = moderator_outer_radius
    
    # Outer target layer
    if z_outer_thickness > 0:
        outer_target_outer_radius = current_radius + z_outer_thickness
    else:
        outer_target_outer_radius = current_radius
    
    # The outermost radius of all layers
    outermost_radius = outer_target_outer_radius

    # Z boundaries
    z_top = openmc.ZPlane(z0=+target_height/2, name='target_top')
    z_bottom = openmc.ZPlane(z0=-target_height/2, name='target_bottom')
    
    # Structure Z boundaries (caps above/below target)
    if struct_thickness > 0:
        struct_top = openmc.ZPlane(z0=+((target_height/2)+struct_thickness), name='struct_top')
        struct_bottom = openmc.ZPlane(z0=-((target_height/2)+struct_thickness), name='struct_bottom')
    else:
        struct_top = z_top
        struct_bottom = z_bottom
    
    # Radial surfaces - only create if needed
    inner_surface = openmc.ZCylinder(r=inner_radius, name='inner_surface')
    outer_surface = openmc.ZCylinder(r=outermost_radius, name='outer_surface')
    
    # Create intermediate surfaces only if their corresponding layers exist
    surfaces_list = [inner_surface]
    
    if struct_thickness > 0 and struct_outer_radius != inner_radius:
        struct_outer_surface = openmc.ZCylinder(r=struct_outer_radius, name='struct_outer')
        surfaces_list.append(struct_outer_surface)
    else:
        struct_outer_surface = inner_surface
    
    if z_inner_thickness > 0 and inner_target_outer_radius != struct_outer_radius:
        inner_target_outer_surface = openmc.ZCylinder(r=inner_target_outer_radius, name='inner_target_outer')
        surfaces_list.append(inner_target_outer_surface)
    else:
        inner_target_outer_surface = struct_outer_surface
    
    if multi_thickness > 0 and multi_outer_radius != inner_target_outer_radius:
        multi_outer_surface = openmc.ZCylinder(r=multi_outer_radius, name='multi_outer')
        surfaces_list.append(multi_outer_surface)
    else:
        multi_outer_surface = inner_target_outer_surface
    
    if moderator_thickness > 0 and moderator_outer_radius != multi_outer_radius:
        moderator_outer_surface = openmc.ZCylinder(r=moderator_outer_radius, name='moderator_outer')
        surfaces_list.append(moderator_outer_surface)
    else:
        moderator_outer_surface = multi_outer_surface
    
    surfaces_list.append(outer_surface)
    
    # Calculate areas for surfaces with non-zero radii differences
    areas = {}
    for surface in surfaces_list:
        if hasattr(surface, 'r') and surface.r > 0:
            areas[surface.name] = 2 * np.pi * surface.r * target_height

    # Create regions and cells - only for layers with thickness > 0
    cells_list = []
    regions_to_exclude = []  # Track regions to exclude from vacuum
    
    # Inner target region (between struct outer and inner target outer)
    if z_inner_thickness > 0:
        inner_target_region = +z_bottom & -z_top & +struct_outer_surface & -inner_target_outer_surface
        inner_target_cell = openmc.Cell(cell_id=3, name='inner_target', fill=materials[0], region=inner_target_region)
        cells_list.append(inner_target_cell)
        regions_to_exclude.append(inner_target_region)
        materials[0].volume = np.pi * (inner_target_outer_radius**2 - struct_outer_radius**2) * target_height
    else:
        inner_target_cell = None
        materials[0].volume = 0
    
    # Multiplier region
    if multi_thickness > 0:
        multi_region = +z_bottom & -z_top & +inner_target_outer_surface & -multi_outer_surface
        multi_cell = openmc.Cell(cell_id=4, name='multi', fill=materials[3], region=multi_region)
        cells_list.append(multi_cell)
        regions_to_exclude.append(multi_region)
        materials[3].volume = np.pi * (multi_outer_radius**2 - inner_target_outer_radius**2) * target_height
    else:
        multi_cell = None
    
    # Moderator region
    if moderator_thickness > 0:
        moderator_region = +z_bottom & -z_top & +multi_outer_surface & -moderator_outer_surface
        moderator_cell = openmc.Cell(cell_id=5, name='moderator', fill=materials[4], region=moderator_region)
        cells_list.append(moderator_cell)
        regions_to_exclude.append(moderator_region)
        materials[4].volume = np.pi * (moderator_outer_radius**2 - multi_outer_radius**2) * target_height
    else:
        moderator_cell = None
    
    # Outer target region
    if z_outer_thickness > 0:
        outer_target_region = +z_bottom & -z_top & +moderator_outer_surface & -outer_surface
        outer_target_cell = openmc.Cell(cell_id=6, name='outer_target', fill=materials[1], region=outer_target_region)
        cells_list.append(outer_target_cell)
        regions_to_exclude.append(outer_target_region)
        materials[1].volume = np.pi * (outer_target_outer_radius**2 - moderator_outer_radius**2) * target_height
    else:
        outer_target_cell = None
        materials[1].volume = 0
    
    # Structure region (wraps around everything - caps + inner ring)
    if struct_thickness > 0:
        # Structure: caps above/below + inner ring between inner_radius and struct_outer
        struct_base_region = +struct_bottom & -struct_top & +inner_surface & -outer_surface
        # Exclude all target/multi/moderator regions from structure
        structure_region = struct_base_region
        for region in regions_to_exclude:
            structure_region = structure_region & ~region
        struct_cell = openmc.Cell(cell_id=7, name='struct', fill=materials[2], region=structure_region)
        cells_list.append(struct_cell)
        # Volume: caps + inner cylinder ring
        struct_volume = (
            2 * np.pi * (outermost_radius**2 - inner_radius**2) * struct_thickness  # two caps
            + np.pi * (struct_outer_radius**2 - inner_radius**2) * target_height     # inner ring
        )
        materials[2].volume = struct_volume
    else:
        struct_cell = None
        materials[2].volume = 0
    
    # Vacuum cylinder region: inside the inner radius, within z boundaries (source region)
    vacuum_cylinder_region = -inner_surface & +z_bottom & -z_top
    vacuum_cylinder_cell = openmc.Cell(cell_id=2, name='vacuum_cylinder', fill=materials[5], region=vacuum_cylinder_region)
    cells_list.insert(0, vacuum_cylinder_cell)
    
    # Water region: everything inside sphere not filled by other cells (outside irradiation chamber)
    water_region = -sphere
    # Exclude the vacuum cylinder
    water_region = water_region & ~vacuum_cylinder_region
    # Exclude structure if it exists
    if struct_thickness > 0:
        water_region = water_region & ~structure_region
    # Exclude all target/multi/moderator regions
    for region in regions_to_exclude:
        water_region = water_region & ~region
    water_cell = openmc.Cell(cell_id=1, name='water', fill=materials[6], region=water_region)
    cells_list.insert(0, water_cell)  # Add water first

    # Create geometry
    universe = openmc.Universe(cells=cells_list)
    geometry = openmc.Geometry(universe)

    # Create plot
    plot = openmc.Plot()
    plot.width = (200.0, 200.0)
    plot.origin = (0.0, 0.0, 0.0)
    plot.basis = 'xz'
    plot.filename = f'geometry_inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_multi{multi_thickness}_moderator{moderator_thickness}_zn{zn64_enrichment*100:.1f}%'
    plot.pixels = (1000, 1000)
    plot.color_by = 'material'

    plots = openmc.Plots([plot])

    # Return cells that have fill material (exclude vacuum)
    output_cells = [c for c in [struct_cell, inner_target_cell, outer_target_cell, moderator_cell, multi_cell] if c is not None]

    return geometry, output_cells, surfaces_list, areas, plots

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

def add_axis_labels(plot_filename, output_dir='.', 
                    z_inner_thickness=5, z_outer_thickness=20, struct_thickness=2,
                    multi_thickness=0, moderator_thickness=0, zn64_enrichment=0.486):
    """
    Add axis labels and legend to geometry plot.
    
    Parameters
    ----------
    plot_filename : str
        Base filename of the plot (without extension)
    output_dir : str
        Directory containing the plot
    z_inner_thickness : float
        Inner target thickness (cm)
    z_outer_thickness : float
        Outer target thickness (cm)
    struct_thickness : float
        Structure thickness (cm)
    multi_thickness : float
        Multiplier thickness (cm)
    moderator_thickness : float
        Moderator thickness (cm)
    zn64_enrichment : float
        Zn-64 enrichment fraction
    """
    plot_path = os.path.join(output_dir, f'{plot_filename}.png')
    if not os.path.exists(plot_path):
        print(f"Warning: Plot file not found: {plot_path}")
        return
    
    img = plt.imread(plot_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display image with proper extent
    ax.imshow(img, extent=[-100, 100, -100, 100], origin='upper')
    
    # Calculate radii for legend
    inner_radius = 5.0
    struct_outer = inner_radius + struct_thickness
    inner_target_outer = struct_outer + z_inner_thickness
    
    if multi_thickness > 0:
        multi_outer = inner_target_outer + multi_thickness
    else:
        multi_outer = inner_target_outer
    
    if moderator_thickness > 0:
        moderator_outer = multi_outer + moderator_thickness
    else:
        moderator_outer = multi_outer
    
    outer_target_outer = moderator_outer + z_outer_thickness
    
    # Add ticks every 10 cm with minor ticks every 5 cm
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
    
    # Create legend with layer information - only include layers with thickness > 0
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label=f'Water (outside chamber)'),
        Patch(facecolor='white', edgecolor='black', label=f'Vacuum Cylinder (r < {inner_radius:.1f} cm)'),
    ]
    
    # Track current radius for legend
    current_r = inner_radius
    
    if struct_thickness > 0:
        legend_elements.append(
            Patch(facecolor='gray', edgecolor='black', label=f'Structure/EUROFER ({current_r:.1f}-{struct_outer:.1f} cm)')
        )
        current_r = struct_outer
    
    if z_inner_thickness > 0:
        legend_elements.append(
            Patch(facecolor='red', edgecolor='black', label=f'Inner Target/Zn-{zn64_enrichment*100:.1f}% ({current_r:.1f}-{inner_target_outer:.1f} cm)')
        )
        current_r = inner_target_outer
    
    if multi_thickness > 0:
        legend_elements.append(
            Patch(facecolor='cyan', edgecolor='black', label=f'Multiplier/Be ({current_r:.1f}-{multi_outer:.1f} cm)')
        )
        current_r = multi_outer
    
    if moderator_thickness > 0:
        legend_elements.append(
            Patch(facecolor='blue', edgecolor='black', label=f'Moderator/Graphite ({current_r:.1f}-{moderator_outer:.1f} cm)')
        )
        current_r = moderator_outer
    
    if z_outer_thickness > 0:
        legend_elements.append(
            Patch(facecolor='orange', edgecolor='black', label=f'Outer Target/Zn-nat ({current_r:.1f}-{outer_target_outer:.1f} cm)')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              framealpha=0.9, edgecolor='black', title='Cell Layers', title_fontsize=10)
    
    # Add title - only show non-zero thicknesses
    title_parts = ['Geometry Cross-Section (XZ Plane)']
    params = []
    if struct_thickness > 0:
        params.append(f'Struct: {struct_thickness}cm')
    if z_inner_thickness > 0:
        params.append(f'Inner: {z_inner_thickness}cm')
    if multi_thickness > 0:
        params.append(f'Multi: {multi_thickness}cm')
    if moderator_thickness > 0:
        params.append(f'Mod: {moderator_thickness}cm')
    if z_outer_thickness > 0:
        params.append(f'Outer: {z_outer_thickness}cm')
    
    if params:
        title_parts.append(', '.join(params))
    
    ax.set_title('\n'.join(title_parts), fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved labeled geometry plot: {plot_path}")

def create_tallies(cells, surfaces, verbose=True):
    """
    Create tallies: surface flux + energy profile per surface; cell flux, energy profile, and heating.

    Parameters:
    -----------
    cells : list
    surfaces : list
    verbose : bool
        If True, print a summary of created tallies (default True).

    Returns:
    --------
    openmc.Tallies
    """
    tallies = openmc.Tallies()
    labels = []  # (tally_name, description) for print

    cell_filter = openmc.CellFilter([cell for cell in cells if cell.fill is not None])

    pts_per_decade = 100
    start, stop = 1e-5, 20e6
    log_start, log_stop = np.log10(start), np.log10(stop)
    npts = int(np.ceil((log_stop - log_start) * pts_per_decade)) + 1
    energy_grid = np.logspace(log_start, log_stop, num=npts)
    energy_filter = openmc.EnergyFilter(energy_grid)

    # --- Surface: current (-> flux/area) and current vs energy (-> flux energy profile) ---
    for surf in surfaces:
        name = getattr(surf, 'name', None) or f'surface_{id(surf)}'

        t = openmc.Tally(name=f'{name}_tally')
        t.filters = [openmc.SurfaceFilter(surf)]
        t.scores = ['current']
        tallies.append(t)
        labels.append((t.name, 'surface current -> flux/area'))

        t = openmc.Tally(name=f'{name}_spectra')
        t.filters = [openmc.SurfaceFilter(surf), energy_filter]
        t.scores = ['current']
        tallies.append(t)
        labels.append((t.name, 'surface neutron energy profile'))

    # --- Cell: flux, flux vs energy, heating ---
    t = openmc.Tally(name='volume_flux')
    t.filters = [cell_filter]
    t.scores = ['flux']
    tallies.append(t)
    labels.append((t.name, 'cell flux'))

    t = openmc.Tally(name='volume_flux_spectra')
    t.filters = [cell_filter, energy_filter]
    t.scores = ['flux']
    tallies.append(t)
    labels.append((t.name, 'cell neutron energy profile'))

    t = openmc.Tally(name='volumetric_heating')
    t.filters = [cell_filter]
    t.scores = ['heating-local']
    tallies.append(t)
    labels.append((t.name, 'cell heating (W/cm³)'))

    def make_tally(name, scores, filters:list=None, nuclides:list=None):
        tally = openmc.Tally(name=name)
        tally.scores = scores
        if filters is not None:
            tally.filters = filters
        if nuclides is not None:
            tally.nuclides = nuclides
        tallies.append(tally)
        labels.append((tally.name, name))
        return tally

    # Zn64(n,p)Cu64, 67Zn(n,p)Cu67, 68Zn(n,d)67Cu
    Cu_tally_tot    = make_tally('Total Cu Production rxn rate',     ['(n,p)', '(n,d)'], nuclides=['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70'])
    Cu_tally        = make_tally('Cu Production rxn rates',          ['(n,p)', '(n,d)'], filters=[cell_filter], nuclides=['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70'])
    Cu_energy_tally = make_tally('Cu Production rxn rates spectrum', ['(n,p)', '(n,d)'], filters=[cell_filter, energy_filter], nuclides=['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70'])

    #Zn64(n,2n)Zn64, 68Zn(n,2n)67Zn, , 68Zn(n,2n)67Zn, 64Zn(n,γ)65Zn, 66Zn(n,2n)65Zn, 66Zn(n,γ)67Zn, 65Zn(n,γ)66Zn...
    Zn_tally_tot    = make_tally('Total Zn rxn rate',     ['(n,gamma)', '(n,2n)'], nuclides=['Zn63', 'Zn64', 'Zn65','Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70'])
    Zn_tally        = make_tally('Zn rxn rates',          ['(n,gamma)', '(n,2n)'], filters=[cell_filter], nuclides=['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70'])
    Zn_energy_tally = make_tally('Zn rxn rates spectrum', ['(n,gamma)', '(n,2n)'], filters=[cell_filter, energy_filter], nuclides=['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70'])

    if verbose:
        print("\nTallies created (label check):")
        for nm, desc in labels:
            print(f"  {nm}: {desc}")

    return tallies

def extract_surface_flux(statepoint_file, source_strength, surfaces, areas):
    """
    Extract surface flux and energy spectra from statepoint file by converting
    current to flux: Flux = (|current| * source_strength) / area.

    Parameters:
    -----------
    statepoint_file : str
        Path to OpenMC statepoint file
    source_strength : float
        Source strength in neutrons/second
    surfaces : list
        List of OpenMC surfaces with '{name}_tally' and '{name}_spectra' tallies
    areas : dict
        Dict of surface name -> area (cm²)

    Returns:
    --------
    tuple : (flux_data, spectra_data, energy_bin_centers, energy_bin_edges)
        flux_data: dict name -> flux (n/cm²/s)
        spectra_data: dict name -> array of flux per energy bin (n/cm²/s)
        energy_bin_centers: 1d array of energy bin centers (eV)
        energy_bin_edges: 1d array of energy bin edges (eV)
    """
    sp = openmc.StatePoint(statepoint_file)
    flux_data = {}
    spectra_data = {}
    energy_bin_centers = None
    energy_bin_edges = None

    for surface in surfaces:
        name = getattr(surface, 'name', None) or f'surface_{id(surface)}'
        tally = sp.get_tally(name=f'{name}_tally')
        current = tally.mean.flatten()[0]
        flux = (np.abs(current) * source_strength) / areas[name]
        flux_data[name] = float(flux)

        spectra_tally = sp.get_tally(name=f'{name}_spectra')
        if energy_bin_edges is None:
            for f in spectra_tally.filters:
                if isinstance(f, openmc.EnergyFilter):
                    v = np.asarray(f.values)
                    energy_bin_edges = np.unique(np.sort(v))  # Get bin edges
                    if len(energy_bin_edges) > 1:
                        energy_bin_centers = (energy_bin_edges[:-1] + energy_bin_edges[1:]) / 2
                    else:
                        energy_bin_centers = energy_bin_edges
                    break
        current_spectra = spectra_tally.mean.flatten()
        flux_spectra = (np.abs(current_spectra) * source_strength) / areas[name]
        spectra_data[name] = np.atleast_1d(flux_spectra)

    return flux_data, spectra_data, energy_bin_centers, energy_bin_edges

'''
def export_final_composition(results, material_id_list=None, output_file='final_composition.csv'):
    """
    Export final material composition to CSV with all materials side-by-side.
    
    Parameters:
    -----------
    results : openmc.deplete.Results
        Depletion results object
    material_id_list : list
        List of material IDs to extract (e.g., ['0', '1'])
    output_file : str
        Output CSV filename
    
    Returns:
    --------
    pd.DataFrame : Composition data with columns for each material
    """
    from scipy.constants import Avogadro
    import numpy as np
    
    if material_id_list is None:
        material_id_list = ['0']
    
    final_step = results[-1]
    material_names = {'0': "Inner Breeding", '1': "Outer Breeding"}
    
    # Collect all nuclides from all materials
    all_nuclides = set()
    material_data = {}
    
    for idx, material_id in enumerate(material_id_list):
        depleted_material = final_step.get_material(str(material_id))
        if depleted_material is None:
            raise ValueError(f"Could not find material with ID={material_id}")
        
        nuclide_atoms = depleted_material.get_nuclide_atoms()
        total_atoms = sum(nuclide_atoms.values())
        
        mat_dict = {}
        for nuclide, atoms in nuclide_atoms.items():
            if atoms > 0:
                all_nuclides.add(nuclide)
                percentage = (atoms / total_atoms) * 100
                
                # Get atomic mass
                try:
                    atomic_mass_amu = openmc.data.atomic_mass(nuclide)
                    mass_grams = (atoms * atomic_mass_amu) / Avogadro
                except:
                    atomic_mass_amu = None
                    mass_grams = None
                
                # Calculate activity (Bq)
                try:
                    half_life_seconds = openmc.data.half_life(nuclide)
                    if half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                        decay_constant = np.log(2) / half_life_seconds
                        activity_Bq = atoms * decay_constant
                    else:
                        activity_Bq = 0.0  # Stable isotope
                except:
                    activity_Bq = 0.0
                
                # Convert to other units
                activity_GBq = activity_Bq / 1e9
                activity_mCi = activity_Bq / 3.7e7
                activity_Ci = activity_Bq / 3.7e10
                
                mat_dict[nuclide] = {
                    'atoms': atoms,
                    'percentage': percentage,
                    'mass_grams': mass_grams,
                    'Bq': activity_Bq,
                    'GBq': activity_GBq,
                    'mCi': activity_mCi,
                    'Ci': activity_Ci
                }
        
        material_data[idx] = mat_dict
    
    # Build combined DataFrame
    rows = []
    for nuclide in sorted(all_nuclides):
        row = {'Nuclide': nuclide}
        
        for idx, material_id in enumerate(material_id_list):
            mat_name = material_names.get(material_id, f"Material_{material_id}")
            if nuclide in material_data[idx]:
                data = material_data[idx][nuclide]
                row[f'Atoms_{mat_name}'] = data['atoms']
                row[f'Percentage_{mat_name}'] = data['percentage']
                row[f'Mass_grams_{mat_name}'] = data['mass_grams']
                row[f'Bq_{mat_name}'] = data['Bq']
                row[f'GBq_{mat_name}'] = data['GBq']
                row[f'mCi_{mat_name}'] = data['mCi']
                row[f'Ci_{mat_name}'] = data['Ci']
            else:
                row[f'Atoms_{mat_name}'] = 0
                row[f'Percentage_{mat_name}'] = 0
                row[f'Mass_grams_{mat_name}'] = 0
                row[f'Bq_{mat_name}'] = 0
                row[f'GBq_{mat_name}'] = 0
                row[f'mCi_{mat_name}'] = 0
                row[f'Ci_{mat_name}'] = 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by total atoms across all materials
    atoms_cols = [f'Atoms_{material_names.get(mid, f"Material_{mid}")}' for mid in material_id_list]
    df['Total_Atoms'] = df[atoms_cols].sum(axis=1)
    df = df.sort_values('Total_Atoms', ascending=False)
    df = df.drop('Total_Atoms', axis=1)
    
    # Export to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\nFinal composition exported to {output_file}")
    print(f"Total number of nuclides: {len(df)}")
    print(f"Materials included: {material_id_list}")
    
    return df
'''

def run_irradiation_simulation(
    target_flux,
    neutron_energy,
    source_neutron_energy,
    num_timesteps,
    timestep_size,
    num_cooldown,
    cooldown_timestep_size,
    particles=10000,
    batches=10,
    zn64_enrichment=0.486,
    z_inner_thickness=15,
    z_outer_thickness=30,
    struct_thickness=2,
    moderator_thickness=10,
    multi_thickness=10,
    zn_cooldown=False,
    original_dir=None,
):
    """
    Run complete irradiation simulation with depletion.

    Material is defined in create_target_material() function.
    Modify that function to change the target material.

    Parameters:
    -----------
    target_flux : float
        Desired neutron flux at surface (n/cm²/s)
    neutron_energy : float
        Neutron energy (MeV)
    num_timesteps : int
        Number of depletion timesteps
    timestep_size : float
        Duration of each timestep (seconds)
    num_cooldown : int
        Number of cooldown timesteps
    cooldown_timestep_size : float
        Duration of each cooldown timestep (seconds)
    particles : int, optional
        Number of particles per batch (default: 10000)
    batches : int, optional
        Number of batches for transport calculation (default: 50)
    zn64_enrichment : float
        Fractional enrichment of Zn-64 (default: 0.486 for natural Zn)
    output_dir : str
        Directory for output files

    Returns:
    --------
    pd.DataFrame : Final composition data after irradiation and after cooldown
        Final composition data after irradiation and after cooldown
    """
    print("="*60)
    print("OpenMC Fusion Neutron Irradiation Simulation")
    print("="*60)
    
    root_dir = os.path.abspath(original_dir)
    if zn_cooldown == False:
        subdir = f'radial_output_inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_multi{multi_thickness}_moderator{moderator_thickness}_zn{zn64_enrichment*100:.1f}%'
    else:
        subdir = f'zinc_cooldown_inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_multi{multi_thickness}_moderator{moderator_thickness}_zn{zn64_enrichment*100:.1f}%'
    output_dir = os.path.join(root_dir, subdir)

    print(f"\nCreating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save original working directory

    # Geometry parameters
    target_height = 100 # cm
    inner_radius = 5 # cm

    print(f"\nGeometry Parameters:")
    print(f"  Target height: {target_height} cm")
    print(f"  Z inner thickness: {z_inner_thickness} cm")
    print(f"  Z outer thickness: {z_outer_thickness} cm")
    print(f"  Struct thickness: {struct_thickness} cm")
    print(f"  Moderator thickness: {moderator_thickness} cm")
    print(f"  Multi thickness: {multi_thickness} cm")
    print(f"  Inner radius: {inner_radius} cm")

    print(f"\nInput Parameters:")
    print(f"  Target flux: {target_flux:.2e} n/cm²/s")
    print(f"  D-T Neutron energy: {neutron_energy} MeV")
    print(f"  Source Neutron energy: {source_neutron_energy} MeV")
    print(f"  Material: Zn (modify create_target_material() to change)")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Timestep size: {timestep_size:.2e} seconds ({timestep_size/3600:.1f} hours)")
    print(f"  Cooldown timesteps: {num_cooldown}")
    print(f"  Cooldown timestep size: {cooldown_timestep_size:.2e} seconds ({cooldown_timestep_size/3600:.1f} hours)")
    print(f"  Total irradiation: {num_timesteps*timestep_size/3600:.1f} hours")
    print(f"  Particles: {particles}")
    print(f"  Batches: {batches}")
    print(f"  Zn-64 enrichment: {zn64_enrichment*100:.1f}%")
    print(f"  Output directory: {output_dir}")


    print(f"\nSource Parameters:")
    print(f"  Source strength: {5e13:.2e} n/s")
    print(f"  Source power: {140:.2e} W")

    # Create materials
    print("\nCreating materials...")
    materials = create_target_material(zn64_enrichment=zn64_enrichment)

    print(f"  Inner material:     {materials[0].name}")
    print(f"  Outer material:     {materials[1].name}")
    print(f"  Struct material:    {materials[2].name}")
    print(f"  Multi material:     {materials[3].name}")
    print(f"  Moderator material: {materials[4].name}")
    print(f"  Vacuum material:    {materials[5].name}")

    # Create geometry
    print("Creating geometry...")
    geometry, cells, surfaces, areas, plots = create_geometry(materials, target_height, z_inner_thickness, z_outer_thickness, struct_thickness, moderator_thickness, multi_thickness, zn64_enrichment)

    # Create source
    print("Creating source...")
    source = create_source(source_neutron_energy, target_height)

    # Create tallies
    print("Creating tallies...")
    tallies = create_tallies(cells, surfaces)

    # Settings
    settings = openmc.Settings()
    settings.source = source
    settings.particles = particles
    settings.batches = batches
    settings.run_mode = 'fixed source'
    
    # Efficiency optimizations for fixed source
    settings.survival_biasing = True  # Improves statistics for thin absorbers
    settings.cutoff = {'weight': 0.25, 'weight_avg': 0.5}  # Weight window cutoff
    
    # Photon transport (disabled for faster neutron-only transport)
    settings.photon_transport = False

    # Create model
    model = openmc.Model(geometry=geometry, settings=settings, materials=materials, tallies=tallies, plots=plots)

    # Get the script directory (where the chain file should be)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to output directory for simulation
    print(f"\nChanging to output directory: {output_dir}")
    try:
        os.chdir(output_dir)

        # Copy chain file to output directory if needed
        chain_file = "JENDL_chain.xml"

        # Check multiple possible locations for the chain file
        possible_locations = [
            os.path.join(script_dir, chain_file),  # Script directory
            os.path.join(root_dir, chain_file),  # Root / original working directory
            chain_file  # Current directory (if already copied)
        ]

        chain_file_source = None
        for location in possible_locations:
            if os.path.exists(location):
                chain_file_source = location
                print(f"Found chain file at: {location}")
                break

        if chain_file_source and not os.path.exists(chain_file):
            print(f"Copying chain file to output directory...")
            shutil.copy2(chain_file_source, chain_file)
        elif not chain_file_source:
            raise FileNotFoundError(
                f"Chain file '{chain_file}' not found in:\n" +
                "\n".join(f"  - {loc}" for loc in possible_locations)
            )

        # Export model files to output directory
        print("Exporting model files...")
        model.materials.export_to_xml()
        model.geometry.export_to_xml()
        model.plots.export_to_xml()
        model.settings.export_to_xml()
        model.tallies.export_to_xml()

        print("Plotting geometry...")
        openmc.plot_geometry()
        plot_filename = f'geometry_inner{z_inner_thickness}_outer{z_outer_thickness}_struct{struct_thickness}_multi{multi_thickness}_moderator{moderator_thickness}_zn{zn64_enrichment*100:.1f}%'
        add_axis_labels(
            plot_filename, 
            output_dir=output_dir,
            z_inner_thickness=z_inner_thickness,
            z_outer_thickness=z_outer_thickness,
            struct_thickness=struct_thickness,
            multi_thickness=multi_thickness,
            moderator_thickness=moderator_thickness,
            zn64_enrichment=zn64_enrichment
        )

        print("Running simulation...")
        openmc.run()

        '''
        # Setup depletion
        print("\nSetting up depletion calculation...")
        operator = openmc.deplete.CoupledOperator(
            model,
            chain_file=chain_file,  # JENDL depletion chain
            normalization_mode='source-rate',
            reduce_chain_level=4,
        )

        time_steps = [timestep_size] * num_timesteps + [cooldown_timestep_size] * num_cooldown
        source_rates = [5e13] * num_timesteps + [0.0] * num_cooldown


        # Run depletion
        print(f"Running depletion for {num_timesteps} timesteps...")
        integrator = openmc.deplete.CF4Integrator(
            operator,
            timesteps=time_steps,
            source_rates=source_rates,
        )
        integrator.integrate()

        # Post-process results
        print("\nPost-processing results...")
        results = openmc.deplete.Results('depletion_results.h5')
        '''
        # Extract and display surface flux from first depletion step
        print("\nFlux Verification (from first depletion step):")
        statepoint_file = f'statepoint.{batches}.h5'
        source_strength = 5e13
        
        # Check if statepoint file exists before trying to extract flux
        if os.path.exists(statepoint_file):
            flux_data, spectra_data, energy_bin_centers, energy_bin_edges = extract_surface_flux(statepoint_file, source_strength, surfaces, areas)
            # Plot surface flux spectra per unit lethargy
            if spectra_data and energy_bin_centers is not None and energy_bin_edges is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Dark2(np.linspace(0, 1, len(spectra_data)))

                # Calculate lethargy width for each bin: Δu = ln(E_high / E_low)
                # Lethargy u = ln(E_ref / E), so Δu = ln(E_high) - ln(E_low) = ln(E_high/E_low)
                lethargy_width = np.log(energy_bin_edges[1:] / energy_bin_edges[:-1])
                
                # Convert energy from eV to MeV for x-axis (matching SHINE format)
                energy_MeV = energy_bin_centers / 1e6  # eV to MeV
            
                for i, (surface_name, flux_spectrum) in enumerate(sorted(spectra_data.items())):
                    # Convert flux per bin to flux per unit lethargy
                    # φ(u) = flux_per_bin / Δu  [n/cm²/s/lethargy]
                    flux_per_lethargy = flux_spectrum / lethargy_width
                    ax.plot(energy_MeV, flux_per_lethargy, lw=1.0, color=colors[i], label=surface_name)

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel("Neutron Energy [MeV]")
                ax.set_ylabel("Neutron flux [n/cm²·s·lethargy]")
                ax.set_title("Neutron Energy Spectrum (SHINE-like format)")
                ax.legend(fontsize=9, loc='upper left')
                ax.grid(True, which="both", ls=":", alpha=0.5)
                
                # Set x-axis limits to match SHINE format (10^-11 to 10^2 MeV)
                ax.set_xlim(1e-11, 1e2)
                
                # Add reference energy annotations (in MeV)
                ax.axvline(x=2.5e-8, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # Thermal 0.025 eV = 2.5e-8 MeV
                ax.axvline(x=14.1, color='orange', linestyle='--', alpha=0.5, linewidth=1)  # D-T 14.1 MeV
                
                # Add text annotations after setting y-limits
                ylims = ax.get_ylim()
                ax.text(5e-8, ylims[1]*0.3, 'Thermal\n0.025 eV', fontsize=8, color='gray', va='top')
                ax.text(10, ylims[1]*0.3, 'D-T\n14.1 MeV', fontsize=8, color='orange', va='top', ha='right')
                
                fig.tight_layout()
                fig.savefig("surface_flux_spectra.png", dpi=300)
                plt.close(fig)
                print("→ Saved surface flux spectra plot → surface_flux_spectra.png")

            print(f"  Target flux: {target_flux:.3e} n/cm²/s")
            for name in sorted(flux_data.keys()):
                flux = flux_data[name]
                ratio = flux / target_flux if target_flux else float('nan')
                print(f"  {name} flux: {flux:.3e} n/cm²/s")
                print(f"  Ratio ({name}/target): {ratio:.3f}")

            # flux_verification.csv: one row per surface (Surface, Flux_n_per_cm2_per_s, Ratio_to_target)
            flux_rows = [
                {'Surface': name, 'Flux_n_per_cm2_per_s': flux_data[name], 'Ratio_to_target': flux_data[name] / target_flux if target_flux else ''}
                for name in sorted(flux_data.keys())
            ]
            flux_df = pd.DataFrame(flux_rows)
            flux_df.to_csv('flux_verification.csv', index=False)
            print(f"\nFlux verification (surface fluxes) exported to flux_verification.csv")
        else:
            print(f"  Warning: Statepoint file '{statepoint_file}' not found. Skipping flux verification.")
            print(f"  (This may be normal if depletion process uses different statepoint naming)")

        '''
        # Export final composition
        df = export_final_composition(results, material_id_list=['0', '1'], output_file='final_composition.csv')
        '''
        print(f"\nReturning to original directory: {root_dir}")
        os.chdir(root_dir)
    finally:
        os.chdir(root_dir)

    print("\n" + "="*60)
    print("Simulation complete!")
    print(f"All output files saved to: {output_dir}")
    print("="*60)


def _run_one_job(kwargs: dict):
    """Single job for parallel execution. Returns whatever run_irradiation_simulation returns."""
    try:
        return run_irradiation_simulation(**kwargs)
    except Exception as e:
        print(f"Job failed: {kwargs.get('z_inner_thickness')} {kwargs.get('z_outer_thickness')} ... : {e}")
        return None

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    # ============================================
    # Configuration - Edit these parameters
    # ============================================
    
    # Parallelization
    RUN_PARALLEL = True
    MAX_JOBS = 8
    ROOT_DIR = os.path.abspath(os.getcwd())
    
    # Source parameters
    # Source power: 140 W → source_strength = 140 W / (14.1 MeV × 1.602e-13 J/MeV) ≈ 6.2e13 n/s
    # Using 5e13 n/s (~113 W)
    # Target flux: ~1.5e10 n/cm²/s from 5e13 n/s source
    # SurfArea(sides cylinder): 2πrh = 2π*5*100 = 3141.6 cm²
    TARGET_FLUX = 1.5e10
    SOURCE_NEUTRON_ENERGY = 14.1  # MeV (D-T fusion)
    
    # Simulation parameters
    PARTICLES = int(10e5)  # 1,000,000 particles
    BATCHES = 10
    
    # ============================================
    # Parameter Sweep Configuration
    # ============================================
    
    # Zn-64 enrichments to run
    ZN64_ENRICHMENTS = [0.486, 0.6, 0.7, 0.9, 0.99]
    
    # Geometry thicknesses (cm)
    Z_INNER_THICKNESSES = [0]       # Inner layer (0 = no inner layer)
    Z_OUTER_THICKNESSES = [20]      # Outer Zn layer
    STRUCT_THICKNESSES = [0]        # Structural material (0 = none)
    
    # Multiplier (Be) and Moderator (C) thicknesses to sweep
    MULTI_MOD_THICKNESSES = [0, 5]  # Be multiplier = C moderator [cm]
    
    # ============================================
    # Build Job List
    # ============================================
    
    jobs = []
    
    for zn64_enrichment in ZN64_ENRICHMENTS:
        for z_inner in Z_INNER_THICKNESSES:
            for z_outer in Z_OUTER_THICKNESSES:
                for struct in STRUCT_THICKNESSES:
                    for thickness in MULTI_MOD_THICKNESSES:
                        jobs.append({
                                "target_flux": TARGET_FLUX,
                                "neutron_energy": SOURCE_NEUTRON_ENERGY,
                                "source_neutron_energy": SOURCE_NEUTRON_ENERGY,
                                "num_timesteps": 1,
                                "timestep_size": 8 * 3600,  # 8 hours (not used directly, analysis varies this)
                                "num_cooldown": 0,
                                "cooldown_timestep_size": 0,
                                "particles": PARTICLES,
                                "batches": BATCHES,
                                "zn64_enrichment": zn64_enrichment,
                                "z_inner_thickness": z_inner,
                                "z_outer_thickness": z_outer,
                                "struct_thickness": struct,
                                "moderator_thickness": thickness,
                                "multi_thickness": thickness,
                                "zn_cooldown": False,
                                "original_dir": ROOT_DIR,
                            })
    
    # ============================================
    # Run Jobs
    # ============================================
    
    print("=" * 70)
    print("FUSION IRRADIATION PARAMETER SWEEP")
    print("=" * 70)
    print(f"Zn-64 enrichments: {ZN64_ENRICHMENTS}")
    print(f"Multi=Mod thicknesses: {MULTI_MOD_THICKNESSES} cm")
    print(f"Total jobs: {len(jobs)}")
    print("=" * 70)
    
    if RUN_PARALLEL and len(jobs) > 1:
        with mp.Pool(processes=min(MAX_JOBS, len(jobs))) as pool:
            results = pool.map(_run_one_job, jobs, chunksize=1)
    else:
        results = [_run_one_job(j) for j in jobs]
    
    print("\n" + "=" * 70)
    print("All runs finished.")
    print(f"Output directories created in: {ROOT_DIR}")
    print("Run simple_analyze.py --base-dir . to analyze all cases")
    print("=" * 70)
