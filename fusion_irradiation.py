"""
OpenMC Planar Fusion Neutron Irradiation Model

This script simulates neutron irradiation of materials using a planar source geometry.
It supports depletion calculations and outputs final material composition.

Input Parameters:
    target_flux: Desired neutron flux at surface (n/cm²/s)
    neutron_energy: Monoenergetic source energy (MeV)
    target_thickness: Target slab thickness (cm)
    num_timesteps: Number of depletion timesteps
    timestep_size: Duration of each timestep (seconds)
    particles: Number of particles per batch
    batches: Number of batches for transport calculation

Note: Material is defined in create_target_material() function (default: Hg-198)
"""

import openmc
import openmc.deplete
import numpy as np
import pandas as pd
import os
import shutil
from scipy.constants import Avogadro

# Physical constants
MEV_TO_JOULES = 1.602176634e-13  # J/MeV


def calculate_source_strength(target_flux, neutron_energy, source_area):
    """
    Calculate source strength and power from desired flux.

    Parameters:
    -----------
    target_flux : float
        Desired neutron flux at target surface (n/cm²/s)
    neutron_energy : float
        Neutron energy (MeV)
    source_area : float
        Source plane area (cm²)

    Returns:
    --------
    dict : Contains 'strength' (n/s) and 'power' (W)
    """
    # Source strength: S = Flux × Area
    source_strength = target_flux * source_area  # neutrons/second

    # Power: P = Energy × Source strength
    power_watts = neutron_energy * MEV_TO_JOULES * source_strength

    return {
        'strength': source_strength,
        'power': power_watts,
        'source_area': source_area
    }


# def create_target_material():
#     """
#     Create target material for irradiation.

#     --------
#     openmc.Material : Configured material marked as depletable
#     """
#     mat = openmc.Material(name='target_material')

#     # Pure Hg-198
#     mat.set_density('g/cm3', 13.534)  # Mercury density
#     mat.add_nuclide('Hg198', 1.0)

#     # Mark material as depletable
#     mat.depletable = True

#     return mat

def create_target_material(zn64_enrichment=0.486, zn67_enrichment=0.2663):
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

    mat = openmc.Material(name='target_material', material_id=1)

    # Enriched Zn-64 - use natural composition for other isotopes
    # Natural Zn composition: Zn64: 48.6%, Zn66: 27.9%, Zn67: 4.1%, Zn68: 18.8%, Zn70: 0.6%
    # Redistribute remaining isotopes proportionally
    # natural_fractions = {'Zn66': 0.279, 'Zn67': 0.041, 'Zn68': 0.188, 'Zn70': 0.0062}
    fractions_64_50 = {'Zn66': 0.277, 'Zn67': 0.040, 'Zn68': 0.177, 'Zn70': 0.002}
    fractions_64_53 = {'Zn66': 0.274, 'Zn67': 0.0398, 'Zn68': 0.155, 'Zn70': 0.0}
    fractions_64_60 = {'Zn66': 0.262, 'Zn67': 0.0277, 'Zn68': 0.1038, 'Zn70': 0.0}
    fractions_64_70 = {'Zn66': 0.247, 'Zn67': 0.016, 'Zn68': 0.03, 'Zn70': 0.0} # 70.6% 64Cu
    fractions_64_80 = {'Zn66': 0.232, 'Zn67': 0.0043, 'Zn68': 0.0, 'Zn70': 0.0}
    fractions_64_90 = {'Zn66': 0.217, 'Zn67': 0.0, 'Zn68': 0.0, 'Zn70': 0.0}
    fractions_64_99 = {'Zn66': 0.0001, 'Zn67': 0.0, 'Zn68': 0.0, 'Zn70': 0.0} # 99.9% 64Cu

    fractions_67_7  = {'Zn64': 0.0005, 'Zn66': 0.342, 'Zn68': 0.557, 'Zn70': 0.023} # 7.3 % 67Cu
    fractions_67_17 = {'Zn64': 0.039, 'Zn66': 0.662, 'Zn68': 0.211, 'Zn70': 0.008} # 17.7 % 67Cu
    fractions_67_50 = {'Zn64': 0.029, 'Zn66': 0.51, 'Zn68': 0.059, 'Zn70': 0.0} 
    fractions_67_70 = {'Zn64': 0.023, 'Zn66': 0.242, 'Zn68': 0.035, 'Zn70': 0.0}   

    zn64_enrichment_map = {
        0.50: fractions_64_50,
        0.53: fractions_64_53,
        0.60: fractions_64_60,
        0.70: fractions_64_70,
        0.80: fractions_64_80,
        0.90: fractions_64_90,
        0.99: fractions_64_99
    }
    
    zn67_enrichment_map = {
        0.073: fractions_67_7,
        0.177: fractions_67_17,
        0.50: fractions_67_50,
        0.70: fractions_67_70
    }
    
    M_Zn64 = openmc.data.atomic_mass('Zn64')
    M_Zn66 = openmc.data.atomic_mass('Zn66')
    M_Zn67 = openmc.data.atomic_mass('Zn67')
    M_Zn68 = openmc.data.atomic_mass('Zn68')
    M_Zn70 = openmc.data.atomic_mass('Zn70')

    if zn67_enrichment == 0.2663 and zn64_enrichment != 0.486: #if Zn64 enriched
        if zn64_enrichment not in zn64_enrichment_map:
            raise ValueError(f"zn64_enrichment {zn64_enrichment} not in map. Available: {list(zn64_enrichment_map.keys())}")
        else:
            fractions = zn64_enrichment_map[zn64_enrichment]
            total_others = sum(fractions.values())
            fraction_66 = 1 - (total_others + zn64_enrichment)
            mat.add_nuclide('Zn64', zn64_enrichment)
            mat.add_nuclide('Zn66', fractions['Zn66'] + fraction_66)
            mat.add_nuclide('Zn67', fractions['Zn67'])
            mat.add_nuclide('Zn68', fractions['Zn68'])
            mat.add_nuclide('Zn70', fractions['Zn70'])

            M_zn_64_enriched = M_Zn64 * zn64_enrichment + (M_Zn66 * (fractions['Zn66'] + fraction_66)) + (M_Zn67 * fractions['Zn67']) + (M_Zn68 * fractions['Zn68']) + (M_Zn70 * fractions['Zn70'])
            zn_64_enriched_density = 7.14 * (M_zn_64_enriched / 65.38)
            print(f"Enriched density of Zn{zn64_enrichment*100:.1f}%: {zn_64_enriched_density} g/cm3")
            mat.set_density('g/cm3', zn_64_enriched_density)

    elif zn64_enrichment == 0.486 and zn67_enrichment != 0.2663: #if Zn67 enriched
        if zn67_enrichment not in zn67_enrichment_map:
            raise ValueError(f"zn67_enrichment {zn67_enrichment} not in map. Available: {list(zn67_enrichment_map.keys())}")
        else:
            fractions = zn67_enrichment_map[zn67_enrichment]
            total_others = sum(fractions.values())
            fraction_66 = 1 - (total_others + zn67_enrichment)
            mat.add_nuclide('Zn64', fractions['Zn64'])
            mat.add_nuclide('Zn66', fractions['Zn66'] + fraction_66)
            mat.add_nuclide('Zn67', zn67_enrichment)
            mat.add_nuclide('Zn68', fractions['Zn68'])
            mat.add_nuclide('Zn70', fractions['Zn70'])

            M_zn_67_enriched = M_Zn64 * fractions['Zn64'] + (M_Zn66 * (fractions['Zn66'] + fraction_66)) + (M_Zn67 * zn67_enrichment) + (M_Zn68 * fractions['Zn68']) + (M_Zn70 * fractions['Zn70'])
            zn_67_enriched_density = 7.14 * (M_zn_67_enriched / 65.38)
            print(f"Enriched density of Zn{zn67_enrichment*100:.1f}%: {zn_67_enriched_density} g/cm3")
            mat.set_density('g/cm3', zn_67_enriched_density)
    elif zn64_enrichment == 0.486 and zn67_enrichment == 0.2663: #if both enriched
        # Natural Zn
        mat.set_density('g/cm3', 7.14)
        mat.add_element('Zn', 1.0)
    # Mark material as depletable
    mat.depletable = True
    mat.temperature = 900  # K
    
    return mat


def create_geometry(target_material, target_thickness, target_size=40.64, source_size=40.64, moderator_thickness=20.0):
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
    sphere = openmc.Sphere(r=500.0, boundary_type='vacuum')

    # Target slab boundaries (top surface at z=0, extends downward)
    z_top = openmc.ZPlane(z0=0.0, name='target_top')
    z_bottom = openmc.ZPlane(z0=-target_thickness, name='target_bottom')
    x_min = openmc.XPlane(x0=-target_size/2)
    x_max = openmc.XPlane(x0=target_size/2)
    y_min = openmc.YPlane(y0=-target_size/2)
    y_max = openmc.YPlane(y0=target_size/2)

    # Moderator slab boundaries
    z_moderator_top = openmc.ZPlane(z0=moderator_thickness, name='moderator_top')
    z_moderator_bottom = openmc.ZPlane(z0=0.0, name='moderator_bottom')
    x_moderator_min = openmc.XPlane(x0=-target_size/2)
    x_moderator_max = openmc.XPlane(x0=target_size/2)
    y_moderator_min = openmc.YPlane(y0=-target_size/2)
    y_moderator_max = openmc.YPlane(y0=target_size/2)

    # Create regions
    # Target region: inside slab boundaries
    target_region = +z_bottom & -z_top & +x_min & -x_max & +y_min & -y_max
    moderator_region = +z_moderator_bottom & -z_moderator_top & +x_moderator_min & -x_moderator_max & +y_moderator_min & -y_moderator_max

    # Vacuum region: inside sphere but outside target
    vacuum_region = -sphere & ~target_region & ~moderator_region

    # Create cells
    target_cell = openmc.Cell(name='target', fill=target_material, region=target_region)

    moderator_mat = openmc.Material(name='moderator', material_id=3)
    moderator_mat.set_density('g/cm3', 3.21)
    moderator_mat.add_element('Si', 1.0, 'ao')
    moderator_mat.add_element('C', 1.0, 'ao')
    moderator_cell = openmc.Cell(name='moderator', fill=moderator_mat, region=moderator_region)

    # Vacuum cell (fills everything else in the sphere)
    vacuum_mat = openmc.Material(name='vacuum', material_id=2)
    vacuum_mat.set_density('g/cm3', 1e-10)  # Near vacuum
    vacuum_mat.add_nuclide('H1', 1.0)

    vacuum_cell = openmc.Cell(name='vacuum', fill=vacuum_mat, region=vacuum_region)

    # Create geometry
    geometry = openmc.Geometry([target_cell, vacuum_cell, moderator_cell])

    # Calculate and set volume
    target_volume = target_size * target_size * target_thickness  # cm³
    target_material.volume = target_volume
    moderator_mat.volume = target_size * target_size * moderator_thickness


    return geometry, target_cell, z_top, z_moderator_top, z_bottom


def create_source(source_params, source_neutron_energy, source_size=40.64, source_distance=80):
    """
    Create planar neutron source positioned above the target.
    Target top surface is at z=0, so source is at z=source_distance.

    Parameters:
    -----------
    source_params : dict
        Output from calculate_source_strength()
    neutron_energy : float
        Neutron energy (MeV)
    source_size : float
        Source x,y dimensions (cm), default 40.64 cm
    source_distance : float
        Distance above target top surface (cm), default 60 cm

    Returns:
    --------
    openmc.IndependentSource
    """
    source = openmc.IndependentSource()

    # Source position: target top is at z=0, source is above at z=source_distance
    source_z = source_distance

    # Planar source positioned above the target
    source.space = openmc.stats.Box(
        lower_left=[-source_size/2, -source_size/2, source_z],
        upper_right=[source_size/2, source_size/2, source_z]
    )

    # Monodirectional - straight down (-z direction)
    source.angle = openmc.stats.Monodirectional(reference_uvw=[0, 0, -1])

    # Monoenergetic at specified energy
    source.energy = openmc.stats.Discrete([source_neutron_energy * 1e6], [1.0])  # Convert MeV to eV


    return source


def create_tallies(entrance_surface, target_cell, moderator_surface, exit_surface):
    """
    Create tallies for flux verification.

    Parameters:
    -----------
    entrance_surface : openmc.Surface
        Target entrance surface for current measurement
    target_cell : openmc.Cell
        Target cell for volume flux

    Returns:
    --------
    openmc.Tallies
    """
    tallies = openmc.Tallies()

    # Surface current tally at entrance
    # (Use 'current' instead of 'flux' - OpenMC restriction for surface tallies)
    surface_tally = openmc.Tally(name='surface_current')
    surface_filter = openmc.SurfaceFilter(entrance_surface)
    surface_tally.filters = [surface_filter]
    surface_tally.scores = ['current']
    tallies.append(surface_tally)

    moderator_tally = openmc.Tally(name='moderator_flux')
    moderator_filter = openmc.SurfaceFilter(moderator_surface)
    moderator_tally.filters = [moderator_filter]
    moderator_tally.scores = ['current']
    tallies.append(moderator_tally)

    exit_tally = openmc.Tally(name='exit_current')
    exit_filter = openmc.SurfaceFilter(exit_surface)
    exit_tally.filters = [exit_filter]
    exit_tally.scores = ['current']
    tallies.append(exit_tally)

    # Volume flux tally
    volume_tally = openmc.Tally(name='volume_flux')
    cell_filter = openmc.CellFilter(target_cell)
    volume_tally.filters = [cell_filter]
    volume_tally.scores = ['flux']
    tallies.append(volume_tally)

    return tallies


def extract_surface_flux(statepoint_file, source_strength, target_area):
    """
    Extract surface flux from statepoint file by converting current to flux.

    For monodirectional beam perpendicular to surface:
    Flux = Current × (Source_strength / Target_area)

    Parameters:
    -----------
    statepoint_file : str
        Path to OpenMC statepoint file
    source_strength : float
        Source strength in neutrons/second
    target_area : float
        Target surface area in cm²

    Returns:
    --------
    tuple : (flux, current) where flux is in n/cm²/s and current is raw tally value
    """
    sp = openmc.StatePoint(statepoint_file)

    # Get surface current tally (particles crossing per source particle)
    tally = sp.get_tally(name='surface_current')
    current = tally.mean.flatten()[0]

    # Get moderator current tally
    mod_tally = sp.get_tally(name='moderator_flux')
    mod_current = mod_tally.mean.flatten()[0]

    # Get exit current tally
    exit_tally = sp.get_tally(name='exit_current')
    exit_current = exit_tally.mean.flatten()[0]

    # Convert current to flux
    # current = particles crossing / source particle
    # flux = particles/(cm²·s) = current × (source_strength / target_area)
    flux = (abs(current) * source_strength) / target_area
    moderator_flux = (abs(mod_current) * source_strength) / target_area
    exit_flux = (abs(exit_current) * source_strength) / target_area

    return flux, current, moderator_flux, exit_flux


def export_final_composition(results, material_id='1', material_name='target_material', output_file='final_composition.csv'):
    """
    Export final material composition to CSV.

    Parameters:
    -----------
    results : openmc.deplete.Results
        Depletion results object
    material_id : str
        Material ID to extract (default: '1')
    material_name : str
        Material name to search for if ID lookup fails (default: 'target_material')
    output_file : str
        Output CSV filename

    Returns:
    --------
    pd.DataFrame : Composition data
    """
    # Get final timestep
    final_step = results[-1]

    # Try to get material by ID
    try:
        depleted_material = final_step.get_material(str(material_id))
    except KeyError:
        print(f"Warning: Material with ID={material_id} not found. Searching by name '{material_name}'...")
        # Search for depletable material by name or property
        depleted_material = None
        for mat_id in final_step.mat_to_ind.keys():
            mat = final_step.get_material(str(mat_id))
            if hasattr(mat, 'name') and mat.name == material_name:
                depleted_material = mat
                print(f"Found material '{material_name}' with ID={mat_id}")
                break

        if depleted_material is None:
            raise ValueError(f"Could not find material with ID={material_id} or name='{material_name}'")

    # Get nuclide atoms from the depleted material
    nuclide_atoms = depleted_material.get_nuclide_atoms()

    # Calculate total atoms
    total_atoms = sum(nuclide_atoms.values())

    # Build data for each nuclide
    mat_data = []
    for nuclide, atoms in nuclide_atoms.items():
        if atoms > 0:  # Only include nuclides with atoms present
            percentage = (atoms / total_atoms) * 100

            # Get atomic mass from OpenMC data
            try:
                atomic_mass_amu = openmc.data.atomic_mass(nuclide)
                mass_grams = (atoms * atomic_mass_amu) / Avogadro
            except:
                atomic_mass_amu = None
                mass_grams = None

            mat_data.append({
                'Nuclide': nuclide,
                'Number_of_Atoms': atoms,
                'Percentage': percentage,
                'Mass_grams': mass_grams
            })

    # Create DataFrame and sort by percentage
    df = pd.DataFrame(mat_data)
    df = df.sort_values('Percentage', ascending=False)

    # Export to CSV
    df.to_csv(output_file, index=False)

    print(f"\nFinal composition exported to {output_file}")
    print(f"Total number of nuclides: {len(df)}")

    return df


def run_irradiation_simulation(
    target_flux,
    neutron_energy,
    source_neutron_energy,
    target_thickness,
    moderator_thickness,
    num_timesteps,
    timestep_size,
    num_cooldown,
    cooldown_timestep_size,
    particles=100,
    batches=3,
    zn64_enrichment=0.486,
    zn67_enrichment=0.2663,
    output_dir=None
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
    target_thickness : float
        Target thickness (cm)
    moderator_thickness : float
        Moderator thickness (cm)
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
    zn67_enrichment : float
        Fractional enrichment of Zn-67 (default: 0.2663 for natural Zn)
    output_dir : str
        Directory for output files

    Returns:
    --------
    tuple : (pd.DataFrame, pd.DataFrame)
        Final composition data after irradiation and after cooldown
    """
    print("="*60)
    print("OpenMC Fusion Neutron Irradiation Simulation")
    print("="*60)

    if output_dir is None:
        enrichment_pct_64 = int(zn64_enrichment * 100) 
        enrichment_pct_67 = int(zn67_enrichment * 100)
        output_dir = f'irradiation_output_{enrichment_pct_64}_{enrichment_pct_67}'
    
    # Create output directory
    print(f"\nCreating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save original working directory
    original_dir = os.getcwd()

    # Geometry parameters
    target_size = 40.64  # cm
    source_size = 40.64  # cm (same as target for direct alignment)

    print(f"\nInput Parameters:")
    print(f"  Target flux: {target_flux:.2e} n/cm²/s")
    print(f"  D-T Neutron energy: {neutron_energy} MeV")
    print(f"  Source Neutron energy: {source_neutron_energy} MeV")
    print(f"  Material: Zn (modify create_target_material() to change)")
    print(f"  Thickness: {target_thickness} cm")
    print(f"  Moderator thickness: {moderator_thickness} cm")
    print(f"  Timesteps: {num_timesteps}")
    print(f"  Timestep size: {timestep_size:.2e} seconds ({timestep_size/3600:.1f} hours)")
    print(f"  Cooldown timesteps: {num_cooldown}")
    print(f"  Cooldown timestep size: {cooldown_timestep_size:.2e} seconds ({cooldown_timestep_size/3600:.1f} hours)")
    print(f"  Total irradiation: {num_timesteps*timestep_size/3600:.1f} hours")
    print(f"  Particles: {particles}")
    print(f"  Batches: {batches}")
    print(f"  Zn-64 enrichment: {zn64_enrichment*100:.1f}%")
    print(f"  Zn-67 enrichment: {zn67_enrichment*100:.1f}%")
    print(f"  Output directory: {output_dir}")

    # Calculate source parameters
    source_area = source_size * source_size  # cm²
    source_params = calculate_source_strength(target_flux, neutron_energy, source_area)

    print(f"\nGeometry:")
    print(f"  Target size: {target_size} × {target_size} cm")
    print(f"  Source size: {source_size} × {source_size} cm (aligned with target)")
    print(f"  Target area: {target_size * target_size} cm²")
    print(f"  Source area: {source_area} cm²")

    print(f"\nSource Parameters:")
    print(f"  Source strength: {source_params['strength']:.2e} n/s")
    print(f"  Source power: {source_params['power']:.2e} W ({source_params['power']/1e6:.2f} MW)")

    # Create materials
    print("\nCreating materials...")
    material = create_target_material(zn64_enrichment=zn64_enrichment, zn67_enrichment=zn67_enrichment)

    # Create geometry
    print("Creating geometry...")
    geometry, target_cell, entrance_surface, moderator_surface, exit_surface = create_geometry(
        material, target_thickness, target_size, source_size, moderator_thickness
    )

    # Create source
    print("Creating source...")
    source = create_source(source_params, source_neutron_energy, source_size)

    # Create tallies
    print("Creating tallies...")
    tallies = create_tallies(entrance_surface, target_cell, moderator_surface, exit_surface)

    # Settings
    settings = openmc.Settings()
    settings.source = source
    settings.particles = particles
    settings.batches = batches
    settings.run_mode = 'fixed source'

    # Create model
    model = openmc.Model(geometry=geometry, settings=settings, tallies=tallies)

    # Get the script directory (where the chain file should be)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to output directory for simulation
    print(f"\nChanging to output directory: {output_dir}")
    os.chdir(output_dir)

    # Copy chain file to output directory if needed
    chain_file = "JENDL_chain.xml"

    # Check multiple possible locations for the chain file
    possible_locations = [
        os.path.join(script_dir, chain_file),  # Script directory
        os.path.join(original_dir, chain_file),  # Original working directory
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
    model.settings.export_to_xml()
    model.tallies.export_to_xml()

    # Setup depletion
    print("\nSetting up depletion calculation...")
    operator = openmc.deplete.CoupledOperator(
        model,
        chain_file=chain_file,  # JENDL depletion chain
        normalization_mode='source-rate',
        reduce_chain_level=4,
    )

    # Create time steps
    time_steps = [timestep_size] * num_timesteps
    source_rates = [source_params['strength']] * num_timesteps  # Constant flux


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

    cool_steps = [cooldown_timestep_size] * num_cooldown
    cool_source_rates = [0.0] * num_cooldown

    print(f"Running post irradiation cooldown depletion for {num_cooldown} timesteps...")
    integrator_cool = openmc.deplete.CF4Integrator(
        operator,
        timesteps=cool_steps,
        source_rates=cool_source_rates,
    )
    
    integrator_cool.integrate()
    print("\nPost-processing cooldown results...")
    results_cool = openmc.deplete.Results('depletion_results.h5')

    # Extract and display surface flux from first depletion step
    print("\nFlux Verification (from first depletion step):")
    statepoint_file = f'statepoint.{batches}.h5'
    target_area = target_size * target_size  # cm²
    measured_flux, raw_current, moderator_flux, exit_flux = extract_surface_flux(statepoint_file, source_params['strength'], target_area)
    print(f"  Raw current tally: {raw_current:.6e}") 
    print(f"  Target flux: {target_flux:.3e} n/cm²/s") 
    print(f"  Measured surface flux: {measured_flux:.3e} n/cm²/s")
    print(f"  Ratio (measured/target): {measured_flux/target_flux:.3f}")
    print(f"  Moderator flux: {moderator_flux:.3e} n/cm²/s")
    print(f"  Exit flux: {exit_flux:.3e} n/cm²/s")

    
    # Save flux verification data to CSV
    flux_data = {
        'Parameter': [
            'Total_Irradiation_Time_seconds',
            'Total_Irradiation_Time_hours',
            'Raw_Current_Tally',
            'Target_Flux_n_per_cm2_per_s',
            'Measured_Surface_Flux_n_per_cm2_per_s',
            'Ratio_Measured_to_Target',
            'Moderator_Flux_n_per_cm2_per_s',
            'Exit_Flux_n_per_cm2_per_s',
            'Target_Area_cm2',
            'Source_Strength_n_per_s',
            'Source_Power_W',
            'Number_of_Timesteps',
            'Timestep_Size_seconds'
        ],
        'Value': [
            num_timesteps*timestep_size,
            num_timesteps*timestep_size/3600,
            raw_current,
            target_flux,
            measured_flux,
            measured_flux/target_flux,
            moderator_flux,
            exit_flux,
            target_area,
            source_params['strength'],
            source_params['power'],
            num_timesteps,
            timestep_size
        ]
    }
    flux_df = pd.DataFrame(flux_data)
    flux_csv_file = 'flux_verification.csv'
    flux_df.to_csv(flux_csv_file, index=False)
    print(f"\nFlux verification data exported to {flux_csv_file}")

    # Export final composition
    df = export_final_composition(results, material_id='1', output_file='final_composition.csv')
    df_cool = export_final_composition(results_cool, material_id='1', output_file='final_composition_cooldown.csv')

    # Display top 10 nuclides
    print("\nTop 10 nuclides by percentage:")
    print(df.head(10).to_string(index=False))

    # Return to original directory
    print(f"\nReturning to original directory: {original_dir}")
    os.chdir(original_dir)

    print("\n" + "="*60)
    print("Simulation complete!")
    print(f"All output files saved to: {output_dir}")
    print("="*60)

    return df, df_cool


if __name__ == '__main__':
    # Example usage: Hg-198 irradiation with fusion neutrons
    # To change material, modify create_target_material() function

    # Input parameters
    #5e13 n/s 
    target_flux =  3.027e10 # n/cm²/s 
    neutron_energy = 17.6  # MeV
    source_neutron_energy = 14.1  # MeV
    target_thickness = 20.32  # cm
    moderator_thickness = 20.0  # cm
    num_timesteps = 16 # run for 8 hours
    timestep_size = 0.5 * 3600  # 30 minutes in seconds
    num_cooldown = 4 # 2 days
    cooldown_timestep_size = 12 * 3600  # 12 hours in seconds
    particles = 100  # particles per batch
    batches = 5  # number of batches
    zinc_enrichment_list = [0.486]
    zinc64_enrichment_list = [0.486, 0.50, 0.53, 0.60, 0.70, 0.80, 0.90, 0.99]
    zinc67_enrichment_list = [0.2663, 0.073, 0.177, 0.50, 0.70]

    for zn64_enrichment in zinc64_enrichment_list:
        print(f"\n{'='*60}")
        print(f"Running simulation with Zn-64 enrichment: {zn64_enrichment*100:.1f}%")
        print(f"{'='*60}")

        results_df, results_df_cool = run_irradiation_simulation(
            target_flux=target_flux,
            neutron_energy=neutron_energy,
            source_neutron_energy=source_neutron_energy,
            target_thickness=target_thickness,
            moderator_thickness=moderator_thickness,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            num_cooldown=num_cooldown,
            cooldown_timestep_size=cooldown_timestep_size,
            particles=particles,
            batches=batches,
            zn64_enrichment=zn64_enrichment,
            zn67_enrichment=0.2663
        )
    for zn67_enrichment in zinc67_enrichment_list:
        print(f"\n{'='*60}")
        print(f"Running simulation with Zn-67 enrichment: {zn67_enrichment*100:.1f}%")
        print(f"{'='*60}")

        results_df, results_df_cool = run_irradiation_simulation(
            target_flux=target_flux,
            neutron_energy=neutron_energy,
            source_neutron_energy=source_neutron_energy,
            target_thickness=target_thickness,
            moderator_thickness=moderator_thickness,
            num_timesteps=num_timesteps,
            timestep_size=timestep_size,
            num_cooldown=num_cooldown,
            cooldown_timestep_size=cooldown_timestep_size,
            particles=particles,
            batches=batches,
            zn64_enrichment=0.486,
            zn67_enrichment=zn67_enrichment
        )

