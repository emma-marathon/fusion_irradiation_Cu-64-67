"""
OpenMC D1S fusion irradiation: neutron activation + photon dose (mSv/s) via decay photons.

Run run_irradiation_simulation() for Zn cylinder in water: isotropic 14.06 MeV source,
8 h at 1.7e13 n/s then cooling steps; dose from photon transport. Fast defaults:
particles=2000, batches=3, n_cycles=1, plot_dose_maps=False, mesh (25,25,1).
"""

import openmc
import openmc.deplete
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import xml.etree.ElementTree as ET
from scipy.constants import Avogadro

from matplotlib.colors import LogNorm
from openmc.deplete import d1s
from openmc.deplete.chain import Chain
from openmc.data.data import half_life

# Physical constants
MEV_TO_JOULES = 1.602176634e-13  # J/MeV

# Pig / Zn target geometry options
PIG_WALL_MATERIAL = 'lead'  # 'lead' or 'bismuth'
PIG_ZN_DENSITY_OVERRIDE_G_CM3 = None  # set to override Zn density (g/cm³)
PIG_GEOMETRY_TYPE = 'default'  # 'default' or 'bismuth_quartz'

# Cylinder-in-water geometry (source -> vacuum bubble -> Zn cylinder, surrounded by water)
SOURCE_TO_CYLINDER_CM = 15.24  # 6 inches: distance from source to nearest cylinder face
SPHERE_RADIUS_CM = 100.0
VACUUM_BUBBLE_RADIUS_CM = 40.0
WATER_RADIUS_CM = 100.0
ZN_CYLINDER_MASS_KG = 0.67  # natural Zn cylinder mass
ZN_CYLINDER_RADIUS_CM = 2.5  # radius of Zn cylinder; height from mass + density

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


def get_zn_fractions(zn64_enrichment):
    """Return dict of Zn isotope fractions for given Zn64 enrichment (0–1)."""
    # Natural abundances (other isotopes), renormalized to sum to (1 - zn64_enrichment)
    natural_other = {'Zn66': 0.279, 'Zn67': 0.041, 'Zn68': 0.188, 'Zn70': 0.006}
    other_sum = sum(natural_other.values())
    remaining = 1.0 - zn64_enrichment
    fracs = {'Zn64': zn64_enrichment}
    for iso, nat in natural_other.items():
        fracs[iso] = remaining * (nat / other_sum)
    return fracs


def calculate_enriched_zn_density(zn64_enrichment):
    """Zn density (g/cm³) for given enrichment; slight mass effect ignored, use ~7.14."""
    return 7.14  # g/cm³


# Natural Zn-64 fraction (~48.6%). Use explicit nuclides so cross_sections.xml lookup works
# (many ENDF libraries list Zn64, Zn66, ... not element "Zn").
NATURAL_ZN64_FRACTION = 0.486


def create_materials(zn64_enrichment=None, include_water=False, wall_material=None):
    """Create materials: Zn target (by nuclide for cross-section lookup), pig wall, optional water."""
    if wall_material is None:
        wall_material = PIG_WALL_MATERIAL
    if zn64_enrichment is None:
        zn64_enrichment = NATURAL_ZN64_FRACTION
    zn_mat = openmc.Material(material_id=1, name='zn_target')
    density = PIG_ZN_DENSITY_OVERRIDE_G_CM3 if PIG_ZN_DENSITY_OVERRIDE_G_CM3 is not None else calculate_enriched_zn_density(zn64_enrichment)
    zn_mat.set_density('g/cm3', density)
    fracs = get_zn_fractions(zn64_enrichment)
    for iso, f in fracs.items():
        zn_mat.add_nuclide(iso, f)
    zn_mat.temperature = 294
    zn_mat.depletable = True

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


def create_geometry(target_material, target_thickness, target_size=10.0, source_size=20.0):
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
        Target x,y dimensions (cm), default 10 cm
    source_size : float
        Source x,y dimensions (cm), default 20 cm (larger than target)

    Returns:
    --------
    tuple : (openmc.Geometry, target_cell, entrance_surface)
    """
    # Create spherical vacuum boundary (100 cm radius, centered at origin)
    sphere = openmc.Sphere(r=100.0, boundary_type='vacuum')

    # Target slab boundaries (top surface at z=0, extends downward)
    z_top = openmc.ZPlane(z0=0.0, name='target_top')
    z_bottom = openmc.ZPlane(z0=-target_thickness, name='target_bottom')
    x_min = openmc.XPlane(x0=-target_size/2)
    x_max = openmc.XPlane(x0=target_size/2)
    y_min = openmc.YPlane(y0=-target_size/2)
    y_max = openmc.YPlane(y0=target_size/2)

    # Create regions
    # Target region: inside slab boundaries
    target_region = +z_bottom & -z_top & +x_min & -x_max & +y_min & -y_max

    # Vacuum region: inside sphere but outside target (void, no material)
    vacuum_region = -sphere & ~target_region

    # Create cells
    target_cell = openmc.Cell(name='target', fill=target_material, region=target_region)
    vacuum_cell = openmc.Cell(name='vacuum', fill=None, region=vacuum_region)

    # Create geometry
    geometry = openmc.Geometry([target_cell, vacuum_cell])

    # Calculate and set volume
    target_volume = target_size * target_size * target_thickness  # cm³
    target_material.volume = target_volume

    return geometry, target_cell, z_top


def _zn_cylinder_height_cm(mass_kg, radius_cm, density_g_cm3=7.14):
    """Height (cm) of a right circular Zn cylinder for given mass (kg) and radius (cm)."""
    vol_cm3 = (mass_kg * 1000.0) / density_g_cm3
    return vol_cm3 / (np.pi * radius_cm**2)


def create_geometry_cylinder_water(zn_mat, water_mat, cylinder_radius_cm=None):
    """
    Create geometry: Zn cylinder (0.67 kg natural Zn) inside a spherical vacuum bubble,
    surrounded by water. Source-to-cylinder distance SOURCE_TO_CYLINDER_CM; cylinder axis along z.

    Parameters
    ----------
    zn_mat : openmc.Material
        Depletable Zn target material
    water_mat : openmc.Material
        Water bath material
    cylinder_radius_cm : float
        Radius of Zn cylinder (cm); height computed from ZN_CYLINDER_MASS_KG and density.

    Returns
    -------
    tuple : (openmc.Geometry, zn_cell)
    """
    if cylinder_radius_cm is None:
        cylinder_radius_cm = ZN_CYLINDER_RADIUS_CM
    density = 7.14  # g/cm³, must match zn_mat
    mass_kg = ZN_CYLINDER_MASS_KG
    h_cm = _zn_cylinder_height_cm(mass_kg, cylinder_radius_cm, density)
    z_front = SOURCE_TO_CYLINDER_CM
    z_back = SOURCE_TO_CYLINDER_CM + h_cm

    sphere_water = openmc.Sphere(r=WATER_RADIUS_CM, boundary_type='vacuum')
    sphere_vacuum = openmc.Sphere(r=VACUUM_BUBBLE_RADIUS_CM)
    z_plane_front = openmc.ZPlane(z0=z_front, name='cylinder_front')
    z_plane_back = openmc.ZPlane(z0=z_back, name='cylinder_back')
    z_cyl = openmc.ZCylinder(r=cylinder_radius_cm)

    # Regions: Zn inside cylinder; vacuum inside bubble (void, no material); water between spheres
    zn_region = -z_plane_back & +z_plane_front & -z_cyl
    vacuum_region = -sphere_vacuum & ~zn_region
    water_region = -sphere_water & +sphere_vacuum

    zn_cell = openmc.Cell(name='zn_cylinder', fill=zn_mat, region=zn_region)
    vacuum_cell = openmc.Cell(name='vacuum_bubble', fill=None, region=vacuum_region)
    water_cell = openmc.Cell(name='water', fill=water_mat, region=water_region)

    geometry = openmc.Geometry([zn_cell, vacuum_cell, water_cell])

    vol_cm3 = np.pi * cylinder_radius_cm**2 * h_cm
    zn_mat.volume = vol_cm3

    return geometry, zn_cell


def create_source(source_params, neutron_energy, source_size=20.0, source_distance=20.0, for_cylinder_water=False):
    """
    Create planar neutron source.
    Planar: source at z=source_distance, direction -z. Cylinder-water: source at z=0, direction +z.

    Parameters:
    -----------
    source_params : dict
        Output from calculate_source_strength()
    neutron_energy : float
        Neutron energy (MeV)
    source_size : float
        Source x,y dimensions (cm)
    source_distance : float
        Distance above target (planar only)
    for_cylinder_water : bool
        If True, source at z=0 directed +z toward cylinder.

    Returns:
    --------
    openmc.IndependentSource
    """
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0, 0, 0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([neutron_energy * 1e6], [1.0])  # MeV -> eV
    source.particle = "neutron"
    return source


def create_tallies_dose_mesh(my_geometry, mesh_dimension=(50, 50, 1), radionuclides=None):
    """
    Create photon dose tally on a regular mesh (xy slice) for D1S dose from photon transport.

    Parameters:
    -----------
    my_geometry : openmc.Geometry
        Model geometry
    mesh_dimension : tuple
        (nx, ny, nz) voxels; default (50,50,1) for faster run, use (100,100,1) for resolution.
    radionuclides : list of str, optional
        If provided, add ParentNuclideFilter so D1S apply_time_correction can be used.

    Returns:
    --------
    tuple : (openmc.Tallies, openmc.RegularMesh) for D1S post-processing
    """
    mesh = openmc.RegularMesh().from_domain(
        my_geometry,
        dimension=list(mesh_dimension),
    )
    energies, pSv_cm2 = openmc.data.dose_coefficients(particle="photon", geometry="AP")
    dose_filter = openmc.EnergyFunctionFilter(
        energies, pSv_cm2, interpolation="cubic"  # ICRP-recommended
    )
    particle_filter = openmc.ParticleFilter(["photon"])
    mesh_filter = openmc.MeshFilter(mesh)
    filters = [particle_filter]
    if radionuclides is not None:
        filters.append(openmc.ParentNuclideFilter(list(radionuclides)))
    filters.extend([mesh_filter, dose_filter])
    dose_tally = openmc.Tally()
    dose_tally.filters = filters
    dose_tally.scores = ["flux"]
    dose_tally.name = "photon_dose_on_mesh"
    return openmc.Tallies([dose_tally]), mesh


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


# Half-lives (s) for Cu activity: Cu-64 12.7006 h, Cu-67 61.83 h
CU64_HALFLIFE_S = 12.7006 * 3600
CU67_HALFLIFE_S = 61.83 * 3600
BQ_PER_MCI = 3.7e7


def compute_run_summary_activity(
    results,
    material_id='1',
    timesteps=None,
    source_rates=None,
    zn_volume_cm3=None,
    zn_density_g_cm3=7.14,
    zn_mass_kg=None,
):
    """
    From depletion results and run parameters, compute Zn/Cu run summary: volume, mass,
    irrad/cool times, Cu-64/Cu-67 activity (mCi, Bq) and purities. Returns dict suitable for print/CSV.
    """
    if zn_mass_kg is None:
        zn_mass_kg = ZN_CYLINDER_MASS_KG
    if zn_volume_cm3 is None:
        zn_volume_cm3 = (zn_mass_kg * 1000.0) / zn_density_g_cm3
    mass_g = zn_mass_kg * 1000.0

    irrad_h = 0.0
    cool_s = 0.0
    if timesteps is not None and source_rates is not None and len(timesteps) == len(source_rates):
        for ts, sr in zip(timesteps, source_rates):
            if sr > 0:
                irrad_h += ts / 3600.0
            else:
                cool_s += ts
    cooldown_days = cool_s / (24.0 * 3600.0)

    final_step = results[-1]
    try:
        depleted_material = final_step.get_material(str(material_id))
    except KeyError:
        depleted_material = None
        for mat_id in final_step.mat_to_ind.keys():
            mat = final_step.get_material(str(mat_id))
            if getattr(mat, 'name', None) == 'zn_target':
                depleted_material = mat
                break
        if depleted_material is None:
            return {
                'zn_volume_cm3': zn_volume_cm3,
                'zn_density_g_cm3': zn_density_g_cm3,
                'zn_mass_g': mass_g,
                'zn_mass_kg': zn_mass_kg,
                'irrad_hours': irrad_h,
                'cooldown_days': cooldown_days,
                'cu64_mCi': None,
                'cu67_mCi': None,
                'cu64_Bq': None,
                'cu67_Bq': None,
                'cu64_atomic_purity': None,
                'cu67_atomic_purity': None,
                'cu64_radionuclide_purity': None,
                'cu67_radionuclide_purity': None,
                'copper_mass_final_g': None,
            }
    nuclide_atoms = depleted_material.get_nuclide_atoms()

    cu_nuclides = {'Cu63', 'Cu64', 'Cu65', 'Cu67'}
    copper_atoms = {n: nuclide_atoms.get(n, 0.0) for n in cu_nuclides}
    cu64_atoms = copper_atoms.get('Cu64', 0.0)
    cu67_atoms = copper_atoms.get('Cu67', 0.0)
    total_cu_atoms = sum(copper_atoms.values())
    radio_cu_atoms = cu64_atoms + cu67_atoms

    lam64 = np.log(2) / CU64_HALFLIFE_S
    lam67 = np.log(2) / CU67_HALFLIFE_S
    cu64_Bq = float(lam64 * cu64_atoms) if cu64_atoms else None
    cu67_Bq = float(lam67 * cu67_atoms) if cu67_atoms else None
    cu64_mCi = (cu64_Bq / BQ_PER_MCI) if cu64_Bq is not None else None
    cu67_mCi = (cu67_Bq / BQ_PER_MCI) if cu67_Bq is not None else None

    cu64_atomic_purity = float(cu64_atoms / total_cu_atoms) if total_cu_atoms else None
    cu67_atomic_purity = float(cu67_atoms / total_cu_atoms) if total_cu_atoms else None
    cu64_radionuclide_purity = float(cu64_atoms / radio_cu_atoms) if radio_cu_atoms else None
    cu67_radionuclide_purity = float(cu67_atoms / radio_cu_atoms) if radio_cu_atoms else None

    copper_mass_final_g = None
    if total_cu_atoms > 0:
        try:
            copper_mass_final_g = sum(
                (nuclide_atoms.get(n, 0.0) * openmc.data.atomic_mass(n)) / Avogadro
                for n in cu_nuclides
            )
        except Exception:
            pass

    return {
        'zn_volume_cm3': zn_volume_cm3,
        'zn_density_g_cm3': zn_density_g_cm3,
        'zn_mass_g': mass_g,
        'zn_mass_kg': zn_mass_kg,
        'irrad_hours': irrad_h,
        'cooldown_days': cooldown_days,
        'cu64_mCi': cu64_mCi,
        'cu67_mCi': cu67_mCi,
        'cu64_Bq': cu64_Bq,
        'cu67_Bq': cu67_Bq,
        'cu64_atomic_purity': cu64_atomic_purity,
        'cu67_atomic_purity': cu67_atomic_purity,
        'cu64_radionuclide_purity': cu64_radionuclide_purity,
        'cu67_radionuclide_purity': cu67_radionuclide_purity,
        'copper_mass_final_g': copper_mass_final_g,
    }


def run_irradiation_simulation(
    output_dir='irradiation_output',
    particles=2000,
    batches=3,
    n_cycles=2,
    plot_dose_maps=False,
    mesh_dimension=(50, 50, 1),
    target_flux=1e16,
    neutron_energy=14.06,
    target_thickness=3.0,
    chain_file_path=None,
):
    """
    Run complete irradiation simulation with depletion.

    Materials are defined in create_materials(). First material is the depletable Zn target.

    Parameters:
    -----------
    target_flux : float
        Desired neutron flux at surface (n/cm²/s)
    neutron_energy : float
        Neutron energy (MeV)
    target_thickness : float
        Target thickness (cm)
    particles : int, optional
        Number of particles per batch (default: 10000)
    batches : int, optional
        Number of batches for transport calculation (default: 3)
    output_dir : str
        Directory for output files
    chain_file_path : str, optional
        Path to depletion chain XML (e.g. JENDL_chain.xml). On a cluster, set this or
        OPENMC_CHAIN_FILE / JENDL_CHAIN_FILE env to the correct chain location.

    Returns:
    --------
    pd.DataFrame : Final composition data
    """
    print("="*60)
    print("OpenMC fixed-source (neutron + photon; depletion commented out)")
    print("="*60)

    # Run output folder: delete if present from a previous run, then create and run inside it
    run_dir = os.path.abspath(output_dir)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    saved_cwd = os.getcwd()
    os.chdir(run_dir)
    print(f"  Particles: {particles} | Batches: {batches} | output in {run_dir}")

    # Resolve depletion chain early (needed for D1S tally ParentNuclideFilter and depletion)
    FAST_CHAIN_FILE = "/workspace/openmc-data/fast.xml"
    THERMAL_CHAIN_FILE = "/workspace/openmc-data/thermal.xml"
    CHAIN_SPECTRUM = "fast"
    if CHAIN_SPECTRUM == "fast":
        chain_file = FAST_CHAIN_FILE
    elif CHAIN_SPECTRUM == "thermal":
        chain_file = THERMAL_CHAIN_FILE
    else:
        raise ValueError(f"Unknown CHAIN_SPECTRUM '{CHAIN_SPECTRUM}'. Use 'fast' or 'thermal'.")
    if not os.path.isfile(chain_file):
        raise FileNotFoundError(
            f"Depletion chain file not found at hard-coded path: {chain_file}. "
            f"Check that it exists on the cluster."
        )
    print(f"  Using chain file: {chain_file}")

    # Materials: Zn (depletable) + water for bath
    print("\nCreating materials...")
    materials = create_materials(include_water=True)
    zn_mat = materials[0]
    water_mat = materials[-1]

    # Geometry: Zn cylinder in vacuum bubble, surrounded by water; target_cell = Zn
    print("Creating geometry (Zn cylinder in vacuum, water bath)...")
    geometry, target_cell = create_geometry_cylinder_water(zn_mat, water_mat)
    print(f"  Zn: {ZN_CYLINDER_MASS_KG} kg, {SOURCE_TO_CYLINDER_CM} cm from source; vacuum r={VACUUM_BUBBLE_RADIUS_CM} cm, water r={WATER_RADIUS_CM} cm")

    # Create source: isotropic point source at origin, 14.06 MeV neutrons
    print("Creating source...")
    my_source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([14.06e6], [1]),
        particle="neutron"
    )

    # Settings: fixed-source neutron + photon transport
    settings = openmc.Settings()
    settings.source = my_source
    settings.particles = particles
    settings.batches = batches
    settings.run_mode = 'fixed source'
    settings.photon_transport = True
    settings.use_decay_photons = True  # D1S: secondary photons from (n,g) etc.

    # Get radionuclides from chain (for D1S tally); need a minimal model for get_radionuclides
    model_for_chain = openmc.Model(geometry=geometry, settings=settings, tallies=openmc.Tallies())
    radionuclides = d1s.get_radionuclides(model_for_chain, chain_file=chain_file)

    # Photon dose tally on mesh with ParentNuclideFilter so D1S apply_time_correction works
    print("Creating tallies...")
    tallies, mesh = create_tallies_dose_mesh(geometry, mesh_dimension=mesh_dimension, radionuclides=radionuclides)

    # Create full model with tallies
    model = openmc.Model(geometry=geometry, settings=settings, tallies=tallies)

    # Export and run inside output_dir (we chdir'd there at start)
    if os.path.isfile("model.xml"):
        os.remove("model.xml")
    print("Exporting model files (current dir)...")
    model.materials.export_to_xml()
    model.geometry.export_to_xml()
    model.settings.export_to_xml()
    model.tallies.export_to_xml()

    # ========== DEPLETION + D1S POST-PROCESSING (turned back on) ==========
    print("\nSetting up depletion (JENDL-based chain + CF4Integrator)...")

    # Reduce chain to nuclides in materials + transmutation/decay products (shortens cross-section read)
    reduce_chain_level = 2  # depth of search from initial materials; None = full chain (slow init)

    operator = openmc.deplete.CoupledOperator(
        model,
        chain_file=chain_file,
        normalization_mode="source-rate",
        reduce_chain_level=reduce_chain_level,
    )

    # D1S pulse schedule: 8 h at 1.7e13 n/s, then 12×5 min cooling; repeated n_cycles (fewer steps = faster)
    irradiation_step = (8 * 3600, 1.7e13)   # 8 hours, source on
    cooling_step = (5 * 60, 0.0)            # 5 min, source off
    cooling_steps = [cooling_step] * 3
    pulse_cycle = [irradiation_step] + cooling_steps
    timesteps_and_source_rates = pulse_cycle * n_cycles
    timesteps = [ts for ts, _ in timesteps_and_source_rates]
    source_rates = [sr for _, sr in timesteps_and_source_rates]

    print(f"Running depletion for {len(timesteps)} steps (D1S schedule)...")
    integrator = openmc.deplete.CF4Integrator(
        operator,
        timesteps=timesteps,
        source_rates=source_rates,
    )
    integrator.integrate()

    # Export and report final composition
    print("\nPost-processing depletion results...")
    results = openmc.deplete.Results("depletion_results.h5")
    df = export_final_composition(results, material_id="1", output_file="final_composition.csv")
    print("\nTop 10 nuclides by percentage:")
    print(df.head(10).to_string(index=False))

    # Run summary: Zn/Cu geometry, irrad/cool times, Cu-64/Cu-67 activity and purities
    act = compute_run_summary_activity(
        results,
        material_id="1",
        timesteps=timesteps,
        source_rates=source_rates,
        zn_density_g_cm3=7.14,
        zn_mass_kg=ZN_CYLINDER_MASS_KG,
    )
    summary_path = "run_summary.csv"
    pd.DataFrame([act]).to_csv(summary_path, index=False)
    print("\nRun summary (saved to {}):".format(summary_path))
    for k, v in act.items():
        print("  {}: {}".format(k, v))

    # D1S dose-rate post-processing: apply time-correction factors to photon dose tally
    print("\nD1S dose rate post-processing...")
    statepoint_filename = f"statepoint.{batches}.h5"
    if not os.path.isfile(statepoint_filename):
        print(f"  WARNING: {statepoint_filename} not found; cannot perform D1S dose post-processing.")
        print("\n" + "="*60)
        print(f"Simulation complete (depletion + photons). Output in {run_dir}")
        print("="*60)
        os.chdir(saved_cwd)
        return df

    with openmc.StatePoint(statepoint_filename) as sp:
        dose_tally_from_sp = sp.get_tally(name="photon_dose_on_mesh")

    radionuclides = d1s.get_radionuclides(model, chain_file=chain_file)
    time_factors = d1s.time_correction_factors(
        nuclides=radionuclides,
        timesteps=timesteps,
        source_rates=source_rates,
        timestep_units="s",
    )

    pico_to_milli = 1e-9
    volume_normalization = mesh.volumes[0][0][0]

    # Dose at final cooling time step (index = last timestep)
    last_index = len(timesteps) - 1
    corrected_tally = d1s.apply_time_correction(
        tally=dose_tally_from_sp,
        time_correction_factors=time_factors,
        index=last_index,
        sum_nuclides=True,
    )
    corrected_mean = corrected_tally.mean.squeeze()
    scaled_corrected_mean = (corrected_mean * pico_to_milli) / volume_normalization

    max_dose = float(scaled_corrected_mean.max())
    print(f"  Max D1S dose at final time step: {max_dose:.3e} mSv/s")

    print("\n" + "="*60)
    print(f"Simulation complete (depletion + photons). Output in {run_dir}")
    print("="*60)
    os.chdir(saved_cwd)
    return df


if __name__ == '__main__':
    # Set cross sections first (so neutron + photon run use JENDL-5)
    os.environ["OPENMC_CROSS_SECTIONS"] = "/workspace/openmc-data/jendl-5-hdf5/cross_sections.xml"
    openmc.config["cross_sections"] = os.environ["OPENMC_CROSS_SECTIONS"]
    print(os.environ["OPENMC_CROSS_SECTIONS"])

    run_irradiation_simulation(
        output_dir='irradiation_output',  # run folder; wiped and recreated each run
        particles=200,
        batches=2,
        n_cycles=1,
        plot_dose_maps=False,
        mesh_dimension=(25, 25, 1),
    )
