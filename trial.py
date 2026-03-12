"""
Trial: 1 kg Zn cylinder inside 0.5 cm thick bismuth chamber, 6 in from D-T point source.
Irradiation: 8 hours on, 1 day cooldown, repeated 10 times. Tracks all activation products
from Zn (and its decay/transmutation products) and from bismuth shielding.
Tallies: Cu-64 and Cu-67 production, Zn-65 production (and all Zn/Bi activation),
photon dose in the bismuth layer (from activation products there).

Dose from photons includes: (1) decay gammas, (2) 511 keV positron-annihilation photons,
(3) bremsstrahlung from electrons/positrons in material (OpenMC thick-target approximation).
Uses source strength/energy from test.py; photon transport, decay photons, and electron_treatment='ttb'.
"""
import os
import glob
import zipfile
from datetime import datetime
import numpy as np
import pandas as pd
import os
from scipy.constants import Avogadro

import openmc
import openmc.deplete
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from openmc.data import half_life

# Default depletion chain: use JENDL_chain.xml if no env or config set (script dir, cwd, or Docker /workspace)
if not os.environ.get('OPENMC_CHAIN_FILE') and not openmc.config.get('chain_file'):
    _chain_name = 'JENDL_chain.xml'
    _candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _chain_name),
        os.path.abspath(_chain_name),  # cwd
        os.path.join('/workspace', _chain_name),  # Docker workspace mount
    ]
    for _path in _candidates:
        if _path and os.path.isfile(_path):
            openmc.config['chain_file'] = _path
            break
    else:
        openmc.config['chain_file'] = _chain_name 


RESULTS_ZIP_BASENAME = 'trial_bismuth_results'

SOURCE_STRENGTH = 1.7e13  # n/s
SOURCE_ENERGY_MEV = 14.1
FUSION_POWER_W = SOURCE_STRENGTH * (SOURCE_ENERGY_MEV * 1e6 * 1.602e-19)  # W from 14.1 MeV neutron

IN_TO_CM = 2.54
SOURCE_TO_CHAMBER_CM = 6.0 * IN_TO_CM  # 15.24 cm: distance from source to nearest chamber surface

# 1 kg Zn cylinder: density 7.14 g/cm³ -> V = 1000/7.14 ≈ 140.06 cm³. Use h=10 cm -> r = sqrt(140/(π*10)) ≈ 2.11 cm
ZN_MASS_G = 1000.0
ZN_DENSITY_G_CM3 = 7.14
ZN_VOLUME_CM3 = ZN_MASS_G / ZN_DENSITY_G_CM3
ZN_HEIGHT_CM = 10.0
ZN_INNER_RADIUS_CM = np.sqrt(ZN_VOLUME_CM3 / (np.pi * ZN_HEIGHT_CM))  # ~2.11 cm

BISMUTH_THICKNESS_CM = 0.5
BISMUTH_OUTER_RADIUS_CM = ZN_INNER_RADIUS_CM + BISMUTH_THICKNESS_CM
# Cylinder axis along z; center x such that nearest bismuth surface to source (0,0,0) is at 6 in
CYLINDER_X_CENTER = SOURCE_TO_CHAMBER_CM + BISMUTH_OUTER_RADIUS_CM

SPHERE_RADIUS_CM = 100.0
Z_LO_CM = -ZN_HEIGHT_CM / 2.0
Z_HI_CM = ZN_HEIGHT_CM / 2.0

# Depletion chain from JENDL: default subdirs under jendl_dir (decay, fpy, neutron)
JENDL_DECAY_SUBDIR = 'jendl5-dec_upd5/jendl5-dec_upd5'
JENDL_FPY_SUBDIR = 'jendl5-fpy_upd8/jendl5-fpy_upd8'
JENDL_NEUTRON_SUBDIR = 'jendl5-n/jendl5-n'
CHAIN_REACTIONS = ('(n,gamma)', '(n,p)', '(n,2n)', '(n,a)', '(n,d)', '(n,np)')


def _print_time(message):
    """Print message with current datetime."""
    print()
    print(message + str(datetime.now()))


def build_depletion_chain(jendl_dir=None, output_xml=None, reactions=None, progress=True):
    """
    Generate a depletion chain from JENDL ENDF files (decay, fission product yield, neutron).

    Requires JENDL data under jendl_dir with subdirs:
      - jendl5-dec_upd5/jendl5-dec_upd5/*.dat
      - jendl5-fpy_upd8/jendl5-fpy_upd8/*.dat
      - jendl5-n/jendl5-n/*.dat

    Parameters
    ----------
    jendl_dir : str, optional
        Root directory containing JENDL subdirs. Default: './JENDL'.
    output_xml : str, optional
        Output chain XML path. Default: 'JENDL_chain.xml'.
    reactions : tuple, optional
        Neutron reactions to include. Default: (n,gamma), (n,p), (n,2n), (n,a), (n,d), (n,np).
    progress : bool
        If True, show progress during chain build.

    Returns
    -------
    str
        Path to the exported chain XML file.
    """
    jendl_dir = os.path.abspath(jendl_dir or './JENDL')
    output_xml = output_xml or 'JENDL_chain.xml'
    reactions = reactions or CHAIN_REACTIONS

    decay_dir = os.path.join(jendl_dir, JENDL_DECAY_SUBDIR)
    fpy_dir = os.path.join(jendl_dir, JENDL_FPY_SUBDIR)
    neutron_dir = os.path.join(jendl_dir, JENDL_NEUTRON_SUBDIR)

    decay_files = glob.glob(os.path.join(decay_dir, '*.dat'))
    fpy_files = glob.glob(os.path.join(fpy_dir, '*.dat'))
    neutron_files = glob.glob(os.path.join(neutron_dir, '*.dat'))

    _print_time('===> Depletion chain build started on ')
    print(f"  JENDL dir: {jendl_dir}")
    print(f"  Decay files: {len(decay_files)}")
    print(f"  FPY files: {len(fpy_files)}")
    print(f"  Neutron files: {len(neutron_files)}")
    if not decay_files or not neutron_files:
        raise FileNotFoundError(
            f"JENDL data not found under {jendl_dir}. "
            f"Expect subdirs: {JENDL_DECAY_SUBDIR}, {JENDL_FPY_SUBDIR}, {JENDL_NEUTRON_SUBDIR}"
        )

    chain = openmc.deplete.Chain.from_endf(
        decay_files,
        fpy_files,
        neutron_files,
        reactions=reactions,
        progress=progress,
    )
    chain.export_to_xml(output_xml)
    _print_time('===> Depletion chain build completed on ')
    print(f"  Chain written to: {os.path.abspath(output_xml)}")
    return os.path.abspath(output_xml)


def create_materials():
    """Zn (natural) and bismuth; both depletable for openmc.deplete. Volumes set on materials for depletion.

    Some cross section libraries used in Docker images do not define elemental Zn
    in their `cross_sections.xml`, which causes an error when calling
    `Material.add_element('Zn', ...)`. To make this script robust across
    libraries, we explicitly add the stable Zn isotopes with their natural
    abundances instead of using the elemental shortcut.
    """
    zn = openmc.Material(material_id=1, name='zn_target')
    zn.set_density('g/cm3', ZN_DENSITY_G_CM3)
    zn.add_element('Zn', 1.0)
    zn.temperature = 294
    zn.depletable = True
    zn.volume = ZN_VOLUME_CM3 

    bi = openmc.Material(material_id=2, name='bismuth')
    bi.set_density('g/cm3', 9.78)
    bi.add_nuclide('Bi209', 1.0)
    bi.temperature = 294
    bi.depletable = True
    bi.volume = np.pi * (BISMUTH_OUTER_RADIUS_CM ** 2 - ZN_INNER_RADIUS_CM ** 2) * ZN_HEIGHT_CM

    return openmc.Materials([zn, bi])


def create_geometry(materials):
    """Vacuum sphere; Zn cylinder + bismuth shell, axis along z, center at (CYLINDER_X_CENTER, 0, 0)."""
    zn_mat, bi_mat = materials[0], materials[1]
    sphere = openmc.Sphere(r=SPHERE_RADIUS_CM, boundary_type='vacuum')
    inner_cyl = openmc.ZCylinder(x0=CYLINDER_X_CENTER, y0=0.0, r=ZN_INNER_RADIUS_CM)
    outer_cyl = openmc.ZCylinder(x0=CYLINDER_X_CENTER, y0=0.0, r=BISMUTH_OUTER_RADIUS_CM)
    z_lo = openmc.ZPlane(z0=Z_LO_CM)
    z_hi = openmc.ZPlane(z0=Z_HI_CM)

    zn_region = -inner_cyl & +z_lo & -z_hi
    bismuth_region = (+inner_cyl & -outer_cyl) & +z_lo & -z_hi
    vacuum_region = -sphere & ~zn_region & ~bismuth_region

    zn_cell = openmc.Cell(cell_id=1, name='zn_target', fill=zn_mat, region=zn_region)
    bi_cell = openmc.Cell(cell_id=2, name='bismuth', fill=bi_mat, region=bismuth_region)
    vac_cell = openmc.Cell(cell_id=3, name='vacuum', fill=None, region=vacuum_region)

    universe = openmc.Universe(cells=[zn_cell, bi_cell, vac_cell])
    return openmc.Geometry(universe), [zn_cell, bi_cell]


def create_source():
    """Point source D-T at origin, 14.1 MeV, isotropic."""
    source = openmc.IndependentSource()
    source.space = openmc.stats.Point((0.0, 0.0, 0.0))
    source.angle = openmc.stats.Isotropic()
    source.energy = openmc.stats.Discrete([SOURCE_ENERGY_MEV * 1e6], [1.0])
    return source


def create_tallies(cells_with_fill, geometry=None):
    """Tallies for depletion: Zn/Bi activation (especially Zn65, Cu64, Cu67), bismuth-layer dose. Optionally mesh dose."""
    tallies = openmc.Tallies()
    zn_nuclides = ['Zn63', 'Zn64', 'Zn65', 'Zn66', 'Zn67', 'Zn68', 'Zn69', 'Zn70']

    zn_cell = next(c for c in cells_with_fill if c.name == 'zn_target')
    zn_filt = openmc.CellFilter([zn_cell])
    # All Zn reactions (activation products from Zn and its products)
    t_zn = openmc.Tally(name='zn_rxn')
    t_zn.filters = [zn_filt]
    t_zn.scores = ['(n,gamma)', '(n,2n)', '(n,p)', '(n,d)']
    t_zn.nuclides = zn_nuclides
    tallies.append(t_zn)

    # Cu-64 and Cu-67 production (Zn64(n,p)Cu64, Zn67(n,p)Cu67; Zn68(n,d)Cu67 in zn_rxn)
    t_cu64 = openmc.Tally(name='Cu64_production')
    t_cu64.filters = [zn_filt]
    t_cu64.scores = ['(n,p)']
    t_cu64.nuclides = ['Zn64']
    tallies.append(t_cu64)
    t_cu67 = openmc.Tally(name='Cu67_production')
    t_cu67.filters = [zn_filt]
    t_cu67.scores = ['(n,p)', '(n,d)']
    t_cu67.nuclides = ['Zn67', 'Zn68']
    tallies.append(t_cu67)

    # Zn-65 production: Zn64(n,gamma)Zn65 (and Zn66(n,2n)Zn65 in zn_rxn)
    t_zn65 = openmc.Tally(name='Zn65_production')
    t_zn65.filters = [zn_filt]
    t_zn65.scores = ['(n,gamma)', '(n,2n)']
    t_zn65.nuclides = ['Zn64', 'Zn66']
    tallies.append(t_zn65)

    bi_cell = next(c for c in cells_with_fill if c.name == 'bismuth')
    bi_filt = openmc.CellFilter([bi_cell])
    # Bismuth activation: Bi-208, Bi-209, Bi-210 (and products)
    bi_nuclides = ['Bi208', 'Bi209', 'Bi210']
    t_bi = openmc.Tally(name='bismuth_rxn')
    t_bi.filters = [bi_filt]
    t_bi.scores = ['(n,gamma)', '(n,2n)']
    t_bi.nuclides = bi_nuclides
    tallies.append(t_bi)

    # Photon dose in bismuth layer (dose from activation products in bismuth)
    energies, pSv_cm2 = openmc.data.dose_coefficients(particle="photon", geometry="AP")
    dose_filter_cell = openmc.EnergyFunctionFilter(energies, pSv_cm2, interpolation="cubic")
    t_bi_dose = openmc.Tally(name='bismuth_layer_dose')
    t_bi_dose.filters = [bi_filt, openmc.ParticleFilter(["photon"]), dose_filter_cell]
    t_bi_dose.scores = ['flux']
    tallies.append(t_bi_dose)

    # Photon dose on regular mesh (xy slice) for D1S time-correction plots
    if geometry is not None:
        mesh = openmc.RegularMesh().from_domain(
            geometry,
            dimension=[100, 100, 1],
        )
        energies, pSv_cm2 = openmc.data.dose_coefficients(particle="photon", geometry="AP")
        dose_filter = openmc.EnergyFunctionFilter(
            energies, pSv_cm2, interpolation="cubic"
        )
        particle_filter = openmc.ParticleFilter(["photon"])
        mesh_filter = openmc.MeshFilter(mesh)
        dose_tally = openmc.Tally()
        dose_tally.filters = [particle_filter, mesh_filter, dose_filter]
        dose_tally.scores = ["flux"]
        dose_tally.name = "photon_dose_on_mesh"
        tallies.append(dose_tally)
        return tallies, mesh
    return tallies, None


def build_model(include_dose_mesh=True):
    """Build OpenMC model: materials, geometry, source, settings (photon transport, decay photons), tallies."""
    materials = create_materials()
    geometry, cells_with_fill = create_geometry(materials)
    source = create_source()
    tallies, mesh = create_tallies(cells_with_fill, geometry=geometry if include_dose_mesh else None)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'
    settings.particles = 10000
    settings.batches = 10
    settings.inactive = 2
    settings.source = source
    settings.statepoint = {'batches': [settings.batches]}
    settings.output = {'summary': True, 'tallies': True}
    # Dose-relevant physics: gammas, positron-annihilation (511 keV), bremsstrahlung from betas in material.
    # - photon_transport: transport all photons (decay gammas, 511 keV from e+ annihilation, bremsstrahlung).
    # - use_decay_photons: use decay photon spectra from activation (D1S-style) instead of prompt only.
    # - electron_treatment='ttb': thick-target bremsstrahlung; electrons/positrons deposit locally and produce
    #   bremsstrahlung photons at birth site; positrons also produce 511 keV annihilation photons (OpenMC 7.1.2).
    settings.photon_transport = True
    settings.use_decay_photons = True
    settings.electron_treatment = 'ttb'

    model = openmc.Model()
    model.materials = materials
    model.geometry = geometry
    model.settings = settings
    model.tallies = tallies
    return model, cells_with_fill, mesh


# Depletion schedule: 8 h irradiation, 1 day cooldown, 10 times (all isotopes from Zn + bismuth).
IRRADIATION_HOURS = 8.0
COOLDOWN_DAYS = 1.0
N_DEPLETION_CYCLES = 10
# One cycle = [irrad_days, cooldown_days]; timestep_units='d'
DEPLETION_TIMESTEPS_DAYS = [IRRADIATION_HOURS / 24.0, COOLDOWN_DAYS] * N_DEPLETION_CYCLES
DEPLETION_SOURCE_RATES = [SOURCE_STRENGTH, 0.0] * N_DEPLETION_CYCLES  # n/s; 0 during cooldown

# D1S pulse schedule (for --d1s-dose): one irradiation step + 24 × 5 min cooling; repeat 4 times.
_IRRADIATION_STEP = (1, 1e18)  # 1 s at 1e18 n/s
_COOLING_STEP = (60 * 5, 0)    # 5 min cooling, no source
_COOLING_STEPS = [_COOLING_STEP] * 24
_PULSE_CYCLE = [_IRRADIATION_STEP] + _COOLING_STEPS
TIMESTEPS_AND_SOURCE_RATES = _PULSE_CYCLE * 4


def _half_life_seconds(nuclide_str: str):
    """Half-life in seconds for legend; uses openmc.data.half_life with fallback for Zn63."""
    try:
        hl = openmc.data.half_life(nuclide_str)
        if hl is None or not np.isfinite(hl) or hl <= 0:
            if nuclide_str == 'Zn63':
                return 38.47 * 60.0  # 38.47 min
            return None
        return float(hl)
    except Exception:
        if nuclide_str == 'Zn63':
            return 38.47 * 60.0
        return None


def _save_geometry_plots(geometry, output_dir):
    """Save geometry setup plots (xy and xz outline) to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    ext_xy = geometry.bounding_box.extent['xy']
    ext_xz = geometry.bounding_box.extent['xz']
    for view, extent in [('xy', ext_xy), ('xz', ext_xz)]:
        try:
            geometry.plot(outline='only', extent=extent, pixels=1_000_000)
            fig = plt.gcf()
            ax = fig.axes[0] if fig.axes else fig.gca()
            ax.set_xlabel("X (cm)")
            ax.set_ylabel("Y (cm)" if view == 'xy' else "Z (cm)")
            ax.set_title(f"Geometry setup ({view})")
            plt.tight_layout()
            path = os.path.join(output_dir, f'geometry_setup_{view}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {path}")
        except Exception as e:
            print(f"  Could not save geometry {view} plot: {e}")


def _zip_results(output_dir, zip_basename=None):
    """Zip all contents of output_dir into trial_bismuth_results.zip (or zip_basename.zip) in parent dir."""
    zip_basename = zip_basename or RESULTS_ZIP_BASENAME
    output_dir = os.path.abspath(output_dir)
    parent = os.path.dirname(output_dir)
    zip_path = os.path.join(parent, f"{zip_basename}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                abspath = os.path.join(root, f)
                arcname = os.path.relpath(abspath, parent)
                zf.write(abspath, arcname)
    print(f"  Zipped results to: {zip_path}")
    return zip_path


def run_d1s_dose(output_dir='trial_depletion'):
    """Run transport once with photon dose mesh tally, then D1S time correction and plot dose at each cooldown time."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model, cells_with_fill, mesh = build_model(include_dose_mesh=True)
    model.output_dir = output_dir

    # Save geometry setup plots (xy and xz)
    _save_geometry_plots(model.geometry, output_dir)

    # D1S: add ParentNuclideFilter to photon tallies (requires chain for radionuclides)
    d1s = openmc.deplete.d1s
    chain_file = os.environ.get('OPENMC_CHAIN_FILE') or openmc.config.get('chain_file')
    if chain_file and os.path.isfile(chain_file):
        d1s.prepare_tallies(model=model, chain_file=chain_file)
    else:
        d1s.prepare_tallies(model=model)

    print(f"Run directory (OpenMC will read XMLs from here): {output_dir}")
    print(f"Zn is defined as nuclides (Zn64, Zn66, Zn67, Zn68, Zn70) for cross_sections compatibility.")
    cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        model.export_to_xml()
        output_path = model.run()
    finally:
        os.chdir(cwd)

    statepoint_path = output_path if isinstance(output_path, str) else os.path.join(output_dir, f'statepoint.{model.settings.batches}.h5')
    if not os.path.isfile(statepoint_path):
        statepoint_path = os.path.join(output_dir, 'statepoint.10.h5')
    if not os.path.isfile(statepoint_path):
        print("Statepoint not found; skipping D1S dose plots.")
        return

    timesteps = [item[0] for item in TIMESTEPS_AND_SOURCE_RATES]
    source_rates = [item[1] for item in TIMESTEPS_AND_SOURCE_RATES]
    radionuclides = d1s.get_radionuclides(model)
    time_factors = d1s.time_correction_factors(
        nuclides=radionuclides,
        timesteps=timesteps,
        source_rates=source_rates,
        timestep_units='s',
    )

    pico_to_milli = 1e-9
    try:
        vol_per_voxel = float(np.asarray(mesh.volumes).flat[0])
    except (AttributeError, IndexError, TypeError):
        # fallback: mesh.volumes might be property returning ndarray
        vol_per_voxel = float(np.asarray(mesh.volumes).ravel()[0]) if hasattr(mesh, 'volumes') else 1.0

    with openmc.StatePoint(statepoint_path) as sp:
        dose_tally_from_sp = sp.get_tally(name='photon_dose_on_mesh')
        geometry = model.geometry

        scaled_max_tally_values = []
        for i_cool in range(1, len(timesteps)):
            corrected_tally = d1s.apply_time_correction(
                tally=dose_tally_from_sp,
                time_correction_factors=time_factors,
                index=i_cool,
                sum_nuclides=True,
            )
            # Max dose rate (mSv/s) at this time step for total-dose-over-time plot
            max_at_step = (np.max(corrected_tally.mean) * pico_to_milli) / vol_per_voxel
            scaled_max_tally_values.append(float(np.asarray(max_at_step).flat[0]))

            if i_cool == 1:
                scaled_max_tally_value = scaled_max_tally_values[0]

            corrected_tally_mean = corrected_tally.get_reshaped_data(value='mean', expand_dims=True).squeeze()
            scaled_corrected_tally_mean = (corrected_tally_mean * pico_to_milli) / vol_per_voxel

            fig, ax1 = plt.subplots(figsize=(6, 4))
            plot_1 = ax1.imshow(
                scaled_corrected_tally_mean.T,
                origin="lower",
                extent=mesh.bounding_box.extent['xy'],
                norm=LogNorm(vmax=scaled_max_tally_value),
            )
            ax2 = geometry.plot(
                outline='only',
                extent=geometry.bounding_box.extent['xy'],
                axes=ax1,
                pixels=1_000_000,
            )
            time_in_mins = round(sum(timesteps[1:i_cool + 1]) / 60.0, 2)
            max_dose_in_timestep = round(float(np.max(scaled_corrected_tally_mean)), 2)
            ax2.set_title(
                f"Dose rate at {time_in_mins} min after irradiation\nMax dose rate: {max_dose_in_timestep} mSv/s"
            )
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_aspect(ax1.get_aspect())
            ax2.set_xlabel("X (cm)")
            ax2.set_ylabel("Y (cm)")
            cbar = plt.colorbar(plot_1, ax=ax1)
            cbar.set_label("Dose [milli Sv per second]")
            plt.tight_layout()
            out_png = os.path.join(output_dir, f'dose_rate_{time_in_mins}_min.png')
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {out_png}")

        # Total max dose rate over time (sharp drop after each shot; buildup of long-lived isotopes)
        time_s = np.cumsum(timesteps[1:])
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(time_s, scaled_max_tally_values, color='C0')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Max Dose Rate (mSv/s)")
        ax1.set_title("Max Dose Rate Over Time")
        ax1.grid(True)
        plt.tight_layout()
        out_total = os.path.join(output_dir, 'max_dose_rate_over_time.png')
        plt.savefig(out_total, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_total}")

        # Per-nuclide dose: get parent nuclides from tally (sum_nuclides=False)
        corrected_tally_nuclides = d1s.apply_time_correction(
            tally=dose_tally_from_sp,
            time_correction_factors=time_factors,
            index=1,
            sum_nuclides=False,
        )
        parent_nuclide_filter = corrected_tally_nuclides.find_filter(openmc.ParentNuclideFilter)
        parent_nuclides = parent_nuclide_filter.bins
        scaled_max_tally_values_per_nuclide = {str(n): [] for n in parent_nuclides}

        for i_cool in range(1, len(timesteps)):
            corrected_tally = d1s.apply_time_correction(
                tally=dose_tally_from_sp,
                time_correction_factors=time_factors,
                index=i_cool,
                sum_nuclides=False,
            )
            mean_vals = np.asarray(corrected_tally.mean).squeeze()
            # shape: (n_nuclides, n_mesh_bins) or similar
            mean_values_per_nuclide = mean_vals.reshape(len(parent_nuclides), -1)
            for i_nuclide, nuclide in enumerate(parent_nuclides):
                max_val = (np.max(mean_values_per_nuclide[i_nuclide]) * pico_to_milli) / vol_per_voxel
                scaled_max_tally_values_per_nuclide[str(nuclide)].append(float(np.asarray(max_val).flat[0]))

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.plot(time_s, scaled_max_tally_values, label='total')
        for nuclide in parent_nuclides:
            nuclide_str = str(nuclide)
            if sum(scaled_max_tally_values_per_nuclide[nuclide_str]) > 2.0:
                hl = _half_life_seconds(nuclide_str)
                label = f"{nuclide_str} half-life={hl:.1e}s" if hl is not None else f"{nuclide_str}"
                ax2.plot(time_s, scaled_max_tally_values_per_nuclide[nuclide_str], label=label)
        ax2.legend()
        ax2.set_xlabel("Time (s)")
        ax2.set_yscale('log')
        ax2.set_ylabel("Max Dose Rate (mSv/s)")
        ax2.set_title("Max Dose Rate Over Time (total and main nuclides)")
        ax2.grid(True)
        plt.tight_layout()
        out_nuclides = os.path.join(output_dir, 'max_dose_rate_per_nuclide.png')
        plt.savefig(out_nuclides, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_nuclides}")

    _zip_results(output_dir)


def _report_activation_and_dose(output_dir):
    """Read last statepoint in output_dir and print Cu64/Cu67/Zn65 production, bismuth activation, bismuth-layer dose."""
    import glob
    statepoints = sorted(glob.glob(os.path.join(output_dir, 'statepoint.*.h5')))
    if not statepoints:
        return
    # Use latest by mtime (last depletion step)
    sp_path = max(statepoints, key=os.path.getmtime)
    try:
        with openmc.StatePoint(sp_path) as sp:
            # Reaction tallies are per source neutron; multiply by SOURCE_STRENGTH for rate per second
            def rate(tally_name):
                t = sp.get_tally(name=tally_name)
                return float(np.sum(t.mean)) * SOURCE_STRENGTH if t is not None else None

            cu64 = rate('Cu64_production')
            cu67 = rate('Cu67_production')
            zn65 = rate('Zn65_production')
            bi_dose_t = sp.get_tally(name='bismuth_layer_dose')
            bi_dose = (float(np.sum(bi_dose_t.mean)) * SOURCE_STRENGTH) if bi_dose_t is not None else None
    except Exception as e:
        print(f"Could not report activation/dose: {e}")
        return
    print("\n--- Activation and dose (last step, rates per second) ---")
    if cu64 is not None:
        print(f"  Cu-64 production (Zn64 n,p): {cu64:.4e} /s")
    if cu67 is not None:
        print(f"  Cu-67 production (Zn67 n,p + Zn68 n,d): {cu67:.4e} /s")
    if zn65 is not None:
        print(f"  Zn-65 production (Zn64 n,g + Zn66 n,2n): {zn65:.4e} /s")
    if bi_dose is not None:
        # flux * dose coeff -> pSv·cm³ per source n; * SOURCE_STRENGTH -> pSv·cm³/s
        print(f"  Bismuth-layer dose (pSv·cm³/s): {bi_dose:.4e}")
    print("---")


def run_depletion(output_dir='trial_depletion'):
    """Run depletion: 8 h irrad + 1 d cooldown × 10; Zn and bismuth activation, Cu64/Cu67/Zn65, bismuth-layer dose."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model, cells_with_fill, _ = build_model(include_dose_mesh=False)
    model.output_dir = output_dir

    # Run from output_dir so OpenMC reads our exported XMLs (nuclide-based Zn), not any
    # materials.xml in the parent dir (e.g. /workspace) that might use element Zn.
    cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        _run_depletion_from_cwd(model, output_dir)
    finally:
        os.chdir(cwd)


def _run_depletion_from_cwd(model, output_dir):
    """Run depletion with cwd already set to output_dir. Caller restores cwd."""
    # Save geometry setup plots (xy and xz)
    _save_geometry_plots(model.geometry, output_dir)

    # Volume calculation for depletion (needed for reaction rates)
    all_cells = list(model.geometry.get_all_cells().values())
    ll = [-SPHERE_RADIUS_CM, -SPHERE_RADIUS_CM, -SPHERE_RADIUS_CM]
    ur = [SPHERE_RADIUS_CM, SPHERE_RADIUS_CM, SPHERE_RADIUS_CM]
    vc = openmc.VolumeCalculation(domains=all_cells, samples=500_000, lower_left=ll, upper_right=ur)
    model.settings.volume_calculations = [vc]

    chain_file = os.environ.get('OPENMC_CHAIN_FILE') or openmc.config.get('chain_file')
    print(f"Run directory (OpenMC will read XMLs from here): {os.getcwd()}")
    print(f"Zn is defined as nuclides (Zn64, Zn66, Zn67, Zn68, Zn70) for cross_sections compatibility.")
    if chain_file and os.path.isfile(chain_file):
        print(f"Chain file: {chain_file}")

    if not chain_file or not os.path.isfile(chain_file):
        print("Warning: No depletion chain file. Set OPENMC_CHAIN_FILE or openmc.config['chain_file'].")
        print("Running transport-only (no depletion).")
        try:
            model.export_to_xml()
            model.calculate_volumes()
            vol_path = os.path.join(output_dir, 'volume_1.h5')
            if os.path.isfile(vol_path):
                vc.load_results(vol_path)
                model.geometry.add_volume_information(vc)
            model.run()
        finally:
            pass
        _zip_results(output_dir)
        return

    # Depletion: 8 h irrad + 1 d cooldown, 10 cycles (source_rate=0 during cooldown)
    op = openmc.deplete.CoupledOperator(
        model,
        chain_file=chain_file,
        normalization_mode='source-rate',
    )
    integrator = openmc.deplete.PredictorIntegrator(
        op,
        DEPLETION_TIMESTEPS_DAYS,
        source_rates=DEPLETION_SOURCE_RATES,
        timestep_units='d',
    )
    integrator.integrate(path=os.path.join(output_dir, 'depletion_results.h5'))
    # Report activation and bismuth-layer dose from last step statepoint
    _report_activation_and_dose(output_dir)
    _zip_results(output_dir)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Trial: 1 kg Zn in 0.5 cm bismuth, 6 in from source; depletion or D1S dose; or build depletion chain from JENDL.')
    p.add_argument('--output', '-o', default='trial_depletion', help='Output directory')
    p.add_argument('--no-deplete', action='store_true', help='Run transport only (no depletion)')
    p.add_argument('--d1s-dose', action='store_true', help='Run D1S: single transport run + photon dose mesh tally + time correction plots')
    p.add_argument('--build-chain', action='store_true', help='Build depletion chain from JENDL ENDF files and exit')
    p.add_argument('--jendl-dir', default=None, help='JENDL root directory (default: ./JENDL); used with --build-chain')
    p.add_argument('--chain-output', default=None, help='Output chain XML path (default: JENDL_chain.xml); used with --build-chain')
    args = p.parse_args()
    if args.build_chain:
        build_depletion_chain(
            jendl_dir=args.jendl_dir,
            output_xml=args.chain_output,
        )
    elif args.d1s_dose:
        run_d1s_dose(output_dir=args.output)
    elif args.no_deplete:
        # Transport only, no dose tally
        model, _, _ = build_model(include_dose_mesh=False)
        os.makedirs(args.output, exist_ok=True)
        model.output_dir = args.output
        cwd = os.getcwd()
        os.chdir(args.output)
        try:
            model.export_to_xml()
            model.run()
        finally:
            os.chdir(cwd)
        print("Transport-only run done.")
    else:
        run_depletion(output_dir=args.output)
