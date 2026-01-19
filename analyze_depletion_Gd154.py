#!/usr/bin/env python3
"""
OpenMC Zinc Enrichment Depletion Analysis 
--------------------------------
Post-processes results from a fusion neutron irradiation simulation.

Features:
- Loads depletion results (depletion_results.h5)
- Plots isotope evolution over time
- Plots total activity and decay heat
- Reads flux tallies from statepoint file
- Exports final composition and isotope ranking to CSV
"""

import openmc
import openmc.deplete
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import Avogadro
from math import log
import os

openmc.config['chain_file'] = os.path.join("irradiation_output", "JENDL_chain.xml")

def export_final_composition(results, material_id='1',
                             output_file='final_composition.csv'):
    """
    Export final isotopic composition from OpenMC depletion results.

    Works with modern OpenMC (0.14+) and CF4Integrator results that
    lack explicit materials or nuclides metadata.

    Parameters
    ----------
    results : openmc.deplete.Results
        Depletion results object
    material_id : str
        Material ID (default: '1')
    output_file : str
        Path to CSV file to write

    Returns
    -------
    pandas.DataFrame
        Final isotopic composition table
    """
    import pandas as pd
    from scipy.constants import Avogadro
    import openmc

    print(" Extracting final composition...")

    # --- Identify candidate nuclides ---
    # We'll probe a list of possible nuclides from the chain file or common isotopes.
    common_nuclides = [
        "H1", "H2", "H3", "He3", "He4",
        "Li6", "Li7", "Be9", "B10", "B11", "C12", "C13", "N14", "O16",
        "Ne20", "Na23", "Mg24", "Al27", "Si28", "Fe56",
        "Ni58", "Ni60", "Cu63", "Cu65",
        "Sm146", "Sm147", "Sm148"
        "Gd147","Gd148","Gd149","Gd150","Gd151",
        "Gd152", "Gd153", "Gd154", "Gd155", "Gd156", "Gd157", "Gd158", "Gd160",
        "Tb159", "Tb160", "Tb161", "Tb162", "Tb163"
    ]

    # --- Probe nuclides by attempting get_atoms() ---
    found_nuclides = []
    for nuc in common_nuclides:
        try:
            _, atoms = results.get_atoms(material_id, nuc)
            if len(atoms) > 0:
                found_nuclides.append(nuc)
        except Exception:
            continue

    if not found_nuclides:
        raise RuntimeError(
            "No nuclides could be queried from depletion results — "
            "verify the chain file and operator setup."
        )

    print(f" Found {len(found_nuclides)} candidate nuclides in results.")

    # --- Extract final composition ---
    final_atoms = {}
    for nuc in found_nuclides:
        try:
            _, atom_array = results.get_atoms(material_id, nuc)
            if atom_array[-1] > 0:
                final_atoms[nuc] = atom_array[-1]
        except Exception:
            continue

    if not final_atoms:
        raise ValueError("No nonzero atom densities found in final step.")

    total_atoms = sum(final_atoms.values())

    # --- Build DataFrame ---
    rows = []
    for nuc, atoms in final_atoms.items():
        pct = 100 * atoms / total_atoms
        try:
            amu = openmc.data.atomic_mass(nuc)
            mass_g = atoms * amu / Avogadro
        except Exception:
            amu, mass_g = None, None
        rows.append({
            'Nuclide': nuc,
            'Number_of_Atoms': atoms,
            'Percentage': pct,
            'Mass_grams': mass_g
        })

    df = pd.DataFrame(rows).sort_values('Percentage', ascending=False)
    df.to_csv(output_file, index=False)

    print(f"\n Final composition exported to {output_file} "
          f"({len(df)} nuclides)")
    return df


from openmc.deplete import Results
r = Results('irradiation_output/depletion_results.h5')
df = export_final_composition(r, output_file='irradiation_output/final_composition.csv')
df.head()


# r = Results('irradiation_output/depletion_results.h5')
# export_final_composition(r, material_id='1', output_file='irradiation_output/final_composition.csv')

# # -------------------------------
# ---- User Configuration ----
# -------------------------------
output_dir = "irradiation_output"
material_id = "1"          # ID of target material
isotopes_to_plot = ['Gd148','Gd149','Gd150','Gd151','Gd152','Gd153','Gd154']
statepoint_batches = 10     # must match your transport batches

# -------------------------------
# ---- Load depletion results ----
# -------------------------------
results_path = os.path.join(output_dir, "depletion_results.h5")
if not os.path.exists(results_path):
    raise FileNotFoundError(f"Missing {results_path}")

results = openmc.deplete.Results(results_path)
# time, _ = results.get_times()
# time = results.get_times()
# time_days = [t / 86400 for t in time]

time = results.get_times()

# --- Heuristic: if max time < 1e5 s, treat it as step count (seconds mis-scaled) ---
if max(time) < 1e5:
    print("Detected times likely not in seconds - scaling to days assuming 30-day steps.")
    time_days = [i * 30 for i in range(len(time))]  # assume 30-day increments
else:
    time_days = [t / 86400 for t in time]

total_days = time_days[-1]
print(f"Number of time steps: {len(time_days)}")
print(f"Total irradiation time: {total_days:.1f} days")

print("="*70)
print("OpenMC Depletion Analysis")
print("="*70)
print(f"Loaded depletion results from: {results_path}")
print(f"Number of time steps: {len(time_days)}")
print(f"Total irradiation time: {time_days[-1]:.1f} days")
print("="*70)

# -------------------------------
# ---- Plot isotope evolution ----
# -------------------------------
plt.figure(figsize=(7,5))
for iso in isotopes_to_plot:
    try:
        _, atoms = results.get_atoms(material_id, iso)
        plt.semilogy(time_days, atoms, label=iso)
    except KeyError:
        print(f"Isotope {iso} not found in results.")
plt.xlabel("Irradiation time [days]")
plt.ylabel("Number of atoms")
plt.title("Isotope evolution in target material")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "isotope_evolution.png"), dpi=300)
plt.close()
print("→ Saved isotope evolution plot")

# -------------------------------
# ---- Plot total activity ----
# -------------------------------
try:
    _, activity = results.get_activity(material_id)
    plt.figure(figsize=(7,5))
    plt.semilogy(time_days, activity)
    plt.xlabel("Irradiation time [days]")
    plt.ylabel("Total Activity [Bq]")
    plt.title("Total Activity vs. Time")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "activity_vs_time.png"), dpi=300)
    plt.close()
    print("→ Saved activity vs time plot")
except Exception as e:
    print(f"Could not compute activity: {e}")

# -------------------------------
# ---- Plot decay heat ----
# -------------------------------
try:
    _, decay_heat = results.get_decay_heat(material_id)
    plt.figure(figsize=(7,5))
    plt.plot(time_days, decay_heat)
    plt.xlabel("Irradiation time [days]")
    plt.ylabel("Decay heat [W]")
    plt.title("Decay Heat vs. Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decay_heat_vs_time.png"), dpi=300)
    plt.close()
    print("→ Saved decay heat vs time plot")
except Exception as e:
    print(f"Could not compute decay heat: {e}")

# -------------------------------
# ---- Load final composition ----
# -------------------------------
final_csv = os.path.join(output_dir, "final_composition.csv")
if os.path.exists(final_csv):
    df = pd.read_csv(final_csv)
    print("\nTop 20 isotopes in final composition:")
    print(df.head(20).to_string(index=False))
else:
    print("\nWarning: final_composition.csv not found.")
    df = None

# -------------------------------
# ---- Flux verification ----
# -------------------------------
try:
    statepoint_file = os.path.join(output_dir, f"statepoint.{statepoint_batches}.h5")
    sp = openmc.StatePoint(statepoint_file)
    tally = sp.get_tally(name='surface_current')
    current = tally.mean.flatten()[0]
    print(f"\nSurface current tally (per source particle): {current:.3e}")
except Exception as e:
    print(f"\nCould not read statepoint flux tally: {e}")

# -------------------------------
# ---- Compute specific activity for key isotope ----
# -------------------------------
target_iso = "Tb161"
try:
    _, atoms = results[-1].get_atoms(material_id, target_iso)
    half_life_days = 6.89
    lam = log(2) / (half_life_days * 86400)
    activity_Bq = atoms * lam
    mass_g = atoms * 161 / Avogadro
    spec_act = activity_Bq / mass_g
    print(f"\nSpecific activity of {target_iso}: {spec_act:.3e} Bq/g")
except KeyError:
    print(f"\n{target_iso} not found in final step.")
except Exception as e:
    print(f"Could not compute specific activity: {e}")

# -------------------------------
# ---- Export isotope evolution table ----
# -------------------------------
records = []
for iso in isotopes_to_plot:
    try:
        _, atoms = results.get_atoms(material_id, iso)
        records.append({'Isotope': iso, **{f"t={int(t)}s": a for t, a in zip(time, atoms)}})
    except KeyError:
        continue

if records:
    df_iso = pd.DataFrame(records)
    df_iso.to_csv(os.path.join(output_dir, "isotope_evolution.csv"), index=False)
    print("→ Exported isotope evolution table to CSV")

print("\nAll analysis complete.")
print(f"Results and plots saved in: {os.path.abspath(output_dir)}")
print("="*70)


#### More plotting stuff


# ---- Gadolinium isotopes to track ----
gd_isotopes = [f"Gd{i}" for i in range(148, 161)]

# ---- Extract time array ----
time = results.get_times()
if max(time) < 1e5:   # seconds too small → treat as step count in days
    time_days = [i * 30 for i in range(len(time))]
else:
    time_days = [t / 86400 for t in time]

# ---- Build isotope evolution dictionary ----
evolution = {}
for iso in gd_isotopes:
    try:
        _, atoms = results.get_atoms('1', iso)
        evolution[iso] = atoms
    except Exception:
        continue  # skip isotopes not in chain

# ---- Check if anything found ----
if not evolution:
    print(" No gadolinium isotopes found in depletion results.")
else:
    # ---- Plot each isotope's evolution ----
    plt.figure(figsize=(9,6))
    for iso, atoms in evolution.items():
        plt.semilogy(time_days, atoms, label=iso)

    plt.xlabel("Irradiation time [days]", fontsize=13)
    plt.ylabel("Number of atoms", fontsize=13)
    plt.title("Evolution of Gadolinium Isotopes (148–160)", fontsize=14)
    plt.legend(loc="best", ncol=2, fontsize=10)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "gd_isotope_evolution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Gd isotope evolution plot → {out_path}")

