#!/usr/bin/env python3
"""
Extended OpenMC Depletion Analysis
----------------------------------
Post-processes a D–T fusion irradiation depletion simulation.

Features:
- Plots isotope evolution over time
- Plots total activity and decay heat
- Exports final composition and isotope ranking to CSV
- Plots the most common reactions / decays
- Plots the top 20 most abundant isotopes over time
"""

import os
import time
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
from adjustText import adjust_text
from openmc.deplete import Results
from math import log
from scipy.constants import Avogadro
import openmc
import openmc.deplete
import h5py


# -------------------------------
# ---- Configuration ----
# -------------------------------




def export_final_composition(results, material_id='1', output_file=None):
    if output_file is None:
        output_file = f"final_composition_{material_id}.csv"
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

    print(" Extracting final composition...")

    # --- Identify candidate nuclides ---
    common_nuclides = [
        "H1", "H2", "H3", "He3", "He4",
        "Li6", "Li7", "Be9", "B10", "B11",
        "C12", "C13", "C14", "N14", "N15", "N16", "O16",
        "Ne20", "Na23", "Mg24", "Al27", "Si28",
        # EUROFER97
        "Fe54", "Fe55", "Fe56", "Fe57", "Fe58", "Fe59", "Fe60",
        "Cr50", "Cr51", "Cr52", "Cr53", "Cr54",
        "W180", "W181", "W182", "W183", "W184", "W185", "W186", "W187",
        "Mn54", "Mn55", "Mn56",
        "Ta179", "Ta180", "Ta181", "Ta182",
        "Ni58", "Ni60", "Ni61", "Ni62", "Ni64",
        # Copper (all)
        "Cu58", "Cu59", "Cu60", "Cu61", "Cu62", "Cu63", "Cu64", "Cu65",
        "Cu66", "Cu67", "Cu68", "Cu69", "Cu70", "Cu71", "Cu72",
        "Cu73", "Cu74", "Cu75", "Cu76", "Cu77", "Cu78", "Cu79", "Cu80", "Cu81",
        # Zinc (all)
        "Zn62", "Zn63", "Zn64", "Zn65", "Zn66", "Zn67", "Zn68", "Zn69", "Zn70",
        "Zn71", "Zn72", "Zn73", "Zn74", "Zn75", "Zn76", "Zn77", "Zn78", "Zn79", "Zn80",
        # Lanthanides
        "Sm146", "Sm147", "Sm148", "Sm149", "Sm150", "Sm152", "Sm154",
        "Gd147", "Gd148", "Gd149", "Gd150", "Gd151",
        "Gd152", "Gd153", "Gd154", "Gd155", "Gd156", "Gd157", "Gd158", "Gd160",
        "Tb159", "Tb160", "Tb161", "Tb162", "Tb163",
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
    
    


def plot_cu_isotopes_evolution(results, output_dir, material_id="1", cu_isotopes=None, time_days=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Cu isotopes evolution ----
    # -------------------------------
    final_atoms_dict = {}
    plt.figure(figsize=(8,6))
    for iso in cu_isotopes:
        try:
            _, atoms = results.get_atoms(material_id, iso)
            final_atoms_dict[iso] = atoms[-1]  # Final atoms after cooldown (last timestep)
            plt.semilogy(time_days, atoms, label=iso)
        except KeyError:
            continue

    # Add vertical line to mark end of irradiation / start of cooldown
    if irradiation_end_idx is not None and irradiation_end_idx < len(time_days):
        plt.axvline(x=time_days[irradiation_end_idx-1], color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='End of Irradiation')
        # Add text annotation
        plt.text(time_days[irradiation_end_idx-1], 1e3, 
                'Cooldown\nstarts', ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel("Time [days]")  # Updated to include both phases
    plt.ylabel("Number of atoms")
    plt.ylim(1e-7, 1e18)    
    plt.title("Evolution of Copper Isotopes (63–71)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    cu_path = os.path.join(output_dir, f"cu_isotope_evolution{material_id}.png")
    plt.savefig(cu_path, dpi=300)
    plt.close()
    print(f"Saved Cu isotope evolution plot {cu_path}")

    return final_atoms_dict  # Returns final atoms after cooldown

def plot_zn_isotopes_activity_evolution(results, output_dir, material_id_list=None, isotopes_to_plot=None, time_days=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Zn isotopes activity evolution with extended decay ----
    # -------------------------------
    
    # Create extended time array: original + 100 days of decay
    final_time = time_days[-1]
    decay_days = 100
    n_decay_points = 100  # Number of points for decay curve
    extended_time = np.concatenate([
        time_days,
        np.linspace(final_time, final_time + decay_days, n_decay_points)
    ])
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(isotopes_to_plot)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(isotopes_to_plot)))
    color_scheme = {}
    
    plt.figure(figsize=(8,6))
    for material_id in material_id_list:
        _, activities = results.get_activity(material_id, units='Bq', by_nuclide=True)
        color_scheme = blue_colors if material_id == '1' else red_colors
        
        for iso_idx, iso in enumerate(isotopes_to_plot):
            activity = []
            for act_dict in activities:
                if iso in act_dict:
                    activity.append(act_dict[iso])
                    idx = activities.index(act_dict)
                else:
                    activity.append(0.0)
            
            activity = np.array(activity)
            mCi_activity = activity / 37000000
            
            # Get final activity value (at last time point)
            final_activity = activity[-1] if len(activity) > 0 else 0.0
            final_mCi_activity = final_activity / 37000000
            
            # Calculate decay for extended period
            if final_activity > 0:
                try:
                    half_life_seconds = openmc.data.half_life(iso)
                    if half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                        decay_constant = np.log(2) / half_life_seconds
                        time_elapsed = extended_time[len(time_days):] - final_time
                        # Exponential decay: A(t) = A₀ * exp(-λ * t)
                        decayed_activity = final_activity * np.exp(-decay_constant * time_elapsed)
                        decayed_mCi_activity = decayed_activity / 37000000
                        extended_activity = np.concatenate([mCi_activity, decayed_mCi_activity])
                    else:
                        # Stable isotope (infinite half-life) - no decay
                        extended_activity = np.concatenate([mCi_activity, np.full(n_decay_points, final_mCi_activity)])
                except Exception:
                    # If half-life lookup fails, extend with constant value
                    extended_activity = np.concatenate([mCi_activity, np.full(n_decay_points, final_mCi_activity)])
            else:
                # No activity - extend with zeros
                extended_activity = np.concatenate([mCi_activity, np.zeros(n_decay_points)])
            
                        
            plt.semilogy(extended_time, extended_activity, label=f"{material_id} - {iso}", color=color_scheme[iso_idx], linewidth=1.5)

            # Add gray text for activity at 60 days if > 1 mCi
            t_label = 60  # days
            if t_label <= extended_time[-1]:
                # Interpolate activity at 60 days
                activity_at_60 = np.interp(t_label, extended_time, extended_activity)
                if activity_at_60 > 1:  # Only label if > 10^0 = 1 mCi
                    plt.text(t_label, activity_at_60, f' {activity_at_60:.1f} mCi', 
                            fontsize=6, color='gray', va='center', ha='left')

    # Add vertical line to mark end of irradiation / start of cooldown
    if irradiation_end_idx is not None and irradiation_end_idx < len(time_days):
        plt.axvline(x=time_days[irradiation_end_idx-1], color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='End of Irradiation')
        # Add text annotation
        plt.text(time_days[irradiation_end_idx-1], plt.ylim()[1]*0.1, 
                'Cooldown\nstarts', ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    #add horizontal colored bands identifyingzones at <=2 mCi C-level, >2 <=100mCi B-level, >100mCi A-level
    plt.axhspan(0, 2, alpha=0.2, color='green', zorder=0)      # C-level: <=2 mCi
    plt.axhspan(2, 100, alpha=0.2, color='yellow', zorder=0)  # B-level: 2-100 mCi
    plt.axhspan(100, 10e6, alpha=0.2, color='red', zorder=0)    # A-level: >100 mCi
    plt.text(1, 1, 'C-level', fontsize=10, fontweight='bold', va='center')
    plt.text(1, 10, 'B-level', fontsize=10, fontweight='bold', va='center')
    plt.text(1, 1000, 'A-level', fontsize=10, fontweight='bold', va='center')

    plt.xlabel("Time [days]")
    plt.ylim(1e-1, 10e6)
    plt.ylabel("Activity [mCi]")
    plt.title("Activity Evolution of Zinc Isotopes (64–71) with >100-Day Decay Projection")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    zn_activity_path = os.path.join(output_dir, f"zn_isotope_activity_evolution.png")
    plt.savefig(zn_activity_path, dpi=300)
    plt.close()
    print(f"→ Saved Zn isotope activity evolution plot → {zn_activity_path}")

    return

def plot_zn_isotopes_SA_evolution(results, output_dir, material_id_list=None, isotopes_to_plot=None, time_days=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Zn isotopes activity evolution with extended decay ----
    # -------------------------------
    
    # Create extended time array: original + 100 days of decay
    final_time = time_days[-1]
    decay_days = 100
    n_decay_points = 100  # Number of points for decay curve
    extended_time = np.concatenate([
        time_days,
        np.linspace(final_time, final_time + decay_days, n_decay_points)
    ])
    
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(isotopes_to_plot)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(isotopes_to_plot)))
    color_scheme = {}
    
    plt.figure(figsize=(8,6))
    for material_id in material_id_list:
        color_scheme = blue_colors if material_id == '1' else red_colors
        total_mass = 0
        for iso in isotopes_to_plot:
            _, atoms = results.get_atoms(material_id, iso)
            mass_g = (atoms *openmc.data.atomic_mass(iso)) / Avogadro
            total_mass += mass_g

        _, activities = results.get_activity(material_id, units='Bq', by_nuclide=True)
        for iso_idx, iso in enumerate(isotopes_to_plot):
            activity = []
            for act_dict in activities:
                if iso in act_dict:
                    activity.append(act_dict[iso])
                else:
                    activity.append(0.0)
            _, atoms = results.get_atoms(material_id, iso)
            mass_g = (atoms *openmc.data.atomic_mass(iso)) / Avogadro
            activity = np.array(activity)
            SA = np.divide(activity, total_mass, out=np.zeros_like(activity), where=(total_mass > 0))
            final_SA = SA[-1] if len(SA) > 0 else 0.0
            
            # Calculate decay for extended period
            if final_SA > 0:
                try:
                    half_life_seconds = openmc.data.half_life(iso)
                    if half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                        decay_constant = np.log(2) / half_life_seconds
                        time_elapsed = extended_time[len(time_days):] - final_time
                        # Exponential decay: A(t) = A₀ * exp(-λ * t)
                        # final_SA is already specific activity, so just decay it directly
                        decayed_SA = final_SA * np.exp(-decay_constant * time_elapsed)
                        extended_SA = np.concatenate([SA, decayed_SA])
                    else:
                        # Stable isotope (infinite half-life) - no decay
                        extended_SA = np.concatenate([SA, np.full(n_decay_points, final_SA)])
                except Exception:
                    # If half-life lookup fails, extend with constant value
                    extended_SA = np.concatenate([SA, np.full(n_decay_points, final_SA)])
            else:
                # No activity - extend with zeros
                extended_SA = np.concatenate([SA, np.zeros(n_decay_points)])
            
            plt.semilogy(extended_time, extended_SA, label=f"{material_id} - {iso}", color=color_scheme[iso_idx], linewidth=1.5)

                        # Add gray text for activity at 60 days if > 1 mCi
            t_label = 60  # days
            if t_label <= extended_time[-1]:
                # Interpolate activity at 60 days
                SA_at_60 = np.interp(t_label, extended_time, extended_SA)
                if SA_at_60 > 1:  # Only label if > 10^0 = 1 mCi
                    plt.text(t_label, SA_at_60, f' {SA_at_60:.1f} mCi', 
                            fontsize=6, color='gray', va='center', ha='left')

    #add horizontal line at 0.1 Bq/g labeled Exclusion Level  
    plt.axhline(y=0.1, color='black', linestyle='--', 
               linewidth=1.5, alpha=0.7, label='Exclusion Level')
    # Position text at the right side, near the exclusion line
    x_pos = extended_time[-1] * 0.95  # Near the right edge
    y_pos = 0.1 * 2.0  # Above the exclusion line (0.1 * 2 = 0.2)
    plt.text(x_pos, y_pos, 
            'Exclusion\nLevel', ha='right', va='bottom', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel("Time [days]")
    plt.ylabel("Specific Activity [Bq/g]")
    plt.title("Specific Activity Evolution of Zinc")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    zn_activity_path = os.path.join(output_dir, f"zn_isotope_specific_activity_evolution.png")
    plt.savefig(zn_activity_path, dpi=300)
    plt.close()
    print(f"→ Saved Zn isotope specific activity evolution plot → {zn_activity_path}")

    return
def plot_cu_isotopes_activity_evolution(results, output_dir, material_id_list=None, cu_isotopes=None, time_days=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Cu isotopes activity evolution ----
    # -------------------------------
    # Create extended time array: original + 100 days of decay
    final_time = time_days[-1]
    decay_days = 1
    n_decay_points = 100  # Number of points for decay curve
    n_irr = len(time_days)
    n_ext = n_irr + n_decay_points   
    extended_time = np.concatenate([
        time_days,
        np.linspace(final_time, final_time + decay_days, n_decay_points)
    ])

    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cu_isotopes)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cu_isotopes)))
    color_scheme = {}
    total_activities = {material_id: [] for material_id in material_id_list}
    activity_results = {material_id: [] for material_id in material_id_list}
    # cooldown_activity_results removed; we extend activity_results in the same dict-per-timestep format
    plt.figure(figsize=(8,6))

    for material_id in material_id_list:
        color_scheme = blue_colors if material_id == '1' else red_colors
        _, activities = results.get_activity(material_id, units='Bq', by_nuclide=True)
        sum_irr = np.zeros(n_irr, dtype=float)           # sum of all Cu (Bq) at each irradiation timestep
        sum_cooldown = np.zeros(n_decay_points, dtype=float)  # sum of all Cu (Bq) at each cooldown timestep
        activity_results[material_id].append(activities)
        cooldown_time = np.linspace(0, decay_days, n_decay_points)
        # One dict per cooldown timestep, same format as activities[t]
        cooldown_dicts = [{} for _ in range(n_decay_points)]

        for iso_idx, iso in enumerate(cu_isotopes):
            activity = []
            for idx, act_dict in enumerate(activities):
                val = act_dict.get(iso, 0.0)
                activity.append(val)
                sum_irr[idx] += val

            activity = np.array(activity)
            mCi_activity = activity / 37000000
            final_activity = activity[-1] if len(activity) > 0 else 0.0
            final_mCi_activity = final_activity / 37000000

            if final_activity > 0:
                try:
                    half_life_seconds = openmc.data.half_life(iso)
                    if half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                        decay_constant = np.log(2) / half_life_seconds
                        cooldown_activity = np.zeros(len(cooldown_time))
                        for i in range(len(cooldown_time)):
                            time_elapsed = cooldown_time[i] * 86400
                            decayed_activity = final_activity * np.exp(-decay_constant * time_elapsed)
                            cooldown_dicts[i][iso] = decayed_activity
                            sum_cooldown[i] += decayed_activity
                            cooldown_activity[i] = decayed_activity / 37000000
                        extended_activity = np.concatenate([mCi_activity, cooldown_activity])
                    else:
                        for i in range(len(cooldown_time)):
                            cooldown_dicts[i][iso] = final_activity
                            sum_cooldown[i] += final_activity
                        extended_activity = np.concatenate([mCi_activity, np.full(n_decay_points, final_mCi_activity)])
                except Exception:
                    for i in range(len(cooldown_time)):
                        cooldown_dicts[i][iso] = final_activity
                        sum_cooldown[i] += final_activity
                    extended_activity = np.concatenate([mCi_activity, np.full(n_decay_points, final_mCi_activity)])
            else:
                for i in range(len(cooldown_time)):
                    cooldown_dicts[i][iso] = 0.0
                extended_activity = np.concatenate([mCi_activity, np.zeros(n_decay_points)])

            plt.semilogy(extended_time, extended_activity, label=f"{material_id} - {iso}", color=color_scheme[iso_idx], linewidth=1.5)

        # Append cooldown in the same format (list of dicts, one dict per timestep)
        activity_results[material_id][0].extend(cooldown_dicts) 

        total_activities[material_id] = np.concatenate([sum_irr, sum_cooldown]).astype(float)
        assert total_activities[material_id].shape[0] == n_ext, (
            "total_activities must have length n_irr+n_decay_points (109)"
        )
        final_total_activity = total_activities[material_id][-1] if len(total_activities[material_id]) > 0 else 0.0

    # Add vertical line to mark end of irradiation / start of cooldown
    if irradiation_end_idx is not None and irradiation_end_idx < len(time_days):
        plt.axvline(x=time_days[irradiation_end_idx-1], color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='End of Irradiation')
        # Add text annotation
        plt.text(time_days[irradiation_end_idx-1], plt.ylim()[1]*0.1, 
                'Cooldown\nstarts', ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel("Time [days]")
    plt.ylabel("Activity [Bq]")
    plt.ylim(1e-7, 1e13)
    plt.title("Activity Evolution of Copper Isotopes (63–71) - Irradiation Phase")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    cu_activity_path = os.path.join(output_dir, f"cu_isotope_activity_evolution.png")
    plt.savefig(cu_activity_path, dpi=300)
    plt.close()
    print(f"→ Saved Cu isotope activity evolution plot → {cu_activity_path}")

    return total_activities, activity_results

def plot_cu_isotopes_purity_evolution(results, output_dir, material_id_list=None, cu_isotopes=None, time_days=None, total_activities=None, activity_results=None, irradiation_end_idx=None):
    if activity_results is None or total_activities is None:
        print("Warning: plot_cu_isotopes_purity_evolution needs activity_results and total_activities from plot_cu_isotopes_activity_evolution")
        return {}, {}

    # Build extended_time (irradiation + cooldown)
    final_time = time_days[-1]
    decay_days = 1
    n_decay_points = 100
    extended_time = np.concatenate([
        time_days,
        np.linspace(final_time, final_time + decay_days, n_decay_points)
    ])
    cooldown_time = np.linspace(0, decay_days, n_decay_points)
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cu_isotopes)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cu_isotopes)))
    purity_results = {mid: None for mid in material_id_list}
    SA_results = {mid: {} for mid in material_id_list}

    plt.figure(figsize=(8, 6))
    for material_id in material_id_list:
        color_scheme = blue_colors if material_id == '1' else red_colors
        # Total copper mass (g)
                # Total copper mass (g): irradiation from get_atoms; cooldown = decay of each Cu isotope
        total_mass = 0.0
        cooldown_masses = np.zeros(n_decay_points)   # before loop, use n_decay_points

        for iso in cu_isotopes:
            _, atoms = results.get_atoms(material_id, iso)
            mass_per_atom = openmc.data.atomic_mass(iso) / Avogadro
            total_mass += (atoms * openmc.data.atomic_mass(iso)) / Avogadro   # irradiation only

            atoms_end = np.atleast_1d(np.asarray(atoms))[-1]
            try:
                half_life_seconds = openmc.data.half_life(iso)
                if half_life_seconds is not None and np.isfinite(half_life_seconds) and half_life_seconds > 0:
                    decay_constant = np.log(2) / half_life_seconds
                    for i in range(n_decay_points):
                        t_sec = cooldown_time[i] * 86400
                        cooldown_masses[i] += atoms_end * mass_per_atom * np.exp(-decay_constant * t_sec)
                else:
                    cooldown_masses[:] += atoms_end * mass_per_atom   # stable
            except Exception:
                cooldown_masses[:] += atoms_end * mass_per_atom

        total_mass = np.concatenate([np.atleast_1d(np.asarray(total_mass, dtype=float)), cooldown_masses])

        # Extended activities: list of dicts, one per timestep (irradiation + cooldown)
        list_of_dicts = activity_results[material_id][0]

        tot_act = np.asarray(total_activities[material_id], dtype=float)
        if tot_act.shape[0] != len(list_of_dicts):
            print(f"Error: total_activities shape {tot_act.shape} does not match list_of_dicts shape {len(list_of_dicts)}")
            tot_act = np.array([sum(d.get(iso, 0.0) for iso in cu_isotopes) for d in list_of_dicts], dtype=float)

        for iso_idx, iso in enumerate(cu_isotopes):
            # Activity (Bq) over all timesteps from imported activity_results
            act = np.array([list_of_dicts[t].get(iso, 0.0) for t in range(len(list_of_dicts))])
            # Specific activity = activity / total copper mass (Bq/g)
            SA = np.divide(act, total_mass, out=np.zeros_like(act, dtype=float), where=(total_mass > 0))
            SA_results[material_id][iso] = SA
            plt.semilogy(extended_time, SA, label=f"{material_id} - {iso}", color=color_scheme[iso_idx], linewidth=1.5)

    plt.xlabel("Time [days]")
    plt.ylabel("Specific Activity [Bq/g]")
    plt.title("Specific Activity Evolution of Copper Isotopes (63–71)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cu_isotope_specific_activity_evolution.png"), dpi=300)
    plt.close()
    print("→ Saved Cu isotope specific activity evolution plot → …")

    # Purity plot (1 - purity) vs extended_time — optional if you only care about final purity
    plt.figure(figsize=(8, 6))
    for material_id in material_id_list:
        color_scheme = blue_colors if material_id == '1' else red_colors
        list_of_dicts = activity_results[material_id][0]
        tot_act = np.asarray(total_activities[material_id], dtype=float)
        # Ensure tot_act length matches list_of_dicts (e.g. 109 = irr + cooldown); recompute from activity_results if not
        if tot_act.shape[0] != len(list_of_dicts):
            tot_act = np.array([sum(d.get(iso, 0.0) for iso in cu_isotopes) for d in list_of_dicts], dtype=float)
        cu64_final, cu67_final = 0.0, 0.0
        for iso in ['Cu64', 'Cu67']:
            if iso not in cu_isotopes:
                continue
            act = np.array([list_of_dicts[t].get(iso, 0.0) for t in range(len(list_of_dicts))])
            purity = np.divide(act, tot_act, out=np.zeros_like(act, dtype=float), where=(tot_act > 0))
            if iso == 'Cu64':
                cu64_final = purity[-1]
            elif iso == 'Cu67':
                cu67_final = purity[-1]
            imp = np.where(1.0 - purity > 0, 1.0 - purity, 1e-30)
            iso_idx = cu_isotopes.index(iso) if iso in cu_isotopes else 0
            plt.semilogy(extended_time, imp, label=f"{material_id} - {iso}", color=color_scheme[iso_idx], linewidth=1.5)
            for t_mark in [0.17, 1.0, 2.0]:
                if t_mark < extended_time[0] or t_mark > extended_time[-1]:
                    continue
                y_mark = np.interp(t_mark, extended_time, imp)
                pct = 100.0 * (1.0 - y_mark)
                plt.plot(t_mark, y_mark, "o", color='black', markersize=4)
                plt.text(t_mark, y_mark * 0.8, f"  {pct:.1f}%", fontsize=7, color='black', va="center")
        purity_results[material_id] = [np.array([cu64_final]), np.array([cu67_final])]

    plt.xlabel("Irradiation time [days]")
    plt.ylabel("1 - Purity (Impurity fraction, log scale)")
    plt.title("Cu64, Cu67 — materials 0 and 1")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cu_isotope_purity_evolution.png"), dpi=300)
    plt.close()
    print("→ Saved Cu isotope purity evolution plot")

    return purity_results, SA_results


def plot_total_activity_evolution(results, output_dir, material_id="1", time_days=None):
    # -------------------------------
    # ---- Plot total activity ---
    # -------------------------------
    try:
        _, activity = results.get_activity(material_id)
        plt.figure(figsize=(7,5))
        plt.semilogy(time_days, activity)
        plt.xlabel("Time [days]")
        plt.ylabel("Total Activity [Bq]")
        plt.title("Total Activity vs. Time")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"activity_vs_time{material_id}.png"), dpi=300)
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
        plt.xlabel("Time [days]")
        plt.ylabel("Decay heat [W]")
        plt.title("Decay Heat vs. Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"decay_heat_vs_time{material_id}.png"), dpi=300)
        plt.close()
        print("→ Saved decay heat vs time plot")
    except Exception as e:
        print(f"Could not compute decay heat: {e}")


def plot_flux_spectra_and_heating_by_cell(statepoint_file, output_dir, time_days=None, batches=None):
    """
    From statepoint tallies:
      (1) Neutron energy spectrum (volume_flux_spectra): one figure, flux vs energy,
          one line per cell, different colors.
      (2) Volumetric heating (volumetric_heating): CSV + one figure with all cells,
          different colors, to compare how dangerous each cell is.

    Parameters
    ----------
    statepoint_file : str
        Path to statepoint (e.g. 'statepoint.2.h5' when run from output_dir).
    output_dir : str
        Directory for figures and CSV.
    material_id : str
        Material ID (default '1').
    cells : list, optional
        List of openmc.Cell; if provided, cell names are used for labels.
    """
    sp = openmc.StatePoint(statepoint_file)
    cells = []
    inner_thickness = float(output_dir.split("_inner")[1].split("_")[0])
    outer_thickness = float(output_dir.split("_outer")[1].split("_")[0])
    moderator_thickness = float(output_dir.split("_moderator")[1].split("_")[0])
    multi_thickness = float(output_dir.split("_multi")[1].split("_")[0])
    # cells struct_cell, inner_target_cell, outer_target_cell, moderator_cell, multi_cell
    cells.append('struct')
    if inner_thickness > 0:
        cells.append('inner_target')
    if outer_thickness > 0:
        cells.append('outer_target')
    if moderator_thickness > 0:
        cells.append('moderator')
    if multi_thickness > 0:
        cells.append('multi')

        # --- 1) Neutron energy spectrum (volume_flux_spectra): cumulative step per cell ---
    try:
        t = sp.get_tally(name='volume_flux_spectra')
        edges = None
        for f in t.filters:
            if isinstance(f, openmc.EnergyFilter):
                v = np.asarray(f.values)
                edges = np.unique(np.sort(v))
                break
        energy_mid = (edges[:-1] + edges[1:]) / 2.0

        mean = np.squeeze(t.mean)
        if mean.ndim == 1:
            n_energy = len(energy_mid)
            n_cells = len(mean) // n_energy
            mean = mean.reshape(n_cells, n_energy)
        n_cells = mean.shape[0]

        colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
        
        # Calculate lethargy width for each bin: Δu = ln(E_high / E_low)
        lethargy_width = np.log(edges[1:] / edges[:-1])

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(n_cells):
            # Convert flux per bin to flux per unit lethargy
            flux_per_lethargy = mean[i, :] / lethargy_width
            ax.plot(energy_mid, flux_per_lethargy, lw=0.8, color=colors[i], label=cells[i] if i < len(cells) else f'Cell {i}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Neutron Energy [eV]")
        ax.set_ylabel("Flux per Unit Lethargy [a.u./lethargy]")
        ax.set_title(f"Neutron Energy Spectrum by Cell: {output_dir}")
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, which="both", ls=":", alpha=0.5)
        
        # Set sensible x-axis limits (thermal to fusion)
        ax.set_xlim(1e-3, 2e7)  # 1 meV to 20 MeV
        
        # Add reference energy annotations
        ax.axvline(x=0.025, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=14.1e6, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        
        fig.tight_layout()
        png = os.path.join(output_dir, f"neutron_energy_spectra_by_cell_{output_dir}.png")
        fig.savefig(png, dpi=300)
        plt.close(fig)
        print(f"→ Saved neutron energy spectrum by cell → {png}")
    except Exception as e:
        print(f"Could not plot neutron energy spectrum by cell: {e}")





def plot_geom_purity_evolution(purity_results_list=None, output_dir_list=None, summary_output_dir=None):

    valid_dirs = [d for d in output_dir_list if d in purity_results_list and purity_results_list[d] is not None]
    n_plots = len(valid_dirs)
    
    if n_plots == 0:
        print("Warning: No valid purity data found for geometry plots")
        return
    
    # Calculate grid dimensions (prefer wider than tall)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Color gradients
    colors_64 = plt.cm.Blues(np.linspace(0.3, 1, 101))
    colors_67 = plt.cm.Reds(np.linspace(0.3, 1, 101))


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    all_purity_cu67_last = []
    all_purity_cu64_last = []

    for i, output_dir in enumerate(output_dir_list):
        if output_dir not in purity_results_list or purity_results_list[output_dir] is None:
            print(f"Warning: plot_geom_purity_evolution: '{output_dir}' not found in purity_results_list, skipping")
            continue
        purity_results = purity_results_list[output_dir]
        purity_cu67_last = purity_results['0'][1][-1]
        purity_cu64_last = purity_results['1'][0][-1]
        all_purity_cu67_last.append(purity_cu67_last)
        all_purity_cu64_last.append(purity_cu64_last)

    max_purity_cu67_last = np.max(all_purity_cu67_last)
    max_purity_cu64_last = np.max(all_purity_cu64_last)
    min_purity_cu67_last = np.min(all_purity_cu67_last)
    min_purity_cu64_last = np.min(all_purity_cu64_last)
    
    for i, output_dir in enumerate(output_dir_list):
        if output_dir not in purity_results_list or purity_results_list[output_dir] is None:
            print(f"Warning: plot_geom_purity_evolution: '{output_dir}' not found in purity_results_list, skipping")
            continue
        purity_results = purity_results_list[output_dir]
        ax = axes[i]

        z_inner_thickness = float(output_dir.split("_inner")[1].split("_")[0])
        z_outer_thickness = float(output_dir.split("_outer")[1].split("_")[0])
        struct_thickness = float(output_dir.split("_struct")[1].split("_")[0])
        moderator_thickness = float(output_dir.split("_moderator")[1].split("_")[0])
        multi_thickness = float(output_dir.split("_multi")[1].split("_")[0])

        inner_radius = 5
        center = (0, 0)
        
        # Get purity values at last timestep
        purity_cu67_last = purity_results['0'][1][-1]  # Material '0', Cu67 (index 1), last timestep
        purity_cu64_last = purity_results['1'][0][-1]  # Material '1', Cu64 (index 0), last timestep
        
        # Blue: Map 97-99.8% (0.97-0.998) to 0-100 (dull to vibrant)
        def blue_purity_to_color_index(purity):
            if purity < min_purity_cu64_last:
                return 0  # Dull for < 97%
            elif purity > max_purity_cu64_last:
                return 100  # Max vibrant for > 99.8%
            else:
                # Map 0.97-0.998 to 0-100
                index = int((purity - min_purity_cu64_last) / (max_purity_cu64_last - min_purity_cu64_last) * 100)
                return min(max(index, 0), 100)

        # Red: Map 1-3% (0.01-0.03) to 0-100 (dull to vibrant)
        def red_purity_to_color_index(purity):
            if purity < min_purity_cu67_last:
                return 0  # Dull for < 1%
            elif purity > max_purity_cu67_last:
                return 100  # Max vibrant for > 3%
            else:
                # Map 0.01-0.03 to 0-100
                index = int((purity - min_purity_cu67_last) / (max_purity_cu67_last - min_purity_cu67_last) * 100)
                return min(max(index, 0), 100)
        
        color_idx_64 = blue_purity_to_color_index(purity_cu64_last)
        color_idx_67 = red_purity_to_color_index(purity_cu67_last)

        # Draw circles for reference (outlines only)
        radii = [
            inner_radius,
            inner_radius + struct_thickness,
            inner_radius + struct_thickness + z_inner_thickness,
            inner_radius + struct_thickness + z_inner_thickness + multi_thickness,
            inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness,
            inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness + z_outer_thickness
        ]
        
        labels = ['Inner', 'Struct', 'Z inner', 'Multi', 'Moderator', 'Z outer']
        for r, label in zip(radii, labels):
            circle = mpatches.Circle(center, r, fill=False, edgecolor='black', linestyle='--', linewidth=0.5)
            ax.add_patch(circle)

        # Draw annulus for Z inner (Cu67 purity)
        r_inner_z = inner_radius + struct_thickness
        r_outer_z = inner_radius + struct_thickness + z_inner_thickness
        annulus_z = mpatches.Annulus(center, r_outer_z, r_outer_z - r_inner_z, color=colors_67[color_idx_67])
        ax.add_patch(annulus_z)

        # Draw annulus for Z outer (Cu64 purity)
        r_inner_z_outer = inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness
        r_outer_z_outer = r_inner_z_outer + z_outer_thickness
        annulus_z_outer = mpatches.Annulus(center, r_outer_z_outer, r_outer_z_outer - r_inner_z_outer, color=colors_64[color_idx_64])
        ax.add_patch(annulus_z_outer)
        
        # Set equal aspect ratio and axis limits
        ax.set_aspect('equal')
        max_radius = radii[-1] * 1.1
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        # Improve title alignment with better spacing
        ax.set_title(f'{output_dir}\nCu67: {purity_cu67_last*100:.2f}%, Cu64: {purity_cu64_last*100:.2f}%', 
                    pad=5, fontsize=6)

    plt.tight_layout(pad=2.0)
    out_dir = summary_output_dir if summary_output_dir is not None else (output_dir_list[0] if output_dir_list else ".")
    output_path = os.path.join(out_dir, "geom_purity_evolution.png")  
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved geom purity evolution → {output_path}")


def plot_geom_activity_evolution(activity_results_list=None, output_dir_list=None, summary_output_dir=None):
    
    valid_dirs = [d for d in output_dir_list if d in activity_results_list and activity_results_list[d] is not None]
    n_plots = len(valid_dirs)
    
    if n_plots == 0:
        print("Warning: No valid purity data found for geometry plots")
        return
    
    # Calculate grid dimensions (prefer wider than tall)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Color gradients
    colors_64 = plt.cm.Greens(np.linspace(0.3, 1, 101))
    colors_67 = plt.cm.RdPu(np.linspace(0.3, 1, 101))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = axes.flatten()  # Flatten to 1D array for easier indexing
    all_activity_cu67_last = []
    all_activity_cu64_last = []
    for i, output_dir in enumerate(output_dir_list):
        if output_dir not in activity_results_list or activity_results_list[output_dir] is None:
            print(f"Warning: plot_geom_activity_evolution: '{output_dir}' not found in activity_results_list, skipping")
            continue
        activity_results = activity_results_list[output_dir]
        activity_cu67_last = activity_results['0'][0][-1]['Cu67'] # ['0'] material_id, [0] list of dicts, [-1] last timestep, {'Cu67': activity}[0] -> activity
        activity_cu64_last = activity_results['1'][0][-1]['Cu64'] # ['1'] material_id, [0] list of dicts, [-1] last timestep, {'Cu64': activity}[0] -> activity
        all_activity_cu67_last.append(activity_cu67_last)
        all_activity_cu64_last.append(activity_cu64_last)
    max_activity_cu67_last = np.max(all_activity_cu67_last)
    max_activity_cu64_last = np.max(all_activity_cu64_last)
    min_activity_cu67_last = np.min(all_activity_cu67_last)
    min_activity_cu64_last = np.min(all_activity_cu64_last)

    
    for i, output_dir in enumerate(output_dir_list):
        if output_dir not in activity_results_list or activity_results_list[output_dir] is None:
            print(f"Warning: plot_geom_activity_evolution: '{output_dir}' not found in activity_results_list, skipping")
            continue
        activity_results = activity_results_list[output_dir]
        ax = axes[i]

        z_inner_thickness = float(output_dir.split("_inner")[1].split("_")[0])
        z_outer_thickness = float(output_dir.split("_outer")[1].split("_")[0])
        struct_thickness = float(output_dir.split("_struct")[1].split("_")[0])
        moderator_thickness = float(output_dir.split("_moderator")[1].split("_")[0])
        multi_thickness = float(output_dir.split("_multi")[1].split("_")[0])

        inner_radius = 5
        center = (0, 0)
        
        # Get purity values at last timestep
        activity_cu67_last = activity_results['0'][0][-1]['Cu67']  # ['0'] material_id, [0] list of dicts, [-1] last timestep, {'Cu67': activity}[0] -> activity
        activity_cu64_last = activity_results['1'][0][-1]['Cu64']  # ['1'] material_id, [0] list of dicts, [-1] last timestep, {'Cu64': activity}[0] -> activity
        
        # Green: Map activity to 0-100 based on min/max Cu64 activity
        def green_activity_to_color_index(activity):
            if activity <= min_activity_cu64_last:
                return 0  # Dullest for min activity
            elif activity >= max_activity_cu64_last:
                return 100  # Brightest for max activity
            else:
                # Map min-max to 0-100
                index = int((activity - min_activity_cu64_last) / (max_activity_cu64_last - min_activity_cu64_last) * 100)
                return min(max(index, 0), 100)

        # Pink: Map activity to 0-100 based on min/max Cu67 activity
        def pink_activity_to_color_index(activity):
            if activity <= min_activity_cu67_last:
                return 0  # Dullest for min activity
            elif activity >= max_activity_cu67_last:
                return 100  # Brightest for max activity
            else:
                # Map min-max to 0-100
                index = int((activity - min_activity_cu67_last) / (max_activity_cu67_last - min_activity_cu67_last) * 100)
                return min(max(index, 0), 100)
        
        color_idx_64 = green_activity_to_color_index(activity_cu64_last)
        color_idx_67 = pink_activity_to_color_index(activity_cu67_last)

        # Draw circles for reference (outlines only)
        radii = [
            inner_radius,
            inner_radius + struct_thickness,
            inner_radius + struct_thickness + z_inner_thickness,
            inner_radius + struct_thickness + z_inner_thickness + multi_thickness,
            inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness,
            inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness + z_outer_thickness
        ]
        
        labels = ['Inner', 'Struct', 'Z inner', 'Multi', 'Moderator', 'Z outer']
        for r, label in zip(radii, labels):
            circle = mpatches.Circle(center, r, fill=False, edgecolor='black', linestyle='--', linewidth=0.5)
            ax.add_patch(circle)

        # Draw annulus for Z inner (Cu67 purity)
        r_inner_z = inner_radius + struct_thickness
        r_outer_z = inner_radius + struct_thickness + z_inner_thickness
        annulus_z = mpatches.Annulus(center, r_outer_z, r_outer_z - r_inner_z, color=colors_67[color_idx_67])
        ax.add_patch(annulus_z)

        # Draw annulus for Z outer (Cu64 purity)
        r_inner_z_outer = inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness
        r_outer_z_outer = r_inner_z_outer + z_outer_thickness
        annulus_z_outer = mpatches.Annulus(center, r_outer_z_outer, r_outer_z_outer - r_inner_z_outer, color=colors_64[color_idx_64])
        ax.add_patch(annulus_z_outer)
        
        # Set equal aspect ratio and axis limits
        ax.set_aspect('equal')
        max_radius = radii[-1] * 1.1
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        # Improve title alignment with better spacing
        activity_cu67_last_Ci = activity_cu67_last * 1.0 / 3.7e10
        activity_cu64_last_Ci = activity_cu64_last * 1.0 / 3.7e10
        ax.set_title(f'{output_dir}\nCu67: {activity_cu67_last_Ci:.2f} Ci, Cu64: {activity_cu64_last_Ci:.2f} Ci', 
                    pad=5, fontsize=6)

    plt.tight_layout(pad=2.0)
    out_dir = summary_output_dir if summary_output_dir is not None else (output_dir_list[0] if output_dir_list else ".")
    output_path = os.path.join(out_dir, "geom_activity_evolution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved geom activity evolution → {output_path}")
    
def plot_production_vs_purity(purity_results_list=None, activity_results_list=None, output_dir_list=None, summary_output_dir=None):
    """
    Simple labeled point plot of production vs purity for all output directories.
    Left plot: Cu64 in outer breeding module (material '1') - cool tones
    Right plot: Cu67 in inner breeding module (material '0') - warm tones
    """
    # Get valid directories
    valid_dirs = []
    for d in output_dir_list:
        if (d in purity_results_list and purity_results_list[d] is not None and
            d in activity_results_list and activity_results_list[d] is not None):
            valid_dirs.append(d)
    
    if len(valid_dirs) == 0:
        print("Warning: No valid data found for production vs purity plots")
        return
    
    # Collect data at 1 day
    cu64_purities = []
    cu64_activities = []
    cu67_purities = []
    cu67_activities = []
    labels = []
    
    for output_dir in valid_dirs:
        try:
            # Get activity at 1 day
            activity_results = activity_results_list[output_dir]
            # activity_results[material_id] is a list: [activities] where activities is list of dicts
            if '1' in activity_results and activity_results['1'] and len(activity_results['1']) > 0:
                activities_list = activity_results['1'][0]  # Get list of dicts
                activity_cu64 = activities_list[-1].get('Cu64', 0.0)  
            else:
                activity_cu64 = 0.0
            
            if '0' in activity_results and activity_results['0'] and len(activity_results['0']) > 0:
                activities_list = activity_results['0'][0]  # Get list of dicts
                activity_cu67 = activities_list[-1].get('Cu67', 0.0)  
            else:
                activity_cu67 = 0.0
            
            # Get purity at 1 day
            purity_results = purity_results_list[output_dir]
            # purity_results[material_id] is a list: [Cu64_array, Cu67_array]
            if '1' in purity_results and purity_results['1'] and len(purity_results['1']) > 0:
                cu64_purity_array = purity_results['1'][0]  # Cu64 is at index 0
                purity_cu64 = cu64_purity_array[-1]  
            else:
                purity_cu64 = 0.0
            
            if '0' in purity_results and purity_results['0'] and len(purity_results['0']) > 1:
                cu67_purity_array = purity_results['0'][1]  # Cu67 is at index 1
                purity_cu67 = cu67_purity_array[-1]  
            else:
                purity_cu67 = 0.0
            
            # Convert activity to Curies
            Bq_to_Ci = 1.0 / 3.7e10
            activity_cu64_Ci = activity_cu64 * Bq_to_Ci
            activity_cu67_Ci = activity_cu67 * Bq_to_Ci
            
            cu64_purities.append(purity_cu64 * 100)  # Convert to percentage
            cu64_activities.append(activity_cu64_Ci)
            cu67_purities.append(purity_cu67 * 100)  # Convert to percentage
            cu67_activities.append(activity_cu67_Ci)
            
            # Create simple label from directory name
            label = output_dir.replace("irr_output_", "").replace("_", " ")
            labels.append(label)
        except Exception as e:
            print(f"Warning: Error processing {output_dir}: {e}")
            continue
    
    if len(cu64_purities) == 0:
        print("Warning: No valid data points collected")
        return
    
    # Create two subplots: Cu64 (left, cool tones) and Cu67 (right, warm tones)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cool color palette for Cu64 (outer, material '1')
    cool_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cu64_purities)))
    
    # Warm color palette for Cu67 (inner, material '0')
    warm_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cu67_purities)))
    
    # Plot Cu64 (left, cool tones)
    texts1 = []
    for i, (purity, activity, label) in enumerate(zip(cu64_purities, cu64_activities, labels)):
        ax1.scatter(purity, activity, c=[cool_colors[i]], s=150, alpha=0.7, 
                   edgecolors='darkblue', linewidth=1.5)
        # Add text label
        t = ax1.annotate(label, (purity, activity), fontsize=5, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points', ha='left')
        texts1.append(t)
    adjust_text(texts1, ax=ax1)
    ax1.set_xlabel('Cu64 Purity [%]', fontsize=12)
    ax1.set_ylabel('Cu64 Production [Ci]', fontsize=12)
    ax1.set_title('Cu64 Production vs Purity (Outer Breeding Module)', fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.4, linestyle='--')
    if min(cu64_purities) > 0:
        ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax1.xaxis.get_major_formatter().set_scientific(False)
    
    # Plot Cu67 (right, warm tones)
    texts2 = []
    for i, (purity, activity, label) in enumerate(zip(cu67_purities, cu67_activities, labels)):
        ax2.scatter(purity, activity, c=[warm_colors[i]], s=150, alpha=0.7, 
                   edgecolors='darkred', linewidth=1.5)
        # Add text label
        t = ax2.annotate(label, (purity, activity), fontsize=7, alpha=0.8, 
                    xytext=(5, 5), textcoords='offset points', ha='left')
        texts2.append(t)
    adjust_text(texts2, ax=ax2)
    ax2.set_xlabel('Cu67 Purity [%]', fontsize=12)
    ax2.set_ylabel('Cu67 Production [Ci]', fontsize=12)
    ax2.set_title('Cu67 Production vs Purity (Inner Breeding Module)', fontsize=14, fontweight='bold')
    ax2.grid(True, which='both', alpha=0.4, linestyle='--')
    if min(cu67_purities) > 0:
        ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax2.xaxis.get_major_formatter().set_scientific(False)
    
    plt.tight_layout()
    out_dir = summary_output_dir if summary_output_dir is not None else (output_dir_list[0] if output_dir_list else ".")
    output_path = os.path.join(out_dir, "production_vs_purity_1day.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved production vs purity at 1 day → {output_path}")

def plot_geom_prod_vs_purity(purity_results_list=None, activity_results_list=None, output_dir_list=None, summary_output_dir=None):
    """
    Plot production vs purity as a function of geometry parameters.
    2 columns (Cu64 external, Cu67 internal) x 4 rows (inner, outer, multi, moderator).
    Each row shows how production vs purity changes when varying one parameter (others at minimum).
    """
    # Get valid directories and parse geometry parameters
    valid_dirs = []
    geometry_params = []  # List of dicts: {inner, outer, multi, moderator, dir_name}
    
    for d in output_dir_list:
        if (d in purity_results_list and purity_results_list[d] is not None and
            d in activity_results_list and activity_results_list[d] is not None):
            try:
                # Parse parameters from directory name
                z_inner = float(d.split("_inner")[1].split("_")[0])
                z_outer = float(d.split("_outer")[1].split("_")[0])
                struct = float(d.split("_struct")[1].split("_")[0])
                multi = float(d.split("_multi")[1].split("_")[0])
                moderator = float(d.split("_moderator")[1].split("_")[0])
                
                valid_dirs.append(d)
                geometry_params.append({
                    'inner': z_inner,
                    'outer': z_outer,
                    'struct': struct,
                    'multi': multi,
                    'moderator': moderator,
                    'dir': d
                })
            except Exception as e:
                print(f"Warning: Could not parse geometry from {d}: {e}")
                continue
    
    if len(valid_dirs) == 0:
        print("Warning: No valid data found for geometry production vs purity plots")
        return
    
    # Find minimum values for each parameter
    min_inner = min(p['inner'] for p in geometry_params)
    min_outer = min(p['outer'] for p in geometry_params)
    min_struct = min(p['struct'] for p in geometry_params)
    min_multi = min(p['multi'] for p in geometry_params)
    min_moderator = min(p['moderator'] for p in geometry_params)
    
    # Group directories by which parameter is varying (others at minimum)
    def is_at_min(params, param_name):
        """Check if all parameters except param_name are at minimum"""
        for key, val in params.items():
            if key == param_name or key == 'dir':
                continue
            min_val = {'inner': min_inner, 'outer': min_outer, 'struct': min_struct,
                      'multi': min_multi, 'moderator': min_moderator}[key]
            if abs(val - min_val) > 0.01:  # Small tolerance for float comparison
                return False
        return True
    
    inner_varying = [p for p in geometry_params if is_at_min(p, 'inner')]
    outer_varying = [p for p in geometry_params if is_at_min(p, 'outer')]
    multi_varying = [p for p in geometry_params if is_at_min(p, 'multi')]
    moderator_varying = [p for p in geometry_params if is_at_min(p, 'moderator')]
    
    # Extract data for each group
    def extract_data(param_list, param_name):
        """Extract purity and activity data for a parameter group"""
        purities_cu64 = []
        activities_cu64 = []
        purities_cu67 = []
        activities_cu67 = []
        param_values = []
        
        for params in sorted(param_list, key=lambda x: x[param_name]):
            output_dir = params['dir']
            try:
                # Get activity at 1 day
                activity_results = activity_results_list[output_dir]
                
                # Cu64 (material '1', outer)
                if '1' in activity_results and activity_results['1'] and len(activity_results['1']) > 0:
                    activities_list = activity_results['1'][0]
                    activity_cu64 = activities_list[-1].get('Cu64', 0.0)  
                else:
                    activity_cu64 = 0.0
                
                # Cu67 (material '0', inner)
                if '0' in activity_results and activity_results['0'] and len(activity_results['0']) > 0:
                    activities_list = activity_results['0'][0]
                    activity_cu67 = activities_list[-1].get('Cu67', 0.0)  
                else:
                    activity_cu67 = 0.0
                
                # Get purity at 1 day
                purity_results = purity_results_list[output_dir]
                
                # Cu64 purity (material '1', index 0)
                if '1' in purity_results and purity_results['1'] and len(purity_results['1']) > 0:
                    cu64_purity_array = purity_results['1'][0]
                    purity_cu64 = cu64_purity_array[-1]  
                else:
                    purity_cu64 = 0.0
                
                # Cu67 purity (material '0', index 1)
                if '0' in purity_results and purity_results['0'] and len(purity_results['0']) > 1:
                    cu67_purity_array = purity_results['0'][1]
                    purity_cu67 = cu67_purity_array[-1]  
                else:
                    purity_cu67 = 0.0
                
                # Convert activity to Curies
                Bq_to_Ci = 1.0 / 3.7e10
                activity_cu64_Ci = activity_cu64 * Bq_to_Ci
                activity_cu67_Ci = activity_cu67 * Bq_to_Ci
                
                purities_cu64.append(purity_cu64 * 100)  # Convert to percentage
                activities_cu64.append(activity_cu64_Ci)
                purities_cu67.append(purity_cu67 * 100)  # Convert to percentage
                activities_cu67.append(activity_cu67_Ci)
                param_values.append(params[param_name])
            except Exception as e:
                print(f"Warning: Error processing {output_dir}: {e}")
                continue
        
        return purities_cu64, activities_cu64, purities_cu67, activities_cu67, param_values
    
    # Extract data for each parameter group
    inner_p64, inner_a64, inner_p67, inner_a67, inner_vals = extract_data(inner_varying, 'inner')
    outer_p64, outer_a64, outer_p67, outer_a67, outer_vals = extract_data(outer_varying, 'outer')
    multi_p64, multi_a64, multi_p67, multi_a67, multi_vals = extract_data(multi_varying, 'multi')
    mod_p64, mod_a64, mod_p67, mod_a67, mod_vals = extract_data(moderator_varying, 'moderator')
    
    # Create 2x4 grid of plots
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # Cool colors for Cu64, warm colors for Cu67
    cool_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
    warm_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
    
    # Row 1: Inner dimensions
    ax1_cu64 = axes[0, 0]
    ax1_cu67 = axes[0, 1]
    
    if len(inner_p64) > 0:
        ax1_cu64.scatter(inner_p64, inner_a64, c=cool_colors[:len(inner_p64)], s=100, alpha=0.7, 
                        edgecolors='darkblue', linewidth=1)
        for i, (p, a, v) in enumerate(zip(inner_p64, inner_a64, inner_vals)):
            ax1_cu64.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax1_cu64.set_xlabel('Cu64 Purity [%]', fontsize=10)
        ax1_cu64.set_ylabel('Cu64 Production [Ci]', fontsize=10)
        ax1_cu64.set_title(f'Inner Thickness (outer={min_outer:.0f}, multi={min_multi:.0f}, mod={min_moderator:.0f})', 
                          fontsize=10, fontweight='bold')
        ax1_cu64.grid(True, alpha=0.3)
        if min(inner_p64) > 0:
            ax1_cu64.set_xscale('log')
        ax1_cu64.set_yscale('log')
    
    if len(inner_p67) > 0:
        ax1_cu67.scatter(inner_p67, inner_a67, c=warm_colors[:len(inner_p67)], s=100, alpha=0.7, 
                        edgecolors='darkred', linewidth=1)
        for i, (p, a, v) in enumerate(zip(inner_p67, inner_a67, inner_vals)):
            ax1_cu67.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax1_cu67.set_xlabel('Cu67 Purity [%]', fontsize=10)
        ax1_cu67.set_ylabel('Cu67 Production [Ci]', fontsize=10)
        ax1_cu67.set_title(f'Inner Thickness (outer={min_outer:.0f}, multi={min_multi:.0f}, mod={min_moderator:.0f})', 
                          fontsize=10, fontweight='bold')
        ax1_cu67.grid(True, alpha=0.3)
        if min(inner_p67) > 0:
            ax1_cu67.set_xscale('log')
        ax1_cu67.set_yscale('log')
    
    # Row 2: Outer dimensions
    ax2_cu64 = axes[1, 0]
    ax2_cu67 = axes[1, 1]
    
    if len(outer_p64) > 0:
        ax2_cu64.scatter(outer_p64, outer_a64, c=cool_colors[:len(outer_p64)], s=100, alpha=0.7, 
                        edgecolors='darkblue', linewidth=1)
        for i, (p, a, v) in enumerate(zip(outer_p64, outer_a64, outer_vals)):
            ax2_cu64.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax2_cu64.set_xlabel('Cu64 Purity [%]', fontsize=10)
        ax2_cu64.set_ylabel('Cu64 Production [Ci]', fontsize=10)
        ax2_cu64.set_title(f'Outer Thickness (inner={min_inner:.0f}, multi={min_multi:.0f}, mod={min_moderator:.0f})', 
                          fontsize=10, fontweight='bold')
        ax2_cu64.grid(True, alpha=0.3)
        if min(outer_p64) > 0:
            ax2_cu64.set_xscale('log')
        ax2_cu64.set_yscale('log')
    
    if len(outer_p67) > 0:
        ax2_cu67.scatter(outer_p67, outer_a67, c=warm_colors[:len(outer_p67)], s=100, alpha=0.7, 
                        edgecolors='darkred', linewidth=1)
        for i, (p, a, v) in enumerate(zip(outer_p67, outer_a67, outer_vals)):
            ax2_cu67.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax2_cu67.set_xlabel('Cu67 Purity [%]', fontsize=10)
        ax2_cu67.set_ylabel('Cu67 Production [Ci]', fontsize=10)
        ax2_cu67.set_title(f'Outer Thickness (inner={min_inner:.0f}, multi={min_multi:.0f}, mod={min_moderator:.0f})', 
                          fontsize=10, fontweight='bold')
        ax2_cu67.grid(True, alpha=0.3)
        if min(outer_p67) > 0:
            ax2_cu67.set_xscale('log')
        ax2_cu67.set_yscale('log')
    
    # Row 3: Multi thicknesses
    ax3_cu64 = axes[2, 0]
    ax3_cu67 = axes[2, 1]
    
    if len(multi_p64) > 0:
        ax3_cu64.scatter(multi_p64, multi_a64, c=cool_colors[:len(multi_p64)], s=100, alpha=0.7, 
                        edgecolors='darkblue', linewidth=1)
        for i, (p, a, v) in enumerate(zip(multi_p64, multi_a64, multi_vals)):
            ax3_cu64.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax3_cu64.set_xlabel('Cu64 Purity [%]', fontsize=10)
        ax3_cu64.set_ylabel('Cu64 Production [Ci]', fontsize=10)
        ax3_cu64.set_title(f'Multi Thickness (inner={min_inner:.0f}, outer={min_outer:.0f}, mod={min_moderator:.0f})', 
                          fontsize=10, fontweight='bold')
        ax3_cu64.grid(True, alpha=0.3)
        if min(multi_p64) > 0:
            ax3_cu64.set_xscale('log')
        ax3_cu64.set_yscale('log')
    
    if len(multi_p67) > 0:
        ax3_cu67.scatter(multi_p67, multi_a67, c=warm_colors[:len(multi_p67)], s=100, alpha=0.7, 
                        edgecolors='darkred', linewidth=1)
        for i, (p, a, v) in enumerate(zip(multi_p67, multi_a67, multi_vals)):
            ax3_cu67.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax3_cu67.set_xlabel('Cu67 Purity [%]', fontsize=10)
        ax3_cu67.set_ylabel('Cu67 Production [Ci]', fontsize=10)
        ax3_cu67.set_title(f'Multi Thickness (inner={min_inner:.0f}, outer={min_outer:.0f}, mod={min_moderator:.0f})', 
                          fontsize=10, fontweight='bold')
        ax3_cu67.grid(True, alpha=0.3)
        if min(multi_p67) > 0:
            ax3_cu67.set_xscale('log')
        ax3_cu67.set_yscale('log')
    
    # Row 4: Moderator thicknesses
    ax4_cu64 = axes[3, 0]
    ax4_cu67 = axes[3, 1]
    
    if len(mod_p64) > 0:
        ax4_cu64.scatter(mod_p64, mod_a64, c=cool_colors[:len(mod_p64)], s=100, alpha=0.7, 
                        edgecolors='darkblue', linewidth=1)
        for i, (p, a, v) in enumerate(zip(mod_p64, mod_a64, mod_vals)):
            ax4_cu64.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax4_cu64.set_xlabel('Cu64 Purity [%]', fontsize=10)
        ax4_cu64.set_ylabel('Cu64 Production [Ci]', fontsize=10)
        ax4_cu64.set_title(f'Moderator Thickness (inner={min_inner:.0f}, outer={min_outer:.0f}, multi={min_multi:.0f})', 
                          fontsize=10, fontweight='bold')
        ax4_cu64.grid(True, alpha=0.3)
        if min(mod_p64) > 0:
            ax4_cu64.set_xscale('log')
        ax4_cu64.set_yscale('log')
    
    if len(mod_p67) > 0:
        ax4_cu67.scatter(mod_p67, mod_a67, c=warm_colors[:len(mod_p67)], s=100, alpha=0.7, 
                        edgecolors='darkred', linewidth=1)
        for i, (p, a, v) in enumerate(zip(mod_p67, mod_a67, mod_vals)):
            ax4_cu67.annotate(f'{v:.0f}', (p, a), fontsize=7, alpha=0.8, 
                            xytext=(5, 5), textcoords='offset points')
        ax4_cu67.set_xlabel('Cu67 Purity [%]', fontsize=10)
        ax4_cu67.set_ylabel('Cu67 Production [Ci]', fontsize=10)
        ax4_cu67.set_title(f'Moderator Thickness (inner={min_inner:.0f}, outer={min_outer:.0f}, multi={min_multi:.0f})', 
                          fontsize=10, fontweight='bold')
        ax4_cu67.grid(True, alpha=0.3)
        if min(mod_p67) > 0:
            ax4_cu67.set_xscale('log')
        ax4_cu67.set_yscale('log')
    
    plt.tight_layout()
    out_dir = summary_output_dir if summary_output_dir is not None else (output_dir_list[0] if output_dir_list else ".")
    output_path = os.path.join(out_dir, "geom_production_vs_purity.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved geometry production vs purity → {output_path}")

def main():
    # ---- User configuration ----
    material_id_list = ["0", "1"]
    isotopes_to_plot = [f"Zn{i}" for i in range(64, 72)]
    cu_isotopes = [f"Cu{i}" for i in range(63, 71)]
    statepoint_batches = 10

    # Discover irr_output_* directories (must contain depletion_results.h5)
    output_dir_list = []
    for dir in os.listdir("."):
        if dir.startswith("irr_output_inner") and os.path.isfile(os.path.join(dir, "depletion_results.h5")):
            output_dir_list.append(dir)
            print(f"Found output directory: {dir}")

    if not output_dir_list:
        print("No irr_output_* directories with depletion_results.h5 found. Exiting.")
        return

    purity_results_list = {d: None for d in output_dir_list}
    activity_results_list = {d: None for d in output_dir_list}
    SA_results_list = {d: None for d in output_dir_list}

    # ---- Per–output_dir processing (writes into each irr_output_* folder) ----
    for output_dir in output_dir_list:
        openmc.config['chain_file'] = os.path.join(output_dir, "JENDL_chain.xml")
        print(f"\n{'='*70}")
        print(f"Processing: {output_dir}")
        print("="*70)

        results_path = os.path.join(output_dir, "depletion_results.h5")
        try:
            results = openmc.deplete.Results(results_path)
        except Exception as e:
            print(f"Failed to load {results_path}: {e}")
            continue

        time_days = np.array(results.get_times())
        time_steps = np.diff(time_days)
        print(f"Time: {results.get_times()}")
        print(f"Time days: {time_days}")
        print(f"Time steps: {time_steps}")

        # Irradiation / cooldown split
        large = np.where(time_steps >= min(time_steps))[0]
        if len(large) > 0:
            typical_irradiation_step = np.median(time_steps[:min(5, len(time_steps))])
            threshold = 2.0 * typical_irradiation_step
            large_steps = np.where(time_steps > threshold)[0]
            if len(large_steps) > 0:
                i = int(large_steps[0])
                irradiation_end_idx = i + 1
                print(f"Irradiation: steps 0–{irradiation_end_idx-1}; cooldown: {irradiation_end_idx}–{len(time_days)-1}")
            else:
                irradiation_end_idx = len(time_days)
        else:
            irradiation_end_idx = len(time_days)

        # Per-material: composition + Cu evolution + total activity
        for material_id in material_id_list:
            print(f"Processing material: {material_id}")
            try:
                export_final_composition(results, material_id=material_id, output_file=os.path.join(output_dir, f"final_composition_{material_id}.csv"))
            except Exception as e:
                print(f"Warning: export_final_composition(material_id={material_id}): {e}")
            try:
                plot_cu_isotopes_evolution(results, output_dir, material_id=material_id, cu_isotopes=cu_isotopes, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            except Exception as e:
                print(f"Warning: plot_cu_isotopes_evolution: {e}")
            try:
                plot_total_activity_evolution(results, output_dir, material_id=material_id, time_days=time_days)
            except Exception as e:
                print(f"Warning: plot_total_activity_evolution: {e}")

        # Flux / heating by cell (optional, needs statepoint)
        try:
            sp_path = os.path.join(output_dir, f"statepoint.{statepoint_batches}.h5")
            if os.path.isfile(sp_path):
                plot_flux_spectra_and_heating_by_cell(sp_path, output_dir, time_days=time_days)
            else:
                print(f"  Skipping flux/heating by cell (no {sp_path})")
        except Exception as e:
            print(f"Warning: plot_flux_spectra_and_heating_by_cell: {e}")

        # Cu activity evolution → total_activities, activity_results (needed for purity and geom plots)
        total_activities, activity_results = None, None
        try:
            total_activities, activity_results = plot_cu_isotopes_activity_evolution(results, output_dir, material_id_list=material_id_list, cu_isotopes=cu_isotopes, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
        except Exception as e:
            print(f"Warning: plot_cu_isotopes_activity_evolution: {e}")
        activity_results_list[output_dir] = activity_results

        # Zn plots
        try:
            plot_zn_isotopes_activity_evolution(results, output_dir, material_id_list=material_id_list, isotopes_to_plot=isotopes_to_plot, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
        except Exception as e:
            print(f"Warning: plot_zn_isotopes_activity_evolution: {e}")
        try:
            plot_zn_isotopes_SA_evolution(results, output_dir, material_id_list=material_id_list, isotopes_to_plot=isotopes_to_plot, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
        except Exception as e:
            print(f"Warning: plot_zn_isotopes_SA_evolution: {e}")

        # Cu purity (uses total_activities and activity_results from this output_dir)
        if total_activities is not None and activity_results is not None:
            try:
                purity_results, SA_results = plot_cu_isotopes_purity_evolution(results, output_dir, material_id_list=material_id_list, cu_isotopes=cu_isotopes, time_days=time_days, total_activities=total_activities, activity_results=activity_results, irradiation_end_idx=irradiation_end_idx)
                purity_results_list[output_dir] = purity_results
                SA_results_list[output_dir] = SA_results
            except Exception as e:
                print(f"Warning: plot_cu_isotopes_purity_evolution: {e}")

        print(f"Completed: {output_dir}")

    # ---- Aggregate/geom plots (use all output_dirs, write into Irr_output_results) ----
    # Summary/geom plots: one folder alongside the irr_output_* run dirs (e.g. ~/irr_output_results when run from ~)
    summary_output_dir = os.path.join(os.getcwd(), "irr_output_results")
    os.makedirs(summary_output_dir, exist_ok=True)

    if purity_results_list:
        try:
            plot_geom_purity_evolution(purity_results_list=purity_results_list, output_dir_list=output_dir_list, summary_output_dir=summary_output_dir)
        except Exception as e:
            print(f"Warning: plot_geom_purity_evolution: {e}")

    if activity_results_list:
        try:
            plot_geom_activity_evolution(activity_results_list=activity_results_list, output_dir_list=output_dir_list, summary_output_dir=summary_output_dir)
        except Exception as e:
            print(f"Warning: plot_geom_activity_evolution: {e}")

    if purity_results_list and activity_results_list:
        try:
            plot_production_vs_purity(purity_results_list=purity_results_list, activity_results_list=activity_results_list, output_dir_list=output_dir_list, summary_output_dir=summary_output_dir)
        except Exception as e:
            print(f"Warning: plot_production_vs_purity: {e}")
        try:
            plot_geom_prod_vs_purity(purity_results_list=purity_results_list, activity_results_list=activity_results_list, output_dir_list=output_dir_list, summary_output_dir=summary_output_dir)
        except Exception as e:
            print(f"Warning: plot_geom_prod_vs_purity: {e}")


if __name__ == '__main__':
    main()


    