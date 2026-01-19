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
from math import log
from scipy.constants import Avogadro
import openmc
import openmc.deplete
import h5py


# -------------------------------
# ---- Configuration ----
# -------------------------------

def analyze_depletion(output_dir, material_id="1", statepoint_batches=3):
    """
    Run complete depletion analysis for a given output directory.
    
    Parameters:
    -----------
    output_dir : str
        Path to the output directory containing depletion_results.h5
    material_id : str
        Material ID to analyze (default: "1")
    statepoint_batches : int
        Number of batches for statepoint file (default: 3)
    """
    results_path = os.path.join(output_dir, "depletion_results.h5")
    chain_file = os.path.join(output_dir, "JENDL_chain.xml")
    isotopes_to_plot = [f"Zn{i}" for i in range(64, 72)]
    cu_isotopes = [f"Cu{i}" for i in range(63, 71)]

    openmc.config['chain_file'] = chain_file

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")

    # -------------------------------
    # ---- Load depletion results ----
    # -------------------------------
    results = openmc.deplete.Results(results_path)
    
    # Get time array first (needed for all cases)
    time = np.array(results.get_times())
    print(f"DEBUG: time array: {time}")
    print(f"DEBUG: time shape: {time.shape}")
    print(f"DEBUG: Number of timesteps in file: {len(time)}")
    time_days = time / 86400 
    
    # Detect when source goes to zero (cooldown phase starts)
    try:
        # Try to read source rates from HDF5 file
        with h5py.File(results_path, 'r') as f:

            time_steps = np.diff(time)
            if len(time_steps) > 0:
                # Find where timestep size jumps (cooldown starts)
                median_step = np.median(time_steps)
                large_steps = np.where(time_steps > median_step * 2)[0]
                irradiation_end_idx = large_steps[0] + 1 if len(large_steps) > 0 else len(time)
            else:
                irradiation_end_idx = len(time)
    except Exception as e:
        print(f"Warning: Could not detect source transition: {e}")
        # Fallback: assume all timesteps are irradiation
        irradiation_end_idx = len(time)

    print(f"Irradiation phase: timesteps 0 to {irradiation_end_idx-1}")
    print(f"Cooldown phase: timesteps {irradiation_end_idx} to {len(time)-1}")

    print("="*70)
    print("OpenMC Depletion Analysis")
    print("="*70)
    print(f"Loaded results from: {results_path}")
    print(f"Number of time steps: {len(time_days)}")
    print(f"Number of irradiation time steps: {irradiation_end_idx}")
    print(f"Number of cooldown time steps: {len(time_days) - irradiation_end_idx}")
    if irradiation_end_idx < len(time_days):
        print(f"Total irradiation time: {time_days[irradiation_end_idx-1]:.1f} days")
        print(f"Total cooldown time: {time_days[-1] - time_days[irradiation_end_idx-1]:.1f} days")
    else:
        print(f"Total irradiation time: {time_days[-1]:.1f} days")
    print("="*70)

    # --- Robustly find all nuclides tracked ---
    try:
        all_nuclides = results.nuclides  # old API (<=0.14)
    except AttributeError:
        # new API: read directly from file
        with h5py.File(results_path, 'r') as f:
            all_nuclides = [n.decode() if isinstance(n, bytes) else str(n)
                            for n in np.array(f['nuclides'])]

    # Get composition AFTER IRRADIATION (end of irradiation phase)
    irradiation_end_atoms = {}
    irradiation_timestep = irradiation_end_idx - 1 if irradiation_end_idx > 0 else 0
    for nuc in all_nuclides:
        try:
            _, atoms = results.get_atoms(material_id, nuc)
            if len(atoms) > irradiation_timestep and atoms[irradiation_timestep] > 0:
                irradiation_end_atoms[nuc] = atoms[irradiation_timestep]
        except Exception:
            continue

    # Get composition AFTER COOLDOWN (final timestep)
    final_step_atoms = {}
    for nuc in all_nuclides:
        try:
            _, atoms = results.get_atoms(material_id, nuc)
            if atoms[-1] > 0:
                final_step_atoms[nuc] = atoms[-1]
        except Exception:
            continue

    if not final_step_atoms:
        raise RuntimeError(f"No nuclides found in final step for material {material_id}")

    # Export composition after irradiation
    if irradiation_end_atoms:
        total_atoms_irr = sum(irradiation_end_atoms.values())
        df_irradiation = (
            pd.DataFrame({
                "Nuclide": list(irradiation_end_atoms.keys()),
                "Number_of_Atoms": list(irradiation_end_atoms.values())
            })
            .assign(Percentage=lambda x: 100 * x["Number_of_Atoms"] / total_atoms_irr)
            .sort_values("Number_of_Atoms", ascending=False)
        )
        final_csv_irr = os.path.join(output_dir, "final_composition_after_irradiation.csv")
        df_irradiation.to_csv(final_csv_irr, index=False)
        print(f"→ Exported final composition after irradiation to {final_csv_irr}")
        print("\nTop 10 isotopes after irradiation:")
        print(df_irradiation.head(10).to_string(index=False))

    # Export composition after cooldown
    total_atoms = sum(final_step_atoms.values())
    df_final = (
        pd.DataFrame({
            "Nuclide": list(final_step_atoms.keys()),
            "Number_of_Atoms": list(final_step_atoms.values())
        })
        .assign(Percentage=lambda x: 100 * x["Number_of_Atoms"] / total_atoms)
        .sort_values("Number_of_Atoms", ascending=False)
    )
    final_csv = os.path.join(output_dir, "final_composition_after_cooldown.csv")
    df_final.to_csv(final_csv, index=False)
    print(f"\n→ Exported final composition after cooldown to {final_csv}")
    print("\nTop 10 isotopes after cooldown:")
    print(df_final.head(10).to_string(index=False))

    return results, results_path, isotopes_to_plot, cu_isotopes, irradiation_end_idx, time_days, final_step_atoms, total_atoms, df_final
    
    

def plot_zn_isotopes_evolution(results, output_dir, material_id="1", isotopes_to_plot=None, time_days=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Zn isotopes evolution ----
    # -------------------------------
    plt.figure(figsize=(8,6))
    for iso in isotopes_to_plot:
        try:
            _, atoms = results.get_atoms(material_id, iso)
            plt.semilogy(time_days, atoms, label=iso)
        except KeyError:
            continue

    # Add vertical line to mark end of irradiation / start of cooldown
    if irradiation_end_idx is not None and irradiation_end_idx < len(time_days):
        plt.axvline(x=time_days[irradiation_end_idx-1], color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label='End of Irradiation')
        # Add text annotation
        plt.text(time_days[irradiation_end_idx-1], plt.ylim()[1]*0.1, 
                'Cooldown\nstarts', ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel("Time [days]")  # Updated to include both phases
    plt.ylabel("Number of atoms")
    plt.title("Evolution of Zinc Isotopes (64–72)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    zn_path = os.path.join(output_dir, "zn_isotope_evolution.png")
    plt.savefig(zn_path, dpi=300)
    plt.close()
    print(f"→ Saved Zn isotope evolution plot → {zn_path}")


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
        plt.text(time_days[irradiation_end_idx-1], plt.ylim()[1]*0.1, 
                'Cooldown\nstarts', ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel("Time [days]")  # Updated to include both phases
    plt.ylabel("Number of atoms")
    plt.ylim(1e-7, 1e30)    
    plt.title("Evolution of Copper Isotopes (63–71)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    cu_path = os.path.join(output_dir, "cu_isotope_evolution.png")
    plt.savefig(cu_path, dpi=300)
    plt.close()
    print(f"→ Saved Cu isotope evolution plot → {cu_path}")

    return final_atoms_dict  # Returns final atoms after cooldown

def plot_cu_isotopes_activity_evolution(results, output_dir, material_id="1", cu_isotopes=None, time_days=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Cu isotopes activity evolution (Irradiation Phase Only) ----
    # -------------------------------
    plt.figure(figsize=(8,6))
    _, activities = results.get_activity(material_id, units='Bq', by_nuclide=True)
    
    # Only plot irradiation phase (before cooldown)
    if irradiation_end_idx is not None and irradiation_end_idx < len(activities):
        activities = activities[:irradiation_end_idx]
        time_plot = time_days[:irradiation_end_idx]
    else:
        time_plot = time_days
    
    total_activities = np.zeros(len(activities))
    for iso in cu_isotopes:
        activity = []
        for act_dict in activities:
            if iso in act_dict:
                activity.append(act_dict[iso])
                idx = activities.index(act_dict)
                total_activities[idx] += act_dict[iso]
            else:
                activity.append(0.0)
        
        activity = np.array(activity)
        
        plt.semilogy(time_plot, activity, label=iso)

    plt.xlabel("Irradiation time [days]")
    plt.ylabel("Activity [Bq]")
    plt.ylim(1e-7, 1e30)
    plt.title("Activity Evolution of Copper Isotopes (63–71) - Irradiation Phase")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    cu_activity_path = os.path.join(output_dir, "cu_isotope_activity_evolution.png")
    plt.savefig(cu_activity_path, dpi=300)
    plt.close()
    print(f"→ Saved Cu isotope activity evolution plot → {cu_activity_path}")

    return total_activities

def plot_cu_isotopes_purity_evolution(results, output_dir, material_id="1", cu_isotopes=None, time_days=None, total_activities=None, irradiation_end_idx=None):
    # -------------------------------
    # ---- Plot Cu isotopic purity (Irradiation Phase Only) ----
    # -------------------------------
    _, activities = results.get_activity(material_id, units='Bq', by_nuclide=True)
    
    # Only plot irradiation phase (before cooldown)
    if irradiation_end_idx is not None and irradiation_end_idx < len(activities):
        activities = activities[:irradiation_end_idx]
        total_activities = total_activities[:irradiation_end_idx]
        time_plot = time_days[:irradiation_end_idx]
    else:
        time_plot = time_days
    
    plt.figure(figsize=(8,6))
    for iso in cu_isotopes:
        purity = []
        for pure_dict in activities:
            if iso in pure_dict:
                idx = activities.index(pure_dict)
                # Purity = activity of isotope / total activity of all isotopes
                if total_activities[idx] > 0:
                    purity.append(pure_dict[iso] / total_activities[idx])
                else:
                    purity.append(0.0)
            else:
                purity.append(0.0)
        
        purity = np.array(purity)
        purity = 1 - purity
        
        plt.semilogy(time_plot, purity, label=iso)

    plt.xlabel("Irradiation time [days]")
    plt.ylabel("1 - Purity (Impurity fraction, log scale)")
    plt.title("Purity Evolution of Copper Isotopes (63–71) - Irradiation Phase")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    cu_purity_path = os.path.join(output_dir, "cu_isotope_purity_evolution.png")
    plt.savefig(cu_purity_path, dpi=300)
    plt.close()
    print(f"→ Saved Cu isotope purity evolution plot → {cu_purity_path}")



''''def plot_cu_isotopes_post_irradiation_activity_evolution(results, output_dir, material_id="1", cu_isotopes=None, time_days=None, total_activities=None, final_atoms_dict=None):""
    # -------------------------------
    # ---- Plot Cu isotopes post-irradiation activity evolution ----
    # -------------------------------
    post_time_hours = np.arange(0, 12 * 24 + 1, 1)  # 0 to 288 hours, 1-hour steps
    post_time_seconds = post_time_hours * 3600
    half_lives_seconds = {}

    for iso in cu_isotopes:
        if iso not in final_atoms_dict or final_atoms_dict[iso] <= 0:
            continue
        try:
            half_life = openmc.data.half_life(iso)
            if half_life is None:
                half_lives_seconds[iso] = np.inf  # Stable
            else:
                half_lives_seconds[iso] = half_life  # Already in seconds
        except:
            half_lives_seconds[iso] = np.inf  # Default to stable if not found

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    activity_total = np.zeros(len(post_time_seconds))
    cu64_activity = np.zeros(len(post_time_seconds))
    cu67_activity = np.zeros(len(post_time_seconds))

    # Plot 1: Atoms decay and accumulate activities
    for iso in cu_isotopes:
        if iso not in final_atoms_dict or final_atoms_dict[iso] <= 0:
            continue
        if iso not in half_lives_seconds:
            continue
        
        half_life = half_lives_seconds[iso]
        if half_life == np.inf:
            continue  # Skip stable isotopes
        
        N0 = final_atoms_dict[iso]  # Initial number of atoms
        decay_constant = np.log(2) / half_life
        
        # Number of atoms at time t: N(t) = N0 * exp(-λ*t)
        atoms_post = N0 * np.exp(-decay_constant * post_time_seconds)
        activity_post = decay_constant * atoms_post
        
        # Accumulate activities (add arrays directly)
        activity_total += activity_post
        if iso == 'Cu64':
            cu64_activity = activity_post  # Cu64 activity (not +=, just assign)
        if iso == 'Cu67':
            cu67_activity = activity_post  # Cu67 activity (not +=, just assign)
        ax1.semilogy(post_time_hours, atoms_post, label=iso)
        ax2.semilogy(post_time_hours, activity_post, label=iso)

    ax1.set_xlabel("Time after irradiation [hours]")
    ax1.set_ylim(10e-5, 10e20)
    ax1.set_ylabel("Number of atoms")
    ax1.set_title("Post-Irradiation Decay of Copper Isotope Atoms (12 days)")
    ax1.legend(ncol=2, fontsize=9)
    ax1.grid(True, which="both", ls=":")

    ax2.set_xlabel("Time after irradiation [hours]")
    ax2.set_ylim(10e-5, 10e20)
    ax2.set_ylabel("Activity [Bq]")
    ax2.set_title("Post-Irradiation Activity Decay of Copper Isotopes (12 days)")
    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True, which="both", ls=":")

    # Plot 3: Cu64 Purity
    cu64_purity = (cu64_activity / activity_total) * 100
    cu67_purity = (cu67_activity / activity_total) * 100

    ax3.plot(post_time_hours, cu64_purity, label='Cu64', linewidth=2, color='red')
    ax3.set_xlabel("Time after irradiation [hours]")
    ax3.set_ylabel("Cu64 Purity [%]")
    ax3.set_title("Cu64 Isotopic Purity Over Time")
    ax3.grid(True, which="both", ls=":")
    ax3.legend(fontsize=10)

    # Create table data at 12-hour intervals
    table_times_hours = np.arange(0, 120, 12)  # Every 12 hours
    table_data = []
    for t_hours in table_times_hours:
        # Find closest index
        idx = np.argmin(np.abs(post_time_hours - t_hours))
        purity_val = cu64_purity[idx]
        table_data.append([f"{t_hours:.0f}", f"{purity_val:.4f}"])

    # Add table to the plot
    table = ax3.table(cellText=table_data,
                    colLabels=['Time [hours]', 'Cu64 Purity [%]'],
                    cellLoc='center',
                    loc='center right',
                    bbox=[0.7, 0.02, 0.28, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.7, 1)

    plt.tight_layout()
    post_activity_path = os.path.join(output_dir, "cu_isotope_post_irradiation_decay.png")
    plt.savefig(post_activity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved Cu isotope post-irradiation decay plot → {post_activity_path}")

    return cu64_purity, cu67_purity, activity_total, post_time_seconds, cu64_activity, cu67_activity '''''


def plot_total_activity_evolution(results, output_dir, material_id="1", time_days=None):
    # -------------------------------
    # ---- Plot total activity ---
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

''''def plot_cu_isotopes_post_irradiation_activity_evolution(results, output_dir, material_id="1", cu_isotopes=None, time_days=None, total_activities=None,):
    # -------------------------------
    # ---- Plot Cu isotopes post-irradiation activity evolution (from depletion cooldown phase) ----
    # -------------------------------
    
    # Get all activities over entire time range
    _, all_activities = results.get_activity(material_id, units='Bq', by_nuclide=True)
    
    # Calculate Cu64 and Cu67 purity over all timesteps to find peak
    cu64_activity_all = np.zeros(len(all_activities))
    cu67_activity_all = np.zeros(len(all_activities))
    activity_total_all = np.zeros(len(all_activities))
    
    for idx, act_dict in enumerate(all_activities):
        for iso in cu_isotopes:
            if iso in act_dict:
                activity_total_all[idx] += act_dict[iso]
                if iso == 'Cu64':
                    cu64_activity_all[idx] = act_dict[iso]
                if iso == 'Cu67':
                    cu67_activity_all[idx] = act_dict[iso]
    
    # Calculate purity over all timesteps
    cu64_purity_all = np.zeros(len(all_activities))
    cu67_purity_all = np.zeros(len(all_activities))
    for idx in range(len(all_activities)):
        if activity_total_all[idx] > 0:
            cu64_purity_all[idx] = cu64_activity_all[idx] / activity_total_all[idx]
            cu67_purity_all[idx] = cu67_activity_all[idx] / activity_total_all[idx]
    
    # Find where Cu64 purity peaks (transitions from increasing to decreasing)
    # Use the maximum purity point as cooldown start
    cu64_peak_idx = np.argmax(cu64_purity_all)
    
    # Also check Cu67 purity peak
    cu67_peak_idx = np.argmax(cu67_purity_all)
    
    # Use the later of the two peaks (or Cu64 if that's the focus)
    cooldown_start_idx = max(cu64_peak_idx, cu67_peak_idx)
    
    print(f"Cu64 purity peak at timestep {cu64_peak_idx} (time: {time_days[cu64_peak_idx]:.2f} days)")
    print(f"Cu67 purity peak at timestep {cu67_peak_idx} (time: {time_days[cu67_peak_idx]:.2f} days)")
    print(f"Using cooldown start at timestep {cooldown_start_idx} (time: {time_days[cooldown_start_idx]:.2f} days)")
    
    # Get cooldown phase data (from peak onwards)
    cooldown_time_days = time_days[cooldown_start_idx:]
    cooldown_time_hours = (cooldown_time_days - time_days[cooldown_start_idx]) * 24  # Time since peak
    
    # Get activities during cooldown phase
    cooldown_activities = all_activities[cooldown_start_idx:]
    
    # Calculate total activity and individual isotope activities during cooldown
    activity_total = np.zeros(len(cooldown_activities))
    cu64_activity = np.zeros(len(cooldown_activities))
    cu67_activity = np.zeros(len(cooldown_activities))
    isotope_activities = {iso: np.zeros(len(cooldown_activities)) for iso in cu_isotopes}
    isotope_atoms = {iso: [] for iso in cu_isotopes}
    
    # Extract activities and atoms for each isotope
    for iso in cu_isotopes:
        try:
            _, atoms = results.get_atoms(material_id, iso)
            cooldown_atoms = atoms[cooldown_start_idx:]
            isotope_atoms[iso] = cooldown_atoms
            
            for idx, act_dict in enumerate(cooldown_activities):
                if iso in act_dict:
                    activity_val = act_dict[iso]
                    isotope_activities[iso][idx] = activity_val
                    activity_total[idx] += activity_val
                    if iso == 'Cu64':
                        cu64_activity[idx] = activity_val
                    if iso == 'Cu67':
                        cu67_activity[idx] = activity_val
        except KeyError:
            continue
    
    # Create figure with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 14))
    
    # Plot 1: Activity evolution during cooldown
    for iso in cu_isotopes:
        if np.any(isotope_activities[iso] > 0):
            ax1.semilogy(cooldown_time_hours, isotope_activities[iso], label=iso)
    
    ax1.set_xlabel("Time after purity peak [hours]")
    ax1.set_ylabel("Activity [Bq]")
    ax1.set_title("Activity Evolution of Copper Isotopes After Purity Peak")
    ax1.legend(ncol=2, fontsize=9)
    ax1.grid(True, which="both", ls=":")
    
    # Plot 2: Atoms evolution during cooldown
    for iso in cu_isotopes:
        if len(isotope_atoms[iso]) > 0 and np.any(isotope_atoms[iso] > 0):
            ax2.semilogy(cooldown_time_hours, isotope_atoms[iso], label=iso)
    
    ax2.set_xlabel("Time after purity peak [hours]")
    ax2.set_ylabel("Number of atoms")
    ax2.set_title("Atoms Evolution of Copper Isotopes After Purity Peak")
    ax2.legend(ncol=2, fontsize=9)
    ax2.grid(True, which="both", ls=":")
    
    # Plot 3: Cu64 Purity (plot 1-purity on log scale)
    cu64_purity = np.zeros(len(cooldown_time_hours))
    for idx in range(len(cooldown_time_hours)):
        if activity_total[idx] > 0:
            cu64_purity[idx] = (cu64_activity[idx] / activity_total[idx])  # Fraction (0-1)
    
    # Plot (1-purity) on log scale to show impurities
    one_minus_purity_cu64 = 1.0 - cu64_purity
    ax3.semilogy(cooldown_time_hours, one_minus_purity_cu64, label='Cu64', linewidth=2, color='red')
    ax3.set_xlabel("Time after purity peak [hours]")
    ax3.set_ylabel("1 - Cu64 Purity (impurity fraction)")
    ax3.set_title("Cu64 Impurity After Purity Peak (1 - Purity, log scale)")
    ax3.grid(True, which="both", ls=":")
    ax3.legend(fontsize=10)
    
    # Plot 4: Cu67 Purity (plot 1-purity on log scale)
    cu67_purity = np.zeros(len(cooldown_time_hours))
    for idx in range(len(cooldown_time_hours)):
        if activity_total[idx] > 0:
            cu67_purity[idx] = (cu67_activity[idx] / activity_total[idx])  # Fraction (0-1)
    
    # Plot (1-purity) on log scale to show impurities
    one_minus_purity_cu67 = 1.0 - cu67_purity
    ax4.semilogy(cooldown_time_hours, one_minus_purity_cu67, label='Cu67', linewidth=2, color='blue')
    ax4.set_xlabel("Time after purity peak [hours]")
    ax4.set_ylabel("1 - Cu67 Purity (impurity fraction)")
    ax4.set_title("Cu67 Impurity After Purity Peak (1 - Purity, log scale)")
    ax4.grid(True, which="both", ls=":")
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    post_activity_path = os.path.join(output_dir, "cu_isotope_cooldown_decay.png")
    plt.savefig(post_activity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved Cu isotope cooldown decay plot → {post_activity_path}")
    
    return cu64_purity, cu67_purity, activity_total, cooldown_time_hours, cu64_activity, cu67_activity'''


def plot_most_common_reactions(results, output_dir, material_id="1", results_path=None):
    # -------------------------------
    # ---- Plot most common reactions that produce Cu isotopes ----
    # -------------------------------
    try:
        with h5py.File(results_path, "r") as f:
            if "reactions" in f and "reaction rates" in f:
                # --- Load names and rate array ---
                reaction_names = [n.decode() if isinstance(n, bytes) else str(n)
                                for n in np.array(f["reactions"])]
                rate_array = np.array(f["reaction rates"])  # shape: (time, mat, rxn, nuc)
                mat_keys = list(f["materials"].keys())
                mat_index = mat_keys.index(material_id) if material_id in mat_keys else 0

                # --- Get nuclide list to find Cu isotopes ---
                if "nuclides" in f:
                    nuclides = [n.decode() if isinstance(n, bytes) else str(n)
                               for n in np.array(f["nuclides"])]
                else:
                    # Fallback: try to get from results object
                    try:
                        nuclides = results.nuclides
                    except AttributeError:
                        nuclides = []
                
                # --- Find indices of Cu isotopes (Cu63-Cu71) ---
                cu_isotopes = [f"Cu{i}" for i in range(63, 72)]
                cu_indices = [i for i, nuc in enumerate(nuclides) if nuc in cu_isotopes]
                
                if not cu_indices:
                    print("⚠️ No Cu isotopes found in nuclide list")
                    return
                
                print(f"Found {len(cu_indices)} Cu isotopes: {[nuclides[i] for i in cu_indices]}")

                # --- Integrate over time and Cu nuclides only for each reaction ---
                # Sum over time (axis 0) and Cu nuclides (axis 3, only Cu indices)
                # rate_array shape: (time, mat, rxn, nuc)
                selected = rate_array[:, mat_index, :, cu_indices]  # shape: (time, rxn, n_cu)
                # Sum over time dimension first, then over Cu isotopes
                cu_production_rates = np.sum(selected, axis=0)  # Sum over time: (rxn, n_cu)
                cu_production_rates = np.sum(cu_production_rates, axis=1)  # Sum over Cu isotopes: (rxn,)
                # --- Sanity check ---
                if len(cu_production_rates) != len(reaction_names):
                    print(f"⚠️ Reaction count mismatch: {len(reaction_names)} names vs {len(cu_production_rates)} rates")
                    min_len = min(len(reaction_names), len(cu_production_rates))
                    reaction_names = reaction_names[:min_len]
                    cu_production_rates = cu_production_rates[:min_len]

                # --- Build DataFrame ---
                df_rxn = pd.DataFrame({
                    "Reaction": reaction_names,
                    "Cu_Production_Rate": cu_production_rates
                }).sort_values("Cu_Production_Rate", ascending=False)

                # Filter out reactions with zero Cu production
                df_rxn = df_rxn[df_rxn["Cu_Production_Rate"] > 0]

                csv_path = os.path.join(output_dir, "reaction_summary_cu_production.csv")
                df_rxn.to_csv(csv_path, index=False)
                print(f"→ Exported Cu-producing reaction summary to {csv_path}")

                # --- Plot top reactions that produce Cu ---
                plt.figure(figsize=(8,5))
                top10 = df_rxn.head(10)
                if len(top10) > 0:
                    plt.barh(top10["Reaction"][::-1],
                            top10["Cu_Production_Rate"][::-1],
                            color="steelblue")
                    plt.xlabel("Integrated Cu Production Rate")
                    plt.title("Top 10 Reaction Channels Producing Cu Isotopes")
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "top_reactions_cu_production.png"), dpi=300)
                    plt.close()
                    print("→ Saved top Cu-producing reaction plot")
                else:
                    print("⚠️ No reactions found that produce Cu isotopes")

            else:
                print("Reaction data not found in depletion_results.h5")

    except Exception as e:
        print(f"Could not extract reaction summary: {e}")
        import traceback
        traceback.print_exc()


### also plot top reactions by nuclide
# -------------------------------
# ---- Breakdown of top 10 reactions by isotope ----
# -------------------------------
from openmc.deplete import Chain
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def build_nuclide_reaction_labels(chain_path, n_rxn, n_nuc):
    """
    Construct (nuclide, reaction) labels to match HDF5 array dimensions.
    Returns list of length n_rxn: ["Gd155 (n,gamma)", "Gd156 (n,2n)", ...]
    """
    chain = Chain.from_xml(chain_path)
    all_pairs = []
    for nuc in chain.nuclides:
        for rxn in nuc.reactions:
            all_pairs.append(f"{nuc.name} ({rxn.type})")

    if len(all_pairs) >= n_rxn:
        return all_pairs[:n_rxn]
    else:
        # repeat or pad if fewer available
        from itertools import cycle
        reps = cycle(all_pairs)
        return [next(reps) for _ in range(n_rxn)]


def plot_most_common_reactions_isotope_breakdown(results, output_dir, material_id="1", results_path=None):
    try:
        with h5py.File(results_path, "r") as f:
            if all(k in f for k in ["reactions", "reaction rates", "nuclides"]):
                # ---- Load nuclide names ----
                raw_nuc = np.array(f["nuclides"])
                nuclide_names = [n.decode() if isinstance(n, bytes) else str(n)
                                for n in raw_nuc.flatten().tolist()]

                # ---- Find Cu isotope indices ----
                cu_isotopes = [f"Cu{i}" for i in range(63, 72)]
                cu_indices = [i for i, nuc in enumerate(nuclide_names) if nuc in cu_isotopes]
                
                if not cu_indices:
                    print("⚠️ No Cu isotopes found in nuclide list")
                    return
                
                print(f"Found {len(cu_indices)} Cu isotopes: {[nuclide_names[i] for i in cu_indices]}")

                # ---- Load and squeeze reaction rates ----
                rate_array = np.squeeze(np.array(f["reaction rates"]))
                print(f"Reaction rate array shape: {rate_array.shape}")
                ndim = rate_array.ndim

                # ---- Integrate over time ----
                if ndim == 4:
                    # (time, mat, reaction, nuclide)
                    mat_keys = list(f["materials"].keys())
                    mat_index = mat_keys.index(material_id) if material_id in mat_keys else 0
                    # Select material and sum over time dimension
                    selected = rate_array[:, mat_index, :, :]  # shape: (time, rxn, nuc)
                    integrated_rxn_nuc = np.sum(selected, axis=0)  # Sum over time: (rxn, nuc)
                elif ndim == 3:
                    # (time, reaction, nuclide)
                    integrated_rxn_nuc = np.sum(rate_array, axis=0)  # Sum over time: (rxn, nuc)
                else:
                    raise ValueError(f"Unexpected shape for reaction rates: {rate_array.shape}")

                n_rxn, n_nuc = integrated_rxn_nuc.shape
                print(f"Detected {n_rxn} reactions × {n_nuc} nuclides")

                # ---- Filter Cu indices to valid range ----
                # cu_indices are indices into nuclide_names, but we need indices < n_nuc
                cu_indices_valid = [idx for idx in cu_indices if idx < n_nuc]
                if len(cu_indices_valid) == 0:
                    print("⚠️ No valid Cu isotope indices found in rate array")
                    return

                print(f"Using {len(cu_indices_valid)} valid Cu isotope indices (out of {len(cu_indices)} found)")

                # ---- Filter to only reactions that produce Cu isotopes ----
                # Sum over Cu isotope indices for each reaction
                cu_production_per_rxn = np.sum(integrated_rxn_nuc[:, cu_indices_valid], axis=1)
                # Only keep reactions that produce Cu
                cu_producing_mask = cu_production_per_rxn > 0
                cu_producing_indices = np.where(cu_producing_mask)[0]
                
                if len(cu_producing_indices) == 0:
                    print("⚠️ No reactions found that produce Cu isotopes")
                    return
                
                print(f"Found {len(cu_producing_indices)} reactions that produce Cu isotopes")
                
                # Filter integrated rates to only Cu-producing reactions
                integrated_rxn_nuc_cu = integrated_rxn_nuc[cu_producing_indices, :]
                cu_production_per_rxn_filtered = cu_production_per_rxn[cu_producing_indices]

                # ---- Fix name mismatches ----
                if len(nuclide_names) != n_nuc:
                    print(f"⚠️ Nuclide name count ({len(nuclide_names)}) != {n_nuc}. Using generic labels.")
                    nuclide_names = [f"Nuclide_{i}" for i in range(n_nuc)]

                # ---- Build nuclide–reaction labels ----
                chain_path = os.path.join(output_dir, "JENDL_chain.xml")
                all_reaction_names = build_nuclide_reaction_labels(chain_path, n_rxn, n_nuc)
                reaction_names = [all_reaction_names[i] for i in cu_producing_indices]
                print(f"→ Built {len(reaction_names)} (nuclide, reaction) labels for Cu-producing reactions.")

                # ---- Identify top reactions that produce Cu ----
                top_rxn_idx = np.argsort(cu_production_per_rxn_filtered)[-10:][::-1]
                top_reactions = [reaction_names[i] for i in top_rxn_idx]

                # ---- Build tidy dataframe with Cu isotopes only ----
                records = []
                for r_idx in top_rxn_idx:
                    reaction = reaction_names[r_idx]
                    # Only include Cu isotopes in the breakdown
                    for cu_idx in cu_indices:
                        val = integrated_rxn_nuc_cu[r_idx, cu_idx]
                        if val > 0:
                            records.append({
                                "Reaction": reaction,
                                "Product_Nuclide": nuclide_names[cu_idx],
                                "Cu_Production_Rate": float(val)
                            })

                df_breakdown = pd.DataFrame(records)

                # ---- Group and sort ----
                df_breakdown = (
                    df_breakdown.sort_values(["Reaction", "Cu_Production_Rate"],
                                            ascending=[True, False])
                    .groupby("Reaction")
                    .head(5)
                    .reset_index(drop=True)
                )

                csv_path = os.path.join(output_dir, "reaction_isotope_breakdown_cu_production.csv")
                df_breakdown.to_csv(csv_path, index=False)
                print(f"→ Exported Cu-producing reaction–product breakdown to {csv_path}")

                # ---- Color palette by reaction type ----
                reaction_types = [r.split("(")[-1].split(")")[0] for r in df_breakdown["Reaction"]]
                unique_types = sorted(set(reaction_types))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
                color_map = dict(zip(unique_types, colors))

                # ---- Plot grouped horizontal bars ----
                plt.figure(figsize=(9,6))
                for i, reaction in enumerate(top_reactions):
                    subset = df_breakdown[df_breakdown["Reaction"] == reaction]
                    if len(subset) == 0:
                        continue
                    rates = subset["Cu_Production_Rate"].values
                    products = subset["Product_Nuclide"].values
                    rx_type = reaction.split("(")[-1].split(")")[0]
                    color = color_map.get(rx_type, "gray")
                    left = np.cumsum(np.concatenate(([0], rates[:-1])))
                    plt.barh([reaction]*len(rates), rates, left=left, color=color, edgecolor="black", alpha=0.8)
                    if len(products) > 0:
                        plt.text(np.sum(rates)*1.02, i, products[0], va='center', fontsize=8)

                # ---- Final touches ----
                plt.xlabel("Integrated Cu Production Rate")
                plt.ylabel("Reaction (nuclide + channel)")
                plt.title("Top 10 Cu-Producing Reactions – Cu Isotope Breakdown (Top 5 per Reaction)")
                plt.grid(True, axis="x", ls=":", alpha=0.5)
                plt.legend(handles=[plt.Line2D([0], [0], color=c, lw=6, label=t) for t, c in color_map.items()],
                        title="Reaction Type", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "reaction_isotope_breakdown_cu_production.png"), dpi=300)
                plt.close()
                print("→ Saved Cu-producing reaction–product breakdown plot")

            else:
                print("⚠️ Reaction/nuclide data not found in depletion_results.h5")

    except Exception as e:
        print(f"Could not compute reaction–isotope breakdown: {e}")
        import traceback
        traceback.print_exc()

def analyze_depletion_zn_enrichment(successful_dirs):
    """
    Plot Cu64 and Cu67 purity and activity evolution across different Zn-64 enrichment cases.
    
    Parameters:
    -----------
    output_dirs : list
        List of output directory paths (e.g., ['irradiation_output_48', 'irradiation_output_50', ...])
    zn64_enrichment_list : list
        List of Zn-64 enrichment fractions (e.g., [0.486, 0.50, 0.60, ...])
    """
    print("="*70)
    print("Analyzing Cu64/Cu67 purity and activity across Zn-64 enrichments")
    print("="*70)
    
    # Dictionary to store data keyed by enrichment percentage
    data_dict = {}
    
    # Process each output directory
    for output_dir in successful_dirs:
        # Extract enrichment percentage from directory name
        try:
            enrichment_pct = int(output_dir.split('_')[-1])
            enrichment = enrichment_pct / 100.0
        except (ValueError, IndexError):
            print(f"Warning: Could not extract enrichment from {output_dir}, skipping...")
            continue
        
        print(f"\nProcessing: {output_dir} (Zn-64 enrichment: {enrichment_pct}%)")
        
        # Check if results exist
        results_path = os.path.join(output_dir, "depletion_results.h5")
        if not os.path.exists(results_path):
            print(f"Warning: {results_path} not found, skipping {output_dir}")
            continue
        
        try:
            # Load depletion results
            results, results_path, isotopes_to_plot, cu_isotopes, irradiation_end_idx, time_days, final_step_atoms, total_atoms, df_final = analyze_depletion(output_dir=output_dir)
            total_activities = plot_cu_isotopes_activity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days)
            final_atoms_dict = plot_cu_isotopes_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days)
            cu64_purity, cu67_purity, activity_total, post_time_seconds, cu64_activity, cu67_activity = plot_cu_isotopes_post_irradiation_activity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, total_activities=total_activities, irradiation_end_idx=irradiation_end_idx)
            post_time_hours = post_time_seconds / 3600
            # Store data with enrichment percentage as key
            data_dict[enrichment_pct] = {
                'cu64_purity': cu64_purity,
                'cu67_purity': cu67_purity,
                'cu64_activity': cu64_activity,
                'cu67_activity': cu67_activity,
                'post_time_hours': post_time_hours,
                'enrichment': enrichment
            }
            
            print(f"Successfully loaded data for {enrichment_pct}% Zn-64 enrichment")
            
        except Exception as e:
            print(f"Error processing {output_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not data_dict:
        print("\nNo data collected. Check that output directories exist and contain depletion_results.h5")
        return
    
    print(f"\nCollected data for {len(data_dict)} enrichment cases")
    print(f"  Enrichment levels: {sorted(data_dict.keys())}%")
    
    # Sort enrichment percentages for consistent color mapping
    sorted_enrichments = sorted(data_dict.keys())
    n_enrichments = len(sorted_enrichments)

    cu64_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_enrichments))
    cu67_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_enrichments))
    cu64_activity_colors = plt.cm.Purples(np.linspace(0.4, 0.9, n_enrichments))
    cu67_activity_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, n_enrichments))
    
    # Create figure with two subplots: purity and activity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Purity evolution
    print("\nPlotting purity evolution...")
    for i, enrichment_pct in enumerate(sorted_enrichments):
        data = data_dict[enrichment_pct]
        
        ax1.plot(data['post_time_hours'], data['cu64_purity'], 
                label=f'Cu64 (Zn-64: {enrichment_pct}%)', 
                linewidth=2.5, color=cu64_colors[i], linestyle='-')
        ax1.plot(data['post_time_hours'], data['cu67_purity'], 
                label=f'Cu67 (Zn-64: {enrichment_pct}%)', 
                linewidth=2.5, color=cu67_colors[i], linestyle='--', alpha=0.8)
    
    ax1.set_xlabel("Time after irradiation [hours]", fontsize=11)
    ax1.set_ylabel("Isotopic Purity [%]", fontsize=11)
    ax1.set_title("Cu64 and Cu67 Purity Evolution vs. Zn-64 Enrichment Level", fontsize=12, fontweight='bold')
    ax1.legend(ncol=2, fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.set_xlim(0, 120)  # Show first 120 hours
    
    # Plot 2: Activity evolution
    print("Plotting activity evolution...")
    for i, enrichment_pct in enumerate(sorted_enrichments):
        data = data_dict[enrichment_pct]
        
        # Plot Cu64 activity (purples)
        ax2.semilogy(data['post_time_hours'], data['cu64_activity'], 
                    label=f'Cu64 (Zn-64: {enrichment_pct}%)', 
                    linewidth=2.5, color=cu64_activity_colors[i], linestyle='-')
        
        # Plot Cu67 activity (oranges)
        ax2.semilogy(data['post_time_hours'], data['cu67_activity'], 
                    label=f'Cu67 (Zn-64: {enrichment_pct}%)', 
                    linewidth=2.5, color=cu67_activity_colors[i], linestyle='--', alpha=0.8)
    
    ax2.set_xlabel("Time after irradiation [hours]", fontsize=11)
    ax2.set_ylabel("Activity [Bq]", fontsize=11)
    ax2.set_title("Cu64 and Cu67 Activity Evolution vs. Zn-64 Enrichment Level", fontsize=12, fontweight='bold')
    ax2.legend(ncol=2, fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.set_xlim(0, 120)  # Show first 120 hours
    
    plt.tight_layout()
    
    # Save plot
    output_path = "cu64_cu67_enrichment_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot → {output_path}")
    print("="*70)
    
    # =====================================================================
    # Export data to Excel for comparison
    # =====================================================================
    print("\nExporting data to CSV...")
    
    # Create DataFrame with time and all data columns
    df_data = {'Time [hours]': data_dict[sorted_enrichments[0]]['post_time_hours']}
    
    # Add all columns for each enrichment
    for enrichment_pct in sorted_enrichments:
        data = data_dict[enrichment_pct]
        df_data[f'Cu64_Purity_Zn{enrichment_pct}%'] = data['cu64_purity']
        df_data[f'Cu67_Purity_Zn{enrichment_pct}%'] = data['cu67_purity']
        df_data[f'Cu64_Activity_Zn{enrichment_pct}%'] = data['cu64_activity']
        df_data[f'Cu67_Activity_Zn{enrichment_pct}%'] = data['cu67_activity']
    
    # Export to CSV
    df = pd.DataFrame(df_data)
    csv_path = "cu64_cu67_enrichment_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"→ Exported comparison data to {csv_path}")
    print("="*70)




def main():
    # Automatically discover available enrichment cases from output directories
    print("="*70)
    print("Discovering available enrichment cases from output directories...")
    print("="*70)
    
    # Find all irradiation_output_* directories
    output_dir_pattern = "irradiation_output_*"
    found_dirs = sorted([d for d in glob.glob(output_dir_pattern) if os.path.isdir(d)])
    
    if not found_dirs:
        print("No irradiation_output_* directories found!")
        print("   Make sure you've run fusion_irradiation.py first.")
        return
    
    # Extract enrichment percentages and create enrichment list
    zn64_enrichment_list = []
    zn67_enrichment_list = []
    output_dirs = []
    
    for dir_path in found_dirs:
        try:
            # Extract enrichment percentages from directory name
            # New format: "irradiation_output_48_26" -> Zn64=48%, Zn67=26%
            # Old format: "irradiation_output_50" -> Zn64=50%, Zn67=natural (0.2663)
            parts = dir_path.split('_')
            
            if len(parts) >= 4:
                # New format with both enrichments
                zn64_pct = int(parts[-2])
                zn67_pct = int(parts[-1])
                zn64_enrichment = zn64_pct / 100.0
                zn67_enrichment = zn67_pct / 100.0
                zn64_enrichment_list.append(zn64_enrichment)
                zn67_enrichment_list.append(zn67_enrichment)
                output_dirs.append(dir_path)
                print(f"  Found: {dir_path} (Zn-64: {zn64_pct}%, Zn-67: {zn67_pct}%)")
            else:
                continue
        except (ValueError, IndexError) as e:
            print(f"  Warning: Could not extract enrichment from {dir_path}, skipping... ({e})")
            continue
    
    # Sort by Zn64 enrichment level (then by Zn67 if same Zn64)
    sorted_indices = sorted(range(len(zn64_enrichment_list)), 
                          key=lambda i: (zn64_enrichment_list[i], zn67_enrichment_list[i]))
    zn64_enrichment_list = [zn64_enrichment_list[i] for i in sorted_indices]
    zn67_enrichment_list = [zn67_enrichment_list[i] for i in sorted_indices]
    output_dirs = [output_dirs[i] for i in sorted_indices]

    print(f"\n✓ Found {len(output_dirs)} enrichment cases:")
    for i, (zn64_enrich, zn67_enrich, output_dir) in enumerate(zip(zn64_enrichment_list, zn67_enrichment_list, output_dirs)):
        print(f"  {i+1}. {output_dir} (Zn-64: {zn64_enrich*100:.1f}%, Zn-67: {zn67_enrich*100:.2f}%)")
    print("="*70)

    successful_64_dirs = []
    successful_67_dirs = []

    
    # Process each discovered directory
    for output_dir, zn64_enrichment in zip(output_dirs, zn64_enrichment_list):
        enrichment_pct = int(zn64_enrichment * 100)
        
        print(f"\n{'='*70}")
        print(f"Processing: {output_dir} (Zn-64: {enrichment_pct}%)")
        print(f"{'='*70}")
        
        # Step 1: Must have analyze_depletion working first
        try:
            results, results_path, isotopes_to_plot, cu_isotopes, irradiation_end_idx, time_days, final_step_atoms, total_atoms, df_final = analyze_depletion(output_dir)
        except Exception as e:
            print(f" Failed to load depletion results for {output_dir}: {e}")
            print(f"   Skipping all analysis for this case...")
            continue
        
        # Step 2: Basic analysis (only needs results)
        try:
            plot_zn_isotopes_evolution(results=results, output_dir=output_dir, isotopes_to_plot=isotopes_to_plot, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            plot_total_activity_evolution(results=results, output_dir=output_dir, time_days=time_days)
            #plot_most_common_reactions(results=results, output_dir=output_dir, results_path=results_path)
            #plot_most_common_reactions_isotope_breakdown(results=results, output_dir=output_dir, results_path=results_path)
        except Exception as e:
            print(f"Warning: Error in basic analysis for {output_dir}: {e}")
            # Continue to try Cu-specific analysis
        
        # Step 3: Get final_atoms_dict (needed for post-irradiation analysis)
        final_atoms_dict = None
        try:
            final_atoms_dict = plot_cu_isotopes_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            if final_atoms_dict is None or len(final_atoms_dict) == 0:
                print(f"Warning: No Cu atoms found in {output_dir}. Skipping Cu-dependent analysis.")
                final_atoms_dict = None
        except Exception as e:
            print(f"Warning: Error getting Cu atoms for {output_dir}: {e}")
            final_atoms_dict = None
        
        # Step 4: Get total_activities (needed for purity and post-irradiation analysis)
        total_activities = None
        try:
            total_activities = plot_cu_isotopes_activity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            if total_activities is None or len(total_activities) == 0:
                print(f"Warning: No Cu activities found in {output_dir}. Skipping activity-dependent analysis.")
                total_activities = None
        except Exception as e:
            print(f"Warning: Error getting Cu activities for {output_dir}: {e}")
            total_activities = None
        
        # Step 5: Purity evolution (needs total_activities)
        if total_activities is not None:
            try:
                plot_cu_isotopes_purity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, total_activities=total_activities, irradiation_end_idx=irradiation_end_idx)
            except Exception as e:
                print(f"Warning: Error in purity evolution plot for {output_dir}: {e}")
        else:
            print(f"Skipping purity evolution (no total_activities available)")
        
        # Step 6: Post-irradiation analysis (needs both total_activities AND final_atoms_dict)
        if total_activities is not None and final_atoms_dict is not None:
            try:
                cu64_purity, cu67_purity, activity_total, post_time_seconds, cu64_activity, cu67_activity = plot_cu_isotopes_post_irradiation_activity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, total_activities=total_activities)
            except Exception as e:
                print(f"Warning: Error in post-irradiation analysis for {output_dir}: {e}")
        else:
            print(f"Skipping post-irradiation analysis (missing prerequisites)")
            if total_activities is None:
                print(f"     - Missing: total_activities")
            if final_atoms_dict is None:
                print(f"     - Missing: final_atoms_dict")
        
        successful_64_dirs.append(output_dir)
        print(f"Completed analysis for {output_dir}")



    
    # Process each discovered directory
    for output_dir, zn67_enrichment in zip(output_dirs, zn67_enrichment_list):
        enrichment_pct = int(zn67_enrichment * 100)
        
        print(f"\n{'='*70}")
        print(f"Processing: {output_dir} (Zn-64: {enrichment_pct}%)")
        print(f"{'='*70}")
        
        # Step 1: Must have analyze_depletion working first
        try:
            results, results_path, isotopes_to_plot, cu_isotopes, irradiation_end_idx, time_days, final_step_atoms, total_atoms, df_final = analyze_depletion(output_dir)
        except Exception as e:
            print(f" Failed to load depletion results for {output_dir}: {e}")
            print(f"   Skipping all analysis for this case...")
            continue        
        
        # Step 2: Basic analysis (only needs results)
        try:
            plot_zn_isotopes_evolution(results=results, output_dir=output_dir, isotopes_to_plot=isotopes_to_plot, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            plot_total_activity_evolution(results=results, output_dir=output_dir, time_days=time_days)
            #plot_most_common_reactions(results=results, output_dir=output_dir, results_path=results_path)
            #plot_most_common_reactions_isotope_breakdown(results=results, output_dir=output_dir, results_path=results_path)
        except Exception as e:
            print(f"Warning: Error in basic analysis for {output_dir}: {e}")
            # Continue to try Cu-specific analysis
        
        # Step 3: Get final_atoms_dict (needed for post-irradiation analysis)
        final_atoms_dict = None
        try:
            final_atoms_dict = plot_cu_isotopes_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            if final_atoms_dict is None or len(final_atoms_dict) == 0:
                print(f"Warning: No Cu atoms found in {output_dir}. Skipping Cu-dependent analysis.")
                final_atoms_dict = None
        except Exception as e:
            print(f"Warning: Error getting Cu atoms for {output_dir}: {e}")
            final_atoms_dict = None
        
        # Step 4: Get total_activities (needed for purity and post-irradiation analysis)
        total_activities = None
        try:
            total_activities = plot_cu_isotopes_activity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, irradiation_end_idx=irradiation_end_idx)
            if total_activities is None or len(total_activities) == 0:
                print(f"Warning: No Cu activities found in {output_dir}. Skipping activity-dependent analysis.")
                total_activities = None
        except Exception as e:
            print(f"Warning: Error getting Cu activities for {output_dir}: {e}")
            total_activities = None
        
        # Step 5: Purity evolution (needs total_activities)
        if total_activities is not None:
            try:
                plot_cu_isotopes_purity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, total_activities=total_activities, irradiation_end_idx=irradiation_end_idx)
            except Exception as e:
                print(f"Warning: Error in purity evolution plot for {output_dir}: {e}")
        else:
            print(f"Skipping purity evolution (no total_activities available)")
        
        # Step 6: Post-irradiation analysis (needs both total_activities AND final_atoms_dict)
        if total_activities is not None and final_atoms_dict is not None:
            try:
                cu64_purity, cu67_purity, activity_total, post_time_seconds, cu64_activity, cu67_activity = plot_cu_isotopes_post_irradiation_activity_evolution(results=results, output_dir=output_dir, cu_isotopes=cu_isotopes, time_days=time_days, total_activities=total_activities)
            except Exception as e:
                print(f"Warning: Error in post-irradiation analysis for {output_dir}: {e}")
        else:
            print(f"Skipping post-irradiation analysis (missing prerequisites)")
            if total_activities is None:
                print(f"     - Missing: total_activities")
            if final_atoms_dict is None:
                print(f"     - Missing: final_atoms_dict")
        
        successful_67_dirs.append(output_dir)
        print(f"Completed analysis for {output_dir}")


    
    # Step 7: Comparison analysis (only if we have successful directories)
    if successful_64_dirs:
        print(f"\n{'='*70}")
        print(f"Running comparison analysis for 64Cu/67Cu purity and activity across {len(successful_64_dirs)} successful cases...")
        print(f"{'='*70}")
        try:
            analyze_depletion_zn_enrichment(successful_dirs=successful_64_dirs)
        except Exception as e:
            print(f"Error in comparison analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{'='*70}")
        print(f"No enriched 64Zn directories were successfully processed!")
        print(f"{'='*70}")
    
    if successful_67_dirs:
        print(f"\n{'='*70}")
        print(f"Running comparison analysis for 67Cu/64Zn purity and activity across {len(successful_67_dirs)} successful cases...")
        print(f"{'='*70}")
        try:
            analyze_depletion_zn_enrichment(successful_dirs=successful_67_dirs)
        except Exception as e:
            print(f"Error in comparison analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{'='*70}")
        print(f"No enriched 67Zn directories were successfully processed!")
        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print(f"Analysis Summary:")
    print(f"  Total directories found: {len(output_dirs)}")
    print(f"  Successfully processed: {len(successful_64_dirs) + len(successful_67_dirs)}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()