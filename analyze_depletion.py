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
# Optional: adjust_text for better label placement
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
from openmc.deplete import Results
from math import log
from scipy.constants import Avogadro
import openmc
import openmc.deplete
import h5py

from utilities  import *
from utilities import _half_life_seconds
# -------------------------------
# ---- Configuration ----
# -------------------------------
def get_initial_targets(statepoint_file, material_id_list=None):
    """
    Get volumes, densities, and initial atom counts per nuclide for materials.
    Uses analytic volumes (from directory name) and Summary for densities/nuclides.
    """
    from scipy.constants import Avogadro
    import numpy as np

    if material_id_list is None:
        material_id_list = [0, 1]

    run_dir = os.path.dirname(os.path.abspath(statepoint_file))
    dir_name = os.path.basename(run_dir)
    summary_path = os.path.join(run_dir, 'summary.h5')
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary.h5 not found: {summary_path}")

    summary = openmc.Summary(summary_path)

    # --- Volumes: analytic from directory name ---
    volumes = compute_volumes_from_dir_name(dir_name)

    # --- Densities from Summary materials ---
    densities = {}
    for mid in material_id_list:
        mat = summary.materials[mid]  # or summary.materials.get_by_id(mid)
        d = mat.density
        densities[mid] = (d[0] if isinstance(d, (list, tuple)) else d)

    # --- Initial atoms per nuclide ---
    atoms_by_material = {}
    for mid in material_id_list:
        mat = summary.materials[mid]
        V = volumes.get(mid)
        rho = densities.get(mid)
        
        if V is None or rho is None:
            atoms_by_material[mid] = {}
            continue

        nuclides = getattr(mat, 'nuclides', []) or []
        if not nuclides:
            atoms_by_material[mid] = {}
            continue

        total_mass = rho * V  # g
        
        # Mean atomic mass (g/mol)
        M_avg = 0.0
        for nuc, frac, _ in nuclides:
            Mi = openmc.data.atomic_mass(nuc)
            M_avg += frac * Mi
        
        if M_avg <= 0:
            atoms_by_material[mid] = {}
            continue

        total_atoms = total_mass * Avogadro / M_avg
        atoms_by_material[mid] = {
            nuc: frac * total_atoms
            for nuc, frac, _ in nuclides
        }

    return volumes, densities, atoms_by_material


def determine_composition(statepoint_file, material_id_list=None, output_file=None, irradiation_time_list=[4, 8, 36]):

    sp = openmc.StatePoint(statepoint_file)

    # initial atoms
    volumes, densities, atoms_by_material = get_initial_targets(statepoint_file, [0, 1])
    print(f"Volume Inner: {volumes[0]}")
    print(f"Volume Outer: {volumes[1]}")
    print(f"Density Inner: {densities[0]}")
    print(f"Density Outer: {densities[1]}")

    results_by_material = {
        0: {
            "initial_atoms": atoms_by_material[0],
            "channel_rr_per_s": build_channel_rr_per_s(sp, cell_id=0, source_strength=5e13),
        },
        1: {
            "initial_atoms": atoms_by_material[1],
            "channel_rr_per_s": build_channel_rr_per_s(sp, cell_id=1, source_strength=5e13),
        },
    }

    irradiation_time_hours_list = [4, 8, 36]
    
    outputs = predict_for_times(
        results_by_material=results_by_material,
        irradiation_time_hours_list=irradiation_time_hours_list,
        material_names={0: "Inner", 1: "Outer"},
        output_prefix=f"{output_file}/transport_only_composition",
    )
    
    return outputs

def plot_cu_isotopes_from_outputs(outputs, output_dir, chamber="Inner", cu_isotopes=None):
    """
    outputs: dict[hours -> DataFrame] from predict_for_times()
    chamber: "Inner" or "Outer" (must match your material_names label)
    """
    if cu_isotopes is None:
        cu_isotopes = [f"Cu{i}" for i in range(63, 72)]

    hours_list = sorted(outputs.keys())
    x_days = np.array(hours_list, dtype=float) / 24.0

    plt.figure(figsize=(8, 6))

    final_atoms_dict = {}
    atoms_col = f"Atoms_{chamber}"

    for iso in cu_isotopes:
        y = []
        for h in hours_list:
            df = outputs[h]
            row = df.loc[df["Nuclide"] == iso]
            atoms = float(row[atoms_col].iloc[0]) if (not row.empty and atoms_col in df.columns) else 0.0
            y.append(atoms)

        y = np.array(y, dtype=float)
        final_atoms_dict[iso] = y[-1] if len(y) else 0.0

        if np.any(y > 0):
            plt.semilogy(x_days, np.maximum(y, 1e-30), label=iso)

    plt.xlabel("Irradiation time [days]")
    plt.ylabel("Number of atoms")
    plt.title(f"Cu isotopes vs irradiation time ({chamber})")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"cu_isotopes_vs_time_{chamber}.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Saved Cu isotope plot {out_png}")
    return final_atoms_dict 

def plot_zn_isotopes_activity_evolution(outputs, output_dir, irradiation_hours=8, cooldown_days=100.0, n_cooldown_points=50, isotopes_to_plot=None, material_names=("Inner", "Outer"),):
    # -------------------------------
    # ---- Plot Zn isotopes activity evolution with extended decay ----
    # -------------------------------
    """
    outputs: dict[hours -> DataFrame] from predict_for_times()
    Plots activity vs total time [days]: during irradiation (0 -> 8h) then cooldown (8h -> 8h+1d).
    """
    import numpy as np

    if isotopes_to_plot is None:
        isotopes_to_plot = [f"Zn{i}" for i in range(64, 71)]

    # --- Time axis: total time [days] ---
    # During irradiation: 0 and each available outputs key <= irradiation_hours
    keys = sorted(k for k in outputs.keys() if float(k) <= float(irradiation_hours))
    if not keys:
        raise KeyError(f"No outputs keys <= {irradiation_hours}. Keys: {list(outputs.keys())}")
    df_end = outputs[keys[-1]]
    cols = list(df_end.columns)
    print(f"  [plot] outputs keys = {list(outputs.keys())}, keys used = {keys}")
    print(f"  [plot] df columns = {cols}")
    for ch in material_names:
        bq_col = f"Bq_{ch}"
        if bq_col in df_end.columns:
            mx = df_end[bq_col].max()
            print(f"  [plot] {bq_col} max = {mx:.4e}")
        else:
            print(f"  [plot] MISSING column {bq_col}")

    t_irr_days = [0.0] + [float(h) / 24.0 for h in keys]
    t_end_irr = float(irradiation_hours) / 24.0

    # Cooldown: t_end_irr -> t_end_irr + cooldown_days
    t_cooldown = np.linspace(0.0, float(cooldown_days), int(n_cooldown_points) + 1)[1:]  # exclude 0
    time_days = np.concatenate([
        np.array(t_irr_days),
        t_end_irr + t_cooldown,
    ])

    plt.figure(figsize=(8, 6))
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(isotopes_to_plot)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(isotopes_to_plot)))

    for chamber in material_names:
        bq_col = f"Bq_{chamber}"
        colors = blue_colors if chamber == "Outer" else red_colors

        for i, iso in enumerate(isotopes_to_plot):
            # During irradiation: 0 at t=0, then from outputs at each key
            act_irr = [0.0]
            for h in keys:
                df = outputs[h]
                row = df.loc[df["Nuclide"] == iso]
                A_Bq = float(row[bq_col].iloc[0]) if (not row.empty and bq_col in df.columns) else 0.0
                act_irr.append(A_Bq)
            act_irr = np.array(act_irr) / 3.7e7  # Bq -> mCi

            # After: decay from end-of-irradiation
            df_end = outputs[keys[-1]]
            row = df_end.loc[df_end["Nuclide"] == iso]
            A0 = float(row[bq_col].iloc[0]) if (not row.empty and bq_col in df_end.columns) else 0.0

            hl = openmc.data.half_life(iso)
            if hl is None or not np.isfinite(hl) or hl <= 0:
                act_cooldown = np.full_like(t_cooldown, A0 / 3.7e7)
            else:
                lam = np.log(2.0) / float(hl)
                act_cooldown = (A0 * np.exp(-lam * t_cooldown * 86400.0)) / 3.7e7

            activity_mCi = np.concatenate([act_irr, act_cooldown])
            activity_mCi = np.maximum(activity_mCi, 1e-30)

            plt.semilogy(time_days, activity_mCi, color=colors[i], label=f"{chamber} - {iso}", linewidth=1.5)

    # Vertical line: end of irradiation / start of cooldown
    plt.axvline(x=t_end_irr, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.text(t_end_irr, plt.ylim()[1] * 0.1, " end irr.\n cooldown starts ", ha="center", va="bottom",
             fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.axhspan(0, 2, alpha=0.2, color='green', zorder=0)      # C-level: <=2 mCi
    plt.axhspan(2, 100, alpha=0.2, color='yellow', zorder=0)  # B-level: 2-100 mCi
    plt.axhspan(100, 10e6, alpha=0.2, color='red', zorder=0)    # A-level: >100 mCi
    plt.text(1, 1, 'C-level', fontsize=10, fontweight='bold', va='center')
    plt.text(1, 10, 'B-level', fontsize=10, fontweight='bold', va='center')
    plt.text(1, 1000, 'A-level', fontsize=10, fontweight='bold', va='center')

    plt.xlabel("Total time [days]")
    plt.ylabel("Activity [mCi]")
    plt.title(f"Zn activity: {irradiation_hours} h irradiation + {cooldown_days} day cooldown")
    plt.ylim(1e-1, 10e6)
    plt.grid(True, which="both", ls=":")
    plt.legend(ncol=2, fontsize=8)

    out_png = os.path.join(output_dir, f"zn_activity_during_after_{irradiation_hours}h_plus_{cooldown_days:g}d.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"→ Saved Zn activity (during + after) → {out_png}")


def plot_zn_isotopes_SA_evolution(outputs, output_dir, irradiation_hours=8, cooldown_days=100.0, n_cooldown_points=100, isotopes_to_plot=None, material_names=("Inner", "Outer")):
    """
    outputs: dict[hours -> DataFrame] from predict_for_times()
    Plots Zn specific activity [Bq/g] vs total time [days]:
    8 h irradiation then 100 day cooldown. Uses outputs only (no depletion results).
    """

    if isotopes_to_plot is None:
        isotopes_to_plot = [f"Zn{i}" for i in range(64, 71)]

    # --- Time axis: total time [days] ---
    keys = sorted(k for k in outputs.keys() if float(k) <= float(irradiation_hours))
    if not keys:
        raise KeyError(f"No outputs keys <= {irradiation_hours}. Keys: {list(outputs.keys())}")

    t_irr_days = [0.0] + [float(h) / 24.0 for h in keys]
    t_end_irr = float(irradiation_hours) / 24.0
    t_cooldown = np.linspace(0.0, float(cooldown_days), int(n_cooldown_points) + 1)[1:]
    time_days = np.concatenate([np.array(t_irr_days), t_end_irr + t_cooldown])

    plt.figure(figsize=(8, 6))
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(isotopes_to_plot)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(isotopes_to_plot)))

    for chamber in material_names:
        bq_col = f"Bq_{chamber}"
        mass_col = f"Mass_grams_{chamber}"
        colors = blue_colors if chamber == "Outer" else red_colors

        # Total mass at end of irradiation (sum over Zn isotopes we plot)
        df_end = outputs[keys[-1]]
        total_mass = 0.0
        for iso in isotopes_to_plot:
            row = df_end.loc[df_end["Nuclide"] == iso]
            if not row.empty and mass_col in df_end.columns:
                total_mass += float(row[mass_col].iloc[0])
        total_mass = max(total_mass, 1e-30)  # avoid div by zero

        for i, iso in enumerate(isotopes_to_plot):
            # During irradiation
            sa_irr = [0.0]
            for h in keys:
                df = outputs[h]
                row = df.loc[df["Nuclide"] == iso]
                bq = float(row[bq_col].iloc[0]) if (not row.empty and bq_col in df.columns) else 0.0
                # Use total_mass at this time step (sum over Zn isotopes)
                m = 0.0
                for iso_m in isotopes_to_plot:
                    r = df.loc[df["Nuclide"] == iso_m]
                    if not r.empty and mass_col in df.columns:
                        m += float(r[mass_col].iloc[0])
                m = max(m, 1e-30)
                sa_irr.append(bq / m)
            sa_irr = np.array(sa_irr)

            # After: decay activity, same total_mass (end-of-irr)
            row = df_end.loc[df_end["Nuclide"] == iso]
            A0 = float(row[bq_col].iloc[0]) if (not row.empty and bq_col in df_end.columns) else 0.0
            hl = openmc.data.half_life(iso)
            if hl is None or not np.isfinite(hl) or hl <= 0:
                sa_cooldown = np.full_like(t_cooldown, A0 / total_mass)
            else:
                lam = np.log(2.0) / float(hl)
                act_cooldown = A0 * np.exp(-lam * t_cooldown * 86400.0)
                sa_cooldown = act_cooldown / total_mass

            SA = np.concatenate([sa_irr, sa_cooldown])
            SA = np.maximum(SA, 1e-30)
            plt.semilogy(time_days, SA, color=colors[i], label=f"{chamber} - {iso}", linewidth=1.5)

            # Label at 60 days if SA > 0.1 Bq/g
            t_label = 60.0
            if t_label <= time_days[-1]:
                sa_60 = np.interp(t_label, time_days, SA)
                if sa_60 > 0.1:
                    plt.text(t_label, sa_60, f" {sa_60:.2f} Bq/g", fontsize=6, color="gray", va="center", ha="left")

    plt.axvline(x=t_end_irr, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.text(t_end_irr, plt.ylim()[1] * 0.1, " end irr.\n cooldown starts ", ha="center", va="bottom",
             fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Exclusion level 0.1 Bq/g
    plt.axhline(y=0.1, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Exclusion Level")
    x_pos = time_days[-1] * 0.95
    plt.text(x_pos, 0.2, "Exclusion\nLevel", ha="right", va="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.xlabel("Total time [days]")
    plt.ylabel("Specific activity [Bq/g]")
    plt.title(f"Zn specific activity: {irradiation_hours} h irr. + {cooldown_days} d cooldown")
    plt.ylim(1e-1, 1e7)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, which="both", ls=":")

    out_png = os.path.join(output_dir, f"zn_SA_{irradiation_hours}h_plus_{cooldown_days:g}d.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"→ Saved Zn SA plot → {out_png}")


def get_total_activities_and_activity_results_from_outputs(
    outputs,
    irradiation_hours=8,
    cooldown_days=1.0,
    n_cooldown_points=100,
    cu_isotopes=None,
    material_id_list=None,
):
    """
    Build total_activities and activity_results from outputs (predict_for_times).
    Uses _half_life_seconds from utilities for decay.
    Returns (total_activities, activity_results, time_days).
    """
    if cu_isotopes is None:
        cu_isotopes = [f"Cu{i}" for i in range(63, 72)]
    if material_id_list is None:
        material_id_list = ["0", "1"]
    mid_to_label = {"0": "Inner", "1": "Outer"}

    keys = sorted(k for k in outputs.keys() if float(k) <= float(irradiation_hours))
    if not keys:
        raise KeyError(f"No outputs keys <= {irradiation_hours}. Keys: {list(outputs.keys())}")

    t_irr_days = [0.0] + [float(h) / 24.0 for h in keys]
    t_end_irr = float(irradiation_hours) / 24.0
    t_cooldown = np.linspace(0.0, float(cooldown_days), int(n_cooldown_points) + 1)[1:]
    time_days = np.concatenate([np.array(t_irr_days), t_end_irr + t_cooldown])

    n_irr = len(t_irr_days)
    n_cool = len(t_cooldown)
    n_ext = n_irr + n_cool

    total_activities = {mid: np.zeros(n_ext, dtype=float) for mid in material_id_list}
    activity_results = {mid: [] for mid in material_id_list}

    for mid in material_id_list:
        label = mid_to_label[mid]
        bq_col = f"Bq_{label}"
        activities_list = []

        for ti, t_d in enumerate(t_irr_days):
            act_dict = {}
            if ti == 0:
                for iso in cu_isotopes:
                    act_dict[iso] = 0.0
            else:
                df = outputs[keys[ti - 1]]
                for iso in cu_isotopes:
                    row = df.loc[df["Nuclide"] == iso]
                    bq = float(row[bq_col].iloc[0]) if (not row.empty and bq_col in df.columns) else 0.0
                    act_dict[iso] = bq
            activities_list.append(act_dict)
            total_activities[mid][ti] = sum(act_dict.values())

        df_end = outputs[keys[-1]]
        cooldown_dicts = [{} for _ in range(n_cool)]
        for iso in cu_isotopes:
            row = df_end.loc[df_end["Nuclide"] == iso]
            A0 = float(row[bq_col].iloc[0]) if (not row.empty and bq_col in df_end.columns) else 0.0
            hl = _half_life_seconds(iso)
            for i in range(n_cool):
                dt_s = t_cooldown[i] * 86400.0
                if hl is not None:
                    lam = np.log(2.0) / hl
                    A = A0 * np.exp(-lam * dt_s)
                else:
                    A = A0
                cooldown_dicts[i][iso] = A

        for i, d in enumerate(cooldown_dicts):
            activities_list.append(d)
            total_activities[mid][n_irr + i] = sum(d.values())

        activity_results[mid] = [activities_list]

    return total_activities, activity_results, time_days


def plot_cu_isotopes_activity_evolution(outputs, output_dir, irradiation_hours=8, cooldown_days=1.0, n_cooldown_points=100, cu_isotopes=None, material_id_list=None):
    """
    Plot Cu isotopes activity evolution using outputs (predict_for_times).
    8 h irradiation + 1 day cooldown. Uses get_total_activities_and_activity_results_from_outputs
    and _half_life_seconds (utilities). Returns (total_activities, activity_results).
    """
    if cu_isotopes is None:
        cu_isotopes = [f"Cu{i}" for i in range(63, 72)]
    if material_id_list is None:
        material_id_list = ["0", "1"]

    total_activities, activity_results, time_days = get_total_activities_and_activity_results_from_outputs(
        outputs,
        irradiation_hours=irradiation_hours,
        cooldown_days=cooldown_days,
        n_cooldown_points=n_cooldown_points,
        cu_isotopes=cu_isotopes,
        material_id_list=material_id_list,
    )

    t_end_irr = float(irradiation_hours) / 24.0
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cu_isotopes)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cu_isotopes)))

    plt.figure(figsize=(8, 6))
    for mid in material_id_list:
        colors = blue_colors if mid == "1" else red_colors
        list_of_dicts = activity_results[mid][0]
        for iso_idx, iso in enumerate(cu_isotopes):
            activity = np.array([d.get(iso, 0.0) for d in list_of_dicts], dtype=float)
            activity = np.maximum(activity, 1e-30)
            plt.semilogy(time_days, activity, label=f"{mid} - {iso}", color=colors[iso_idx], linewidth=1.5)

    plt.axvline(x=t_end_irr, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.text(t_end_irr, plt.ylim()[1] * 0.1, " end irr.\n cooldown starts ", ha="center", va="bottom",
             fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.xlabel("Total time [days]")
    plt.ylabel("Activity [Bq]")
    plt.title(f"Cu activity: {irradiation_hours} h irr. + {cooldown_days} d cooldown (from outputs)")
    plt.ylim(1e-7, 1e13)
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    out_png = os.path.join(output_dir, f"cu_isotope_activity_evolution_{irradiation_hours}h_plus_{cooldown_days:g}d.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"→ Saved Cu isotope activity evolution (from outputs) → {out_png}")
    return total_activities, activity_results


def plot_cu_isotopes_purity_evolution(outputs, output_dir, irradiation_hours=8, cooldown_days=1.0, n_cooldown_points=100, cu_isotopes=None, material_id_list=None):
    """
    Plot Cu isotopes purity and SA evolution using outputs (predict_for_times).
    8 h irradiation + 1 day cooldown. Uses get_total_activities_and_activity_results_from_outputs
    and _half_life_seconds (utilities). Returns (purity_results, SA_results).
    """
    if cu_isotopes is None:
        cu_isotopes = [f"Cu{i}" for i in range(63, 72)]
    if material_id_list is None:
        material_id_list = ["0", "1"]
    mid_to_label = {"0": "Inner", "1": "Outer"}

    # Get activities and time from get_total_activities_and_activity_results_from_outputs
    total_activities, activity_results, time_days = get_total_activities_and_activity_results_from_outputs(
        outputs,
        irradiation_hours=irradiation_hours,
        cooldown_days=cooldown_days,
        n_cooldown_points=n_cooldown_points,
        cu_isotopes=cu_isotopes,
        material_id_list=material_id_list,
    )

    t_end_irr = float(irradiation_hours) / 24.0
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cu_isotopes)))
    red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(cu_isotopes)))
    purity_results = {mid: None for mid in material_id_list}
    SA_results = {mid: {} for mid in material_id_list}

    # Get keys from outputs
    keys = sorted(k for k in outputs.keys() if float(k) <= float(irradiation_hours))
    if not keys:
        raise KeyError(f"No outputs keys <= {irradiation_hours}. Keys: {list(outputs.keys())}")

    # Specific Activity plot
    plt.figure(figsize=(8, 6))
    for material_id in material_id_list:
        color_scheme = blue_colors if material_id == '1' else red_colors
        label = mid_to_label[material_id]
        atoms_col = f"Atoms_{label}"
        mass_col = f"Mass_grams_{label}"
        
        # Get atoms during irradiation from outputs
        n_irr = len([0.0] + [float(h) / 24.0 for h in keys])
        n_cool = n_cooldown_points
        total_mass = np.zeros(n_irr + n_cool)
        
        # During irradiation: get atoms from outputs DataFrames
        for ti in range(n_irr):
            mass_at_t = 0.0
            if ti == 0:
                for iso in cu_isotopes:
                    mass_at_t += 0.0
            else:
                df = outputs[keys[ti - 1]]
                for iso in cu_isotopes:
                    row = df.loc[df["Nuclide"] == iso]
                    if not row.empty and mass_col in df.columns:
                        mass_at_t += float(row[mass_col].iloc[0])
                    elif not row.empty and atoms_col in df.columns:
                        atoms = float(row[atoms_col].iloc[0])
                        mass_per_atom = openmc.data.atomic_mass(iso) / Avogadro
                        mass_at_t += atoms * openmc.data.atomic_mass(iso) / Avogadro
            total_mass[ti] = max(mass_at_t, 1e-30)
        
        # During cooldown: decay atoms from end of irradiation
        df_end = outputs[keys[-1]]
        cooldown_masses = np.zeros(n_cool)
        t_cooldown = np.linspace(0.0, float(cooldown_days), int(n_cooldown_points) + 1)[1:]
        
        for iso in cu_isotopes:
            row = df_end.loc[df_end["Nuclide"] == iso]
            if not row.empty:
                if mass_col in df_end.columns:
                    mass_end = float(row[mass_col].iloc[0])
                elif atoms_col in df_end.columns:
                    atoms_end = float(row[atoms_col].iloc[0])
                    mass_end = atoms_end * openmc.data.atomic_mass(iso) / Avogadro
                else:
                    mass_end = 0.0
                
                mass_per_atom = openmc.data.atomic_mass(iso) / Avogadro
                hl = _half_life_seconds(iso)
                for i in range(n_cool):
                    dt_s = t_cooldown[i] * 86400.0
                    if hl is not None:
                        lam = np.log(2.0) / hl
                        atoms_decayed = (mass_end / mass_per_atom) * np.exp(-lam * dt_s)
                        cooldown_masses[i] += atoms_decayed * mass_per_atom
                    else:
                        cooldown_masses[i] += mass_end
        
        total_mass[n_irr:] = cooldown_masses
        
        # Get activities from activity_results
        list_of_dicts = activity_results[material_id][0]
        tot_act = np.asarray(total_activities[material_id], dtype=float)
        if tot_act.shape[0] != len(list_of_dicts):
            tot_act = np.array([sum(d.get(iso, 0.0) for iso in cu_isotopes) for d in list_of_dicts], dtype=float)

        for iso_idx, iso in enumerate(cu_isotopes):
            act = np.array([list_of_dicts[t].get(iso, 0.0) for t in range(len(list_of_dicts))])
            SA = np.divide(act, total_mass, out=np.zeros_like(act, dtype=float), where=(total_mass > 0))
            SA_results[material_id][iso] = SA
            plt.semilogy(time_days, SA, label=f"{material_id} - {iso}", color=color_scheme[iso_idx], linewidth=1.5)

    plt.axvline(x=t_end_irr, color="gray", linestyle="--", linewidth=1, alpha=0.8)
    plt.text(t_end_irr, plt.ylim()[1] * 0.1, " end irr.\n cooldown starts ", ha="center", va="bottom",
             fontsize=8, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.xlabel("Time [days]")
    plt.ylabel("Specific Activity [Bq/g]")
    plt.title("Specific Activity Evolution of Copper Isotopes (63–71)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cu_isotope_specific_activity_evolution.png"), dpi=300)
    plt.close()
    print("→ Saved Cu isotope specific activity evolution plot → …")

    # Purity plot (1 - purity) vs time_days
    plt.figure(figsize=(8, 6))
    for material_id in material_id_list:
        color_scheme = blue_colors if material_id == '1' else red_colors
        list_of_dicts = activity_results[material_id][0]
        
        # Get Cu64 and Cu67 activities directly (don't check if in cu_isotopes)
        act_cu64 = np.array([d.get('Cu64', 0.0) for d in list_of_dicts], dtype=float)
        act_cu67 = np.array([d.get('Cu67', 0.0) for d in list_of_dicts], dtype=float)
        
        # Calculate total activity as sum of ONLY Cu64 and Cu67 (for purity calculation)
        tot_act_cu64_cu67 = act_cu64 + act_cu67
        
        cu64_final, cu67_final = 0.0, 0.0
        
        # Plot Cu64 purity
        if np.any(act_cu64 > 0) or np.any(act_cu67 > 0):
            purity_cu64 = np.divide(act_cu64, tot_act_cu64_cu67, out=np.zeros_like(act_cu64, dtype=float), where=(tot_act_cu64_cu67 > 0))
            cu64_final = purity_cu64[-1]
            imp_cu64 = np.where(1.0 - purity_cu64 > 0, 1.0 - purity_cu64, 1e-30)
            iso_idx = cu_isotopes.index('Cu64') if 'Cu64' in cu_isotopes else 0
            plt.semilogy(time_days, imp_cu64, label=f"{material_id} - Cu64", color=color_scheme[iso_idx], linewidth=1.5)
            for t_mark in [0.17, 1.0, 2.0]:
                if t_mark < time_days[0] or t_mark > time_days[-1]:
                    continue
                y_mark = np.interp(t_mark, time_days, imp_cu64)
                pct = 100.0 * (1.0 - y_mark)
                plt.plot(t_mark, y_mark, "o", color='black', markersize=4)
                plt.text(t_mark, y_mark * 0.8, f"  {pct:.1f}%", fontsize=7, color='black', va="center")
        
        # Plot Cu67 purity
        if np.any(act_cu67 > 0) or np.any(act_cu64 > 0):
            purity_cu67 = np.divide(act_cu67, tot_act_cu64_cu67, out=np.zeros_like(act_cu67, dtype=float), where=(tot_act_cu64_cu67 > 0))
            cu67_final = purity_cu67[-1]
            imp_cu67 = np.where(1.0 - purity_cu67 > 0, 1.0 - purity_cu67, 1e-30)
            iso_idx = cu_isotopes.index('Cu67') if 'Cu67' in cu_isotopes else (len(cu_isotopes) - 1 if cu_isotopes else 0)
            plt.semilogy(time_days, imp_cu67, label=f"{material_id} - Cu67", color=color_scheme[iso_idx], linewidth=1.5)
            for t_mark in [0.17, 1.0, 2.0]:
                if t_mark < time_days[0] or t_mark > time_days[-1]:
                    continue
                y_mark = np.interp(t_mark, time_days, imp_cu67)
                pct = 100.0 * (1.0 - y_mark)
                plt.plot(t_mark, y_mark, "o", color='black', markersize=4)
                plt.text(t_mark, y_mark * 0.8, f"  {pct:.1f}%", fontsize=7, color='black', va="center")
        
        purity_results[material_id] = [np.array([cu64_final]), np.array([cu67_final])]

    plt.xlabel("Total time [days]")
    plt.ylabel("1 - Purity (Impurity fraction, log scale)")
    plt.title("Cu64, Cu67 — materials 0 and 1")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cu_isotope_purity_evolution.png"), dpi=300)
    plt.close()
    print("→ Saved Cu isotope purity evolution plot")

    return purity_results, SA_results


def plot_flux_spectra_and_heating_by_cell(statepoint_file, output_dir):
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
    """Plot geometry purity evolution - only plots directories with valid purity data."""
    
    # Filter to only valid directories with proper data structure
    valid_dirs = []
    for d in output_dir_list:
        if d not in purity_results_list or purity_results_list[d] is None:
            continue
        purity_data = purity_results_list[d]
        # Validate data structure
        if not isinstance(purity_data, dict):
            print(f"Warning: plot_geom_purity_evolution: Invalid data type for '{d}': {type(purity_data)}")
            continue
        if '0' not in purity_data or '1' not in purity_data:
            print(f"Warning: plot_geom_purity_evolution: Missing material keys for '{d}'")
            continue
        if purity_data['0'] is None or purity_data['1'] is None:
            print(f"Warning: plot_geom_purity_evolution: None values for '{d}'")
            continue
        valid_dirs.append(d)
    
    n_plots = len(valid_dirs)
    
    if n_plots == 0:
        print("Warning: No valid purity data found for geometry plots")
        return
    
    # Calculate grid dimensions - limit columns for readability
    n_cols = min(5, int(np.ceil(np.sqrt(n_plots))))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Color gradients
    colors_64 = plt.cm.Blues(np.linspace(0.3, 1, 101))
    colors_67 = plt.cm.Reds(np.linspace(0.3, 1, 101))

    # Scale figure size based on number of plots
    fig_width = n_cols * 4
    fig_height = n_rows * 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle single plot case
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Collect purity values for color scaling
    all_purity_cu67_last = []
    all_purity_cu64_last = []

    for output_dir in valid_dirs:
        purity_results = purity_results_list[output_dir]
        try:
            purity_cu67_last = purity_results['0'][1][-1]
            purity_cu64_last = purity_results['1'][0][-1]
            all_purity_cu67_last.append(purity_cu67_last)
            all_purity_cu64_last.append(purity_cu64_last)
        except (IndexError, KeyError, TypeError) as e:
            print(f"Warning: Error extracting purity for '{output_dir}': {e}")

    if not all_purity_cu67_last or not all_purity_cu64_last:
        print("Warning: No valid purity values extracted")
        plt.close()
        return

    max_purity_cu67_last = np.max(all_purity_cu67_last)
    max_purity_cu64_last = np.max(all_purity_cu64_last)
    min_purity_cu67_last = np.min(all_purity_cu67_last)
    min_purity_cu64_last = np.min(all_purity_cu64_last)
    
    # Color mapping functions
    def blue_purity_to_color_index(purity):
        if max_purity_cu64_last == min_purity_cu64_last:
            return 50
        if purity < min_purity_cu64_last:
            return 0
        elif purity > max_purity_cu64_last:
            return 100
        else:
            index = int((purity - min_purity_cu64_last) / (max_purity_cu64_last - min_purity_cu64_last) * 100)
            return min(max(index, 0), 100)

    def red_purity_to_color_index(purity):
        if max_purity_cu67_last == min_purity_cu67_last:
            return 50
        if purity < min_purity_cu67_last:
            return 0
        elif purity > max_purity_cu67_last:
            return 100
        else:
            index = int((purity - min_purity_cu67_last) / (max_purity_cu67_last - min_purity_cu67_last) * 100)
            return min(max(index, 0), 100)
    
    # Plot each valid directory
    for plot_idx, output_dir in enumerate(valid_dirs):
        purity_results = purity_results_list[output_dir]
        ax = axes[plot_idx]

        try:
            z_inner_thickness = float(output_dir.split("_inner")[1].split("_")[0])
            z_outer_thickness = float(output_dir.split("_outer")[1].split("_")[0])
            struct_thickness = float(output_dir.split("_struct")[1].split("_")[0])
            moderator_thickness = float(output_dir.split("_moderator")[1].split("_")[0])
            multi_thickness = float(output_dir.split("_multi")[1].split("_")[0])
        except (IndexError, ValueError) as e:
            print(f"Warning: Error parsing directory name '{output_dir}': {e}")
            continue

        inner_radius = 5
        center = (0, 0)
        
        # Get purity values at last timestep
        purity_cu67_last = purity_results['0'][1][-1]
        purity_cu64_last = purity_results['1'][0][-1]
        
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
        
        for r in radii:
            circle = mpatches.Circle(center, r, fill=False, edgecolor='black', linestyle='--', linewidth=0.5)
            ax.add_patch(circle)

        # Draw annulus for Z inner (Cu67 purity)
        r_inner_z = inner_radius + struct_thickness
        r_outer_z = inner_radius + struct_thickness + z_inner_thickness
        if r_outer_z > r_inner_z:
            annulus_z = mpatches.Annulus(center, r_outer_z, r_outer_z - r_inner_z, color=colors_67[color_idx_67])
            ax.add_patch(annulus_z)

        # Draw annulus for Z outer (Cu64 purity)
        r_inner_z_outer = inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness
        r_outer_z_outer = r_inner_z_outer + z_outer_thickness
        if r_outer_z_outer > r_inner_z_outer:
            annulus_z_outer = mpatches.Annulus(center, r_outer_z_outer, r_outer_z_outer - r_inner_z_outer, color=colors_64[color_idx_64])
            ax.add_patch(annulus_z_outer)
        
        # Set equal aspect ratio and axis limits
        ax.set_aspect('equal')
        max_radius = radii[-1] * 1.1
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        
        # Cleaner title with key parameters
        title = f"in={z_inner_thickness}, out={z_outer_thickness}, mod={moderator_thickness}\nCu67: {purity_cu67_last*100:.2f}%, Cu64: {purity_cu64_last*100:.2f}%"
        ax.set_title(title, pad=8, fontsize=8)

    # Hide unused subplots
    for j in range(len(valid_dirs), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    out_dir = summary_output_dir if summary_output_dir is not None else (output_dir_list[0] if output_dir_list else ".")
    output_path = os.path.join(out_dir, "geom_purity_evolution.png")  
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"→ Saved geom purity evolution → {output_path}")


def plot_geom_activity_evolution(activity_results_list=None, output_dir_list=None, summary_output_dir=None):
    """Plot geometry activity evolution - only plots directories with valid activity data."""
    
    # Filter to only valid directories with proper data structure
    valid_dirs = []
    for d in output_dir_list:
        if d not in activity_results_list or activity_results_list[d] is None:
            continue
        activity_data = activity_results_list[d]
        # Validate data structure - must be a dict with proper structure
        if not isinstance(activity_data, dict):
            print(f"Warning: plot_geom_activity_evolution: Invalid data type for '{d}': {type(activity_data)}")
            continue
        if '0' not in activity_data or '1' not in activity_data:
            print(f"Warning: plot_geom_activity_evolution: Missing material keys for '{d}'")
            continue
        # Check that the nested structure is correct
        try:
            _ = activity_data['0'][0][-1]['Cu67']
            _ = activity_data['1'][0][-1]['Cu64']
            valid_dirs.append(d)
        except (IndexError, KeyError, TypeError) as e:
            print(f"Warning: plot_geom_activity_evolution: Invalid data structure for '{d}': {e}")
            continue
    
    n_plots = len(valid_dirs)
    
    if n_plots == 0:
        print("Warning: No valid activity data found for geometry plots")
        return
    
    # Calculate grid dimensions - limit columns for readability
    n_cols = min(5, int(np.ceil(np.sqrt(n_plots))))
    n_rows = int(np.ceil(n_plots / n_cols))
    
    # Color gradients
    colors_64 = plt.cm.Greens(np.linspace(0.3, 1, 101))
    colors_67 = plt.cm.RdPu(np.linspace(0.3, 1, 101))

    # Scale figure size based on number of plots
    fig_width = n_cols * 4
    fig_height = n_rows * 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle single plot case
    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Collect activity values for color scaling
    all_activity_cu67_last = []
    all_activity_cu64_last = []

    for output_dir in valid_dirs:
        activity_results = activity_results_list[output_dir]
        try:
            activity_cu67_last = activity_results['0'][0][-1]['Cu67']
            activity_cu64_last = activity_results['1'][0][-1]['Cu64']
            all_activity_cu67_last.append(activity_cu67_last)
            all_activity_cu64_last.append(activity_cu64_last)
        except (IndexError, KeyError, TypeError) as e:
            print(f"Warning: Error extracting activity for '{output_dir}': {e}")

    if not all_activity_cu67_last or not all_activity_cu64_last:
        print("Warning: No valid activity values extracted")
        plt.close()
        return

    max_activity_cu67_last = np.max(all_activity_cu67_last)
    max_activity_cu64_last = np.max(all_activity_cu64_last)
    min_activity_cu67_last = np.min(all_activity_cu67_last)
    min_activity_cu64_last = np.min(all_activity_cu64_last)
    
    # Color mapping functions
    def green_activity_to_color_index(activity):
        if max_activity_cu64_last == min_activity_cu64_last:
            return 50
        if activity <= min_activity_cu64_last:
            return 0
        elif activity >= max_activity_cu64_last:
            return 100
        else:
            index = int((activity - min_activity_cu64_last) / (max_activity_cu64_last - min_activity_cu64_last) * 100)
            return min(max(index, 0), 100)

    def pink_activity_to_color_index(activity):
        if max_activity_cu67_last == min_activity_cu67_last:
            return 50
        if activity <= min_activity_cu67_last:
            return 0
        elif activity >= max_activity_cu67_last:
            return 100
        else:
            index = int((activity - min_activity_cu67_last) / (max_activity_cu67_last - min_activity_cu67_last) * 100)
            return min(max(index, 0), 100)
    
    # Plot each valid directory
    for plot_idx, output_dir in enumerate(valid_dirs):
        activity_results = activity_results_list[output_dir]
        ax = axes[plot_idx]

        try:
            z_inner_thickness = float(output_dir.split("_inner")[1].split("_")[0])
            z_outer_thickness = float(output_dir.split("_outer")[1].split("_")[0])
            struct_thickness = float(output_dir.split("_struct")[1].split("_")[0])
            moderator_thickness = float(output_dir.split("_moderator")[1].split("_")[0])
            multi_thickness = float(output_dir.split("_multi")[1].split("_")[0])
        except (IndexError, ValueError) as e:
            print(f"Warning: Error parsing directory name '{output_dir}': {e}")
            continue

        inner_radius = 5
        center = (0, 0)
        
        # Get activity values at last timestep
        activity_cu67_last = activity_results['0'][0][-1]['Cu67']
        activity_cu64_last = activity_results['1'][0][-1]['Cu64']
        
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
        
        for r in radii:
            circle = mpatches.Circle(center, r, fill=False, edgecolor='black', linestyle='--', linewidth=0.5)
            ax.add_patch(circle)

        # Draw annulus for Z inner (Cu67 activity)
        r_inner_z = inner_radius + struct_thickness
        r_outer_z = inner_radius + struct_thickness + z_inner_thickness
        if r_outer_z > r_inner_z:
            annulus_z = mpatches.Annulus(center, r_outer_z, r_outer_z - r_inner_z, color=colors_67[color_idx_67])
            ax.add_patch(annulus_z)

        # Draw annulus for Z outer (Cu64 activity)
        r_inner_z_outer = inner_radius + struct_thickness + z_inner_thickness + multi_thickness + moderator_thickness
        r_outer_z_outer = r_inner_z_outer + z_outer_thickness
        if r_outer_z_outer > r_inner_z_outer:
            annulus_z_outer = mpatches.Annulus(center, r_outer_z_outer, r_outer_z_outer - r_inner_z_outer, color=colors_64[color_idx_64])
            ax.add_patch(annulus_z_outer)
        
        # Set equal aspect ratio and axis limits
        ax.set_aspect('equal')
        max_radius = radii[-1] * 1.1
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
        
        # Convert to Curies for display
        activity_cu67_last_Ci = activity_cu67_last / 3.7e10
        activity_cu64_last_Ci = activity_cu64_last / 3.7e10
        
        # Cleaner title with key parameters
        title = f"in={z_inner_thickness}, out={z_outer_thickness}, mod={moderator_thickness}\nCu67: {activity_cu67_last_Ci:.2f} Ci, Cu64: {activity_cu64_last_Ci:.2f} Ci"
        ax.set_title(title, pad=8, fontsize=8)

    # Hide unused subplots
    for j in range(len(valid_dirs), len(axes)):
        axes[j].set_visible(False)

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
    if HAS_ADJUST_TEXT:
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
    if HAS_ADJUST_TEXT:
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


def _find_statepoint(output_dir):
    """Statepoint for outputs-based run: openmc_simulation_n0.h5 or first statepoint.*.h5."""
    p = os.path.join(output_dir, "openmc_simulation_n0.h5")
    if os.path.isfile(p):
        return p
    cand = glob.glob(os.path.join(output_dir, "statepoint.*.h5"))
    return cand[0] if cand else None

from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


def has_outputs_run(d):
    """Check if directory has required output files."""
    sp = _find_statepoint(d)
    return sp is not None and os.path.isfile(os.path.join(d, "summary.h5"))


def process_single_directory(output_dir, config):
    """Process a single output directory - returns results for aggregation."""
    irradiation_hours = config['irradiation_hours']
    cooldown_days_cu = config['cooldown_days_cu']
    cooldown_days_zn = config['cooldown_days_zn']
    n_cooldown_points = config['n_cooldown_points']
    isotopes_to_plot = config['isotopes_to_plot']
    cu_isotopes = config['cu_isotopes']
    material_id_list = config['material_id_list']
    material_names = config['material_names']
    
    print(f"\n{'='*70}\nProcessing: {output_dir}\n{'='*70}")
    
    sp_path = _find_statepoint(output_dir)
    if not sp_path:
        print(f"Warning: No statepoint found for {output_dir}")
        return output_dir, None, None
    
    try:
        outputs = determine_composition(sp_path, output_file=output_dir, 
                                        material_id_list=[0, 1], 
                                        irradiation_time_list=[4, 8, 36])
    except Exception as e:
        print(f"Failed determine_composition for {output_dir}: {e}")
        return output_dir, None, None
    
    # Run individual plots
    try:
        plot_zn_isotopes_activity_evolution(
            outputs, output_dir,
            irradiation_hours=irradiation_hours, cooldown_days=cooldown_days_zn,
            n_cooldown_points=n_cooldown_points, isotopes_to_plot=isotopes_to_plot,
            material_names=material_names,
        )
    except Exception as e:
        print(f"Warning: plot_zn_isotopes_activity_evolution: {e}")
    
    try:
        plot_zn_isotopes_SA_evolution(
            outputs, output_dir,
            irradiation_hours=irradiation_hours, cooldown_days=cooldown_days_zn,
            n_cooldown_points=n_cooldown_points, isotopes_to_plot=isotopes_to_plot,
            material_names=material_names,
        )
    except Exception as e:
        print(f"Warning: plot_zn_isotopes_SA_evolution: {e}")
    
    try:
        plot_cu_isotopes_activity_evolution(
            outputs, output_dir,
            irradiation_hours=irradiation_hours, cooldown_days=cooldown_days_cu,
            n_cooldown_points=n_cooldown_points, cu_isotopes=cu_isotopes,
            material_id_list=material_id_list,
        )
    except Exception as e:
        print(f"Warning: plot_cu_isotopes_activity_evolution: {e}")
    
    try:
        plot_flux_spectra_and_heating_by_cell(sp_path, output_dir)
    except Exception as e:
        print(f"Warning: plot_flux_spectra_and_heating_by_cell: {e}")
    
    # Return the results needed for aggregate plots
    try:
        purity_results, SA_results = plot_cu_isotopes_purity_evolution(
            outputs, output_dir, 
            irradiation_hours=irradiation_hours, cooldown_days=cooldown_days_cu,
            n_cooldown_points=n_cooldown_points, cu_isotopes=cu_isotopes,
            material_id_list=material_id_list,
        )
        print(f"✓ Completed: {output_dir}")
        return output_dir, purity_results, SA_results
    except Exception as e:
        print(f"Warning: plot_cu_isotopes_purity_evolution for {output_dir}: {e}")
        return output_dir, None, None


def main(parallel=True, aggregate_only=False, n_workers=4):
    """
    Main function to run analysis.
    
    Parameters
    ----------
    parallel : bool
        If True, process directories in parallel. Default True.
    aggregate_only : bool
        If True, skip individual processing and only run aggregate plots 
        (requires cached results from previous run). Default False.
    n_workers : int
        Number of parallel workers. Default 4.
    """
    # Configuration
    config = {
        'irradiation_hours': 8,
        'cooldown_days_cu': 1.0,
        'cooldown_days_zn': 100.0,
        'n_cooldown_points': 100,
        'isotopes_to_plot': [f"Zn{i}" for i in range(64, 71)],
        'cu_isotopes': [f"Cu{i}" for i in range(64, 68)],
        'material_id_list': ["0", "1"],
        'material_names': ("Inner", "Outer"),
    }
    
    cache_file = 'aggregate_results_cache.pkl'
    
    if aggregate_only:
        # Load cached results
        if not os.path.isfile(cache_file):
            print(f"Error: Cache file '{cache_file}' not found. Run without --aggregate-only first.")
            return
        
        print(f"Loading cached results from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        purity_results_list = cache['purity_results_list']
        activity_results_list = cache['activity_results_list']
        all_dirs = cache['all_dirs']
        print(f"Loaded results for {len(all_dirs)} directories")
    
    else:
        # Find all directories
        dirs = [x for x in os.listdir(".") if os.path.isdir(x) and x.startswith("irrad_output_inner")]
        all_dirs = [d for d in dirs if has_outputs_run(d)]
        
        if not all_dirs:
            print("No irrad_output_inner* directories with valid output files found. Exiting.")
            return
        
        print(f"Found {len(all_dirs)} directories to process")
        
        purity_results_list = {}
        activity_results_list = {}
        
        if parallel and len(all_dirs) > 1:
            # Process in parallel
            actual_workers = min(n_workers, len(all_dirs))
            print(f"Processing {len(all_dirs)} directories with {actual_workers} workers...")
            
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                futures = {executor.submit(process_single_directory, d, config): d 
                           for d in all_dirs}
                
                for future in as_completed(futures):
                    try:
                        output_dir, purity_results, SA_results = future.result()
                        if purity_results is not None:
                            purity_results_list[output_dir] = purity_results
                            activity_results_list[output_dir] = SA_results
                    except Exception as e:
                        print(f"Error processing directory: {e}")
        else:
            # Process sequentially
            print(f"Processing {len(all_dirs)} directories sequentially...")
            for output_dir in all_dirs:
                output_dir, purity_results, SA_results = process_single_directory(output_dir, config)
                if purity_results is not None:
                    purity_results_list[output_dir] = purity_results
                    activity_results_list[output_dir] = SA_results
        
        # Save results for future aggregate-only runs
        print(f"\nSaving results cache to {cache_file}...")
        cache = {
            'purity_results_list': purity_results_list,
            'activity_results_list': activity_results_list,
            'all_dirs': all_dirs
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        print(f"✓ Saved cache with {len(purity_results_list)} valid results")
    
    # ---- Aggregate plots ----
    print("\n" + "="*70)
    print("Running aggregate plots...")
    print("="*70)
    
    summary_output_dir = os.path.join(os.getcwd(), "irrad_output_results")
    os.makedirs(summary_output_dir, exist_ok=True)
    
    # Plot geometry purity evolution
    try:
        plot_geom_purity_evolution(
            purity_results_list=purity_results_list,
            output_dir_list=all_dirs,
            summary_output_dir=summary_output_dir,
        )
    except Exception as e:
        print(f"Warning: plot_geom_purity_evolution: {e}")
    
    # Plot geometry activity evolution
    try:
        plot_geom_activity_evolution(
            activity_results_list=activity_results_list,
            output_dir_list=all_dirs,
            summary_output_dir=summary_output_dir,
        )
    except Exception as e:
        print(f"Warning: plot_geom_activity_evolution: {e}")
    
    # Plot production vs purity
    try:
        plot_production_vs_purity(
            purity_results_list=purity_results_list,
            activity_results_list=activity_results_list,
            output_dir_list=all_dirs,
            summary_output_dir=summary_output_dir,
        )
    except Exception as e:
        print(f"Warning: plot_production_vs_purity: {e}")
    
    # Plot geometry production vs purity
    try:
        plot_geom_prod_vs_purity(
            purity_results_list=purity_results_list,
            activity_results_list=activity_results_list,
            output_dir_list=all_dirs,
            summary_output_dir=summary_output_dir,
        )
    except Exception as e:
        print(f"Warning: plot_geom_prod_vs_purity: {e}")
    
    print("\n" + "="*70)
    print(f"✓ All processing complete!")
    print(f"  - Processed: {len(purity_results_list)}/{len(all_dirs)} directories")
    print(f"  - Results saved to: {summary_output_dir}")
    print("="*70)


if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    parallel = '--sequential' not in sys.argv
    aggregate_only = '--aggregate-only' in sys.argv
    
    # Check for worker count
    n_workers = 4
    for arg in sys.argv:
        if arg.startswith('--workers='):
            try:
                n_workers = int(arg.split('=')[1])
            except ValueError:
                pass
    
    main(parallel=parallel, aggregate_only=aggregate_only, n_workers=n_workers)

