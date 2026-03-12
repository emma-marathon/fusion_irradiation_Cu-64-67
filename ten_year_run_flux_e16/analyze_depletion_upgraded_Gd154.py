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
import numpy as np
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
output_dir = "irradiation_output"
results_path = os.path.join(output_dir, "depletion_results.h5")
chain_file = os.path.join(output_dir, "JENDL_chain.xml")
material_id = "1"
statepoint_batches = 10
isotopes_to_plot = [f"Gd{i}" for i in range(148, 161)]

openmc.config['chain_file'] = chain_file

if not os.path.exists(results_path):
    raise FileNotFoundError(f"Missing {results_path}")

# -------------------------------
# ---- Load depletion results ----
# -------------------------------
results = openmc.deplete.Results(results_path)
time = np.array(results.get_times())
time_days = time / 86400 if np.max(time) > 1e5 else np.arange(len(time)) * 30
total_days = time_days[-1]

print("="*70)
print("OpenMC Depletion Analysis")
print("="*70)
print(f"Loaded results from: {results_path}")
print(f"Number of time steps: {len(time_days)}")
print(f"Total irradiation time: {total_days:.1f} days")
print("="*70)


# -------------------------------
# ---- Export final composition ----
# -------------------------------
print("\nExtracting final composition...")

# --- Robustly find all nuclides tracked ---
try:
    all_nuclides = results.nuclides  # old API (<=0.14)
except AttributeError:
    # new API: read directly from file
    import h5py
    with h5py.File(results_path, 'r') as f:
        all_nuclides = [n.decode() if isinstance(n, bytes) else str(n)
                        for n in np.array(f['nuclides'])]

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

total_atoms = sum(final_step_atoms.values())

df_final = (
    pd.DataFrame({
        "Nuclide": list(final_step_atoms.keys()),
        "Number_of_Atoms": list(final_step_atoms.values())
    })
    .assign(Percentage=lambda x: 100 * x["Number_of_Atoms"] / total_atoms)
    .sort_values("Number_of_Atoms", ascending=False)
)

final_csv = os.path.join(output_dir, "final_composition.csv")
df_final.to_csv(final_csv, index=False)
print(f"→ Exported final composition to {final_csv}")

# Show top 10 in terminal
print("\nTop 10 isotopes in final composition:")
print(df_final.head(10).to_string(index=False))

# -------------------------------
# ---- Plot top 20 isotope traces ----
# -------------------------------
top20 = df_final.head(20)["Nuclide"].tolist()
plt.figure(figsize=(9,6))
for iso in top20:
    try:
        _, atoms = results.get_atoms(material_id, iso)
        plt.semilogy(time_days, atoms, label=iso)
    except KeyError:
        continue

plt.xlabel("Irradiation time [days]")
plt.ylabel("Number of atoms")
plt.title("Time Evolution of Top 20 Final Isotopes")
plt.legend(fontsize=8, ncol=2)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
out_path = os.path.join(output_dir, "top20_isotopes_evolution.png")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"→ Saved top 20 isotope evolution plot → {out_path}")

# -------------------------------
# ---- Plot Gd isotopes evolution ----
# -------------------------------
plt.figure(figsize=(8,6))
for iso in isotopes_to_plot:
    try:
        _, atoms = results.get_atoms(material_id, iso)
        plt.semilogy(time_days, atoms, label=iso)
    except KeyError:
        continue

plt.xlabel("Irradiation time [days]")
plt.ylabel("Number of atoms")
plt.title("Evolution of Gadolinium Isotopes (148–160)")
plt.legend(ncol=2, fontsize=9)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
gd_path = os.path.join(output_dir, "gd_isotope_evolution.png")
plt.savefig(gd_path, dpi=300)
plt.close()
print(f"→ Saved Gd isotope evolution plot → {gd_path}")

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
# ---- Plot most common reactions ----
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

            # --- Integrate over time, nuclides, and materials for each reaction ---
            # result: one number per reaction
            integrated_rates = np.sum(rate_array[:, mat_index, :, :], axis=(0, 2)).flatten()

            # --- Sanity check ---
            if len(integrated_rates) != len(reaction_names):
                print(f"⚠️ Reaction count mismatch: {len(reaction_names)} names vs {len(integrated_rates)} rates")
                min_len = min(len(reaction_names), len(integrated_rates))
                reaction_names = reaction_names[:min_len]
                integrated_rates = integrated_rates[:min_len]

            # --- Build DataFrame ---
            df_rxn = pd.DataFrame({
                "Reaction": reaction_names,
                "Integrated_Rate": integrated_rates
            }).sort_values("Integrated_Rate", ascending=False)

            csv_path = os.path.join(output_dir, "reaction_summary.csv")
            df_rxn.to_csv(csv_path, index=False)
            print(f"→ Exported reaction summary to {csv_path}")

            # --- Plot top reactions ---
            plt.figure(figsize=(8,5))
            top10 = df_rxn.head(10)
            plt.barh(top10["Reaction"][::-1],
                     top10["Integrated_Rate"][::-1],
                     color="steelblue")
            plt.xlabel("Integrated Reaction Rate")
            plt.title("Top 10 Reaction Channels")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_reactions.png"), dpi=300)
            plt.close()
            print("→ Saved top reaction plot")

        else:
            print("⚠️ Reaction data not found in depletion_results.h5")

except Exception as e:
    print(f"Could not extract reaction summary: {e}")


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


# =====================================================================
# Main reaction breakdown analysis
# =====================================================================

try:
    with h5py.File(results_path, "r") as f:
        if all(k in f for k in ["reactions", "reaction rates", "nuclides"]):
            # ---- Load nuclide names ----
            raw_nuc = np.array(f["nuclides"])
            nuclide_names = [n.decode() if isinstance(n, bytes) else str(n)
                             for n in raw_nuc.flatten().tolist()]

            # ---- Load and squeeze reaction rates ----
            rate_array = np.squeeze(np.array(f["reaction rates"]))
            print(f"Reaction rate array shape: {rate_array.shape}")
            ndim = rate_array.ndim

            # ---- Integrate over time ----
            if ndim == 4:
                # (time, mat, reaction, nuclide)
                mat_keys = list(f["materials"].keys())
                mat_index = mat_keys.index(material_id) if material_id in mat_keys else 0
                integrated_rxn_nuc = np.sum(rate_array[:, mat_index, :, :], axis=0)
            elif ndim == 3:
                # (time, reaction, nuclide)
                integrated_rxn_nuc = np.sum(rate_array, axis=0)
            else:
                raise ValueError(f"Unexpected shape for reaction rates: {rate_array.shape}")

            n_rxn, n_nuc = integrated_rxn_nuc.shape
            print(f"Detected {n_rxn} reactions × {n_nuc} nuclides")

            # ---- Fix name mismatches ----
            if len(nuclide_names) != n_nuc:
                print(f"⚠️ Nuclide name count ({len(nuclide_names)}) != {n_nuc}. Using generic labels.")
                nuclide_names = [f"Nuclide_{i}" for i in range(n_nuc)]

            # ---- Build nuclide–reaction labels ----
            chain_path = os.path.join(output_dir, "JENDL_chain.xml")
            reaction_names = build_nuclide_reaction_labels(chain_path, n_rxn, n_nuc)
            print(f"→ Built {len(reaction_names)} (nuclide, reaction) labels.")

            # ---- Identify top reactions ----
            total_per_rxn = np.sum(integrated_rxn_nuc, axis=1)
            top_rxn_idx = np.argsort(total_per_rxn)[-10:][::-1]
            top_reactions = [reaction_names[i] for i in top_rxn_idx]

            # ---- Build tidy dataframe with proper nuclide names ----
            records = []
            for r_idx in top_rxn_idx:
                reaction = reaction_names[r_idx]
                for n_idx, nuc in enumerate(nuclide_names):
                    val = integrated_rxn_nuc[r_idx, n_idx]
                    if val > 0:
                        records.append({
                            "Reaction": reaction,
                            "Product_Nuclide": nuc,
                            "Integrated_Rate": float(val)
                        })

            df_breakdown = pd.DataFrame(records)

            # ---- Group and sort ----
            df_breakdown = (
                df_breakdown.sort_values(["Reaction", "Integrated_Rate"],
                                         ascending=[True, False])
                .groupby("Reaction")
                .head(5)
                .reset_index(drop=True)
            )

            csv_path = os.path.join(output_dir, "reaction_isotope_breakdown.csv")
            df_breakdown.to_csv(csv_path, index=False)
            print(f"→ Exported reaction–product breakdown to {csv_path}")

            # ---- Color palette by reaction type ----
            reaction_types = [r.split("(")[-1].split(")")[0] for r in df_breakdown["Reaction"]]
            unique_types = sorted(set(reaction_types))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
            color_map = dict(zip(unique_types, colors))

            # ---- Plot grouped horizontal bars ----
            plt.figure(figsize=(9,6))
            for i, reaction in enumerate(top_reactions):
                subset = df_breakdown[df_breakdown["Reaction"] == reaction]
                rates = subset["Integrated_Rate"].values
                products = subset["Product_Nuclide"].values
                rx_type = reaction.split("(")[-1].split(")")[0]
                color = color_map.get(rx_type, "gray")
                left = np.cumsum(np.concatenate(([0], rates[:-1])))
                plt.barh([reaction]*len(rates), rates, left=left, color=color, edgecolor="black", alpha=0.8)
                if len(products) > 0:
                    plt.text(np.sum(rates)*1.02, i, products[0], va='center', fontsize=8)

            # ---- Final touches ----
            plt.xlabel("Integrated Reaction Rate")
            plt.ylabel("Reaction (nuclide + channel)")
            plt.title("Top 10 Reactions – Product Isotope Breakdown (Top 5 per Reaction)")
            plt.grid(True, axis="x", ls=":", alpha=0.5)
            plt.legend(handles=[plt.Line2D([0], [0], color=c, lw=6, label=t) for t, c in color_map.items()],
                       title="Reaction Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "reaction_isotope_breakdown.png"), dpi=300)
            plt.close()
            print("→ Saved improved reaction–product breakdown plot")

        else:
            print("⚠️ Reaction/nuclide data not found in depletion_results.h5")

except Exception as e:
    print(f"Could not compute reaction–isotope breakdown: {e}")

    