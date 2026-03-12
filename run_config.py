"""
DASHBOARD : Edit RUN_CASES to set run cases.
Run using python run.py cycles through each case, saves to runcase_1, runcase_2, ...
"""

import os

# ==================== SHARED (all cases) ====================
RUN_BASE_DIR = os.path.abspath(os.getcwd())
TARGET_HEIGHT_CM = 100.0
INNER_RADIUS_CM = 5.0
SOURCE_NEUTRON_ENERGY_MEV = 14.1
# Fusion source: 5e13 n/s; power from 14.1 MeV neutron (not 17.6 MeV total D-T Q-value)
SOURCE_STRENGTH = 5.0e13  # n/s
FUSION_POWER_W = SOURCE_STRENGTH * (SOURCE_NEUTRON_ENERGY_MEV * 1e6 * 1.602e-19)  # W

# Economics (shared)
ECON_CU64_PRICE_PER_MCI = 15.0
ECON_CU67_PRICE_PER_MCI = 60.0
ECON_SHINE_SOURCE_PURCHASE = 5_000_000.0   # FLARE one-time cost [USD]; revenue/NPV use this + feedstock
ECON_SHINE_SOURCE_OPERATIONS_ANNUAL = 1_000_000.0  # OPEX [USD/yr]
ECON_IRRAD_HOURS = 26280  # 3 years (e.g. Zn-65 1-yr plots)
ECON_COOLDOWN_DAYS = 1
ECON_MARKET_CU64_CI_PER_YEAR = 2000.0  # for production_vs_purity "global demand" annotation

# Simple analyze (shared)
IRRADIATION_HOURS = [1, 4, 8, 12, 16, 24, 72, 98, 8760, 26280]  # 12, 16 for 8/12/16h prod-vs-purity; 98 h for purity shelf-life; 26280 = 3 years
COOLDOWN_DAYS = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7]  # 1.5 for 8h-by-cooldown plot; extended for purity fall-to-99.9%

# FLARE NPV: load production (g/yr) from cu_summary and run financial analyses
RUN_FLARE_NPV = True
# Contingency: only natural (49.17%), 71%, 99% Zn-64; fixed $/kg ($1000, $3151, $12121). No interpolation.
# When True, flare_npv filters to these 3 enrichments and uses contingency loading cost; NPV only at these points.
FLARE_NPV_CONTINGENCY = True
# Which irradiation duration to use for NPV production rates (g/yr is scaled from this run length)
# Options: 1, 4, 8, 24, 72, 98, 8760, 26280 (must match a value in IRRADIATION_HOURS used in simple_analyze)
NPV_IRRAD_HOURS = 1  # use 1 h irradiation rates (scale-up); set to 4, 8, 8760, etc. to use those runs
# Cooldown for NPV production basis: 0 = EOI activity only (no decay); use 0 so g/yr uses mCi at end of irradiation
NPV_COOLDOWN_DAYS = 0
# When True, Cu-64 revenue only when purity >= 99.9% and plot titles show purity; applied to sell_all and market_cap
NPV_PURITY_CAP_64 = False

# RUN CASES

# Add or remove dicts; each runs to its own runcase_N folder.
# Keys override shared defaults. Required: RUN_MODE, thicknesses, enrichments.
# Optional: COMPLEX_GEOM = True — cylindrical stack inside 3 ft vacuum box, 4 in Al, ~19k gal water, concrete.
# Optional: RUN_BOTH_GEOM = True — run each geometry twice (normal and complex) for this case.

RUN_CASES = [
    # Case 1: Single Zn-64 (Cu-64 production, outer chamber only)
    {
        'name': 'runcase_64',
        'RUN_MODE': 'single_zn64',
        'ZN64_ENRICHMENTS': [0.4917, 0.53, 0.71, 0.76, 0.81, 0.91, 0.99,0.999],  # natural, 71%, 99% for FLARE_NPV_CONTINGENCY
        'ZN67_ENRICHMENTS': [0.0404],
        'Z_INNER_THICKNESSES': [0],
        'Z_OUTER_THICKNESSES': [1, 5, 10, 15, 20],
        'STRUCT_THICKNESSES': [0.5],
        'BORON_THICKNESSES': [1],
        'MULTI_THICKNESSES': [0],
        'MODERATOR_THICKNESSES': [0],
        'PARTICLES': int(1e3),
        'BATCHES': 10,
        'OUTPUT_PREFIX': 'irrad_output',
        'RUN_PARALLEL': True,
        'MAX_JOBS': 4,
        'ZN_WASTE_CASE_INDEX': 0,
        # Zn-65 waste: nat Zn (49.2%), 10 cm Zn, 0.5 struct, 1 boron
        'ZN_WASTE_CASE_DIR': 'irrad_output_single_cu64_inner0_outer10_struct0.5_boron1_multi0_moderator0_zn64_49.2%',
    },

    #Case 2: Single Zn-67 (Cu-67 production, outer chamber only)
    #{
    #    'name': 'runcase_67',
    #    'RUN_MODE': 'single_zn67',
    #    'ZN64_ENRICHMENTS': [0.4917],
    #    'ZN67_ENRICHMENTS': [0.0404, 0.073, 0.117],
    #    'Z_INNER_THICKNESSES': [0],
    #    'Z_OUTER_THICKNESSES': [1, 5, 10, 15, 20],
    #    'STRUCT_THICKNESSES': [0.5],
    #    'BORON_THICKNESSES': [1],
    #    'MULTI_THICKNESSES': [0],
    #    'MODERATOR_THICKNESSES': [0],
    #    'PARTICLES': int(1e3),
    #    'BATCHES': 10,
    #    'OUTPUT_PREFIX': 'irrad_output',
    #    'RUN_PARALLEL': True,
    #    'MAX_JOBS': 4,
    #    'ZN_WASTE_CASE_INDEX': 0,
    #    'ZN_WASTE_CASE_DIR': 'irrad_output_single_cu67_inner0_outer10_struct0.5_boron1_multi0_moderator0_zn67_4.0%',
    #},

    # Case 3: Complex geometry 
    #{
    #    'name': 'runcase_complex',
    #    'RUN_MODE': 'single_zn64',
    #    'ZN64_ENRICHMENTS': [0.4917],
    #    'ZN67_ENRICHMENTS': [0.0404],
    #    'Z_INNER_THICKNESSES': [0],
    #    'Z_OUTER_THICKNESSES': [10],
    #    'STRUCT_THICKNESSES': [0.5],
    #    'BORON_THICKNESSES': [1],
    #    'MULTI_THICKNESSES': [0],
    #    'MODERATOR_THICKNESSES': [0],
    #    'PARTICLES': int(1e3),
    #    'BATCHES': 10,
    #    'OUTPUT_PREFIX': 'irrad_output',
    #    'RUN_BOTH_GEOM': True,
    #    'RUN_PARALLEL': False,
    #    'ZN_WASTE_CASE_INDEX': 0,
    #    'ZN_WASTE_CASE_DIR': 'irrad_output_single_cu64_inner0_outer10_struct0.5_boron1_multi0_moderator0_zn64_49.2%',
    #},

]
