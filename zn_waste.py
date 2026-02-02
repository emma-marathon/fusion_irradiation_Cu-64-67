"""
Zn-65 Waste Analysis and Comparison with Lu-177m

Compares radiation safety, dose, storage, and shielding requirements between:
- Zn-65 (waste from Cu-64/Cu-67 fusion production)
- Lu-177m (known waste from medical Lu-177 production)

Reference data sources:
- ICRP Publication 119 (Dose Coefficients)
- IAEA Safety Standards (Exempt/Clearance levels)
- Medical isotope production literature
- NRC 10 CFR Part 20 (Occupational Dose Limits)
- NRC 10 CFR 35.92 (Decay-in-Storage)

================================================================================
REGULATORY FRAMEWORK AND KEY CITATIONS
================================================================================

1. DECAY-IN-STORAGE ELIGIBILITY (10 CFR 35.92)
----------------------------------------------
From NRC 10 CFR 35.92 - Decay-in-Storage:

"(a) A licensee may hold byproduct material with a physical half-life of 
LESS THAN OR EQUAL TO 120 DAYS for decay-in-storage before disposal 
without regard to its radioactivity if it—

    (1) Monitors byproduct material at the surface before disposal and 
        determines that its radioactivity cannot be distinguished from the 
        background radiation level with an appropriate radiation detection 
        survey meter set on its most sensitive scale and with no interposed 
        shielding; and

    (2) Removes or obliterates all radiation labels, except for radiation 
        labels on materials that are within containers and that will be 
        managed as biomedical waste after they have been released from 
        the licensee.

(b) A licensee shall retain a record of each disposal permitted under 
    paragraph (a) of this section in accordance with § 35.2092."

KEY IMPLICATION FOR THIS ANALYSIS:
- Zn-65 (t½ = 244 days) EXCEEDS 120-day threshold → NOT eligible for decay-in-storage
- Lu-177m (t½ = 160 days) EXCEEDS 120-day threshold → NOT eligible for decay-in-storage
- Lu-177 (t½ = 6.65 days) IS eligible for decay-in-storage
- Cu-64 (t½ = 12.7 hours) IS eligible for decay-in-storage
- Cu-67 (t½ = 61.83 hours) IS eligible for decay-in-storage

Both Zn-65 and Lu-177m require authorized disposition paths other than simple 
decay-in-storage under standard medical license provisions.

2. OCCUPATIONAL DOSE LIMITS (10 CFR 20.1201)
--------------------------------------------
From 10 CFR 20.1201 - Occupational dose limits for adults:

"(a) The licensee shall control the occupational dose to individual adults, 
except for planned special exposures under § 20.1206, to the following limits:

    (1) An annual limit, which is the more limiting of—
        (i) The total effective dose equivalent being equal to 5 rems (0.05 Sv); or
        (ii) The sum of the deep-dose equivalent and the committed dose 
             equivalent to any individual organ or tissue other than the 
             lens of the eye being equal to 50 rems (0.5 Sv).

    (2) The annual limits to the lens of the eye, to the skin of the whole body, 
        and to the skin of the extremities, which are:
        (i) A lens dose equivalent of 15 rems (0.15 Sv), and
        (ii) A shallow-dose equivalent of 50 rem (0.5 Sv) to the skin of the 
             whole body or to the skin of any extremity."

3. ANNUAL LIMIT ON INTAKE (ALI) VALUES
--------------------------------------
From 10 CFR 20 Appendix B - Table 1 (Occupational Values):

Zn-65 (Atomic No. 30):
    - Class: Y, all compounds
    - Oral Ingestion ALI: 4E+2 µCi (400 µCi = 14.8 MBq)
    - Inhalation ALI: 3E+2 µCi (300 µCi = 11.1 MBq)
    - Inhalation DAC: 1E-7 µCi/ml (3.7 Bq/ml)
    - Air Effluent Concentration: 4E-10 µCi/ml
    - Water Effluent Concentration: 5E-6 µCi/ml
    - Sewer Release: 5E-5 µCi/ml

Lu-177m (Atomic No. 71):
    - Class: W (see Lu-169)
    - Inhalation ALI: 7E+2 µCi (700 µCi = 25.9 MBq)
    - Inhalation DAC: 5E-8 µCi/ml
    - Air Effluent Concentration: 1E-5 µCi/ml
    - Water Effluent Concentration: 1E-4 µCi/ml

NOTE: The ALI represents the intake that would result in either:
    (a) A committed effective dose equivalent of 5 rem (50 mSv), or
    (b) A committed dose equivalent of 50 rem (500 mSv) to any organ

4. BIOLOGICAL SHIELD REQUIREMENTS
---------------------------------
From SHINE PSAR Section 4b.2 (Radioisotope Production Facility Biological Shield):

"The production facility biological shield (PFBS) provides a barrier to protect 
SHINE facility personnel, members of the public, and various components and 
equipment of the SHINE facility by reducing radiation exposure."

Key shielded areas include:
    - Supercell (for Mo extraction, purification, and packaging)
    - Process tanks
    - Pipe chases
    - Waste processing cells
    - UREX cell
    - Thermal denitration (TDN) cell
    - Pump room hot cell
    - PVVS cell
    - NGRS shielded cell

"The RPF biological shields are provided to ensure that the projected radiation 
dose rates and accumulated doses in occupied areas do not exceed the limits of 
10 CFR Part 20 and the guidelines of the facility ALARA program."

From ISG Augmenting NUREG-1537, Parts 1 and 2, Section 4b.2:
    - Shield design must maintain dose rates < 2.5 mrem/hr (25 µSv/hr) in 
      continuously occupied areas
    - Shield design must maintain dose rates < 100 mrem/hr (1 mSv/hr) at 
      facility boundary during normal operations

5. LOW-LEVEL RADIOACTIVE WASTE (LLRW) CLASSIFICATION
----------------------------------------------------
From 10 CFR 61.55 - Waste Classification:

LLRW is defined as radioactive waste NOT included in:
    - Spent nuclear fuel
    - High-level waste
    - Transuranic waste
    - Uranium and thorium mill tailings

Zn-65 and Lu-177m are classified as:
    - Class A waste (lowest activity, least restrictive disposal)
    - Can be disposed of in near-surface disposal facilities
    - Subject to 10 CFR 61 concentration limits

Class A Concentration Limits (10 CFR 61.55 Table 1):
    - Zn-65: Not specifically listed (falls under "all other nuclides")
    - General Class A limit: Activity concentration should allow decay to 
      unrestricted release levels within 100 years

6. ALARA PROGRAM REQUIREMENTS
-----------------------------
From 10 CFR 20.1101(b):

"The licensee shall use, to the extent practical, procedures and engineering 
controls based upon sound radiation protection principles to achieve occupational 
doses and doses to members of the public that are as low as is reasonably 
achievable (ALARA)."

ALARA investigation levels (typical facility administrative limits):
    - External dose: 100 mrem/month (1 mSv/month)
    - Internal dose: 40 DAC-hours per month
    - Extremity dose: 5 rem/month (50 mSv/month)

7. INHALATION CLASS DEFINITIONS (10 CFR 20 / ICRP)
-------------------------------------------------
From 10 CFR 20 Appendix B and ICRP Publication 30/134:

The inhalation class refers to the lung clearance half-time after inhalation
of airborne radioactive material (aerosol AMAD = 1 µm):

CLASS D (Day):
    - Clearance half-time: < 10 days
    - Material clears from pulmonary region quickly
    - Lower lung dose, higher systemic dose
    - Examples: Most soluble compounds (chlorides, nitrates)

CLASS W (Week):
    - Clearance half-time: 10-100 days
    - Moderate retention in lungs
    - Balanced lung/systemic dose contribution
    - Examples: Lu-177m (as oxide/hydroxide compounds)
    - Lu compounds: Oxides moderately solite in lung fluid

CLASS Y (Year):
    - Clearance half-time: > 100 days
    - Long retention in pulmonary region
    - Higher lung dose contribution
    - Examples: Zn-65 (ALL COMPOUNDS per 10 CFR 20)
    - Zinc metabolizes slowly from lung tissue

REGULATORY SIGNIFICANCE FOR Zn-65 vs Lu-177m:
    - Zn-65 is CLASS Y → Lower ALI for inhalation (300 µCi vs 700 µCi)
    - Zn-65 higher DAC (1E-7 µCi/ml) vs Lu-177m (5E-8 µCi/ml)
    - Both require respiratory protection in airborne contamination scenarios
    - Class Y means Zn-65 poses GREATER inhalation hazard per unit activity

ICRP Publication 134 (2016) Update:
    Modern ICRP uses absorption Types F/M/S instead of D/W/Y:
    - Type F (Fast): Similar to Class D, absorption half-time ~10 min
    - Type M (Moderate): Similar to Class W, ~140 days
    - Type S (Slow): Similar to Class Y, ~7000 days
    Zinc is assigned Type M for most compounds in ICRP 134.

8. HANDLING PROTOCOLS FOR Zn-65 (BGSU RSO / NRC Guidance)
---------------------------------------------------------
Reference: BGSU Radiation Safety Office - Zn-65 Fact Sheet

RADIATION CHARACTERISTICS:
    - Half-life: 243.8 days
    - Beta: 330 keV (max)
    - Gamma: 1,116 keV (50.6% intensity)
    - Critical organ: Bone marrow
    - Biological half-life: ~933 days (bone)

SHIELDING REQUIREMENTS:
    - Beta: 0.1 cm Plexiglas or 0.05 cm Aluminum
    - Gamma/X-ray: 16 cm Concrete OR 3 cm Lead minimum
    - Use lead bricks for storage of significant quantities
    - Lead shielding MANDATORY for gamma protection

HANDLING PROCEDURES:
    - Use tongs/forceps to maintain distance from sources
    - Never handle with bare hands
    - Work behind lead shielding or in shielded enclosure
    - Disposable gloves, lab coat, safety glasses required
    - Work over absorbent paper with plastic backing
    - Use secondary containment for liquid sources

DETECTION METHODS (in order of preference):
    1. Radiation survey meter with energy-compensated GM pancake detector
    2. Ion chamber survey meter
    3. Liquid scintillation counting (for wipe tests)
    Note: Standard GM may under-respond to high-energy gamma

DOSE LIMITS (10 CFR 20.1201):
    - Radiation workers: 5 rem/year whole body
    - Non-radiation workers: 0.1 rem/year
    - Pregnant workers: 0.5 rem total during pregnancy
    - Minors: 0.1 rem/year

SPILL PROCEDURES:
    1. Alert others and evacuate immediate area
    2. Establish 5-meter barrier minimum
    3. Mark area as radiation hazard
    4. Contact Radiation Safety Officer immediately
    5. Do NOT attempt cleanup without RSO guidance

9. WASTE DISPOSAL COMPANIES AND PATHWAYS
----------------------------------------
COMMERCIAL LLRW DISPOSAL FACILITIES:

Waste Control Specialists (WCS) - Andrews, Texas:
    - Texas Compact Waste Facility (CWF)
    - Only commercial facility for Class A, B, C LLRW in USA
    - Operational since 2012
    - Contact: www.wcstexas.com
    - Accepts: Medical, industrial, research waste

EnergySolutions - Clive, Utah:
    - Class A waste only
    - Large volume disposal for lower-activity waste
    - Contact: www.energysolutions.com

ESTIMATED DISPOSAL COSTS (WCS Texas - historical rates):
    - Class A Compactable: ~$35/cubic foot
    - Class A Non-Compactable: ~$130/cubic foot
    - Class A High Dose Rate: ~$222/cubic foot
    - Additional surcharges for curie content
    - Transportation costs separate

    NOTE: Current rates available from TCEQ:
    Texas Commission on Environmental Quality
    30 TAC §336.1310 (rate schedule)
    Contact: radmat@tceq.texas.gov or (512) 239-6466

WASTE BROKER SERVICES:

Perma-Fix Environmental Services:
    - Full-service radioactive waste broker
    - Medical isotope disposal specialty
    - Services: Characterization, packaging, transport, disposal
    - Global network: USA, Canada, Mexico, UK, EU
    - Contact: www.perma-fix.com

Studsvik:
    - Waste treatment and repository management
    - Processing services for volume reduction
    - Contact: www.studsvik.com

COST COMPARISON (Zn-65 vs Lu-177m):

Zn-65 DISPOSAL CHALLENGES:
    - Higher shielding requirement (3 cm Pb vs 0.2 cm for Lu-177m)
    - Heavier containers needed → higher shipping costs
    - Longer storage needed → higher interim storage costs
    - Less established pathway than medical Lu-177m

ESTIMATED COST FACTORS:
    Base disposal: Similar Class A rates apply to both
    Shielding surcharge: Zn-65 ~3-5x higher due to gamma energy
    Container/transport: Zn-65 ~2-3x higher (heavier shielding)
    Total estimated: Zn-65 may cost 2-4x more per GBq than Lu-177m

Lu-177m PATHWAY:
    - Medical isotope facilities have established contracts
    - Standard medical waste disposal protocols apply
    - Well-characterized waste stream
    - Economies of scale from high production volume

10. SHINE FACILITY COMPATIBILITY ASSESSMENT
-------------------------------------------
Reference: NUREG-2183, SHINE PSAR Chapter 11

SHINE FACILITY WASTE MANAGEMENT:
    - Designed for Mo-99 production (~3000 6-day Ci/week)
    - Includes comprehensive radioactive waste management system
    - LLRW generation from target solution processing
    - Waste processing cells with appropriate shielding

SHIELDED AREAS IN SHINE RPF:
    - Supercell (Mo extraction, purification, packaging)
    - Process tanks
    - Pipe chases
    - Waste processing cells
    - UREX cell
    - Thermal denitration (TDN) cell
    - Pump room hot cell
    - PVVS cell
    - NGRS shielded cell

COMPATIBILITY FOR Zn-65:

The SHINE facility's biological shield design (PFBS) provides:
    - Shielding for high-activity fission product handling
    - Hot cells designed for gamma-emitting isotopes
    - ALARA program with dose rate targets

HOWEVER, KEY DIFFERENCES FOR Zn-65:
    1. GAMMA ENERGY: Zn-65 (1.116 MeV) vs typical fission products
       - SHINE shields optimized for fission spectrum (mostly <1 MeV)
       - May need additional shielding for Zn-65 gamma
    
    2. HALF-LIFE: Zn-65 (244 days) vs short-lived fission products
       - Most SHINE waste decays quickly (hours-days)
       - Zn-65 requires longer interim storage before disposal
    
    3. CHEMICAL FORM: Metallic Zn vs aqueous target solution
       - Different handling requirements
       - Different contamination/cleanup protocols

CONCLUSION - FACILITY ADEQUACY:
    - SHINE-type hot cells COULD handle Zn-65 with modifications
    - Additional lead shielding likely needed for processing areas
    - Separate interim storage facility needed for long decay
    - New waste disposal contracts would be required
    - Existing SHINE waste pathways do NOT automatically apply

RECOMMENDATION:
    For fusion-produced Zn-65 at GBq levels:
    - Use 3+ cm lead shielding for handling operations
    - Plan for 5-10 year interim storage before disposal
    - Establish relationship with licensed waste broker (e.g., Perma-Fix)
    - Budget 2-4x more for disposal vs equivalent Lu-177m activity

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================
# Physical Constants and DCFs
# ============================================
#
# Reference: 10 CFR 20 Appendix B, ICRP Publication 119
#
# ALI (Annual Limit on Intake) represents intake causing:
#   - 5 rem (50 mSv) committed effective dose equivalent, OR
#   - 50 rem (500 mSv) committed dose equivalent to any organ
#
# DAC (Derived Air Concentration) = ALI / (2000 hr × 1.2 m³/hr)
#   where 2000 hr = working hours/year, 1.2 m³/hr = breathing rate
# ============================================

# Half-lives (seconds)
# KEY REGULATORY NOTE: 10 CFR 35.92 allows decay-in-storage ONLY for t½ ≤ 120 days
HALF_LIVES = {
    'Zn65': 244.0 * 86400,      # 244 days - EXCEEDS 120d limit, requires authorized disposition
    'Lu177m': 160.4 * 86400,    # 160.4 days - EXCEEDS 120d limit, requires authorized disposition
    'Lu177': 6.647 * 86400,     # 6.647 days - Eligible for decay-in-storage
    'Cu64': 12.7 * 3600,        # 12.7 hours - Eligible for decay-in-storage
    'Cu67': 61.83 * 3600,       # 61.83 hours - Eligible for decay-in-storage
}

# Decay-in-storage eligibility (10 CFR 35.92)
DECAY_IN_STORAGE_LIMIT_DAYS = 120  # Half-life must be ≤ 120 days
DECAY_IN_STORAGE_ELIGIBLE = {
    iso: (hl / 86400) <= DECAY_IN_STORAGE_LIMIT_DAYS 
    for iso, hl in HALF_LIVES.items()
}

# Decay constants (s^-1)
DECAY_CONSTANTS = {iso: np.log(2) / hl for iso, hl in HALF_LIVES.items()}

# Annual Limit on Intake (ALI) - from 10 CFR 20 Appendix B Table 1
# ALI represents intake causing 5 rem (50 mSv) CEDE or 50 rem to any organ
ALI_INGESTION_uCi = {
    'Zn65': 400,      # 4E+2 µCi - Class Y, all compounds
    'Lu177m': 700,    # 7E+2 µCi - Class W (see Lu-169)
    'Lu177': 2000,    # 2E+3 µCi
}

ALI_INHALATION_uCi = {
    'Zn65': 300,      # 3E+2 µCi - Class Y
    'Lu177m': 700,    # 7E+2 µCi - Class W, Bone surface (1E+1), LLI wall (1E+2)
    'Lu177': 2000,    # 2E+3 µCi
}

# Derived Air Concentration (DAC) - from 10 CFR 20 Appendix B
# DAC = ALI / (2000 work-hours × 1.2 m³/hr breathing rate)
DAC_uCi_per_ml = {
    'Zn65': 1e-7,     # 1E-7 µCi/ml
    'Lu177m': 5e-8,   # 5E-8 µCi/ml
    'Lu177': 9e-7,    # 9E-7 µCi/ml
}

# Effluent Concentration Limits - from 10 CFR 20 Appendix B Table 2
# These are concentrations that, if inhaled/ingested continuously for 1 year,
# would produce 50 mrem (0.5 mSv) total effective dose equivalent
EFFLUENT_AIR_uCi_per_ml = {
    'Zn65': 4e-10,    # 4E-10 µCi/ml
    'Lu177m': 1e-5,   # 1E-5 µCi/ml (significantly higher due to lower gamma)
}

EFFLUENT_WATER_uCi_per_ml = {
    'Zn65': 5e-6,     # 5E-6 µCi/ml
    'Lu177m': 1e-4,   # 1E-4 µCi/ml
}

# Sewer release limits - from 10 CFR 20 Appendix B Table 3
SEWER_RELEASE_uCi_per_ml = {
    'Zn65': 5e-5,     # 5E-5 µCi/ml (Monthly Average Concentration)
    'Lu177m': 1e-4,   # 1E-4 µCi/ml
}

# Dose Conversion Factors (Sv/Bq) - ICRP 119
# These relate activity intake to committed effective dose
DCF_INGESTION = {
    'Zn65': 3.9e-9,   # Higher due to bone uptake (Zn metabolism)
    'Lu177m': 1.7e-9, # Lower, but bone surface dose is limiting organ
    'Lu177': 4.7e-10,
}

DCF_INHALATION = {
    'Zn65': 2.9e-9,   # Class Y (Year retention)
    'Lu177m': 1.0e-8, # Class W (Week retention) - higher lung dose
    'Lu177': 1.0e-9,
}

# External dose rate coefficient (µSv/hr at 1m per MBq)
# Based on gamma energies and intensities
# Reference: ICRP Publication 74, Radionuclide Transformations
EXTERNAL_DOSE_RATE = {
    'Zn65': 0.29,     # High due to 1.116 MeV gamma (50.6% intensity)
    'Lu177m': 0.018,  # Lower energy gammas (208 keV, 113 keV)
    'Lu177': 0.002,   # Very low (208 keV, 6.4% intensity)
}

# Gamma energies for shielding calculations
# HVL = Half-Value Layer (thickness to reduce intensity by 50%)
# Reference: NCRP Report 151, Structural Shielding Design
GAMMA_ENERGIES = {
    'Zn65': {
        'E_MeV': 1.116, 
        'intensity': 0.506, 
        'HVL_Pb_cm': 1.1,      # Lead HVL at 1.1 MeV
        'HVL_concrete_cm': 6.1  # Standard concrete (2.35 g/cm³)
    },
    'Lu177m': {
        'E_MeV': 0.208, 
        'intensity': 0.11, 
        'HVL_Pb_cm': 0.2,      # Much easier to shield
        'HVL_concrete_cm': 2.5
    },
}

# Regulatory limits (IAEA BSS, 10 CFR 20, 10 CFR 61)
# Exempt Activity: Below this, material can be released without regulatory control
EXEMPT_ACTIVITY_Bq = {
    'Zn65': 1e4,      # 10 kBq (IAEA RS-G-1.7)
    'Lu177m': 1e4,    # 10 kBq
}

# Clearance Level: Activity concentration below which material can be cleared
# This is the target for decay-in-storage to reach
CLEARANCE_LEVEL_Bq_per_g = {
    'Zn65': 100,      # 100 Bq/g (IAEA RS-G-1.7)
    'Lu177m': 100,    # 100 Bq/g
}

# 10 CFR 20.1201 Occupational Dose Limits (annual)
OCCUPATIONAL_LIMITS = {
    'TEDE_rem': 5,            # Total Effective Dose Equivalent
    'organ_rem': 50,          # Committed dose to any organ
    'lens_rem': 15,           # Lens of eye
    'skin_rem': 50,           # Shallow dose to skin
}

# Dose rate limits for shielded areas (SHINE PSAR criteria)
DOSE_RATE_LIMITS = {
    'continuously_occupied_mrem_hr': 2.5,    # 25 µSv/hr
    'controlled_area_mrem_hr': 100,          # 1 mSv/hr
    'unrestricted_area_mrem_hr': 2,          # 20 µSv/hr (public)
}

# Inhalation Classes (10 CFR 20 / ICRP Publication 30)
# Clearance half-time from pulmonary region:
#   Class D: < 10 days
#   Class W: 10-100 days
#   Class Y: > 100 days
INHALATION_CLASS = {
    'Zn65': 'Y',       # ALL zinc compounds - long lung retention
    'Lu177m': 'W',     # Lutetium oxides/hydroxides - moderate retention
    'Lu177': 'W',      # Same as Lu-177m
}

# Lung clearance half-times (days) - typical values
LUNG_CLEARANCE_HALFLIFE_DAYS = {
    'Zn65': 500,       # Class Y - very slow clearance
    'Lu177m': 50,      # Class W - moderate clearance
}

# ============================================
# WASTE DISPOSAL COSTS (Reference: WCS Texas / TCEQ)
# ============================================
# NOTE: These are historical/estimated values. Contact TCEQ for current rates.
# Texas Commission on Environmental Quality: 30 TAC §336.1310
# Phone: (512) 239-6466, Email: radmat@tceq.texas.gov

DISPOSAL_COST_PER_CUFT = {
    'Class_A_compactable': 35,       # $/ft³
    'Class_A_noncompactable': 130,   # $/ft³
    'Class_A_high_dose': 222,        # $/ft³
}

# Surcharge factors (estimated multipliers based on activity/shielding)
DISPOSAL_SURCHARGE_FACTORS = {
    'Zn65': {
        'shielding_factor': 3.5,     # Higher due to 1.116 MeV gamma
        'container_factor': 2.0,      # Heavier shielded containers
        'transport_factor': 1.5,      # Heavier packages
    },
    'Lu177m': {
        'shielding_factor': 1.0,     # Baseline (lower energy gamma)
        'container_factor': 1.0,      # Standard containers
        'transport_factor': 1.0,      # Standard packages
    },
}

# Estimated total cost multiplier (Zn-65 vs Lu-177m baseline)
ESTIMATED_COST_MULTIPLIER_ZN65 = 2.5  # Zn-65 costs ~2-4x more than Lu-177m

# Waste broker contact information
WASTE_BROKERS = {
    'Perma-Fix': {
        'website': 'www.perma-fix.com',
        'services': 'Full-service radioactive waste broker, medical isotope specialty',
        'regions': 'USA, Canada, Mexico, UK, EU',
    },
    'WCS_Texas': {
        'website': 'www.wcstexas.com',
        'services': 'Class A/B/C LLRW disposal facility',
        'location': 'Andrews, Texas',
    },
    'EnergySolutions': {
        'website': 'www.energysolutions.com',
        'services': 'Class A LLRW disposal',
        'location': 'Clive, Utah',
    },
}

# Typical medical Lu-177m production rates (for comparison)
# A typical Lu-177 production facility produces ~50-200 GBq Lu-177 per batch
# Lu-177m is ~0.01-0.1% of Lu-177 activity produced
TYPICAL_LU177M_PER_BATCH_GBq = 0.05  # ~50 MBq Lu-177m per production batch
BATCHES_PER_YEAR = 200  # Typical production facility
ANNUAL_LU177M_GBq = TYPICAL_LU177M_PER_BATCH_GBq * BATCHES_PER_YEAR  # ~10 GBq/year


def read_zn65_from_simulation(sim_dir, irradiation_hours=8760, cooldown_days=0):
    """
    Read Zn-65 data from a simulation output directory.
    
    Parameters:
    -----------
    sim_dir : str
        Path to simulation directory (e.g., 'simple_output_inner0_outer20_struct0_multi0_moderator0_zn99.0%')
    irradiation_hours : float
        Irradiation time in hours (default: 8760 = 1 year)
    cooldown_days : float
        Cooldown time in days after irradiation (default: 0)
    
    Returns:
    --------
    dict : {'zn65_activity_Bq': float, 'zn65_mass_g': float, 'params': dict}
    """
    import os
    import glob
    
    # Find statepoint file
    sp_patterns = [
        os.path.join(sim_dir, 'statepoint.*.h5'),
        os.path.join(sim_dir, 'statepoint.h5'),
    ]
    
    sp_file = None
    for pattern in sp_patterns:
        matches = glob.glob(pattern)
        if matches:
            sp_file = sorted(matches)[-1]  # Latest statepoint
            break
    
    if sp_file is None:
        raise FileNotFoundError(f"No statepoint file found in {sim_dir}")
    
    print(f"Reading from: {sp_file}")
    
    # Import OpenMC and utilities
    import openmc
    from utilities import build_channel_rr_per_s, compute_volumes_from_dir_name
    
    # Parse directory name for parameters
    dir_name = os.path.basename(sim_dir)
    params = {}
    
    try:
        if '_inner' in dir_name:
            params['inner'] = float(dir_name.split('_inner')[1].split('_')[0])
        else:
            params['inner'] = 0
        
        if '_outer' in dir_name:
            params['outer'] = float(dir_name.split('_outer')[1].split('_')[0])
        else:
            params['outer'] = 20
        
        if '_struct' in dir_name:
            params['struct'] = float(dir_name.split('_struct')[1].split('_')[0])
        else:
            params['struct'] = 0
        
        if '_multi' in dir_name:
            params['multi'] = float(dir_name.split('_multi')[1].split('_')[0])
        else:
            params['multi'] = 0
        
        if '_moderator' in dir_name:
            params['moderator'] = float(dir_name.split('_moderator')[1].split('_')[0])
        else:
            params['moderator'] = 0
        
        if '_zn' in dir_name:
            zn_str = dir_name.split('_zn')[1].replace('%', '')
            params['zn_enrichment'] = float(zn_str) / 100.0
        else:
            params['zn_enrichment'] = 0.486
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not fully parse directory name: {e}")
    
    # Get volumes
    volumes = compute_volumes_from_dir_name(dir_name)
    outer_volume_cm3 = volumes.get(1, 0)  # Outer target material ID = 1
    
    # Get density from summary.h5 (fallback to natural Zn 7.14 g/cm³)
    zn_density = get_material_density_from_statepoint(sp_file, material_id=1)
    if zn_density is None:
        zn_density = 7.14  # g/cm³ natural Zn
    zn_mass_g = outer_volume_cm3 * zn_density
    
    # Open statepoint and get reaction rates
    sp = openmc.StatePoint(sp_file)
    
    # Get Zn-65 production rate from reaction rates
    # Zn-65 is produced by: Zn64(n,γ)Zn65 and Zn66(n,2n)Zn65
    channel_rr = build_channel_rr_per_s(sp, cell_id=1, source_strength=5e13)
    
    zn65_production_rate = (
        channel_rr.get("Zn64 (n,gamma) Zn65", 0) +
        channel_rr.get("Zn66 (n,2n) Zn65", 0)
    )
    
    print(f"  Zn-65 production channels:")
    print(f"    Zn64(n,γ)Zn65: {channel_rr.get('Zn64 (n,gamma) Zn65', 0):.3e} atoms/s")
    print(f"    Zn66(n,2n)Zn65: {channel_rr.get('Zn66 (n,2n) Zn65', 0):.3e} atoms/s")
    print(f"    Total production: {zn65_production_rate:.3e} atoms/s")
    
    # Calculate Zn-65 activity using Bateman equation
    # For production + decay: N(t) = P/λ * (1 - exp(-λt))
    # Activity A(t) = λ * N(t) = P * (1 - exp(-λt))
    
    lam_zn65 = DECAY_CONSTANTS['Zn65']  # 1/s
    irrad_time_s = irradiation_hours * 3600
    
    # Atoms at end of irradiation (Bateman)
    if lam_zn65 > 0:
        zn65_atoms_eoi = (zn65_production_rate / lam_zn65) * (1 - np.exp(-lam_zn65 * irrad_time_s))
    else:
        zn65_atoms_eoi = zn65_production_rate * irrad_time_s
    
    zn65_activity_eoi_Bq = zn65_atoms_eoi * lam_zn65
    
    # Apply cooldown decay
    cooldown_s = cooldown_days * 86400
    zn65_activity_Bq = zn65_activity_eoi_Bq * np.exp(-lam_zn65 * cooldown_s)
    
    print(f"  Irradiation: {irradiation_hours} hours ({irradiation_hours/8760:.2f} years)")
    print(f"  Cooldown: {cooldown_days} days")
    print(f"  Zn-65 atoms at EOI: {zn65_atoms_eoi:.3e}")
    print(f"  Zn-65 activity at EOI: {zn65_activity_eoi_Bq:.3e} Bq ({zn65_activity_eoi_Bq/1e9:.3f} GBq)")
    print(f"  Zn-65 activity after cooldown: {zn65_activity_Bq:.3e} Bq ({zn65_activity_Bq/1e9:.3f} GBq)")
    print(f"  Zn mass: {zn_mass_g:.1f} g ({zn_mass_g/1000:.2f} kg)")
    
    return {
        'zn65_activity_Bq': zn65_activity_Bq,
        'zn65_mass_g': zn_mass_g,
        'zn65_production_rate': zn65_production_rate,
        'irradiation_hours': irradiation_hours,
        'cooldown_days': cooldown_days,
        'params': params,
        'volume_cm3': outer_volume_cm3,
    }


def calculate_activity_Bq(atoms, isotope):
    """Calculate activity in Bq from atom count."""
    lam = DECAY_CONSTANTS.get(isotope, 0)
    return atoms * lam


def calculate_activity_decay(initial_activity_Bq, isotope, time_days):
    """Calculate activity after decay time."""
    lam = DECAY_CONSTANTS.get(isotope, 0)
    time_s = time_days * 86400
    return initial_activity_Bq * np.exp(-lam * time_s)


def time_to_clearance(initial_activity_Bq, mass_g, isotope):
    """Calculate days to reach clearance level (100 Bq/g)."""
    clearance = CLEARANCE_LEVEL_Bq_per_g.get(isotope, 100)
    target_activity = clearance * mass_g
    
    if initial_activity_Bq <= target_activity:
        return 0
    
    lam = DECAY_CONSTANTS.get(isotope, 0)
    if lam <= 0:
        return np.inf
    
    # A(t) = A0 * exp(-λt) = target
    # t = -ln(target/A0) / λ
    t_s = -np.log(target_activity / initial_activity_Bq) / lam
    return t_s / 86400  # days


def shielding_thickness(activity_MBq, isotope, target_dose_rate_uSv_hr=2.5, material='Pb'):
    """
    Calculate shielding thickness to reduce dose rate to target.
    
    Parameters:
    -----------
    activity_MBq : float
        Source activity in MBq
    isotope : str
        Isotope name
    target_dose_rate_uSv_hr : float
        Target dose rate at 1m (default 2.5 µSv/hr for controlled area)
    material : str
        'Pb' (lead) or 'concrete'
    
    Returns:
    --------
    float : Required thickness in cm
    """
    unshielded_rate = EXTERNAL_DOSE_RATE.get(isotope, 0.1) * activity_MBq
    
    if unshielded_rate <= target_dose_rate_uSv_hr:
        return 0
    
    gamma_data = GAMMA_ENERGIES.get(isotope, {'HVL_Pb_cm': 1.0, 'HVL_concrete_cm': 5.0})
    
    if material == 'Pb':
        hvl = gamma_data['HVL_Pb_cm']
    else:
        hvl = gamma_data['HVL_concrete_cm']
    
    # Number of HVLs needed
    ratio = unshielded_rate / target_dose_rate_uSv_hr
    n_hvl = np.log2(ratio)
    
    return n_hvl * hvl


def create_comparison_plots(zn65_activity_Bq, zn65_mass_g, output_dir='.'):
    """
    Create focused waste comparison plots: Zn-65 vs Lu-177m.
    
    Based on EJNMMI Physics 2023 guidance on Lu-177m waste management:
    "Dealing with dry waste disposal issues associated with 177mLu impurities"
    
    Key comparison metrics (per radiation safety literature):
    1. Activity decay timeline (half-life comparison)
    2. External dose rate (primary operational concern)
    3. Time to clearance (regulatory milestone)
    4. ALI/DAC comparison (occupational limits)
    
    Parameters:
    -----------
    zn65_activity_Bq : float
        Zn-65 activity in Bq (from your simulation)
    zn65_mass_g : float
        Mass of Zn target material in grams
    output_dir : str
        Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to convenient units
    zn65_GBq = zn65_activity_Bq / 1e9
    zn65_mCi = zn65_activity_Bq / 3.7e7
    
    # Reference Lu-177m: Use EJNMMI Physics 2023 values
    # Typical Lu-177 patient treatment: 7.4 GBq, Lu-177m contamination ~0.02-0.04%
    # Annual facility waste estimate: ~10-50 GBq Lu-177m (10 GBq used here)
    lu177m_GBq = 10.0  # Reference: typical annual medical facility waste
    lu177m_Bq = lu177m_GBq * 1e9
    lu177m_mass_g = 100  # Typical waste mass scale
    
    print(f"\n{'='*70}")
    print("Zn-65 WASTE ANALYSIS vs Lu-177m REFERENCE")
    print(f"{'='*70}")
    print(f"Your Zn-65: {zn65_GBq:.3f} GBq ({zn65_mCi:.1f} mCi) in {zn65_mass_g:.0f}g")
    print(f"Reference Lu-177m: {lu177m_GBq:.1f} GBq (typical annual medical facility)")
    print(f"Source: EJNMMI Phys 2023, 10 CFR 20, ICRP 119")
    print(f"{'='*70}\n")
    
    # ========================================
    # SINGLE FIGURE: Key Comparisons (2x2)
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Zn-65 Waste Comparison: Your Sample vs Medical Lu-177m Reference\n'
                 f'Zn-65: {zn65_GBq:.2f} GBq ({zn65_mass_g:.0f}g) | Lu-177m: {lu177m_GBq:.1f} GBq (ref)\n'
                 f'Sources: ICRP 119, 10 CFR 20 Appendix B, EJNMMI Phys 2023', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Time array (0 to 5 years)
    days = np.linspace(0, 5*365, 500)
    
    # Calculate activity decay
    zn65_decay = calculate_activity_decay(zn65_activity_Bq, 'Zn65', days)
    lu177m_decay = calculate_activity_decay(lu177m_Bq, 'Lu177m', days)
    
    # --- Plot 1: Activity Decay Over Time ---
    ax = axes[0, 0]
    ax.semilogy(days/365, zn65_decay/1e9, 'b-', linewidth=2.5, label=f'Zn-65 (t½=244d)')
    ax.semilogy(days/365, lu177m_decay/1e9, 'r--', linewidth=2.5, label=f'Lu-177m (t½=160d)')
    
    # Clearance level lines
    zn_clearance_Bq = CLEARANCE_LEVEL_Bq_per_g['Zn65'] * zn65_mass_g
    lu_clearance_Bq = CLEARANCE_LEVEL_Bq_per_g['Lu177m'] * lu177m_mass_g
    ax.axhline(y=zn_clearance_Bq/1e9, color='b', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.axhline(y=lu_clearance_Bq/1e9, color='r', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Mark clearance time
    t_clear_zn = time_to_clearance(zn65_activity_Bq, zn65_mass_g, 'Zn65')
    t_clear_lu = time_to_clearance(lu177m_Bq, lu177m_mass_g, 'Lu177m')
    ax.axvline(x=t_clear_zn/365, color='b', linestyle=':', alpha=0.5)
    ax.axvline(x=t_clear_lu/365, color='r', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Activity (GBq)', fontsize=11, fontweight='bold')
    ax.set_title(f'(a) Activity Decay\n'
                 f'Zn-65 clearance: {t_clear_zn/365:.1f}y | Lu-177m: {t_clear_lu/365:.1f}y', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 5)
    
    # --- Plot 2: External Dose Rate (PRIMARY SAFETY METRIC) ---
    ax = axes[0, 1]
    zn65_external = (zn65_decay/1e6) * EXTERNAL_DOSE_RATE['Zn65']  # µSv/hr at 1m
    lu177m_external = (lu177m_decay/1e6) * EXTERNAL_DOSE_RATE['Lu177m']
    
    ax.semilogy(days/365, zn65_external, 'b-', linewidth=2.5, 
                label=f'Zn-65 (1.116 MeV, 50.6%)')
    ax.semilogy(days/365, lu177m_external, 'r--', linewidth=2.5, 
                label=f'Lu-177m (0.208 MeV, 11%)')
    
    # Reference levels per 10 CFR 20.1301
    ax.axhline(y=2.5, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
               label='Controlled area limit (2.5 µSv/hr)')
    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.8, linewidth=2, 
               label='Unrestricted area (0.5 µSv/hr)')
    
    ax.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Dose Rate at 1m, unshielded (µSv/hr)', fontsize=11, fontweight='bold')
    ax.set_title('(b) External Dose Rate (Key Safety Metric)\n'
                 'Zn-65 has 16x higher dose rate per GBq due to γ energy', 
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 5)
    
    # --- Plot 3: Shielding Requirement ---
    ax = axes[1, 0]
    activities_GBq = np.logspace(-2, 2, 100)
    pb_zn = [shielding_thickness(a*1000, 'Zn65', 2.5, 'Pb') for a in activities_GBq]
    pb_lu = [shielding_thickness(a*1000, 'Lu177m', 2.5, 'Pb') for a in activities_GBq]
    
    ax.semilogx(activities_GBq, pb_zn, 'b-', linewidth=2.5, 
                label=f'Zn-65 (HVL={GAMMA_ENERGIES["Zn65"]["HVL_Pb_cm"]:.2f} cm Pb)')
    ax.semilogx(activities_GBq, pb_lu, 'r--', linewidth=2.5, 
                label=f'Lu-177m (HVL={GAMMA_ENERGIES["Lu177m"]["HVL_Pb_cm"]:.2f} cm Pb)')
    
    # Mark your activity
    shield_pb_zn = shielding_thickness(zn65_GBq*1000, 'Zn65', 2.5, 'Pb')
    shield_pb_lu = shielding_thickness(lu177m_GBq*1000, 'Lu177m', 2.5, 'Pb')
    ax.plot(zn65_GBq, shield_pb_zn, 'bo', markersize=12, markeredgecolor='black', 
            label=f'Your Zn-65: {shield_pb_zn:.1f} cm')
    ax.plot(lu177m_GBq, shield_pb_lu, 'rs', markersize=12, markeredgecolor='black',
            label=f'Ref Lu-177m: {shield_pb_lu:.1f} cm')
    
    ax.set_xlabel('Activity (GBq)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Lead Thickness (cm)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Lead Shielding to 2.5 µSv/hr at 1m\n'
                 '(NCRP 151 HVL data)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(pb_zn) * 1.1)
    
    # --- Plot 4: Key Metrics Comparison Table ---
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate key metrics
    t_clear_zn = time_to_clearance(zn65_activity_Bq, zn65_mass_g, 'Zn65')
    t_clear_lu = time_to_clearance(lu177m_Bq, lu177m_mass_g, 'Lu177m')
    
    shield_pb_zn = shielding_thickness(zn65_GBq*1000, 'Zn65', 2.5, 'Pb')
    shield_pb_lu = shielding_thickness(lu177m_GBq*1000, 'Lu177m', 2.5, 'Pb')
    
    shield_conc_zn = shielding_thickness(zn65_GBq*1000, 'Zn65', 2.5, 'concrete')
    shield_conc_lu = shielding_thickness(lu177m_GBq*1000, 'Lu177m', 2.5, 'concrete')
    
    # Simplified table - focus on key metrics per EJNMMI Physics 2023 guidance
    # and 10 CFR 20 Appendix B
    table_data = [
        ['Metric', 'Zn-65', 'Lu-177m', 'Comment'],
        ['Half-life', '244 days', '160 days', 'Zn-65 persists 1.5x longer'],
        ['Your Activity', f'{zn65_GBq:.2f} GBq', f'{lu177m_GBq:.1f} GBq (ref)', f'{zn65_GBq/lu177m_GBq:.1f}x annual med.'],
        ['γ Energy', '1.116 MeV', '0.208 MeV', 'Zn-65 much harder to shield'],
        ['Dose Rate Factor', '0.29 µSv/hr/MBq', '0.018 µSv/hr/MBq', '16x higher for Zn-65'],
        ['Pb Shielding', f'{shield_pb_zn:.1f} cm', f'{shield_pb_lu:.1f} cm', f'{shield_pb_zn/max(shield_pb_lu,0.1):.0f}x more Pb needed'],
        ['Clearance Time', f'{t_clear_zn/365:.1f} years', f'{t_clear_lu/365:.1f} years', f'{t_clear_zn/t_clear_lu:.1f}x longer storage'],
        ['', '', '', ''],
        ['10 CFR 20 (Occ.)', '', '', ''],
        ['ALI (inhalation)', '300 µCi', '700 µCi', 'Zn-65 more restrictive'],
        ['Inhalation Class', 'Y (lung >100d)', 'W (10-100d)', 'Zn-65 longer lung retention'],
        ['Decay-in-Storage', 'NO (>120d)', 'NO (>120d)', 'Both need LLRW disposal'],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.28, 0.22, 0.22, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color section header row
    for i in range(4):
        table[(8, i)].set_facecolor('#E0E0E0')
        table[(8, i)].set_text_props(fontweight='bold')
    
    # Highlight key warning rows
    for i in range(4):
        table[(11, i)].set_facecolor('#FFCCCC')  # Light red - both need LLRW
    
    ax.set_title('(d) Key Comparison Summary\n(10 CFR 20, ICRP 119, EJNMMI Phys 2023)', 
                 fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, 'zn65_waste_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/zn65_waste_comparison.png")
    
    # ========================================
    # Print Summary Report with Key Metrics
    # ========================================
    print(f"\n{'='*70}")
    print("SUMMARY REPORT - Zn-65 WASTE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nYour Zn-65 Production:")
    print(f"  Activity: {zn65_GBq:.3f} GBq ({zn65_mCi:.1f} mCi)")
    print(f"  Mass: {zn65_mass_g:.1f} g")
    print(f"  Specific Activity: {zn65_activity_Bq/zn65_mass_g:.2e} Bq/g")
    
    print(f"\n{'='*70}")
    print("REGULATORY STATUS (10 CFR 35.92 - Decay-in-Storage)")
    print(f"{'='*70}")
    print(f"  Threshold for decay-in-storage: t½ ≤ 120 days")
    print(f"  Zn-65 half-life: 244 days → NOT ELIGIBLE for simple decay-in-storage")
    print(f"  Lu-177m half-life: 160 days → NOT ELIGIBLE for simple decay-in-storage")
    print(f"  ")
    print(f"  IMPLICATION: Both require authorized disposition pathway")
    print(f"               (LLRW disposal facility, licensed waste broker, etc.)")
    
    print(f"\n{'='*70}")
    print("COMPARISON TO MEDICAL Lu-177m WASTE")
    print(f"{'='*70}")
    print(f"  Typical annual medical Lu-177m waste: {lu177m_GBq:.1f} GBq")
    print(f"  Your Zn-65: {zn65_GBq:.2f} GBq")
    print(f"  Ratio: Your Zn-65 is {zn65_GBq/lu177m_GBq:.1f}x the annual Lu-177m production")
    
    print(f"\n{'='*70}")
    print("STORAGE REQUIREMENTS (to reach 100 Bq/g clearance)")
    print(f"{'='*70}")
    print(f"  Zn-65: {t_clear_zn/365:.1f} years to reach clearance level")
    print(f"  Lu-177m: {t_clear_lu/365:.1f} years to reach clearance level")
    print(f"  Factor: {t_clear_zn/t_clear_lu:.1f}x longer storage for Zn-65")
    
    print(f"\n{'='*70}")
    print("SHIELDING REQUIREMENTS (per SHINE PSAR / NUREG-1537)")
    print(f"{'='*70}")
    print(f"  Target: 2.5 µSv/hr (25 µrem/hr) at 1m for continuously occupied areas")
    print(f"  ")
    print(f"  Lead shielding:")
    print(f"    Zn-65: {shield_pb_zn:.1f} cm (due to 1.116 MeV gamma, 50.6% intensity)")
    print(f"    Lu-177m: {shield_pb_lu:.1f} cm (0.208 MeV gamma, 11% intensity)")
    print(f"    Factor: {shield_pb_zn/max(shield_pb_lu,0.1):.0f}x MORE lead for Zn-65")
    print(f"  ")
    print(f"  Concrete shielding:")
    print(f"    Zn-65: {shield_conc_zn:.1f} cm")
    print(f"    Lu-177m: {shield_conc_lu:.1f} cm")
    print(f"    Factor: {shield_conc_zn/max(shield_conc_lu,0.1):.0f}x MORE concrete for Zn-65")
    
    print(f"\n{'='*70}")
    print("10 CFR 20 APPENDIX B - OCCUPATIONAL LIMITS")
    print(f"{'='*70}")
    print(f"  Annual Limit on Intake (ALI) - Oral Ingestion:")
    print(f"    Zn-65: 400 µCi ({400*37:.0f} kBq)")
    print(f"    Lu-177m: 700 µCi ({700*37:.0f} kBq)")
    print(f"    Your Zn-65 = {zn65_mCi*1000/400:.1f} ALIs (ingestion)")
    print(f"  ")
    print(f"  Derived Air Concentration (DAC):")
    print(f"    Zn-65: 1E-7 µCi/ml")
    print(f"    Lu-177m: 5E-8 µCi/ml")
    
    print(f"\n{'='*70}")
    print("INHALATION CLASS COMPARISON (10 CFR 20 Appendix B)")
    print(f"{'='*70}")
    print(f"  Zn-65:  CLASS Y (Year) - ALL compounds")
    print(f"          Lung clearance half-time: >100 days (~500 days typical)")
    print(f"          → Higher lung dose per unit inhaled")
    print(f"          → Lower ALI (300 µCi) despite higher DAC")
    print(f"  ")
    print(f"  Lu-177m: CLASS W (Week)")
    print(f"           Lung clearance half-time: 10-100 days (~50 days)")
    print(f"           → Moderate lung retention")
    print(f"           → Higher ALI (700 µCi)")
    print(f"  ")
    print(f"  IMPLICATION: Zn-65 poses GREATER inhalation hazard per unit activity")
    print(f"               Respiratory protection more critical for Zn-65 aerosols")

    print(f"\n{'='*70}")
    print("WASTE DISPOSAL COST ESTIMATE")
    print(f"{'='*70}")
    print(f"  Reference: WCS Texas (30 TAC §336.1310), Perma-Fix Environmental")
    print(f"  ")
    print(f"  Base Class A disposal rates (historical):")
    print(f"    - Compactable: ~$35/ft³")
    print(f"    - Non-compactable: ~$130/ft³")
    print(f"    - High dose rate: ~$222/ft³")
    print(f"  ")
    print(f"  Zn-65 COST MULTIPLIERS vs Lu-177m baseline:")
    print(f"    - Shielding factor: ~3.5x (1.116 MeV gamma requires more Pb)")
    print(f"    - Container factor: ~2.0x (heavier shielded containers)")
    print(f"    - Transport factor: ~1.5x (heavier packages)")
    print(f"    - ESTIMATED TOTAL: Zn-65 costs ~2.5-4x more per GBq")
    print(f"  ")
    # Rough cost estimate
    volume_ft3 = 1.0  # Assume 1 ft³ waste volume
    base_cost_lu = DISPOSAL_COST_PER_CUFT['Class_A_high_dose']
    base_cost_zn = base_cost_lu * ESTIMATED_COST_MULTIPLIER_ZN65
    print(f"  For your {zn65_GBq:.2f} GBq Zn-65 (assuming 1 ft³ volume):")
    print(f"    Estimated disposal: ${base_cost_zn:.0f} - ${base_cost_zn*2:.0f}")
    print(f"    Equivalent Lu-177m: ${base_cost_lu:.0f} - ${base_cost_lu*1.5:.0f}")

    print(f"\n{'='*70}")
    print("SHINE FACILITY COMPATIBILITY")
    print(f"{'='*70}")
    print(f"  SHINE Mo-99 facility design (NUREG-2183, PSAR Section 4b.2):")
    print(f"    - Hot cells designed for fission product handling")
    print(f"    - Gamma shielding optimized for fission spectrum (<1 MeV mostly)")
    print(f"    - Short-lived waste streams (hours to days)")
    print(f"  ")
    print(f"  Zn-65 DIFFERENCES requiring facility modifications:")
    print(f"    1. GAMMA ENERGY: 1.116 MeV exceeds typical fission product energies")
    print(f"       → Additional 1-2 cm Pb may be needed in processing areas")
    print(f"    2. HALF-LIFE: 244 days vs hours/days for fission products")
    print(f"       → Separate interim storage needed (5-10 years)")
    print(f"    3. WASTE PATHWAY: Not covered by existing SHINE disposal contracts")
    print(f"       → New broker relationship required (recommend Perma-Fix)")
    print(f"  ")
    print(f"  CONCLUSION: Existing SHINE-type facility could handle Zn-65")
    print(f"              BUT requires shielding upgrades and new waste contracts")

    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")
    if zn65_GBq > lu177m_GBq:
        print(f"  ⚠ Your Zn-65 activity EXCEEDS typical annual medical Lu-177m waste")
    else:
        print(f"  ✓ Your Zn-65 activity is within typical medical waste volumes")
    
    print(f"  ")
    print(f"  SHIELDING: Zn-65's 1.116 MeV gamma requires significantly more shielding")
    print(f"             (~{shield_pb_zn/max(shield_pb_lu,0.1):.0f}x more Pb) than Lu-177m due to:")
    print(f"             - Higher gamma energy (1.116 vs 0.208 MeV)")
    print(f"             - Higher intensity (50.6% vs 11%)")
    print(f"  ")
    print(f"  INHALATION: Zn-65 is Class Y (>100 day lung clearance)")
    print(f"              Lu-177m is Class W (10-100 day lung clearance)")
    print(f"              → Zn-65 poses greater respiratory hazard")
    print(f"  ")
    print(f"  STORAGE: Zn-65 requires ~{t_clear_zn/t_clear_lu:.1f}x longer storage due to")
    print(f"           244-day half-life vs 160 days for Lu-177m")
    print(f"  ")
    print(f"  DISPOSAL: Neither qualifies for 10 CFR 35.92 decay-in-storage")
    print(f"            Both require LLRW Class A disposal or equivalent")
    print(f"            Lu-177m has established medical waste pathways")
    print(f"            Zn-65 from fusion production needs new broker contract")
    print(f"  ")
    print(f"  COST: Expect Zn-65 disposal to cost 2.5-4x more than Lu-177m")
    print(f"        due to shielding, container, and transport requirements")
    
    print(f"\n{'='*70}")
    print("RECOMMENDED CONTACTS FOR DISPOSAL")
    print(f"{'='*70}")
    print(f"  Perma-Fix Environmental Services (waste broker)")
    print(f"    www.perma-fix.com")
    print(f"    Full-service: characterization, packaging, transport, disposal")
    print(f"  ")
    print(f"  Waste Control Specialists (disposal facility)")
    print(f"    www.wcstexas.com")
    print(f"    Andrews, Texas - Class A/B/C LLRW disposal")
    print(f"  ")
    print(f"  TCEQ (current rate information)")
    print(f"    radmat@tceq.texas.gov or (512) 239-6466")
    print(f"    30 TAC §336.1310 rate schedule")
    print(f"{'='*70}\n")
    
    # Save comprehensive summary CSV with regulatory context
    summary_df = pd.DataFrame({
        'Metric': [
            'Activity (GBq)', 
            'Activity (mCi)', 
            'Mass (g)', 
            'Specific Activity (Bq/g)',
            'Half-life (days)',
            '10 CFR 35.92 Eligible (≤120d)',
            'Inhalation Class',
            'Lung Clearance Half-time',
            'Time to Clearance (years)',
            'Pb Shielding (cm) for 2.5 µSv/hr', 
            'Concrete Shielding (cm)',
            'ALI Ingestion (µCi)',
            'ALI Inhalation (µCi)',
            'DAC (µCi/ml)',
            'Ratio vs Medical Lu-177m',
            'Disposal Classification',
            'Est. Disposal Cost Multiplier',
            'SHINE Facility Compatible',
        ],
        'Zn-65': [
            f'{zn65_GBq:.3f}', 
            f'{zn65_mCi:.1f}', 
            f'{zn65_mass_g:.1f}', 
            f'{zn65_activity_Bq/zn65_mass_g:.2e}',
            '244',
            'NO',
            'Y (Year)',
            '>100 days (~500d)',
            f'{t_clear_zn/365:.1f}',
            f'{shield_pb_zn:.1f}', 
            f'{shield_conc_zn:.1f}', 
            '400',
            '300',
            '1E-7',
            f'{zn65_GBq/lu177m_GBq:.1f}x',
            'LLRW Class A',
            '2.5-4x',
            'With modifications',
        ],
        'Lu-177m (ref)': [
            f'{lu177m_GBq:.3f}', 
            f'{lu177m_GBq*1e9/3.7e7:.1f}', 
            f'{lu177m_mass_g:.1f}',
            f'{lu177m_Bq/lu177m_mass_g:.2e}', 
            '160',
            'NO',
            'W (Week)',
            '10-100 days (~50d)',
            f'{t_clear_lu/365:.1f}',
            f'{shield_pb_lu:.1f}', 
            f'{shield_conc_lu:.1f}', 
            '700',
            '700',
            '5E-8',
            '1.0x (baseline)',
            'Medical LLRW',
            '1.0x (baseline)',
            'Yes',
        ],
        'Regulatory Reference': [
            '-',
            '-',
            '-',
            '10 CFR 61.55',
            '-',
            '10 CFR 35.92',
            '10 CFR 20 App B / ICRP 30',
            'ICRP Publication 134',
            'IAEA RS-G-1.7',
            'NUREG-1537 / SHINE PSAR',
            'NCRP Report 151',
            '10 CFR 20 App B Table 1',
            '10 CFR 20 App B Table 1',
            '10 CFR 20 App B Table 1',
            '-',
            '10 CFR 61',
            'WCS Texas / TCEQ',
            'SHINE PSAR Section 4b.2',
        ]
    })
    summary_df.to_csv(os.path.join(output_dir, 'waste_comparison_summary.csv'), index=False)
    print(f"Saved: {output_dir}/waste_comparison_summary.csv")


def main():
    """
    Main function - use with your Zn-65 data from simple_analyze.py
    
    Usage:
        # Read from simulation directory:
        python zn_waste.py --from-dir simple_output_inner0_outer20_struct0_multi0_moderator0_zn99.0%
        
        # With custom irradiation/cooldown times:
        python zn_waste.py --from-dir ./radial_output_... --irrad-hours 8760 --cooldown-days 30
        
        # Direct input (legacy):
        python zn_waste.py --zn65-activity-Bq 1e12 --zn65-mass-g 1000
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Zn-65 Waste Analysis and Lu-177m Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Radial case (inner=0, outer=20, struct=0, multi=5, mod=5, enrich=48.6%%):
  python zn_waste.py --case "0 20 0 5 5 48.6%%"
  
  # Read from simulation output directory (1 year irradiation):
  python zn_waste.py --from-dir simple_output_inner0_outer20_struct0_multi0_moderator0_zn99.0%
  
  # Radial with custom irradiation (8 h) and cooldown (1 day):
  python zn_waste.py --case "0 20 0 5 5 48.6%%" --irrad-hours 8 --cooldown-days 1
  
  # Direct input of activity and mass:
  python zn_waste.py --zn65-activity-Bq 1e12 --zn65-mass-g 1000
        """
    )
    
    # Option 1: Read from simulation directory
    parser.add_argument("--from-dir", type=str, default=None,
                        help="Path to simulation output directory (e.g., simple_output_* or radial_output_*)")
    parser.add_argument("--case", type=str, default=None,
                        help="Radial case: 'INNER OUTER STRUCT MULTI MOD ENRICH%%' e.g. '0 20 0 5 5 48.6%%'")
    parser.add_argument("--radial-base", type=str, default=".",
                        help="Base directory for radial_output_* folders when using --case (default: .)")
    parser.add_argument("--irrad-hours", type=float, default=8760,
                        help="Irradiation time in hours (default: 8760 = 1 year)")
    parser.add_argument("--cooldown-days", type=float, default=0,
                        help="Cooldown time in days after irradiation (default: 0)")
    
    # Option 2: Direct input
    parser.add_argument("--zn65-activity-Bq", type=float, default=None,
                        help="Zn-65 activity in Bq (direct input, alternative to --from-dir)")
    parser.add_argument("--zn65-mass-g", type=float, default=1000,
                        help="Mass of Zn target material in grams (default: 1000)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="waste_analysis",
                        help="Output directory for plots (default: waste_analysis)")
    
    args = parser.parse_args()
    
    # Resolve --case to --from-dir for radial results
    if args.case:
        parts = args.case.strip().split()
        if len(parts) < 6:
            parser.error("--case requires 'INNER OUTER STRUCT MULTI MOD ENRICH%%' e.g. '0 20 0 5 5 48.6%%'")
        inner = float(parts[0])
        outer = float(parts[1])
        struct = float(parts[2])
        multi = float(parts[3])
        mod = float(parts[4])
        enrich_str = parts[5].rstrip('%')
        enrich = float(enrich_str)
        # Format to match fusion_irradiation output (integers for whole numbers)
        def _fmt(v):
            return int(v) if v == int(v) else v
        dir_name = f"radial_output_inner{_fmt(inner)}_outer{_fmt(outer)}_struct{_fmt(struct)}_multi{_fmt(multi)}_moderator{_fmt(mod)}_zn{enrich}%"
        args.from_dir = os.path.join(os.path.abspath(args.radial_base), dir_name)
        print(f"Case {args.case} -> {args.from_dir}")
    
    # Determine Zn-65 activity and mass
    if args.from_dir:
        # Read from simulation directory
        print("=" * 70)
        print("READING ZN-65 DATA FROM SIMULATION")
        print("=" * 70)
        
        sim_data = read_zn65_from_simulation(
            args.from_dir,
            irradiation_hours=args.irrad_hours,
            cooldown_days=args.cooldown_days
        )
        
        zn65_activity_Bq = sim_data['zn65_activity_Bq']
        zn65_mass_g = sim_data['zn65_mass_g']
        
        print("=" * 70)
        
    elif args.zn65_activity_Bq is not None:
        # Direct input
        zn65_activity_Bq = args.zn65_activity_Bq
        zn65_mass_g = args.zn65_mass_g
        
    else:
        parser.error("Either --from-dir or --zn65-activity-Bq is required")
    
    # Create comparison plots
    create_comparison_plots(
        zn65_activity_Bq=zn65_activity_Bq,
        zn65_mass_g=zn65_mass_g,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    # Example usage with typical values from your simulation
    # You can run this directly with test values, or use command line args
    
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default example: 1 GBq Zn-65 from 1 kg Zn target
        print("=" * 70)
        print("ZN-65 WASTE ANALYSIS TOOL")
        print("=" * 70)
        print()
        print("Usage options:")
        print()
        print("  1. Radial case (inner=0, outer=20, struct=0, multi=5, mod=5, enrich=48.6%):")
        print("     python zn_waste.py --case \"0 20 0 5 5 48.6%\"")
        print()
        print("  2. Read from simulation directory:")
        print("     python zn_waste.py --from-dir simple_output_inner0_outer20_struct0_multi0_moderator0_zn99.0%")
        print()
        print("  3. With custom irradiation (8 h) and cooldown (1 day):")
        print("     python zn_waste.py --case \"0 20 0 5 5 48.6%\" --irrad-hours 8 --cooldown-days 1")
        print()
        print("  4. Direct input of activity and mass:")
        print("     python zn_waste.py --zn65-activity-Bq 1e12 --zn65-mass-g 1000")
        print()
        print("Running with example values (1 GBq Zn-65, 1 kg Zn)...")
        print("=" * 70)
        print()
        
        # Example: ~1 GBq Zn-65 (typical after 1 year irradiation)
        create_comparison_plots(
            zn65_activity_Bq=1e9,  # 1 GBq
            zn65_mass_g=1000,      # 1 kg
            output_dir="waste_analysis_example"
        )
