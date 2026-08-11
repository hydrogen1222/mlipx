# Molecular Dynamics Settings
# mlipx - VASP-style input

# Calculation Type
CALC_TYPE = MD
TASK = omat

# Model Settings
MODEL_PATH = uma-s-1.pt
DEVICE = cuda              # MD benefits from GPU
INFERENCE_MODE = turbo     # Turbo mode for better performance

# MD Settings
MD_ENSEMBLE = NVT          # NVT or NVE (Velocity Verlet)
TEMPERATURE = 300.0        # Temperature in Kelvin
TIMESTEP = 1.0             # Time step in femtoseconds
STEPS = 10000              # Number of production MD steps
EQUILIBRATION_STEPS = 0    # Same-ensemble steps before production
THERMOSTAT = LANGEVIN      # LANGEVIN, BUSSI, or NHC (NVT only)
FRICTION = 0.001           # Langevin friction (1/fs)
BUSSI_TAU = 1000.0         # Bussi/CSVR coupling time (fs)
NHC_TDAMP = 100.0          # Nose-Hoover-chain damping time (fs)
NHC_TCHAIN = 3             # Nose-Hoover chain length
NHC_TLOOP = 1              # Thermostat integration substeps
SAVE_INTERVAL = 10         # Save trajectory every N steps

# Pre-relaxation before MD
# PRE_RELAX = .TRUE.        # default: .TRUE. for NVT, .FALSE. for NVE
# PRE_RELAX_STEPS = 50
# PRE_RELAX_FMAX = 0.1

# Output Control
WRITE_TRAJECTORY = .TRUE.
OUTPUT_FORMAT = VASP

# Notes:
# - NVT uses only the coupling parameter for the selected THERMOSTAT
# - For NVE ensemble, initial temperature is set but not controlled
# - Thermostat choice/coupling can influence dynamical and transport properties;
#   check thermostat sensitivity for transport-oriented calculations
# - Pre-relaxation is on by default for NVT, off for NVE (override with PRE_RELAX)
# - TIMESTEP of 1 fs is typical, can increase to 2 fs for light elements
# - Turbo mode is recommended for MD (1.5-2x faster)
# - Use cuda device for MD (much faster than CPU)
