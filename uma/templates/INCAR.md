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
MD_ENSEMBLE = NVT          # NVT (Langevin) or NVE (Velocity Verlet)
TEMPERATURE = 300.0        # Temperature in Kelvin
TIMESTEP = 1.0             # Time step in femtoseconds
STEPS = 10000              # Number of MD steps
FRICTION = 0.001           # Friction coefficient for NVT (1/fs)
SAVE_INTERVAL = 10         # Save trajectory every N steps

# Output Control
WRITE_TRAJECTORY = .TRUE.
OUTPUT_FORMAT = VASP

# Notes:
# - For NVT ensemble, FRICTION controls the thermostat strength
# - For NVE ensemble, initial temperature is set but not controlled
# - TIMESTEP of 1 fs is typical, can increase to 2 fs for light elements
# - Turbo mode is recommended for MD (1.5-2x faster)
# - Use cuda device for MD (much faster than CPU)
