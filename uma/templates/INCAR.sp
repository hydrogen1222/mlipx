# Single Point Calculation Settings
# mlipx - VASP-style input

# Calculation Type
CALC_TYPE = SP
TASK = omat

# Model Settings
MODEL_PATH = uma-s-1.pt
DEVICE = cpu
INFERENCE_MODE = default

# Output Control
WRITE_FORCES = .TRUE.
WRITE_STRESS = .TRUE.
OUTPUT_FORMAT = VASP

# Notes:
# - TASK: omat (materials), omol (molecules), oc20 (catalysis), odac (MOFs), omc (molecular crystals)
# - DEVICE: cpu or cuda
# - INFERENCE_MODE: default (general) or turbo (fast, for MD)
# - For omol, add charge and spin to structure file or set in script
