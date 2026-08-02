# MD post-processing: trajectories, core analyses, and solid electrolytes

The analysis stack is model-independent and has three layers:

1. `TrajectoryDataset` normalizes new and legacy mlipx runs, ASE trajectories,
   and XDATCAR files, including PBC unwrapping, timing metadata, and available
   per-frame energy/force/stress data. Legacy `.traj` files can reconstruct the
   thermodynamic table when `md.csv` is absent.
2. Dependency-light mlipx tasks provide validation, thermodynamic summaries,
   RMSD/RMSF, partial RDF/coordination numbers, time-origin-averaged MSD,
   3-D density, VACF, and VDOS.
3. Native adapters call kinisi for covariance-aware Bayesian diffusion and
   conductivity, and GEMDAT for sites, occupancies, jumps, free-energy volumes,
   collective motion, and percolating pathways.

Install only what is needed:

```bash
uv sync --extra analysis       # matplotlib plots
uv sync --extra transport      # kinisi
uv sync --extra electrolyte    # GEMDAT
uv sync --extra analysis-all   # all of the above
```

Post-processing does not load an MLIP model, so one analysis environment can
process trajectories produced by all four engine environments.

## CLI examples

```bash
# Core tasks (the default task set)
uv run mlipx analyze results/my-md --mobile Li --framework Ge,P,S

# Select tasks and partial RDFs
uv run mlipx analyze results/my-md \
  --tasks validate msd rdf density --mobile Li --framework Ge,P,S \
  --rdf-pair Li-Li --rdf-pair Li-S --fit-start-ps 40

# Publication-oriented transport uncertainty
uv run mlipx analyze results/my-md \
  --tasks transport --mobile Li --temperature 800 --charge 1 \
  --fit-start-ps 40

# GEMDAT solid-electrolyte analysis
uv run mlipx analyze results/my-md \
  --tasks electrolyte --mobile Li --temperature 800 \
  --gemdat-resolution 0.5 --percolation xyz
```

For a standalone trajectory with no timing metadata, pass the interval between
stored frames—not the integrator time step:

```bash
uv run mlipx analyze old-run --mobile Li --frame-interval-fs 100
```

Known migration sites can be supplied with `--sites sites.cif`; this is often
more defensible than automatic density segmentation. Use `--site-radius` and
`--minimal-residence` to make the site/jump definition explicit.

## Results and provenance

Each result is stored under `analysis/<task>/<parameter-hash>/`. CSV contains
tables, NPZ contains multidimensional arrays, optional PNG files contain plots,
CIF contains inferred/occupied sites, and `density.vasp` / `free_energy.vasp`
are VASP volumetric files.

Every directory contains `metadata.json` with the input SHA-256, full
parameters, trajectory timing/PBC interpretation, package versions, warnings,
outputs, and summary. This provenance record makes a number traceable to its
input and method. Identical work is cached; `--force` recomputes it.

The built-in OLS Einstein fit is intended for rapid inspection. MSD values at
different lag times are correlated; use the kinisi task when quantitative
uncertainty matters. Likewise, automatically inferred GEMDAT sites and
percolation depend on trajectory length, temperature, voxel resolution, and
segmentation threshold and require stability checks.

## Python API

```python
from mlipx.analysis import TrajectoryDataset, analyze_run

dataset = TrajectoryDataset.load("results/my-md")
print(dataset.validation_report())

outputs = analyze_run(
    "results/my-md",
    tasks=["msd", "rdf", "density"],
    mobile="Li",
    framework="Ge,P,S",
    rdf_pairs=[("Li", "Li"), ("Li", "S")],
    plots=False,
)
```

`mlipx.analysis.core.arrhenius_fit` provides a deterministic multi-temperature
fit. `mlipx.analysis.transport.kinisi_arrhenius` is available when posterior
uncertainty propagation is required.

Upstream API references: [kinisi Analyze](https://kinisi.readthedocs.io/en/latest/analyze.html),
[GEMDAT Trajectory](https://gemdat.readthedocs.io/en/latest/api/gemdat_trajectory/),
and [GEMDAT simulation metrics](https://gemdat.readthedocs.io/en/latest/api/gemdat_simulation_metrics/).
