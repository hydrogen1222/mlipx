# Analysis v2

Analysis v2 consumes an mlipx run directory, an ASE `.traj`, or an XDATCAR
without importing a calculator, Torch, FairChem, MACE, DeepMD, or
TensorPotential. For an mlipx run, `raw/trajectory.traj` and `raw/md.csv` are
preferred over interoperability exports.

From the repository root, install the lightweight analysis dependencies into
the environment that will run the analysis:

```bash
python -m pip install -e './mlipx[analysis]'
```

Transport and GEMDAT remain optional:

```bash
python -m pip install -e './mlipx[analysis,transport]'
python -m pip install -e './mlipx[analysis,electrolyte]'
```

## Start with validation

```bash
mlipx analyze RUN validate
```

Validation reports the exact time axis, MD timestep, saved-frame stride, frame
interval, PBC, fixed/variable cell, position convention, velocities, phase
range, Nyquist frequency, source run status, and separate eligibility for RDF,
MSD/transport, and VACF. Nonuniform time is never replaced by a median interval.

External trajectories whose coordinate convention cannot be inferred are
reported as `unknown`. Declare it only when known:

```bash
mlipx analyze trajectory.traj validate \
  --positions-convention wrapped --frame-interval-fs 10
```

## Equilibration and production

MD accepts `--equilibration-steps N` followed by `--steps M` production steps.
Both phases use the same ensemble, integrator/thermostat, timestep, temperature,
and velocities. The producer records phase per saved frame and the production
boundary in `artifacts.json`.

RDF, MSD, transport, and density analyses use production frames by default.
`--include-equilibration` is available for diagnostics; mlipx does not attempt
automatic equilibrium or plateau detection.

Legacy trajectories may not contain phase metadata. Such trajectories are
treated entirely as production data; this is a compatibility rule, not an
equilibration claim. Inspect `thermo` first and use `--start-frame` and
`--stop-frame` to select a justified range. These options count saved frames,
not MD integration steps.

## Core commands

```bash
mlipx analyze RUN thermo
mlipx analyze RUN rdf --center Li --neighbor S --rmax 6 --cn-cutoff 3
mlipx analyze RUN msd --mobile Li --axes x,y,z,xy,xyz \
  --drift-reference nonmobile
mlipx analyze RUN density --mobile Li --spacing 0.25
mlipx analyze RUN vacf --species Li
mlipx analyze RUN spectrum --species Li --taper one-sided-cosine
```

Publication transport uses kinisi 2.x and requires an explicit fit start. For
long, frequently saved trajectories, provide both lag-grid controls; they
sparsify kinisi's evaluated lag times without downsampling the trajectory:

```bash
mlipx analyze RUN transport --mobile Li --charge 1 \
  --drift-reference nonmobile --fit-start-ps 40 \
  --lag-step-ps 2 --lag-stop-ps 200 --random-seed 0
```

The `2 ps` value is a current LGPS working parameter after a 1 ps versus 2 ps
sensitivity check, not a universal default. Prefer `RUN` over a bare exported
trajectory because the run directory carries the saved-frame interval,
coordinate convention, production phase, and temperature. For an external
trajectory, declare known source semantics explicitly, for example
`--positions-convention wrapped --frame-interval-fs 10 --temperature-K 700`;
do not choose a convention merely to pass validation.

kinisi reconstructs periodic displacements from the ASE frames it receives.
For an exact unwrapped source, mlipx checks every saved-frame displacement
against ASE's periodic minimum-image reconstruction before invoking kinisi and
fails closed if image history would be lost. The result records this backend
semantics. Wrapped sources retain the existing unwrap-safety guard.

Bulk RDF uses ordered A centers, excludes self pairs for A=A, and refuses a
radius beyond half the minimum triclinic face height. Coordination cutoff is
always explicit. Density output contains both occupancy probability (sum 1)
and number density in A^-3 (volume integral equals the number of selected
mobile particles).

MSD uses unwrapped coordinates. Wrapped fixed-cell trajectories are
reconstructed with consecutive fractional minimum images and receive an
unwrap ambiguity diagnostic. Variable-cell transport is unsupported because
the current contract does not separate affine lattice deformation from atomic
migration. A simple diffusion fit is produced over the full analysis lag range
by default. Supply both `--fit-start-ps` and `--fit-stop-ps` to restrict that
diagnostic fit to an explicit window. Its name and metadata mark it as
diagnostic, not a covariance-aware publication estimate; the default
full-range fit is not an automatic diffusive-regime detector.

VACF requires stored velocities and uniform sampling. Positions are not
differentiated to invent velocities. The optional spectrum uses a one-sided
taper with `w(0)=1`, preserves negative estimates, records the Nyquist limit,
and is named a VACF-derived velocity spectrum—not a harmonic phonon DOS.

## Outputs and reproducibility

Every task writes under `RUN/analysis/TASK/REQUEST_HASH/`:

```text
request.json
provenance.json
results.json
task-specific CSV/NPZ
PNG and SVG when applicable
diagnostics.json when applicable
```

Successful transport additionally writes `kinisi_arrays.npz`,
`transport_summary.csv`, `transport_msd.png`, and `transport_msd.svg`. The
transport plot shows the kinisi MSD and the Bayesian diffusion regression
window, without drawing an OLS fit line. The CLI prints the tracer posterior
mean/standard deviation/95% credible interval, fit window, lag grid, and
Nernst–Einstein tracer conductivity. The latter also includes a posterior
summary formed by linear propagation of the kinisi tracer-D posterior. This
uncertainty excludes model, finite-size, replica, temperature, volume,
Nernst–Einstein approximation, and ion-correlation uncertainty; it is not a
total physical uncertainty.

The request hash includes the source fingerprint, selection, range, axes,
drift definition, scientific parameters, and backend versions. A completed
identical request is reused unless `--force` is supplied. Failures write
`error.json`; no half-finished plot is presented as success.

## References

- MDAnalysis MSD documentation, especially its unwrapped-coordinate and
  diffusive-window guidance: <https://docs.mdanalysis.org/2.9.0/documentation_pages/analysis/msd.html>
- SciPy Hann documentation (a symmetric Hann has zero-valued ends):
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html>
