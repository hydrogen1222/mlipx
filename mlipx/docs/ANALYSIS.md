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

The request hash includes the source fingerprint, selection, range, axes,
drift definition, scientific parameters, and backend versions. A completed
identical request is reused unless `--force` is supplied. Failures write
`error.json`; no half-finished plot is presented as success.

## References

- MDAnalysis MSD documentation, especially its unwrapped-coordinate and
  diffusive-window guidance: <https://docs.mdanalysis.org/2.9.0/documentation_pages/analysis/msd.html>
- SciPy Hann documentation (a symmetric Hann has zero-valued ends):
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html>
