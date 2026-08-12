# Transport definitions and limitations

## Trajectory prerequisites

Publication transport currently requires a fixed-cell, 3-D periodic trajectory
with finite coordinates, at least four production frames, a uniform time axis,
and an explicit wrapped/unwrapped convention. An mlipx run marked failed,
aborted, or cancelled is rejected. Imported trajectories have no mlipx run
status, so their completeness remains the user's responsibility. Wrapped
coordinates are reconstructed by consecutive minimum images; the reported
unwrap safety ratio is a heuristic and cannot prove that no hidden multiple
cell crossing occurred between sparse frames.

The kinisi ASE backend consumes periodic ASE frames and reconstructs
displacements from periodic/scaled coordinates. Therefore an exact unwrapped
source is not treated as if kinisi directly preserved its image counters.
Before calling kinisi, mlipx compares every saved-frame displacement from the
exact unwrapped source with ASE's periodic minimum-image reconstruction. A
non-equivalent interval is rejected because the backend would lose image
history; save frames more densely or use the native mlipx MSD for that source.
When the check passes, the result records that the backend reconstruction was
equivalent, but that exact unwrapped coordinates were not consumed directly.

Drift correction is never silently enabled:

- `none`: raw mobile-particle displacement;
- `nonmobile`: kinisi removes the mean displacement of all nonmobile atoms;
- `indices`: kinisi removes the mean displacement of explicitly selected
  reference atoms.

The chosen indices/species and backend semantics are stored in results.

## MSD and tracer diffusion

For selected axes with dimensionality `d`:

```text
MSD_axes(t) = <sum_axis [r_i(t0+t)-r_i(t0)]^2>_(i,t0)
D_axes = slope(MSD_axes) / (2 d)
```

`mlipx analyze ... msd` supplies directional visualization and an optional
explicit-range OLS diagnostic. It does not automatically identify the correct
diffusive regime. The local log-log slope is a diagnostic only.

The publication-oriented scalar tracer estimate uses kinisi 2.x. The 20 ps
fit start below is only an example; choose it from the equilibrated trajectory
and the observed diffusive regime rather than copying it blindly:

```bash
mlipx analyze RUN transport --mobile Li --charge 1 \
  --drift-reference nonmobile --fit-start-ps 20 \
  --lag-step-ps 1 --lag-stop-ps 200 --random-seed 0
```

The adapter passes either the recorded MD timestep and saved-frame stride or,
when those are unavailable, the explicit frame interval with stride one. The
result records this mapping, kinisi version, fit start, random seed, posterior
mean/standard deviation/95% interval, and both m^2/s and cm^2/s.

For long, frequently saved trajectories, provide an explicit kinisi lag grid
with `--lag-step-ps` and `--lag-stop-ps`. The two options must be supplied
together. The lag grid controls which time intervals kinisi evaluates; it does
not downsample the trajectory, so all saved frames and their time origins are
preserved. `--fit-start-ps` is separate: it selects the first lag used by the
diffusion regression, while the lag grid selects the available lag values.
Choose the grid from the observed diffusive regime and perform a convergence
check (for example, compare 0.5, 1, and 2 ps spacing) for each material and
trajectory. kinisi's documentation notes that evaluating every possible time
interval can be excessive for its full covariance analysis. When no explicit
grid is supplied, mlipx preserves kinisi's native behavior for small grids but
refuses pathological dense defaults and asks for explicit parameters.

The custom-grid metadata distinguishes `requested_step_ps` and
`nominal_step_ps` from `actual_step_ps`. If `fit_start_ps` is not on the
regular grid, mlipx inserts it for the regression and records
`fit_start_inserted: true`, `is_uniform_grid: false`, and
`actual_step_ps: null`; the nominal step must not be mistaken for every
adjacent spacing.

References:

- McCluskey et al., *kinisi: Bayesian analysis of mass transport from molecular
  dynamics simulations*, JOSS 9, 5984 (2024), DOI 10.21105/joss.05984.
- McCluskey, Coles, and Morgan, *Accurate Estimation of Diffusion Coefficients
  and their Uncertainties from Computer Simulation*, DOI
  10.1021/acs.jctc.4c01249.

## Nernst–Einstein tracer conductivity

The reported quantity is deliberately named `sigma_NE_tracer`:

```text
n = N_mobile / V
q = z e
sigma_NE_tracer = n q^2 D_tracer / (k_B T)
```

`z` is required and is never guessed from the element. Temperature is the
production mean when stored, or an explicit input for imported trajectories;
300 K is not a fallback. Results include S/m, S/cm, and mS/cm.

Nernst–Einstein tracer conductivity is not automatically equal to experimental
or collective ionic conductivity. Correlated motion can matter. Optional
kinisi collective conductivity is explicitly named `sigma_collective`; mlipx
does not expose an ambiguous field named `f` and does not automatically compute
a Haven ratio.

The tracer Nernst–Einstein result includes posterior summaries in S/m, S/cm,
and mS/cm. These are a linear propagation of the kinisi tracer-D posterior
with number density, charge, volume, and temperature held fixed. They do not
include model, finite-size, replica, temperature, volume, approximation, or
ion-ion-correlation uncertainty; the fields are not a claim about total
physical uncertainty. The legacy scalar conductivity fields remain and equal
the corresponding posterior means.

Each successful transport analysis also writes `transport_summary.csv`,
`transport_msd.png`, and `transport_msd.svg` alongside `kinisi_arrays.npz`.
The plot shows the kinisi MSD and the actual Bayesian diffusion fit window;
it deliberately does not draw an ordinary-least-squares fit line. The CLI
prints the tracer posterior, credible interval, fit window, lag-grid summary,
and Nernst–Einstein posterior after the usual output path.

For an mlipx run directory, prefer the run-level source:

```bash
mlipx analyze RUN transport \
  --mobile Li --charge 1 --drift-reference nonmobile \
  --fit-start-ps 40 --lag-step-ps 2 --lag-stop-ps 200 --random-seed 0
```

The `2 ps` grid is a current LGPS working parameter after a 1 ps versus 2 ps
sensitivity check, not a universal scientific default. A run directory carries
the trajectory timestep, save stride, frame interval, coordinate convention,
production phase, and temperature metadata. For an external trajectory, pass
the values only when they are known:

```bash
mlipx analyze XDATCAR transport --mobile Li --charge 1 \
  --positions-convention wrapped --frame-interval-fs 10 \
  --temperature-K 700 --fit-start-ps 40 \
  --lag-step-ps 2 --lag-stop-ps 200
```

Do not select `wrapped` or `unwrapped` merely to pass validation; the value
must describe the file's actual coordinate semantics.

If a Haven ratio is added to a downstream analysis, its formula must be stored,
for example `H_R = D_tracer / D_sigma`, together with the inverse. A GEMDAT
metric is not silently reinterpreted as a generic literature correction factor.

## Arrhenius fitting

Arrhenius input must represent independent temperature runs:

```text
D(T) = D0 exp[-Ea/(k_B T)]
```

At least three temperatures are recommended. Two points are allowed with a
strong warning. When diffusion standard deviations are supplied, the linear
fit weights `ln(D)` with `sigma_lnD ~= sigma_D/D`. Extrapolated temperatures
are marked and the simulated temperature range is retained.

Trajectory duration alone is not a universal quality threshold: atom count,
diffusion rate, temperature, correlation, system size, and fit regime all
affect uncertainty.
