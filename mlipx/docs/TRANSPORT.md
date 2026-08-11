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
  --drift-reference nonmobile --fit-start-ps 20 --random-seed 0
```

The adapter passes either the recorded MD timestep and saved-frame stride or,
when those are unavailable, the explicit frame interval with stride one. The
result records this mapping, kinisi version, fit start, random seed, posterior
mean/standard deviation/95% interval, and both m^2/s and cm^2/s.

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
