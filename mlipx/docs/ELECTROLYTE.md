# GEMDAT electrolyte mechanism analysis

GEMDAT is an optional mechanism-analysis backend. mlipx uses kinisi for the
primary covariance-aware scalar transport estimate and GEMDAT for site mapping,
transitions, jumps, jump matrices, collective-jump diagnostics, and percolating
pathways.

Install and run:

```bash
python -m pip install -e './mlipx[analysis,electrolyte]'
mlipx analyze RUN electrolyte --mobile Li --sites Li_sites.cif \
  --jump-dimensions 3 --percolation-axes xyz
```

A site source is mandatory: an explicit CIF in the CLI/TUI, explicit
fractional site coordinates through the Python adapter, or a separately and
explicitly requested GEMDAT density-peak segmentation. Site discovery is not
silently performed.

`jump_dimensions` controls the `2*d*N*t` dimensional factor in GEMDAT jump
diffusivity. `percolation_axes` controls only the directions tested for path
connectivity. Changing percolation axes cannot change jump dimensionality,
tracer diffusion, or conductivity; this separation has a permanent regression
test.

Outputs retain the site source, matching inputs, thresholds, GEMDAT version,
sites/reference structures, transition/jump tables and matrices, density/free
energy arrays, and per-axis pathway coordinates/barriers. GEMDAT endpoint
tracer diffusivity is not promoted over the kinisi estimate.

Reference: Lavrinenko et al., *GEMDAT: a Python toolkit for site-resolved
diffusion analysis in solid-state molecular dynamics*, npj Computational
Materials (2026), DOI 10.1038/s41524-026-02133-7.
