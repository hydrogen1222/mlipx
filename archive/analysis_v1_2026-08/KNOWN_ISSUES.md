# Known scientific issues in Analysis v1

This list records known reasons the archived implementation must not be used
for scientific results.

- GEMDAT percolation axes were incorrectly coupled to the dimensional factor
  used for jump diffusivity. Percolation axes and diffusion dimensionality
  must be independent inputs.
- The historical VACF-to-VDOS estimator multiplied a normalized VACF by a
  standard Hann window whose first value is zero, destroying `Cvv(0) = 1`.
  It also clipped negative spectrum values, masking estimator or sampling
  failures.
- The default MSD fit used a heuristic tail window without demonstrating a
  diffusive regime or accounting for correlated errors.
- Irregularly sampled trajectories could be approximated by a median time
  interval even for algorithms requiring a uniform time grid.
- Variable-cell unwrapping did not explicitly separate affine cell motion
  from particle diffusion.
- The kinisi conductivity interface defaulted ionic charge to `+1`, which is
  scientifically unsafe for multivalent ions. A future interface must require
  charge explicitly.

Archive is a reference, not a specification for Analysis v2.
