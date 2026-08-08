# mlipx development contract

1. Scientific correctness has priority over speed and feature count.
2. Fail closed when physical assumptions are not satisfied.
3. Never silently change units, PBC semantics, model task/head, or dtype.
4. Never mix absolute energies from incompatible model tasks/reference levels.
5. Prefer ASE/upstream algorithms over reimplementing MD integrators.
6. Add a regression or known-answer test for scientific fixes when feasible.
7. `archive/` is historical reference only and must never be imported.
8. Advanced trajectory analysis is currently out of scope.
9. Do not add features without an explicit request.
10. Preserve reproducible raw MLMD trajectories and provenance.
