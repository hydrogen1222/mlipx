"""Historical adsorption-energy convenience API (unsupported snapshot).

ARCHIVED / DO NOT USE FOR SCIENTIFIC RESULTS. The implementation could mix
absolute energies from incompatible model tasks/reference levels. It is kept
only to document the removed public surface; see Git history for its original
context and tests.
"""

FORMULA = "E_adsorbed - E_gas - E_surface"


def calculate_adsorption_energy(*args, **kwargs):
    """Unavailable historical entry point."""
    raise RuntimeError(
        "Archived workflow: adsorption energy requires a validated, common "
        "energy reference and is not supported by current mlipx."
    )
