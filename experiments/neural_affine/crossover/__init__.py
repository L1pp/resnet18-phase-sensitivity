"""Clean-room Neural Affine crossover helpers.

The package deliberately keeps the saved-field decomposition independent from
the training runner.  ``delta_analysis`` only depends on NumPy and the Python
standard library, so it can be run on the local CPU against frozen fields.
"""

from .delta_analysis import (
    DEFAULT_COORD_SCALE,
    DEFAULT_SEEDS,
    FieldSource,
    LoadedField,
    SeedFieldBundle,
    analyze_from_sources,
    analyze_three_seed_fields,
    atomic_save_npz,
    atomic_write_json,
    load_field,
    load_seed_bundle,
    load_three_seed_fields,
)

__all__ = [
    "DEFAULT_COORD_SCALE",
    "DEFAULT_SEEDS",
    "FieldSource",
    "LoadedField",
    "SeedFieldBundle",
    "analyze_from_sources",
    "analyze_three_seed_fields",
    "atomic_save_npz",
    "atomic_write_json",
    "load_field",
    "load_seed_bundle",
    "load_three_seed_fields",
]
