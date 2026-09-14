"""Read ordered DeepAllo residue-contact features from a plain DAT file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_RESIDUE_LABEL = re.compile(r"^[A-Za-z0-9]+:-?\d+$")


@dataclass(frozen=True)
class ContactFeatures:
    simulation_pairs: list[tuple[str, str]]


def _validate_pairs(pairs: list[tuple[str, str]], path: Path) -> None:
    if not pairs:
        raise ValueError(f"No residue-contact pairs found in {path}.")
    seen = set()
    for feature_number, pair in enumerate(pairs, start=1):
        for label in pair:
            if not _RESIDUE_LABEL.fullmatch(label):
                raise ValueError(
                    f"{path}: feature {feature_number}: invalid residue label "
                    f"{label!r}; expected RESNAME:RESID."
                )
        if pair in seen:
            raise ValueError(
                f"{path}: duplicate ordered residue-contact pair {pair}."
            )
        seen.add(pair)


def load_contact_features(path: str | Path) -> ContactFeatures:
    """Load ordered pairs from a two-column, whitespace-separated DAT file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Contact feature file not found: {path}")
    if path.suffix.casefold() not in {".dat", ".txt"}:
        raise ValueError(
            f"Unsupported contact feature format {path.suffix!r}: {path}. "
            "Use a two-column .dat file."
        )
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly two whitespace-"
                    "separated fields: RESNAME:RESID RESNAME:RESID."
                )
            pairs.append((fields[0], fields[1]))
    _validate_pairs(pairs, path)
    return ContactFeatures(simulation_pairs=pairs)
