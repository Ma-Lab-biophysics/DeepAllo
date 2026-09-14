#!/usr/bin/env python3
"""Build one minimal DeepAllo feature file from two GetContacts frequencies.

The inputs are residue-frequency TSV files written by
``get_contact_frequencies.py`` after hydrogen-bond and salt-bridge contacts
have been calculated for every replica. The output contains only the two
ordered simulation-numbered residue columns required by DeepAllo.

For reproduction with archived model weights, use
``--match-archived-order``. This verifies that the newly selected pair set is
identical to the corresponding archived DAT feature list and writes it in the
archived feature order. By default, the reference is the file with the same
name under ``descriptors/``; use ``--reference-features`` to specify another
reference. If the reference is unavailable, the script warns and writes the
normal deterministic order instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


# All values below use the residue numbering in the simulation topology.
MOTOR_RESIDUE_MIN = 1
MOTOR_RESIDUE_MAX = 780
EXCLUDE_RANGES = (
    (1, 80),
    (200, 211),
    (290, 450),
    (504, 642),
    (731, 734),
)
MIN_SEQUENCE_SEPARATION = 5
MIN_STATE_FREQUENCY = 0.10
MIN_FREQUENCY_DIFFERENCE = 0.10
ARCHIVED_FEATURE_NAMES = {
    "Apo_vs_Mava_features.dat",
    "Apo_vs_OM_features.dat",
}


@dataclass(frozen=True)
class Residue:
    chain: str
    name: str
    number: int

    @property
    def label(self) -> str:
        return f"{self.name}:{self.number}"


@dataclass(frozen=True)
class Feature:
    residue1: str
    residue2: str
    absolute_frequency_difference: float

    @property
    def pair(self) -> tuple[str, str]:
        return self.residue1, self.residue2


def parse_residue(value: str, source: Path, line_number: int) -> Residue:
    """Parse a GetContacts residue identifier such as ``X:GLU:773``."""
    fields = value.strip().split(":")
    if len(fields) != 3:
        raise ValueError(
            f"{source}:{line_number}: expected CHAIN:RESNAME:RESID, "
            f"found {value!r}. Use residue-frequency output from "
            "get_contact_frequencies.py, not atom-level dynamic contacts."
        )
    chain, residue_name, residue_number = fields
    try:
        number = int(residue_number)
    except ValueError as exc:
        raise ValueError(
            f"{source}:{line_number}: invalid residue number in {value!r}."
        ) from exc
    return Residue(chain=chain, name=residue_name, number=number)


def in_excluded_range(number: int) -> bool:
    return any(lower <= number <= upper for lower, upper in EXCLUDE_RANGES)


def canonical_pair(residue1: Residue, residue2: Residue) -> tuple[str, str]:
    """Return the same residue-label orientation used by the archived files."""
    return tuple(sorted((residue1.label, residue2.label)))


def load_frequency_tsv(
    path: Path,
    motor_chain: str | None,
) -> dict[tuple[str, str], float]:
    """Read and filter one GetContacts residue-frequency TSV."""
    if not path.is_file():
        raise FileNotFoundError(f"GetContacts frequency file not found: {path}")

    frequencies: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"{path}:{line_number}: expected at least three "
                    "tab-separated fields."
                )

            residue1 = parse_residue(fields[0], path, line_number)
            residue2 = parse_residue(fields[1], path, line_number)
            try:
                frequency = float(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid contact frequency "
                    f"{fields[2]!r}."
                ) from exc
            if not 0.0 <= frequency <= 1.0:
                raise ValueError(
                    f"{path}:{line_number}: frequency must be between 0 and 1."
                )

            if motor_chain is not None and (
                residue1.chain != motor_chain or residue2.chain != motor_chain
            ):
                continue
            if not (
                MOTOR_RESIDUE_MIN <= residue1.number <= MOTOR_RESIDUE_MAX
                and MOTOR_RESIDUE_MIN <= residue2.number <= MOTOR_RESIDUE_MAX
            ):
                continue
            if (
                in_excluded_range(residue1.number)
                or in_excluded_range(residue2.number)
            ):
                continue
            if abs(residue2.number - residue1.number) < MIN_SEQUENCE_SEPARATION:
                continue

            pair = canonical_pair(residue1, residue2)
            if pair in frequencies:
                raise ValueError(
                    f"{path}:{line_number}: duplicate residue pair {pair}. "
                    "Expected one residue-level frequency per pair."
                )
            frequencies[pair] = frequency

    if not frequencies:
        raise ValueError(f"No contacts remained after filtering {path}.")
    return frequencies


def select_features(
    frequencies1: dict[tuple[str, str], float],
    frequencies2: dict[tuple[str, str], float],
) -> list[Feature]:
    """Apply the SI frequency filters and return a deterministic feature list."""
    selected: list[Feature] = []
    for pair in set(frequencies1) | set(frequencies2):
        frequency1 = frequencies1.get(pair, 0.0)
        frequency2 = frequencies2.get(pair, 0.0)
        if (
            frequency1 < MIN_STATE_FREQUENCY
            and frequency2 < MIN_STATE_FREQUENCY
        ):
            continue
        absolute_difference = abs(frequency1 - frequency2)
        if absolute_difference + 1e-12 < MIN_FREQUENCY_DIFFERENCE:
            continue
        selected.append(
            Feature(
                residue1=pair[0],
                residue2=pair[1],
                absolute_frequency_difference=absolute_difference,
            )
        )

    # A secondary residue-label sort makes tied frequency differences stable.
    selected.sort(
        key=lambda feature: (
            -feature.absolute_frequency_difference,
            feature.residue1,
            feature.residue2,
        )
    )
    return selected


def load_archived_pairs(path: Path) -> list[tuple[str, str]]:
    """Read ordered simulation-numbered pairs from an archived DAT file."""
    if not path.is_file():
        raise FileNotFoundError(f"Archived feature file not found: {path}")
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected two residue labels."
                )
            pairs.append((fields[0], fields[1]))
    if not pairs:
        raise ValueError(f"No archived residue pairs found in {path}.")
    return pairs


def match_archived_order(
    features: list[Feature],
    reference_path: Path,
) -> list[Feature]:
    """Verify the pair set and return features in archived model order."""
    archived_pairs = load_archived_pairs(reference_path)
    by_pair = {feature.pair: feature for feature in features}
    generated_pairs = set(by_pair)
    reference_pairs = set(archived_pairs)
    if generated_pairs != reference_pairs:
        missing = sorted(reference_pairs - generated_pairs)
        extra = sorted(generated_pairs - reference_pairs)
        raise ValueError(
            f"Feature set differs from {reference_path}. "
            f"Missing archived pairs: {missing[:10]}; "
            f"new extra pairs: {extra[:10]}."
        )
    return [by_pair[pair] for pair in archived_pairs]


def write_feature_dat(path: Path, features: list[Feature]) -> None:
    """Write one ordered residue pair per line as tab-separated plain text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# residue_1_sim\tresidue_2_sim\n")
        for feature in features:
            handle.write(f"{feature.residue1}\t{feature.residue2}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state1-frequency",
        type=Path,
        required=True,
        help="Residue-frequency TSV for the first state.",
    )
    parser.add_argument(
        "--state2-frequency",
        type=Path,
        required=True,
        help="Residue-frequency TSV for the second state.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output two-column DAT feature file.",
    )
    parser.add_argument(
        "--reference-features",
        type=Path,
        default=None,
        help=(
            "Archived DAT file used with --match-archived-order. Defaults to "
            "the file with the output filename under descriptors/."
        ),
    )
    parser.add_argument(
        "--motor-chain",
        help=(
            "Optional GetContacts chain identifier for the myosin motor. "
            "Residues outside 1-780 are excluded regardless."
        ),
    )
    parser.add_argument(
        "--match-archived-order",
        action="store_true",
        help=(
            "Match supplied archived DAT feature lists when available. If a "
            "reference is absent, warn and write deterministic order."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    frequencies1 = load_frequency_tsv(args.state1_frequency, args.motor_chain)
    frequencies2 = load_frequency_tsv(args.state2_frequency, args.motor_chain)
    features = select_features(frequencies1, frequencies2)
    output_path = args.output
    reference = (
        args.reference_features
        if args.reference_features is not None
        else script_dir / output_path.name
    )

    protected_paths = {
        (script_dir / filename).resolve()
        for filename in ARCHIVED_FEATURE_NAMES
    }
    if args.reference_features is not None:
        protected_paths.add(reference.resolve())
    if output_path.resolve() in protected_paths:
        raise ValueError(
            f"Refusing to overwrite archived feature file: {output_path}. "
            "Write to a separate path, such as generated_features/<name>.dat."
        )

    if args.match_archived_order:
        if reference.is_file():
            features = match_archived_order(features, reference)
            verification = f"; pair set and order matched {reference}"
        else:
            print(
                f"WARNING: archived feature list not found: {reference}. "
                f"Writing {output_path} in deterministic frequency-based "
                "order; do not use it with archived model weights."
            )
            verification = "; archived order unavailable; deterministic order used"
    else:
        verification = ""

    write_feature_dat(output_path, features)
    print(f"Wrote {len(features)} features to {output_path}{verification}")


if __name__ == "__main__":
    main()
