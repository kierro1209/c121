"""Week 2: Intro to pysam and pandas

Quick start (no pysam needed for isolated exercise checks):
    python wk2_worksheet.py --exercise load_snps
    -> check a specific exercise, e.g. load_snps from Exercise 1
    
    python wk2_worksheet.py --exercise all
    -> check every exercise

Submit your completed worksheet to Bruin Learn under the Week 2 worksheet assignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_COLUMNS = ["read_name", "position", "observation", "phred"]

def load_snps(path) -> pd.DataFrame:
    """Exercise 1: Read a tab-separated SNP table into a pandas DataFrame.
    Preserve all columns and row order. Assume the table is valid."""
    return pd.read_csv(path, sep="\t")


def get_maf(snps: pd.DataFrame, position: int) -> float:
    """Exercise 2: Return the MAF for the given 1-based SNP position.
    Select the row where snps["pos"] equals position, then read its "maf".
    Assume the position appears exactly once in this single-contig table.
    Example: get_maf(snps, 6815) from Exercise 2 returns 0.3959 for the provided data."""
    return float(snps.loc[snps["pos"] == position, "maf"].iloc[0])


def to_bam_interval(position) -> tuple[int, int]:
    """Exercise 3: Convert a positive integer, 1-based SNP position into
    a 0-based, half-open (start, stop) tuple for exactly one base.
    Example: 6815 -> (6814, 6815).
    Assume the input is a valid positive integer."""
    return position - 1, position


def reference_base(ref, chrom, position) -> str:
    """Exercise 4: Return the reference base at a 1-based position.
    Use to_bam_interval() from Exercise 3 and ref.fetch(chrom, start, stop).
    Assume the contig and position are valid."""
    start, stop = to_bam_interval(position)
    return ref.fetch(chrom, start, stop)


def extract_observation(pileup_read, position) -> dict[str, str | int] | None:
    """Exercise 5: Return one observation dict, or None when no base exists.
    Skip is_del, is_refskip, query_position=None, missing query_sequence,
    or missing query_qualities. Get the AlignedSegment from .alignment.
    Dict keys: read_name, position (original 1-based), observation, phred.
    Use query_position to index the stored sequence AND qualities.
    Do not index the read with the reference coordinate."""
    if pileup_read.is_del or pileup_read.is_refskip:
        return None

    query_position = pileup_read.query_position
    alignment = pileup_read.alignment
    if (
        query_position is None
        or alignment.query_sequence is None
        or alignment.query_qualities is None
    ):
        return None

    return {
        "read_name": alignment.query_name,
        "position": position,
        "observation": alignment.query_sequence[query_position],
        "phred": int(alignment.query_qualities[query_position]),
    }


def collect_site_observations(bam, chrom, position) -> list[dict[str, str | int]]:
    """Exercise 6: Return a list of observation dicts for one SNP.
    Use to_bam_interval() from Exercise 3, then bam.pileup(chrom, start, stop,
    truncate=True, min_base_quality=0, ignore_orphans=False).
    Loop over columns, then column.pileups. Call extract_observation()
    from Exercise 5 for each entry; append only non-None results.
    Consume pileup columns inside the iterator; do not save them for later."""
    start, stop = to_bam_interval(position)
    observations = []
    for column in bam.pileup(
        chrom,
        start,
        stop,
        truncate=True,
        min_base_quality=0,
        ignore_orphans=False,
    ):
        for pileup_read in column.pileups:
            observation = extract_observation(pileup_read, position)
            if observation is not None:
                observations.append(observation)
    return observations


def collect_all_observations(bam, snps) -> list[dict[str, str | int]]:
    """Exercise 7: Return one flat list of observations for all SNP rows.
    Use snps.itertuples(index=False), row.chr, and int(row.pos).
    Call collect_site_observations() from Exercise 6 and extend the output list.
    Preserve SNP order. Sites without observations contribute no rows."""
    observations = []
    for row in snps.itertuples(index=False):
        observations.extend(
            collect_site_observations(bam, row.chr, int(row.pos))
        )
    return observations


def build_dataframe(rows) -> pd.DataFrame:
    """Exercise 8: Make a DataFrame with OUTPUT_COLUMNS in that order.
    Even rows=[] must produce these four column names."""
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def save_results(df, output_path) -> None:
    """Exercise 9: Save df as CSV without a pandas row index.
    Return None. Hint: df.to_csv(..., index=False)."""
    df.to_csv(output_path, index=False)



# PROVIDED infrastructure: leave this section unchanged.
EXERCISES = (
    "load_snps", "get_maf", "to_bam_interval", "reference_base", "extract_observation",
    "collect_site_observations", "collect_all_observations", "build_dataframe",
    "save_results",
)


def check_exercise(name) -> None:
    """Check exactly one student function; dependencies use known fixtures."""
    expected = {"read_name": "toy_read", "position": 3, "observation": "G", "phred": 30}
    record = SimpleNamespace(query_name="toy_read", query_sequence="ACGT",
                             query_qualities=[10, 20, 30, 40])
    entry = SimpleNamespace(is_del=False, is_refskip=False, query_position=2,
                            alignment=record)
    with TemporaryDirectory(prefix="wk2_check_") as folder:
        folder = Path(folder)
        if name == "load_snps":
            path = folder / "snps.tsv"
            path.write_text("chr\tpos\tref\ncontigA\t3\tG\n")
            actual = load_snps(path)
            assert actual.to_dict("records") == [{"chr": "contigA", "pos": 3, "ref": "G"}]
        elif name == "get_maf":
            snps = pd.DataFrame({"pos": [9, 3, 5], "maf": [0.1, 0.3959, 0.25]})
            original = snps.copy(deep=True)
            for position, expected_maf in ((3, 0.3959), (9, 0.1), (5, 0.25)):
                result = get_maf(snps, position)
                assert isinstance(result, float), "Return one float, not a Series"
                assert abs(result - expected_maf) < 1e-12
            pd.testing.assert_frame_equal(snps, original)
        elif name == "to_bam_interval":
            assert to_bam_interval(1) == (0, 1)
            assert to_bam_interval(6815) == (6814, 6815)
        elif name == "reference_base":
            def fetch(chrom, start, stop):
                assert (chrom, start, stop) == ("contigA", 2, 3)
                return "g"
            with patch.dict(globals(), to_bam_interval=lambda pos: (2, 3)):
                assert reference_base(SimpleNamespace(fetch=fetch), "contigA", 3) == "g"
        elif name == "extract_observation":
            assert extract_observation(entry, 3) == expected
            for changes in ({"is_del": True}, {"is_refskip": True}, {"query_position": None}):
                other = SimpleNamespace(**(vars(entry) | changes))
                assert extract_observation(other, 3) is None
            for field in ("query_sequence", "query_qualities"):
                other_record = SimpleNamespace(**(vars(record) | {field: None}))
                other = SimpleNamespace(**(vars(entry) | {"alignment": other_record}))
                assert extract_observation(other, 3) is None
        elif name == "collect_site_observations":
            skipped = SimpleNamespace(label="skip")
            def pileup(chrom, start, stop, **kwargs):
                assert (chrom, start, stop) == ("contigA", 2, 3)
                assert kwargs == {"truncate": True, "min_base_quality": 0, "ignore_orphans": False}
                yield SimpleNamespace(pileups=[entry, skipped])
            def fixture_observation(item, pos):
                assert pos == 3
                return expected.copy() if item is entry else None
            with patch.dict(globals(), to_bam_interval=lambda pos: (2, 3),
                            extract_observation=fixture_observation):
                assert collect_site_observations(SimpleNamespace(pileup=pileup), "contigA", 3) == [expected]
                assert collect_site_observations(SimpleNamespace(pileup=lambda *a, **k: iter(())), "contigA", 3) == []
        elif name == "collect_all_observations":
            snps = pd.DataFrame({"chr": ["contigA"] * 3, "pos": ["3", "4", "5"]})
            bam_fixture = object()
            calls = []
            def fixture_site(bam, chrom, pos):
                assert bam is bam_fixture and chrom == "contigA" and isinstance(pos, int)
                calls.append(pos)
                return [expected.copy(), expected.copy()] if pos == 3 else []
            with patch.dict(globals(), collect_site_observations=fixture_site):
                assert collect_all_observations(bam_fixture, snps) == [expected, expected]
                assert calls == [3, 4, 5]
                assert collect_all_observations(bam_fixture, snps.iloc[:0]) == []
        elif name == "build_dataframe":
            actual = build_dataframe([expected])
            assert list(actual.columns) == OUTPUT_COLUMNS
            assert actual.to_dict("records") == [expected]
            empty = build_dataframe([])
            assert empty.empty and list(empty.columns) == OUTPUT_COLUMNS
        elif name == "save_results":
            path = folder / "result.csv"
            df = pd.DataFrame([expected], columns=OUTPUT_COLUMNS)
            save_results(df, path)
            assert path.read_text().splitlines()[0] == ",".join(OUTPUT_COLUMNS)
            assert pd.read_csv(path).to_dict("records") == [expected]


def run_checks(selection) -> int:
    names = EXERCISES if selection == "all" else (selection,)
    failed = 0
    for name in names:
        try:
            check_exercise(name)
        except NotImplementedError:
            print(f"SKIP {name}: implement this exercise when ready.")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    return int(failed > 0)


def prepare_indexes(bam_path, fasta_path) -> None:
    """Create missing BAM/FASTA sidecars. BAM must already be coordinate sorted."""
    import pysam
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        indexed = bam.has_index()
    if not indexed:
        pysam.index(str(bam_path))
    if not Path(str(fasta_path) + ".fai").exists():
        pysam.faidx(str(fasta_path))


def run_pipeline(args) -> int:
    # Detect untouched stubs BEFORE writing indexes or output files.
    missing = [name for name in EXERCISES if name not in ("get_maf", "reference_base")
               and "TODO: implement" in globals()[name].__code__.co_consts]
    if missing:
        print("Pipeline needs: " + ", ".join(missing))
        print("Use --exercise FUNCTION_NAME to work on one function at a time.")
        return 1
    import pysam
    snps = load_snps(args.snps)
    prepare_indexes(args.bam, args.fasta)
    with pysam.AlignmentFile(str(args.bam), "rb") as bam, pysam.FastaFile(str(args.fasta)) as ref:
        rows = collect_all_observations(bam, snps)
    final_df = build_dataframe(rows)
    save_results(final_df, args.output)
    print(f"Wrote {len(final_df):,} observations from {len(snps):,} SNP sites to {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--exercise", choices=("all",) + EXERCISES, default="all")
    mode.add_argument("--run", action="store_true", help="Run the complete pipeline on the provided data")
    parser.add_argument("--bam", type=Path, default=DATA_DIR / "out_sort.bam")
    parser.add_argument("--fasta", type=Path, default=DATA_DIR / "chr1_1e6_2e6.fasta")
    parser.add_argument("--snps", type=Path, default=DATA_DIR / "putative_snps.tsv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "result.csv")
    args = parser.parse_args()
    if not args.run:
        return run_checks(args.exercise)
    try:
        return run_pipeline(args)
    except NotImplementedError:
        print("An exercise is unfinished. Run --exercise all to find it.")
        return 1
    except (ImportError, OSError, ValueError) as exc:
        print(f"Cannot run pipeline: {exc}")
        print("Install pandas and pysam; verify input paths and that BAM is coordinate sorted.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
