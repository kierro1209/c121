from pathlib import Path

import pandas as pd
import pysam


OUTPUT_COLUMNS = ["read_name", "position", "observation", "phred"]


def observations_at_position(bam, chromosome, position):
    """Return read observations at one 1-based SNP position."""
    start = position - 1
    observations = []

    for column in bam.pileup(
        chromosome,
        start,
        position,
        truncate=True,
        min_base_quality=0,
        ignore_orphans=False,
    ):
        for pileup_read in column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue

            query_position = pileup_read.query_position
            read = pileup_read.alignment
            if (
                query_position is None
                or read.query_sequence is None
                or read.query_qualities is None
            ):
                continue

            observations.append(
                {
                    "read_name": read.query_name,
                    "position": position,
                    "observation": read.query_sequence[query_position],
                    "phred": int(read.query_qualities[query_position]),
                }
            )

    return observations


def main():
    output_directory = Path(__file__).resolve().parent
    data_directory = output_directory
    if not (data_directory / "out_sort.bam").exists():
        data_directory = output_directory / "pj1.1_data" / "fixtures"

    snps = pd.read_csv(data_directory / "putative_snps.tsv", sep="\t")
    rows = []

    with pysam.AlignmentFile(data_directory / "out_sort.bam", "rb") as bam:
        for snp in snps.itertuples(index=False):
            rows.extend(
                observations_at_position(bam, snp.chr, int(snp.pos))
            )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result.to_csv(output_directory / "result.csv", index=False)


if __name__ == "__main__":
    main()
