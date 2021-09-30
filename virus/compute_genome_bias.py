import os
import sys
import logging

logger = logging.getLogger(__name__)

import click
import pandas as pd
import numpy as np

sys.path.append("..")
from utils.sequence_utils import GenomeBiasCollectingService


@click.command()
@click.option(
    "--input_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the dataframe of virus-host associations",
    default=f"{os.getcwd()}/../data/virus_sequence_data.csv".replace("\\", "/"),
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path that will hold a dataframe mapping virus taxon name and id from the associations dataframe to sequence",
    default=f"{os.getcwd()}/../data/viral_genome_bias.csv".replace("\\", "/"),
)
@click.option(
    "--logger_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to logging file",
    default="collect_sequence_data.log",
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    help="boolean indicating weather script should be executed in debug mode",
    default=False,
)
def compute_genome_bias(
    input_path: click.Path,
    output_path: click.Path,
    logger_path: click.Path,
    debug_mode: np.float64,
):
    # initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(logger_path)),
        ],
    )

    virus_sequence_df = pd.read_csv(input_path)
    genomic_bias_df = pd.DataFrame()

    # collect genomic bias features
    for index, row in virus_sequence_df.iterrows():
        record = {"taxon_name": row.taxon_name, "accession": row.accession}
        genomic_sequence = row.sequence
        coding_sequence = GenomeBiasCollectingService.extract_coding_sequence(
            genomic_sequence=row.sequence, coding_regions=row.cds
        )
        genomic_features = GenomeBiasCollectingService.collect_genomic_bias_features(
            genome_sequence=genomic_sequence,
            coding_sequence=coding_sequence,
        )
        record.update(genomic_features)
        genomic_bias_df = genomic_bias_df.append(record, ignore_index=True)

    genomic_bias_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    compute_genome_bias()
