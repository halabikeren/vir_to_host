import multiprocessing
import os
import sys
import logging

logger = logging.getLogger(__name__)

import click
import pandas as pd
import numpy as np

sys.path.append("..")
from utils.sequence_utils import GenomeBiasCollectingService
from utils.parallelization_service import ParallelizationService


@click.command()
@click.option(
    "--virus_sequence_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the dataframe of virus-host associations",
    default=f"{os.getcwd()}/../data/virus_data.csv".replace("\\", "/"),
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path that will hold a dataframe mapping virus taxon name and id from the associations dataframe to sequence",
    default=f"{os.getcwd()}/../data/virus_data_united.csv".replace("\\", "/"),
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
    virus_sequence_data_path: click.Path,
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
            logging.FileHandler(logger_path),
        ],
    )

    # read data
    virus_sequence_data = pd.read_csv(virus_sequence_data_path)
    virus_sequence_data.sort_values("taxon_name", inplace=True)

    # correct annotations in df
    genomic_bias_df = ParallelizationService.parallelize(
        df=virus_sequence_data,
        func=GenomeBiasCollectingService.compute_genome_bias_features,
        num_of_processes=multiprocessing.cpu_count() - 1,
    )

    genomic_bias_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    compute_genome_bias()
