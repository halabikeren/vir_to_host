import multiprocessing
import os
import sys
import logging
from functools import partial

logger = logging.getLogger(__name__)

import click
import pandas as pd
import numpy as np

sys.path.append("..")
from utils.sequence_utils import SequenceCollectingUtils
from utils.parallelization_service import ParallelizationService


def report_missing_data(virus_data: pd.DataFrame):
    for source in ["refseq", "genbank"]:
        viruses_with_acc_and_missing_data = virus_data.loc[
            (virus_data.source == source)
            & (virus_data.accession.notna())
            & (
                (virus_data.sequence.isna())
                | (virus_data.cds.isna())
                | (virus_data.annotation.isna())
            )
        ]
        logger.info(
            f"# viruses with {source} accessions and missing data = {viruses_with_acc_and_missing_data.shape[0]}"
        )

    viruses_with_no_acc_and_missing_data = list(
        virus_data.loc[virus_data.accession.isna(), "taxon_name"].unique()
    )
    logger.info(
        f"# viruses viruses with no accession and missing data = {len(viruses_with_no_acc_and_missing_data)}\n\n"
    )

    logger.info(f"missing records across each columns=\n{virus_data.isnull().sum()}")


@click.command()
@click.option(
    "--virus_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the dataframe of virus sequence accessions if available",
    default=f"{os.getcwd()}/../data/virus_sequence_data_template.csv".replace(
        "\\", "/"
    ),
)
@click.option(
    "--ncbi_seq_data_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path to ncbi ftp db parsed data",
    default=f"{os.getcwd()}/../data/databases/ncbi_viral_seq_data.csv".replace(
        "\\", "/"
    ),
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path that will hold a dataframe mapping virus taxon name and id from the associations dataframe to sequence",
    default=f"{os.getcwd()}/../data/virus_sequence_data.csv".replace("\\", "/"),
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
def correct_sequence_data(
    virus_data_path: click.Path,
    ncbi_seq_data_path: click.Path,
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
    virus_data = pd.read_csv(virus_data_path)

    # divide data to: data with refseq/genbank accessions, data with gi accessions to convert to actual accessions and data with no accessions
    complete_data = virus_data.loc[virus_data.sequence.notna()]
    data_with_accession = virus_data.loc[
        (virus_data.accession.notna()) & (virus_data.sequence.isna())
    ]
    data_with_no_accession = virus_data.loc[
        virus_data.accession.isna() & (virus_data.sequence.isna())
    ]

    # complement data with accessions
    data_with_accession = ParallelizationService.parallelize(
        df=data_with_accession,
        func=partial(
            SequenceCollectingUtils.fill_missing_data_by_acc,
        ),
        num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
    )

    # complement data without accessions with data from ncbi ftp dataframe
    if os.path.exists(ncbi_seq_data_path):
        ncbi_ftp_data = pd.read_csv(ncbi_seq_data_path)
        ncbi_ftp_data.set_index("taxon_id", inplace=True)
        data_with_no_accession.set_index("taxon_id", inplace=True)
        for column in data_with_no_accession.columns:
            if column in ncbi_ftp_data.columns and column != "taxon_id":
                data_with_no_accession[column].fillna(
                    value=ncbi_ftp_data[column].to_dict(), inplace=True
                )
        data_with_no_accession.reset_index()

    # concat data
    virus_data = pd.concat([complete_data, data_with_accession, data_with_no_accession])

    # for additional missing data, complement using ncbi esearch queries
    virus_complete_data = virus_data.loc[virus_data.accession.notna()]
    virus_missing_data = virus_data.loc[virus_data.accession.isna()]
    virus_missing_data = ParallelizationService.parallelize(
        df=virus_missing_data,
        func=partial(
            SequenceCollectingUtils.fill_missing_data_by_organism,
        ),
        num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
    )

    virus_data = pd.concat([virus_complete_data, virus_missing_data])

    virus_data.to_csv(output_path, index=False)

    virus_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    correct_sequence_data()
