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
                | (virus_data.keywords.isna())
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


@click.command()
@click.option(
    "--virus_data_path",
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
def collect_sequence_data(
    virus_data_path: click.Path,
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
    for col in virus_data.columns:
        if col != "virus_taxon_name" and virus_data[col].dtypes == object:
            virus_data[col] = virus_data[col].str.upper()

    # correct accessions, if needed
    virus_data["virus_refseq_accession"] = virus_data["virus_refseq_accession"].apply(
        lambda x: x.replace("*", "") if type(x) is str else x
    )
    virus_data["virus_genbank_accession"] = virus_data["virus_refseq_accession"].apply(
        lambda x: x.replace("*", "") if type(x) is str else x
    )

    # flatten df
    flattened_virus_data = ParallelizationService.parallelize(
        df=virus_data,
        func=partial(SequenceCollectingUtils.flatten_sequence_data),
        num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
    )

    # report missing data
    report_missing_data(virus_data=flattened_virus_data)

    # complete missing data
    flattened_virus_missing_data = flattened_virus_data.loc[
        (flattened_virus_data.accession.notna())
        & (
            (flattened_virus_data.sequence.isna())
            | (flattened_virus_data.cds.isna())
            | (flattened_virus_data.annotation.isna())
        )
    ]
    if flattened_virus_missing_data.shape[0] > 0:
        logger.info(
            f"complementing missing data by accessions for {flattened_virus_missing_data.shape[0]} records"
        )
        flattened_virus_missing_data = ParallelizationService.parallelize(
            df=flattened_virus_missing_data,
            func=partial(
                SequenceCollectingUtils.fill_missing_data_by_acc,
            ),
            num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
        )
        flattened_virus_missing_data.set_index("taxon_name", inplace=True)
        for col in flattened_virus_missing_data.columns:
            if col not in ["taxon_name", "accession"]:
                flattened_virus_data[col].fillna(
                    value=flattened_virus_missing_data[col].to_dict(), inplace=True
                )
        flattened_virus_data.reset_index(inplace=True)
        flattened_virus_data.to_csv(output_path, index=False)

        # report missing data
        report_missing_data(virus_data=flattened_virus_data)

    # complete missing data with direct api requests
    virus_missing_data = flattened_virus_data.loc[
        flattened_virus_data["accession"].isna()
    ]
    if virus_missing_data.shape[0] > 0:
        logger.info(
            f"complementing missing data by name for {virus_missing_data.shape[0]} records"
        )
        virus_missing_data = ParallelizationService.parallelize(
            df=virus_missing_data,
            func=partial(
                SequenceCollectingUtils.fill_missing_data_by_id,
                data_prefix="",
                id_field="taxon_name",
            ),
            num_of_processes=np.min(
                [multiprocessing.cpu_count() - 1, 3]
            ),  # here, allow less cpus because each process can file multiple requests at the same time
        )
        flattened_virus_data.set_index("taxon_name", inplace=True)
        virus_missing_data.set_index("taxon_name", inplace=True)
        for col in flattened_virus_data.columns:
            if col != "taxon_name":
                flattened_virus_data[col].fillna(
                    value=virus_missing_data[col].to_dict(), inplace=True
                )

        # report missing data
        report_missing_data(virus_data=flattened_virus_data)

    flattened_virus_data.reset_index(inplace=True)
    flattened_virus_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    collect_sequence_data()
