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

    viruses_with_missing_genbank_data = virus_data.loc[
        (virus_data.virus_genbank_accession.notna())
        & (
            (virus_data.virus_genbank_sequence.isna())
            | (virus_data.virus_genbank_cds.isna())
        )
    ]

    viruses_with_missing_refseq_data = virus_data.loc[
        (virus_data.virus_refseq_accession.notna())
        & (
            (virus_data.virus_refseq_sequence.isna())
            | (virus_data.virus_refseq_cds.isna())
        )
    ]

    viruses_with_missing_gi_data = virus_data.loc[
        (virus_data.virus_gi_accession.notna())
        & (
            (virus_data.virus_genbank_cds.isna())
            & (virus_data.virus_genbank_sequence.isna())
        )
        | (
            (virus_data.virus_refseq_cds.isna())
            & (virus_data.virus_refseq_sequence.isna())
        )
        & (
            ~virus_data.virus_taxon_name.isin(
                viruses_with_missing_genbank_data.virus_taxon_name.unique()
            )
        )
        & (
            ~virus_data.virus_taxon_name.isin(
                viruses_with_missing_refseq_data.virus_taxon_name.unique()
            )
        )
    ]
    viruses_with_missing_data = virus_data.loc[
        (virus_data.virus_genbank_accession.isna())
        & (virus_data.virus_refseq_accession.isna())
        & (virus_data.virus_gi_accession.isna())
    ]
    logger.info(
        f"#missing viruses with gb acc = {viruses_with_missing_genbank_data.shape[0]}\n"
        f"#missing viruses with refseq acc = {viruses_with_missing_refseq_data.shape[0]}\n"
        f"#missing viruses with gi acc = {viruses_with_missing_gi_data.shape[0]}\n"
        f"# missing viruses with no acc = {viruses_with_missing_data.shape[0]}"
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
@click.option(
    "--cpus_num",
    type=click.INT,
    help="number of cpus to parallelize over",
    default=multiprocessing.cpu_count() - 1,
)
def collect_sequence_data(
    virus_data_path: click.Path,
    output_path: click.Path,
    logger_path: click.Path,
    debug_mode: np.float64,
    cpus_num: int,
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

    # report missing data
    report_missing_data(virus_data=virus_data)

    # extract sequence data from refseq accessions
    refseq_virus_data = virus_data.loc[
        (virus_data.virus_refseq_accession.notna())
        & (
            (virus_data.virus_refseq_sequence.isna())
            | (virus_data.virus_refseq_cds.isna())
        )
    ]
    refseq_virus_data = ParallelizationService.parallelize(
        df=refseq_virus_data,
        func=partial(
            SequenceCollectingUtils.extract_missing_data_from_ncbi_api_by_unique_acc,
            acc_field_name="virus_refseq_accession",
        ),
        num_of_processes=cpus_num,
    )
    virus_data.set_index("virus_taxon_name", inplace=True)
    refseq_virus_data.set_index("virus_taxon_name", inplace=True)
    for col in virus_data.columns:
        if col != "virus_taxon_name":
            virus_data[col].fillna(value=refseq_virus_data[col].to_dict(), inplace=True)
    virus_data.reset_index(inplace=True)
    virus_data.to_csv(output_path, index=False)

    # report missing data
    report_missing_data(virus_data=virus_data)

    # extract sequence data from genbank accessions
    genbank_virus_data = virus_data.loc[
        (virus_data.virus_genbank_accession.notna())
        & (
            (virus_data.virus_genbank_sequence.isna())
            | (virus_data.virus_genbank_cds.isna())
        )
    ]
    genbank_virus_data = ParallelizationService.parallelize(
        df=genbank_virus_data,
        func=partial(
            SequenceCollectingUtils.extract_missing_data_from_ncbi_api_by_unique_acc,
            acc_field_name="virus_genbank_accession",
        ),
        num_of_processes=cpus_num,
    )
    virus_data.set_index("virus_taxon_name", inplace=True)
    genbank_virus_data.set_index("virus_taxon_name", inplace=True)
    for col in virus_data.columns:
        if col != "virus_taxon_name":
            virus_data[col].fillna(
                value=genbank_virus_data[col].to_dict(), inplace=True
            )
    virus_data.reset_index(inplace=True)
    virus_data.to_csv(output_path, index=False)

    # report missing data
    report_missing_data(virus_data=virus_data)

    # extract sequence data from gi accessions
    gi_virus_data = virus_data.loc[
        (virus_data.virus_gi_accession.notna())
        & (
            (
                (virus_data.virus_refseq_sequence.isna())
                & (virus_data.virus_refseq_cds.isna())
            )
            | (
                (virus_data.virus_genbank_sequence.isna())
                & (virus_data.virus_genbank_cds.isna())
            )
        )
    ]
    gi_virus_data = ParallelizationService.parallelize(
        df=gi_virus_data,
        func=partial(
            SequenceCollectingUtils.extract_missing_data_from_ncbi_api_by_gi_acc,
            data_prefix="virus",
            acc_field_name="virus_gi_accession",
        ),
        num_of_processes=cpus_num,
    )
    virus_data.set_index("virus_taxon_name", inplace=True)
    gi_virus_data.set_index("virus_taxon_name", inplace=True)
    for col in virus_data.columns:
        if col != "virus_taxon_name":
            virus_data[col].fillna(value=gi_virus_data[col].to_dict(), inplace=True)
    virus_data.reset_index(inplace=True)
    virus_data.to_csv(output_path, index=False)

    # report missing data
    report_missing_data(virus_data=virus_data)

    # complete missing data with direct api requests
    virus_missing_data = virus_data.loc[
        (virus_data["virus_gi_accession"].isna())
        & (virus_data["virus_refseq_accession"].isna())
        & (virus_data["virus_genbank_accession"].isna())
    ]
    virus_missing_data = ParallelizationService.parallelize(
        df=virus_missing_data,
        func=partial(
            SequenceCollectingUtils.extract_missing_data_from_ncbi_api,
            data_prefix="virus",
            id_field="taxon_name",
        ),
        num_of_processes=cpus_num,
    )
    virus_data.set_index("virus_taxon_name", inplace=True)
    virus_missing_data.set_index("virus_taxon_name", inplace=True)
    for col in virus_data.columns:
        if col != "virus_taxon_name":
            virus_data[col].fillna(
                value=virus_missing_data[col].to_dict(), inplace=True
            )
    virus_data.reset_index(inplace=True)
    virus_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    collect_sequence_data()
