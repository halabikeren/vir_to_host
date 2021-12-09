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
    "--input_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the dataframe of virus sequence accessions if available",
    required=True,
)
@click.option(
    "--ncbi_seq_data_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path to ncbi ftp db parsed data",
    required=False,
    default=f"{os.getcwd()}/../data/databases/ncbi_viral_seq_data.csv".replace(
        "\\", "/"
    ),
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path that will hold a dataframe with the sequence data corresponding to the accessions",
    required=True,
)
@click.option(
    "--logger_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to logging file",
    required=False,
    default="collect_sequence_data.log",
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    help="boolean indicating weather script should be executed in debug mode",
    required=False,
    default=False,
)
@click.option(
    "--collect_for_all",
    type=click.BOOL,
    help="indicator weather accessions should be collected for all taxa in the input df or only for those without available accessions",
    required=False,
    default=True
)
@click.option(
    "--use_multiprocessing",
    type=click.BOOL,
    help="indicator weather multiprocessing should be used or not",
    required=False,
    default=True
)
def collect_complementary_genomic_data(
    input_path: click.Path,
    ncbi_seq_data_path: click.Path,
    output_path: click.Path,
    logger_path: click.Path,
    debug_mode: np.float64,
    collect_for_all: bool,
    use_multiprocessing: bool,
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
    virus_data = pd.read_csv(input_path)
    has_accessions = virus_data.loc[virus_data.accession.notna()].shape[0] > 0

    if not collect_for_all: # this data consist of viral species for either on sequence or a single sequence is available.

        # divide data to: data with accession and sequence, data with accesison but without sequence, and data without accession
        complete_data = virus_data.loc[virus_data.sequence.notna()]
        logger.info(f"# records with no missing data = {complete_data.shape[0]}")

        data_with_no_accession = virus_data.loc[virus_data.accession.isna()]
        logger.info(f"# records with no accession data = {data_with_no_accession.shape[0]}")

        # complement data without accession - first by taxon name, and in case of failure - by species name
        logger.info(
            f"missing data before completion of accession by tax names:\n{virus_data.isna().sum()}"
        )

        if use_multiprocessing:
            data_with_no_accession = ParallelizationService.parallelize(
                df=data_with_no_accession,
                func=SequenceCollectingUtils.fill_missing_data_by_organism,
                num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
            )
        else:
            data_with_no_accession = pd.read_csv(SequenceCollectingUtils.fill_missing_data_by_organism(df=data_with_no_accession))

        logger.info(
            f"missing data after completion of accession by tax names:\n{virus_data.isna().sum()}"
        )

        # unite collect data with original data
        virus_data = pd.concat([complete_data, data_with_no_accession])

        # divide data to: data with accession and sequence, data with accession but without sequence, and data without accession
        rest_of_data = virus_data.loc[(virus_data.accession.isna()) | (virus_data.sequence.notna())]

        data_with_accession = virus_data.loc[(virus_data.accession.notna()) & (virus_data.sequence.isna())]
        logger.info(f"# records with accession but missing sequence data = {data_with_accession.shape[0]}")

        # complement data with accessions
        logger.info(
            f"missing data before completion of sequence data by accession:\n{virus_data.isna().sum()}"
        )

        if use_multiprocessing:
            data_with_accession = ParallelizationService.parallelize(
                df=data_with_accession,
                func=SequenceCollectingUtils.fill_missing_data_by_acc,
                num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
            )
        else:
            data_with_accession = pd.read_csv(SequenceCollectingUtils.fill_missing_data_by_acc(df=data_with_no_accession))

        logger.info(
            f"missing data after completion of sequence data by accession:\n{virus_data.isna().sum()}"
        )

        virus_data = pd.concat([rest_of_data, data_with_accession])
        virus_data.to_csv(output_path, index=False)

        # complement data without accessions with data from ncbi ftp dataframe
        if os.path.exists(str(ncbi_seq_data_path)):
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
        virus_data = pd.concat([rest_of_data, data_with_accession])

        # complement missing accessions from ncbi ftp accessions list at: "/groups/itay_mayrose/halabikeren/vir_to_host/data/databases/ncbi_viral_genome_accessions/taxid10239.nbr"
        # do manually for curation (only 38 instances in total)

        logger.info(
            f"missing data before completion by accession:\n{virus_data.isna().sum()}"
        )

        # for additional missing data, complement using ncbi esearch queries
        logger.info(
            "complementing data with no accessions using esearch api queries to ncbi"
        )
        virus_complete_data = virus_data.loc[virus_data.accession.notna()]
        virus_missing_data = virus_data.loc[virus_data.accession.isna()]

        if use_multiprocessing:
            virus_missing_data = ParallelizationService.parallelize(
                df=virus_missing_data,
                func=SequenceCollectingUtils.fill_missing_data_by_organism,
                num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
            )
        else:
            virus_missing_data = pd.read_csv(SequenceCollectingUtils.fill_missing_data_by_organism(df=virus_missing_data))

        virus_data = pd.concat([virus_complete_data, virus_missing_data])

        logger.info(
            f"missing data before completion by accession:\n{virus_data.isna().sum()}"
        )

        virus_data.to_csv(output_path, index=False)

    else:

        virus_complete_data = virus_data.loc[virus_data.sequence.notna()]


        # for additional missing data, complement using ncbi esearch queries

        logger.info(
            f"missing data before completion:\n{virus_data.isna().sum()}"
        )

        if not has_accessions:
            virus_additional_data = virus_data[["taxon_name", "taxon_id", "species_name", "species_id"]].drop_duplicates()
            for col in virus_data.columns:
                if col not in virus_additional_data.columns:
                    virus_additional_data[col] = np.nan

            logger.info(
                f"complementing data with no accessions using esearch api queries to ncbi for {virus_additional_data.shape[0]} tax ids"
            )

            if use_multiprocessing:
                virus_additional_data = ParallelizationService.parallelize(
                    df=virus_additional_data,
                    func=SequenceCollectingUtils.fill_missing_data_by_organism,
                    num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
                )
            else:
                virus_additional_data = pd.read_csv(SequenceCollectingUtils.fill_missing_data_by_organism(df=virus_additional_data))

            virus_data = pd.concat([virus_complete_data, virus_additional_data])

        else:


            virus_data_without_accessions = virus_data.loc[virus_data.accession.isna()]
            virus_data_with_accessions = virus_data.loc[virus_data.accession.notna()]

            if virus_data_without_accessions.shape[0] > 0:
                logger.info(
                    f"complementing data with no accessions using esearch api queries to ncbi for {virus_data_without_accessions.shape[0]} tax ids"
                )

                if use_multiprocessing:
                    virus_data_without_accessions = ParallelizationService.parallelize(
                        df=virus_data_without_accessions,
                        func=SequenceCollectingUtils.fill_missing_data_by_organism,
                        num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
                    )
                else:
                    virus_data_without_accessions = pd.read_csv(SequenceCollectingUtils.fill_missing_data_by_organism(df=virus_data_without_accessions))

                virus_data = pd.concat([virus_data_without_accessions, virus_data_with_accessions])
                virus_data.to_csv(output_path)

            if virus_data_with_accessions.shape[0] > 0:
                logger.info(
                    f"complementing data with accessions using efetch api queries to ncbi for {virus_data_with_accessions.shape[0]} accessions"
                )

                if use_multiprocessing:
                    virus_data_with_accessions = ParallelizationService.parallelize(
                        df=virus_data_with_accessions,
                        func=SequenceCollectingUtils.fill_missing_data_by_acc,
                        num_of_processes=np.min([multiprocessing.cpu_count() - 1, 10]),
                    )
                else:
                    virus_data_with_accessions = pd.read_csv(SequenceCollectingUtils.fill_missing_data_by_acc(df=virus_data_with_accessions))

                virus_data = pd.concat([virus_data_without_accessions, virus_data_with_accessions])

        logger.info(
            f"missing data after completion:\n{virus_data.isna().sum()}"
        )

        virus_data.drop_duplicates(subset=["accession"], inplace=True)
        virus_data.to_csv(output_path)

if __name__ == "__main__":
    collect_complementary_genomic_data()
