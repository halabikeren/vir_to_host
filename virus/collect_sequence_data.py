import multiprocessing
import sys

import logging
from functools import partial

from Bio import Entrez

logger = logging.getLogger(__name__)

import click
import pandas as pd
import numpy as np

sys.path.append("..")
from utils.sequence_utils import SequenceCollectingUtils
from utils.parallelization_service import ParallelizationService


def extract_missing_data_from_ncbi_api(df: pd.DataFrame, data_prefix: str, id_field: str) -> pd.DataFrame:
    """
    :param df: dataframe with items were all the fields except for the id field are missing
    :param data_prefix: data prefix for each column in the dataframe
    :param id_field: name of id field to be indexed, without the data prefix
    :return: the dataframe with values from the ncbi api, if available
    """
    ids = list(df[f"{data_prefix}_{id_field}"].unique())
    logger.info(f"complementing missing accession data from ncbi nucleotide api for {len(ids)} records")
    id_to_raw_data = {name: Entrez.read(
        Entrez.esearch(db="nucleotide", term=f"({name}[Organism]) AND complete genome[Text Word]", retmode="xml"))
        for name in ids}
    logger.info(f"data extraction from ncbi nucleotide api is complete")
    id_to_gi_acc = {name: id_to_raw_data[name]['IdList'][0] for name in id_to_raw_data if
                    len(id_to_raw_data[name]['IdList']) > 0}
    df.set_index(f"{data_prefix}_{id_field}", inplace=True)
    gi_accession_field_name = f"{data_prefix}_gi_accession"
    if gi_accession_field_name not in df.columns:
        df[gi_accession_field_name] = np.nan
    df[gi_accession_field_name].fillna(value=id_to_gi_acc, inplace=True)
    df.reset_index(inplace=True)

    # do batch request on additional data
    logger.info(
        f"complementing missing sequence data from ncbi nucleotide api for {len(list(id_to_gi_acc.values()))} records")
    ncbi_raw_data = list(
        Entrez.parse(Entrez.efetch(dx="nucleotide", id=",".join(list(id_to_gi_acc.values())), retmode="xml")))
    logger.info(f"data extraction from ncbi nucleotide api is complete")

    # process raw data
    logger.info(f"filling dataframe with complementary data from ncbi api")
    gi_acc_to_raw_data = {
        [item.split("|")[1] for item in ncbi_raw_data[i]['GBSeq_other-seqids'] if 'gi' in item][0]: ncbi_raw_data[i]
        for i in range(len(ncbi_raw_data))}
    gi_acc_to_refseq_acc = {
        gi_acc: [item.split("|")[1] for item in gi_acc_to_raw_data[gi_acc]['GBSeq_other-seqids'] if 'ref' in item][
            0] for gi_acc in gi_acc_to_raw_data if
        len([item for item in gi_acc_to_raw_data[gi_acc]['GBSeq_other-seqids'] if 'ref' in item]) > 0}
    gi_acc_to_genbank_acc = {
        gi_acc: [item.split("|")[1] for item in gi_acc_to_raw_data[gi_acc]['GBSeq_other-seqids'] if 'gb' in item][0]
        for gi_acc in gi_acc_to_raw_data if
        len([item for item in gi_acc_to_raw_data[gi_acc]['GBSeq_other-seqids'] if 'gb' in item]) > 0}
    gi_acc_to_refseq_seq = {gi_acc: gi_acc_to_raw_data[gi_acc]['GBSeq_sequence'] for gi_acc in gi_acc_to_refseq_acc}
    gi_acc_to_refseq_cds = {gi_acc: ";".join(
        [
            feature["GBFeature_location"]
            for feature in gi_acc_to_raw_data[gi_acc]["GBSeq_feature-table"]
            if feature["GBFeature_key"] == "CDS"
        ]
    ) for gi_acc in gi_acc_to_refseq_acc}
    gi_acc_to_genbank_seq = {gi_acc: gi_acc_to_raw_data[gi_acc]['GBSeq_sequence'] for gi_acc in
                             gi_acc_to_genbank_acc}
    gi_acc_to_genbank_cds = {gi_acc: ";".join(
        [
            feature["GBFeature_location"]
            for feature in gi_acc_to_raw_data[gi_acc]["GBSeq_feature-table"]
            if feature["GBFeature_key"] == "CDS"
        ]
    ) for gi_acc in gi_acc_to_genbank_acc}

    # fill dataframe with the collected data
    df.set_index(gi_accession_field_name, inplace=True)
    df[f"{data_prefix}_refseq_accession"].fillna(value=gi_acc_to_refseq_acc, inplace=True)
    df[f"{data_prefix}_refseq_sequence"].fillna(value=gi_acc_to_refseq_seq, inplace=True)
    df[f"{data_prefix}_refseq_cds"].fillna(value=gi_acc_to_refseq_cds, inplace=True)
    df[f"{data_prefix}_genbank_accession"].fillna(value=gi_acc_to_genbank_acc, inplace=True)
    df[f"{data_prefix}_genbank_sequence"].fillna(value=gi_acc_to_genbank_seq, inplace=True)
    df[f"{data_prefix}_genbank_cds"].fillna(value=gi_acc_to_genbank_cds, inplace=True)
    df.reset_index(inplace=True)
    logger.info(f"dataframe filling is complete")

    return df

@click.command()
@click.option(
    "--databases_source_dir",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="directory that holds the data collected databases",
    default="../data/databases/virus/",
)
@click.option(
    "--associations_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the dataframe of virus-host associations",
    default="../data/associations_united.csv",
)
@click.option(
    "--output_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path that will hold a dataframe mapping virus taxon name and id from the associations dataframe to sequence",
    default="../data/virus_data.csv",
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
        databases_source_dir: click.Path,
        associations_path: click.Path,
        output_path: click.Path,
        logger_path: click.Path,
        debug_mode: np.float64,
):
    # initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logger_path), ],
    )

    # associations = pd.read_csv(associations_path)
    # virus_data = associations[
    #     [
    #         "virus_taxon_name",
    #         "virus_taxon_id",
    #         "virus_genbank_accession",
    #         "virus_refseq_accession",
    #         "virus_gi_accession",
    #     ]
    # ]
    # virus_data = virus_data.groupby(["virus_taxon_name", "virus_taxon_id"])[
    #     [
    #         col
    #         for col in virus_data.columns
    #         if col not in ["virus_taxon_name", "virus_taxon_id"]
    #     ]
    # ].apply(",".join)
    # virus_data["virus_refseq_accession"] = virus_data[
    #     "virus_refseq_accession"
    # ].str.upper()
    # virus_data["virus_genbank_accession"] = virus_data[
    #     "virus_genbank_accession"
    # ].str.upper()
    #
    # SequenceCollectingUtils.extract_ictv_accessions(
    #     df=virus_data,
    #     ictv_data_path=f"{databases_source_dir}/ICTVDB/ictvdb_sequence_acc.xlsx",
    # )
    #
    # # extract sequence data from refseq
    # virus_data["virus_refseq_sequence"] = np.nan
    # SequenceCollectingUtils.extract_refseq_sequences(
    #     df=virus_data, refseq_data_dir=f"{databases_source_dir}/REFSEQ/"
    # )
    #
    # # extract sequence data from genbank via viprdb
    # virus_data["virus_genbank_sequence"] = np.nan
    # virus_data = pd.read_csv(output_path)
    # SequenceCollectingUtils.extract_genbank_sequences(
    #     df=virus_data, genbank_data_dir=f"{databases_source_dir}/viprdb/"
    # )
    #
    # # extract sequence data from gi accessions
    # gi_accessions = re.split(";|,", (",".join(virus_data.virus_gi_accession.dropna())))
    # record_gi_acc_to_seq = SequenceCollectingUtils.get_gi_sequences_from_ncbi(
    #     gi_accessions=gi_accessions
    # )
    # virus_data["virus_gi_sequences"] = virus_data["virus_gi_accession"].apply(
    #     lambda x: SequenceCollectingUtils.get_gi_sequences_from_df(gi_accessions=x, gi_acc_to_seq=record_gi_acc_to_seq)
    # )
    #
    # # extract cds locations for sequences with available data
    # logger.info("extracting refseq coding sequences locations")
    # virus_refseq_accessions = [
    #     acc.split(":")[-1].replace("*", "")
    #     for acc in virus_data.virus_refseq_accession.dropna().unique()
    # ]
    # virus_refseq_acc_to_cds = SequenceCollectingUtils.get_coding_regions(
    #     virus_refseq_accessions
    # )
    # virus_data["virus_refseq_cds"] = virus_data["virus_refseq_accession"].apply(
    #     lambda x: SequenceCollectingUtils.get_cds(accessions=x, acc_to_cds=virus_refseq_acc_to_cds)
    # )
    # virus_data.to_csv(output_path, index=False)
    # logger.info("refseq coding sequences locations extraction is complete")
    #
    # logger.info("extracting genbank coding sequences locations")
    # virus_genbank_accessions = [
    #     acc.split(":")[-1].replace("*", "")
    #     for acc in virus_data.virus_genbank_accession.dropna().unique()
    # ]
    # virus_genbank_acc_to_cds = SequenceCollectingUtils.get_coding_regions(
    #     virus_genbank_accessions
    # )
    # virus_data["virus_genbank_cds"] = virus_data["virus_genbank_accession"].apply(
    #     lambda x: SequenceCollectingUtils.get_cds(accessions=x, acc_to_cds=virus_genbank_acc_to_cds)
    # )
    # virus_data.to_csv(output_path, index=False)
    # logger.info("genbank coding sequences locations extraction is complete")
    #
    # logger.info("extracting gi coding sequences locations")
    # virus_gi_accessions = [
    #     ",".join(acc.split(";"))
    #     for acc in virus_data.virus_gi_accession.dropna().unique()
    # ]
    # virus_gi_acc_to_cds = SequenceCollectingUtils.get_coding_regions(
    #     virus_gi_accessions
    # )
    # virus_data["virus_gi_cds"] = virus_data["virus_gi_accession"].apply(
    #     lambda x: SequenceCollectingUtils.get_cds(accessions=x, acc_to_cds=virus_gi_acc_to_cds)
    # )
    # virus_data.to_csv(output_path, index=False)
    # logger.info("gi coding sequences locations extraction is complete")

    virus_data = pd.read_csv(output_path)

    # complete missing data with direct api requests
    virus_missing_data = virus_data.loc[
        (virus_data["virus_refseq_sequence"].isna()) & (virus_data["virus_genbank_sequence"].isna())]
    virus_missing_data = ParallelizationService.parallelize(df=virus_missing_data,
                                                            func=partial(
                                                                extract_missing_data_from_ncbi_api,
                                                                data_prefix="virus", id_field="taxon_name"),
                                                            num_of_processes=multiprocessing.cpu_count())

    virus_data.update(virus_missing_data)
    virus_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    collect_sequence_data()
