import json
import os
import sys
import re
import shutil

import logging

logger = logging.getLogger(__name__)

import click
import pandas as pd
import numpy as np
import wget
from Bio import Entrez, SeqIO

Entrez.email = "halabikeren@gmail.com"

from utils.data_collecting_utils import DataCleanupUtils, SequenceCollectingUtils


def extract_ictv_accessions(df: pd.DataFrame, ictv_data_path: str):
    """
    :param df: dataframe to fill accessions in
    :param ictv_data_path: path to ictv data
    :return: none, alters the dataframe inplace
    """
    # process accessions from ICTV-DB
    ictv_db = pd.read_excel(ictv_data_path)
    ictv_db["Virus name(s)"] = ictv_db["Virus name(s)"].str.lower()

    # fill genbank accessions
    relevant_df = ictv_db.loc[
        (ictv_db["Virus name(s)"].isin(df.virus_taxon_name.unique()))
        & (ictv_db["Virus GENBANK accession"].notna())
    ]
    genbank_accessions = relevant_df.set_index("Virus name(s)")[
        "Virus GENBANK accession"
    ].to_dict()
    df.set_index("virus_taxon_name", inplace=True)
    df["virus_genbank_accession"].fillna(value=genbank_accessions, inplace=True)
    df.reset_index(inplace=True)

    # fill refseq accessions
    relevant_df = ictv_db.loc[
        (ictv_db["Virus name(s)"].isin(df.virus_taxon_name.unique()))
        & (ictv_db["Virus REFSEQ accession"].notna())
    ]
    refseq_accessions = relevant_df.set_index("Virus name(s)")[
        "Virus REFSEQ accession"
    ].to_dict()
    df.set_index("virus_taxon_name", inplace=True)
    df["virus_refseq_accession"].fillna(value=refseq_accessions, inplace=True)
    df.reset_index(inplace=True)


def extract_refseq_sequences(df: pd.DataFrame, refseq_data_dir: str):
    """

    :param df: dataframe to fill with refseq accessions
    :param refseq_data_dir: directory holding refseq sequence files, downloaded from ncbi ftp services
    :return: none, changes the dataframe inplace
    """

    logger.info(
        f"missing refseq accessions before ={df.loc[df.virus_refseq_accession.isna()].shape[0]}\nmissing sequences before = {df.loc[df.virus_refseq_sequence.isna()].shape[0]}"
    )

    # complement missing accessions
    refseq_accessions = pd.read_csv(
        f"{refseq_data_dir}/refseq_ncbi_accessions.tbl", sep="\t"
    )
    refseq_accessions["Genome"] = refseq_accessions["Genome"].str.strip().str.lower()
    complete_refseq_accessions = refseq_accessions.loc[
        refseq_accessions["RefSeq type"] == "complete"
    ]
    vir_to_acc = complete_refseq_accessions.set_index("Genome")["Accession"].to_dict()

    df.set_index("virus_taxon_name", inplace=True)
    df["virus_refseq_accession"].fillna(value=vir_to_acc, inplace=True)
    df["virus_refseq_accession"].replace("-", np.nan, inplace=True)
    df.reset_index(inplace=True)

    # extract sequence data from available files
    sequence_paths = [
        f"{refseq_data_dir}/{path}"
        for path in os.listdir(refseq_data_dir)
        if ".fna" in path or ".fasta" in path
    ]
    sequences = []
    for path in sequence_paths:
        sequences += list(SeqIO.parse(path, format="fasta"))
    accession_to_sequence = {item.id.split(".")[0]: item.seq for item in sequences}

    df.loc[df["virus_refseq_accession"].notna(), "refseq_sequence"] = df.loc[
        df["virus_refseq_accession"].notna(), "virus_refseq_accession"
    ].apply(
        lambda x: SequenceCollectingUtils.get_sequence(
            x, sequence_data=accession_to_sequence
        )
    )

    # complement missing data via ftp request
    missing_refseq_accessions_data = df.loc[
        (df["virus_refseq_accession"].notna()) & (df["refseq_sequence"].isna()),
        "virus_refseq_accession",
    ]
    missing_refseq_accessions_accessions = ",".join(
        missing_refseq_accessions_data.apply(
            SequenceCollectingUtils.get_accession
        ).values
    )

    missing_refseq_accessions_unique_accessions = ",".join(
        list(set(missing_refseq_accessions_accessions.split(",")))
    )
    records = list(
        Entrez.parse(
            Entrez.efetch(
                db="nucleotide",
                id=missing_refseq_accessions_unique_accessions,
                retmode="xml",
            )
        )
    )
    complementary_accession_to_sequence = {
        records[i]["GBSeq_locus"]: records[i]["GBSeq_sequence"]
        for i in range(len(records))
        if "GBSeq_sequence" in records[i]
    }
    df.loc[df["virus_refseq_accession"].notna(), "refseq_sequence"] = df.loc[
        df["virus_refseq_accession"].notna(), "virus_refseq_accession"
    ].apply(
        lambda x: SequenceCollectingUtils.get_sequence(
            x, sequence_data=complementary_accession_to_sequence
        )
    )

    # complement data of viruses with missing refseq accessions via the ncbi ftp services
    with open(f"{refseq_data_dir}/refseq_ftp_viruses.json", "r") as infile:
        viruses_to_ftp_dir = json.load(infile)
    ftp_ncbi_virus_to_seq = dict()
    regex = re.compile("(\w*_\d*.fna)")
    for virus in [
        virus
        for virus in df.loc[df.refseq_sequence.isna(), "virus_taxon_name"]
        if virus in viruses_to_ftp_dir
    ]:
        os.makedirs(f"{refseq_data_dir}/{virus}", exist_ok=True)
        os.chdir(f"{refseq_data_dir}/{virus}")
        try:
            virus_dir = f"https://ftp.ncbi.nlm.nih.gov/genomes/Viruses/{viruses_to_ftp_dir[virus]}"
            download_info = wget.download(virus_dir)
            with open(download_info, "r") as infile:
                file_name = regex.search(infile.read()).group(1)
            filename = wget.download(f"{virus_dir}/{file_name}")
            ftp_ncbi_virus_to_seq[virus] = [
                str(list(SeqIO.parse(filename, format="fasta"))[0].id),
                str(list(SeqIO.parse(filename, format="fasta"))[0].seq),
            ]
        except:
            print(virus)
        os.chdir(refseq_data_dir)
        shutil.rmtree(virus)

        df.loc[(df.virus_refseq_accession.notna()) & (df.refseq_sequence.isna())][
            ["virus_refseq_accession", "virus_refseq_sequence"]
        ] = df.loc[(df.virus_refseq_accession.notna()) & (df.refseq_sequence.isna())][
            ["virus_taxon_name"]
        ].apply(
            lambda x: SequenceCollectingUtils.get_seq_data_from_virus_name(
                x, ftp_ncbi_virus_to_seq
            ),
            axis=1,
            result_type="expand",
        )

        logger.info(
            f"missing refseq accessions after ={df.loc[df.virus_refseq_accession.isna()].shape[0]}\nmissing sequences after = {df.loc[df.virus_refseq_sequence.isna()].shape[0]}"
        )


def extract_genbank_sequences(df, genbank_data_dir):
    """
    :param df: dataframe to fill with genbank sequences
    :param genbank_data_dir: directory holding genbank sequences downloaded from vipr-db
    :return: none, alters the dataframe inplace
    """
    # paths = [
    #     f"{genbank_data_dir}/{path}"
    #     for path in os.listdir(genbank_data_dir)
    #     if ".fasta" in path
    # ]
    # for path in paths:
    #     logger.info(
    #         f"path={path}\n#missing sequences={df.loc[(df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna())].shape[0]}"
    #     )
    #     (
    #         virus_taxon_name_to_seq,
    #         virus_taxon_name_to_gb,
    #     ) = SequenceCollectingUtils.get_sequence_info(path)
    #     df.set_index("virus_taxon_name", inplace=True)
    #     df["virus_genbank_accession"].fillna(value=virus_taxon_name_to_gb, inplace=True)
    #     df["genbank_sequence"].fillna(value=virus_taxon_name_to_seq, inplace=True)
    #     logger.info(
    #         f"#missing sequences={df.loc[(df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna())].shape[0]}\n\n"
    #     )
    #     df.reset_index(inplace=True)

    # complement missing data using api requests
    logger.info(
        f"#missing sequences before genbank API search={df.loc[(df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna())].shape[0]}\n\n"
    )

    viruses_with_missing_sequences = df.loc[
        (df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna()),
        "virus_taxon_name",
    ].unique()
    batch_size = 1000
    virus_names_batches = [
        viruses_with_missing_sequences[i : i + batch_size]
        for i in range(0, len(viruses_with_missing_sequences), batch_size)
    ]
    suffix = ") NOT gene[Text Word]) NOT protein[Text Word]) NOT partial[Text Word]"
    text_queries = [
        "(((" + " OR".join([f"({name}[Organism])" for name in batch]) + suffix
        for batch in virus_names_batches
    ]
    record_ids = []

    for query in text_queries:
        record_ids += Entrez.read(
            Entrez.esearch(db="nucleotide", term=query, retmax=10, idtype="acc")
        )["IdList"]

    records = list(
        Entrez.parse(
            Entrez.efetch(db="nucleotide", id=",".join(record_ids), retmode="xml")
        )
    )
    (
        virus_name_to_acc,
        virus_name_to_seq,
    ) = SequenceCollectingUtils.extract_genome_data_from_entrez_result(records)
    df.set_index("virus_taxon_name", inplace=True)
    df["virus_genbank_accession"].fillna(value=virus_name_to_acc, inplace=True)
    df["virus_genbank_sequence"].fillna(value=virus_name_to_seq, inplace=True)
    df.reset_index(inplace=True)

    logger.info(
        f"#missing sequences after genbank API search={df.loc[(df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna())].shape[0]}\n\n"
    )


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
    default="../associations/associations_united.csv",
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
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logger_path),],
    )

    # associations = pd.read_csv(associations_path)
    # virus_data = associations[
    #     [
    #         "virus_taxon_name",
    #         "virus_taxon_id",
    #         "virus_genbank_accession",
    #         "virus_refseq_accession",
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
    # extract_ictv_accessions(
    #     df=virus_data,
    #     ictv_data_path=f"{databases_source_dir}/ICTVDB/ictvdb_sequence_acc.xlsx",
    # )
    #
    # # extract sequence data from refseq
    # virus_data["virus_refseq_sequence"] = np.nan
    # extract_refseq_sequences(
    #     df=virus_data, refseq_data_dir=f"{databases_source_dir}/REFSEQ/"
    # )
    #
    # # extract sequence data from genbank via viprdb
    # virus_data["virus_genbank_sequence"] = np.nan
    virus_data = pd.read_csv(output_path)
    extract_genbank_sequences(
        df=virus_data, genbank_data_dir=f"{databases_source_dir}/viprdb/"
    )

    virus_data.to_csv(output_path, index=False)


if __name__ == "__main__":
    collect_sequence_data()
