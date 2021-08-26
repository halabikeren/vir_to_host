import json
import logging
import multiprocessing
import os
import shutil
import typing as t
import re
from enum import Enum
from functools import partial
import wget

import Bio
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from Bio.Seq import Seq

from .parallelization_service import ParallelizationService

logger = logging.getLogger(__name__)

NUCLEOTIDES = ["A", "C", "G", "T"]
STOP_CODONS = Bio.Data.CodonTable.standard_dna_table.stop_codons
CODONS = list(Bio.Data.CodonTable.standard_dna_table.forward_table.keys()) + STOP_CODONS
AMINO_ACIDS = set(Bio.Data.CodonTable.standard_dna_table.forward_table.values())


class SequenceType(Enum):
    GENOME = 1
    CDS = 2
    PROTEIN = 3


class DinucleotidePositionType(Enum):
    REGULAR = 1
    BRIDGE = 2
    NONBRIDGE = 3


class GenomeType(Enum):
    RNA = 0
    DNA = 1
    UNKNOWN = np.nan


class SequenceCollectingUtils:
    @staticmethod
    def get_sequence(record: str, sequence_data):
        regex = re.compile("(\w*_\s*\d*)")
        accessions = [item.group(1).replace(" ", "") for item in regex.finditer(record)]
        seq = ""
        for acc in accessions:
            if acc in sequence_data:
                seq += str(sequence_data[acc])
            else:
                print(f"record={record}, missing acc={acc}")
                return np.nan
        if len(seq) == 0:
            print(f"record={record}")
            return np.nan
        return seq

    @staticmethod
    def get_accession(record: str):
        regex = re.compile("(\w*_\s*\d*)")
        accessions = [
            item.group(1)
                .replace(" ", "")
                .replace("L: ", "")
                .replace("M: ", "")
                .replace("S: ", "")
                .replace("; ", ",")
            for item in regex.finditer(record)
        ]
        return ",".join(accessions)

    @staticmethod
    def extract_accessions(accession_records):
        regex = re.compile("(\w*\s*\d*)")
        accessions = []
        for record in accession_records:
            record_accessions = [
                item.group(1).replace(" ", "") for item in regex.finditer(record)
            ]
            accessions += record_accessions
        return accessions

    @staticmethod
    def get_seq_data_from_virus_name(name, seq_data):
        acc, seq = np.nan, np.nan
        if name.values[0] in seq_data:
            data = seq_data[name.values[0]]
            acc = data[0]
            seq = data[1]
        return acc, seq

    @staticmethod
    def get_sequence_info(path: str) -> t.Tuple[t.Dict[str, str], t.Dict[str, str]]:
        sequence_data = list(SeqIO.parse(path, format="fasta"))
        name_regex = re.compile(
            "gb\:([^|]*)\|Organism\:([^|]*)\|.*?Strain Name\:([^|]*)"
        )
        virus_taxon_name_to_seq = {
            name_regex.search(item.description).group(2).lower()
            + " "
            + name_regex.search(item.description).group(3).lower(): str(item.seq)
            for item in sequence_data
        }
        virus_taxon_name_to_gb = {
            name_regex.search(item.description).group(2).lower()
            + " "
            + name_regex.search(item.description)
                .group(3)
                .lower(): name_regex.search(item.description)
                .group(1)
            for item in sequence_data
        }
        return virus_taxon_name_to_seq, virus_taxon_name_to_gb

    @staticmethod
    def extract_sequences_from_record(
            sequences: dict, record: t.List[t.Dict[str, str]], type: SequenceType
    ):
        if type == SequenceType.GENOME:
            sequences[type].append(record[0]["GBSeq_sequence"])
        else:
            coding_regions = [
                record[0]["GBSeq_feature-table"][i]
                for i in range(len(record[0]["GBSeq_feature-table"]))
                if record[0]["GBSeq_feature-table"][i]["GBFeature_key"] == "CDS"
            ]
            for coding_region in coding_regions:
                if type == SequenceType.CDS:
                    cds_start = int(
                        coding_region["GBInterval_from"][0]["GBInterval_from"]
                    )
                    cds_end = int(coding_region["GBInterval_from"][0]["GBInterval_to"])
                    genome = record[0]["GBSeq_sequence"]
                    if "complement" in coding_region["GBFeature_location"]:
                        genome = genome.complement()
                    cds = genome[cds_start: cds_end + 1]
                    sequences[type].append(cds)
                else:
                    protein = coding_region["GBFeature_quals"][-1]["GBQualifier_value"]
                    sequences[type].append(protein)

    @staticmethod
    def extract_genome_data_from_entrez_result(
            entrez_result: t.List[t.Dict],
    ) -> t.Tuple[t.Dict[str, str], t.Dict[str, str]]:
        virus_taxon_name_to_acc = dict()
        virus_taxon_name_to_seq = dict()
        for record in entrez_result:
            if (
                    record["GBSeq_definition"]
                    and record["GBSeq_organism"] not in virus_taxon_name_to_acc
                    and record["GBSeq_organism"] not in virus_taxon_name_to_seq
            ):
                virus_taxon_name_to_acc[str(record["GBSeq_organism"]).lower()] = record[
                    "GBSeq_locus"
                ]
                virus_taxon_name_to_seq[str(record["GBSeq_organism"]).lower()] = str(
                    record["GBSeq_sequence"]
                )
        return virus_taxon_name_to_acc, virus_taxon_name_to_seq

    @staticmethod
    def get_gi_sequences_from_ncbi(
            gi_accessions: t.List[str], batch_size: int = 500
    ) -> t.Dict[str, str]:
        """
        :param gi_accessions: list of gi accessions
        :param batch_size: api query btach size
        :return: dictionary mapping accessions to sequences
        """
        gi_accession_queries = [
            ",".join(gi_accessions[i: i + batch_size])
            for i in range(0, len(gi_accessions), batch_size)
        ]
        records = []
        for gi_accession_query in gi_accession_queries:
            records += list(
                Entrez.parse(
                    Entrez.efetch(db="nucleotide", id=gi_accession_query, retmode="xml")
                )
            )
        record_gi_acc_to_seq = dict()
        for record in records:
            for acc_data in record["GBSeq_other-seqids"]:
                if "gi" in acc_data:
                    acc = acc_data.split("|")[-1]
                    seq = record["GBSeq_sequence"]
                    record_gi_acc_to_seq[acc] = seq
        return record_gi_acc_to_seq

    @staticmethod
    def get_coding_regions(
            accessions: t.List[str], batch_size: int = 500
    ) -> t.Dict[str, str]:
        """
        :param accessions: list if refseq or genbank accessions
        :param batch_size: batch size for making queries to Entrez
        :return: dictionary mapping accession to a list of coding regions
        """
        queries = [
            ",".join(accessions[i: i + batch_size])
            for i in range(0, len(accessions), batch_size)
        ]
        records = []
        for query in queries:
            records += list(
                Entrez.parse(
                    Entrez.efetch(db="nucleotide", id=query, retmode="xml"),
                    validate=False,
                )
            )
        acc_to_coding_regions = {
            record["GBSeq_locus"]: ";".join(
                [
                    feature["GBFeature_location"]
                    for feature in record["GBSeq_feature-table"]
                    if feature["GBFeature_key"] == "CDS"
                ]
            )
            for record in records
        }
        return acc_to_coding_regions

    @staticmethod
    def extract_coding_sequence(
            genome_sequence: str, coding_regions: str
    ) -> t.List[str]:
        """
        :param genome_sequence: genomic sequence
        :param coding_regions: list of coding sequence regions in the form of join(a..c,c..d,...)
        :return: the coding sequence
        """
        coding_region_regex = re.compile("(\d*)\.\.(\d*)")
        coding_sequences = []
        for cds in coding_regions.split(";"):
            coding_sequence = ""
            for match in coding_region_regex.finditer(cds):
                start = int(match.group(1))
                end = int(match.group(2))
                assert (end - start) % 3 == 0
                coding_sequence += genome_sequence[start: end + 1]
            coding_sequences.append(coding_sequence)
        return coding_sequences

    @staticmethod
    def get_name_to_ncbi_genome_accession(names: t.List[str]) -> t.Dict[str, str]:
        """
        :param names: list of organism names to find accessions for
        :return: map of organism names to their genomic sequence accessions, if exist
        """
        raw_get_data = {name: Entrez.read(
            Entrez.esearch(db="nucleotide", term=f"({name}[Organism]) AND complete genome[Text Word]", retmode="xml"))
            for name in names}
        relevant_raw_get_data = {name: raw_get_data[name]['IdList'][0] for name in raw_get_data if
                                 len(raw_get_data[name]['IdList']) > 0}
        return raw_get_data

    @staticmethod
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
        logger.info(f"complementing missing sequence data from ncbi nucleotide api for {len(list(id_to_gi_acc.values()))} records")
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def extract_genbank_sequences(df, genbank_data_dir):
        """
        :param df: dataframe to fill with genbank sequences
        :param genbank_data_dir: directory holding genbank sequences downloaded from vipr-db
        :return: none, alters the dataframe inplace
        """
        paths = [
            f"{genbank_data_dir}/{path}"
            for path in os.listdir(genbank_data_dir)
            if ".fasta" in path
        ]
        for path in paths:
            logger.info(
                f"path={path}\n#missing sequences={df.loc[(df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna())].shape[0]}"
            )
            (
                virus_taxon_name_to_seq,
                virus_taxon_name_to_gb,
            ) = SequenceCollectingUtils.get_sequence_info(path)
            df.set_index("virus_taxon_name", inplace=True)
            df["virus_genbank_accession"].fillna(value=virus_taxon_name_to_gb, inplace=True)
            df["genbank_sequence"].fillna(value=virus_taxon_name_to_seq, inplace=True)
            logger.info(
                f"#missing sequences={df.loc[(df.virus_genbank_sequence.isna()) & (df.virus_refseq_sequence.isna())].shape[0]}\n\n"
            )
            df.reset_index(inplace=True)

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

    @staticmethod
    def get_gi_sequences_from_df(
        gi_accessions: t.Union[float, str], gi_acc_to_seq: t.Dict[str, str]
    ):
        """
        :param gi_accessions: gi accessions string
        :param gi_acc_to_seq: dictionary mapping gio accessions to sequences
        :return: string of gi sequences
        """
        if type(gi_accessions) is not str:
            return np.nan
        gi_accessions_lst = (
            gi_accessions.split(";") if ";" in gi_accessions else [gi_accessions]
        )
        sequences = ",".join(
            [gi_acc_to_seq[acc] for acc in gi_accessions_lst if acc in gi_acc_to_seq]
        )
        if len(sequences) == 0:
            return np.nan
        return sequences

    @staticmethod
    def get_cds(accessions: str, acc_to_cds: t.Dict[str, str]) -> str:
        if type(accessions) is not str:
            return np.nan
        accessions_lst = [
            acc.split(":")[-1].replace("*", "") for acc in re.split(",|;", accessions)
        ]
        cds = [acc_to_cds[acc] for acc in accessions_lst if acc in acc_to_cds]
        return ",".join(cds)


class GenomeBiasCollectingService:
    @staticmethod
    def get_dinucleotides_by_range(coding_sequence: str, seq_range: range):
        """
        :param coding_sequence: coding sequence
        :param seq_range: range for sequence window
        :return: a sequence of bridge / non-bridge dinucleotides depending on requested range
        """
        dinuc_sequence = "".join([coding_sequence[i: i + 2] for i in seq_range])
        return dinuc_sequence

    @staticmethod
    def compute_dinucleotide_bias(
            coding_sequences: t.List[str],
            computation_type: DinucleotidePositionType = DinucleotidePositionType.BRIDGE,
    ):
        """
        :param coding_sequences: list of coding sequences
        :param computation_type: can be either regular, or limited to bridge or non-bridge positions
        :return: dinucleotide bias dictionary
        dinculeotide bias computed according to https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        computation_type options:
            BRIDGE - consider only dinucleotide positions corresponding to bridges between codons (one is the last pos of a codon and the next is the first of another)
            NONBRIDGE - consider only dinucleotide positions do not correspond to bridges between codons
            REGULAR - consider all dinucleotide positions"""
        avg_dinucleotide_biases = dict()
        dinucleotide_biases_dicts = []
        for sequence in coding_sequences:
            dinuc_sequence = sequence
            if (
                    computation_type == DinucleotidePositionType.BRIDGE
            ):  # limit the sequence to bridge positions only
                dinuc_sequence = GenomeBiasCollectingService.get_dinucleotides_by_range(
                    sequence, range(2, len(sequence) - 2, 3)
                )
            elif computation_type == DinucleotidePositionType.NONBRIDGE:
                dinuc_sequence = GenomeBiasCollectingService.get_dinucleotides_by_range(
                    sequence, range(0, len(sequence) - 2, 3)
                )
            nucleotide_count = {
                "A": dinuc_sequence.count("A"),
                "C": dinuc_sequence.count("C"),
                "G": dinuc_sequence.count("G"),
                "T": dinuc_sequence.count("T"),
            }
            nucleotide_total_count = len(dinuc_sequence)
            assert nucleotide_total_count > 0
            dinucleotide_total_count = len(sequence) / 2
            assert dinucleotide_total_count > 0
            dinucleotide_biases = dict()
            for nuc_i in nucleotide_count.keys():
                for nuc_j in nucleotide_count.keys():
                    dinucleotide = nuc_i + "p" + nuc_j
                    dinucleotide_biases[
                        computation_type.name + "_" + dinucleotide + "_bias"
                        ] = (sequence.count(dinucleotide) / dinucleotide_total_count) / (
                            nucleotide_count[nuc_i]
                            / nucleotide_total_count
                            * nucleotide_count[nuc_j]
                            / nucleotide_total_count
                    )
            dinucleotide_biases_dicts.append(dinucleotide_biases)

        # average dinucleotide biases across genomic sequences
        for dinucleotide in dinucleotide_biases_dicts[0]:
            avg_dinucleotide_biases[dinucleotide] = np.mean(
                [
                    dinucleotide_biases_dict[dinucleotide]
                    for dinucleotide_biases_dict in dinucleotide_biases_dicts
                ]
            )

        return avg_dinucleotide_biases

    @staticmethod
    def compute_codon_bias(coding_sequences: t.List[str]) -> t.Dict[str, float]:
        """
        :param coding_sequences: list of coding sequences
        :return: the codon bias computation described in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        """
        sequence = "".join(coding_sequences)
        codon_biases = dict()
        for codon in CODONS:
            if codon not in STOP_CODONS:
                aa = Bio.Data.CodonTable.standard_dna_table.forward_table[codon]
                other_codons = [
                    codon
                    for codon in CODONS
                    if codon not in STOP_CODONS
                       and Bio.Data.CodonTable.standard_dna_table.forward_table[codon]
                       == aa
                ]
                codon_biases[codon + "_bias"] = sequence.count(codon) / np.sum(
                    [sequence.count(c) for c in other_codons]
                )
        return codon_biases

    @staticmethod
    def compute_diaa_bias(coding_sequences: t.List[str]) -> t.Dict[str, float]:
        """
        :param coding_sequences: list of coding sequences
        :return: the diaa biases, similar to compute_dinucleotide_bias
        """
        sequence = "".join([str(Seq(seq).translate()) for seq in coding_sequences])
        diaa_biases = dict()
        total_diaa_count = len(sequence) / 2
        total_aa_count = len(sequence)
        for aa_i in AMINO_ACIDS:
            for aa_j in AMINO_ACIDS:
                diaa = aa_i + aa_j
                diaa_biases[diaa + "_bias"] = (
                                                      sequence.count(diaa) / total_diaa_count
                                              ) / (
                                                      sequence.count(aa_i)
                                                      / total_aa_count
                                                      * sequence.count(aa_j)
                                                      / total_aa_count
                                              )
                if diaa_biases[diaa + "_bias"] == 0:
                    diaa_biases[diaa + "_bias"] += 0.0001
        return diaa_biases

    @staticmethod
    def compute_codon_pair_bias(
            coding_sequences: t.List[str], diaa_bias: t.Dict[str, float]
    ) -> t.Dict[str, float]:
        """
        :param coding_sequences: list of coding sequences
        :param diaa_bias: dictionary mapping diaa to its bias
        :return: dictionary mapping each dicodon to its bias
        codon pair bias measured by the codon pair score (CPS) as shown in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        the denominator is obtained by multiplying the count od each codon with the bias of the respective amino acid pair
        """
        sequence = "".join(coding_sequences)
        codon_count = dict()
        for codon in CODONS:
            codon_count[codon] = sequence.count(codon)
        codon_pair_scores = dict()
        for codon_i in CODONS:
            for codon_j in CODONS:
                if codon_i not in STOP_CODONS and codon_j not in STOP_CODONS:
                    codon_pair = codon_i + codon_j
                    codon_pair_count = sequence.count(codon_pair)
                    denominator = (
                            codon_count[codon_i]
                            * codon_count[codon_j]
                            * diaa_bias[
                                f"{str(Seq(codon_i).translate())}{str(Seq(codon_j).translate())}_bias"
                            ]
                    )
                    if denominator == 0:
                        diaa_bias = diaa_bias[
                            f"{str(Seq(codon_i).translate())}{str(Seq(codon_j).translate())}_bias"
                        ]
                        print(
                            f"codon_count[{codon_i}]={codon_count[codon_i]}, codon_count[{codon_j}]={codon_count[codon_j]}, diaa_bias={diaa_bias}"
                        )
                        pass
                    else:
                        codon_pair_scores[codon_pair + "_bias"] = np.log(
                            codon_pair_count / denominator
                        )
        return codon_pair_scores

    @staticmethod
    def collect_genomic_bias_features(
            genome_sequences: t.List[str], coding_sequences: t.List[str]
    ):
        """
        :param genome_sequences: list of genomic sequences
        :param coding_sequences: list of coding sequences
        :return: dictionary with genomic features to be added as a record to a dataframe
        """
        dinucleotide_biases = GenomeBiasCollectingService.compute_dinucleotide_bias(
            coding_sequences=genome_sequences,
            computation_type=DinucleotidePositionType.REGULAR,
        )
        id_genomic_traits = dict(dinucleotide_biases)
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                coding_sequences=genome_sequences,
                computation_type=DinucleotidePositionType.BRIDGE,
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                coding_sequences=genome_sequences,
                computation_type=DinucleotidePositionType.NONBRIDGE,
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_codon_bias(
                coding_sequences=coding_sequences
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_diaa_bias(
                coding_sequences=coding_sequences
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_codon_pair_bias(
                coding_sequences=coding_sequences, diaa_bias=id_genomic_traits
            )
        )
        return id_genomic_traits
