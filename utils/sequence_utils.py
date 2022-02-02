import logging
import os
import typing as t
import re
from collections import defaultdict
from enum import Enum
from functools import partial
from time import sleep

import Bio
import subprocess

import http

from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from urllib.error import HTTPError
import sys

sys.path.append("..")
from settings import get_settings

tqdm.pandas()

from Bio.Data import CodonTable
import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

NUCLEOTIDES = ["A", "C", "G", "T"]
STOP_CODONS = CodonTable.standard_dna_table.stop_codons
CODONS = list(CodonTable.standard_dna_table.forward_table.keys()) + STOP_CODONS
AMINO_ACIDS = list(set(CodonTable.standard_dna_table.forward_table.values())) + [
    "O",
    "S",
    "U",
    "T",
    "W",
    "Y",
    "V",
    "B",
    "Z",
    "X",
    "J",
]
ENTREZ_RETMAX = 50


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


class AnnotationType(Enum):
    UNDEFINED = 0
    GENE = 1
    CDS = 2
    PEPTIDE = 3
    UTR3 = 4
    UTR5 = 5


class SequenceCollectingUtils:
    @staticmethod
    def parse_ncbi_sequence_raw_data_by_unique_acc(ncbi_raw_data: t.List[t.Dict[str, str]]) -> t.List[t.Dict[str, str]]:
        """
        :param ncbi_raw_data: raw data from api efetch call to ncbi api
        :return: parsed ncbi data
        """

        acc_to_seq = {
            record["GBSeq_accession-version"].split(".")[0]: record["GBSeq_sequence"]
            for record in ncbi_raw_data
            if "GBSeq_sequence" in record
        }

        acc_to_organism = {
            record["GBSeq_accession-version"].split(".")[0]: record["GBSeq_organism"]
            for record in ncbi_raw_data
            if "GBSeq_organism" in record
        }

        acc_to_cds = {
            record["GBSeq_accession-version"].split(".")[0]: ";".join(
                [
                    feature["GBFeature_location"]
                    for feature in record["GBSeq_feature-table"]
                    if feature["GBFeature_key"] == "CDS"
                ]
            )
            for record in ncbi_raw_data
        }
        for key in list(acc_to_cds.keys()):
            if acc_to_cds[key] == "":
                acc_to_cds.pop(key, None)
        acc_to_annotation = {
            record["GBSeq_accession-version"].split(".")[0]: record["GBSeq_definition"]
            for record in ncbi_raw_data
            if "GBSeq_definition" in record
        }
        acc_to_keywords = {
            record["GBSeq_accession-version"].split(".")[0]: record["GBSeq_keywords"]
            for record in ncbi_raw_data
            if "GBSeq_keywords" in record
        }
        parsed_data = [acc_to_seq, acc_to_organism, acc_to_cds, acc_to_annotation, acc_to_keywords]

        return parsed_data

    @staticmethod
    def fill_ncbi_data_by_unique_acc(
        df: pd.DataFrame, parsed_data: t.List[t.Dict[str, str]], index_field_name: str = "taxon_name"
    ):
        """
        :param df: dataframe to fill
        :param parsed_data: parsed data to fill df with
        :param index_field_name: name of field to index by
        :return: nothing. changes the df inplace
        """

        acc_to_seq = parsed_data[0]
        acc_to_organism = parsed_data[1]
        acc_to_cds = parsed_data[2]
        acc_to_annotation = parsed_data[3]
        acc_to_keywords = parsed_data[4]

        for col in ["sequence", "cds", "annotation", "keywords", "category", "accession_organism"]:
            if col not in df.columns:
                df[col] = np.nan

        # replace values in acc field to exclude version number
        df["accession"] = df["accession"].apply(
            lambda x: str(x).split(".")[0].replace(" ", "").replace("*", "") if pd.notna(x) else x
        )

        df.set_index("accession", inplace=True)
        old_missing_seq_num = df["sequence"].isna().sum()
        logger.info(f"# extracted sequences = {len(acc_to_seq.keys())}")
        df["sequence"].fillna(value=acc_to_seq, inplace=True)
        new_missing_seq_num = df["sequence"].isna().sum()
        logger.info(f"# of added sequences from pid {os.getpid()} = {old_missing_seq_num - new_missing_seq_num}")

        old_missing_organisms_num = df["accession_organism"].isna().sum()
        logger.info(f"# extracted organisms = {len(acc_to_organism.keys())}")
        df["accession_organism"].fillna(value=acc_to_organism, inplace=True)
        new_missing_organisms_num = df["accession_organism"].isna().sum()
        logger.info(
            f"# of added organisms from pid {os.getpid()} = {old_missing_organisms_num - new_missing_organisms_num}"
        )

        old_missing_cds_num = df["cds"].isna().sum()
        logger.info(f"# extracted cds = {len(acc_to_cds.keys())}")
        df["cds"].fillna(value=acc_to_cds, inplace=True)
        new_missing_cds_num = df["cds"].isna().sum()
        logger.info(f"# of added cds from pid {os.getpid()} = {old_missing_cds_num - new_missing_cds_num}")

        old_missing_annotations_num = df["annotation"].isna().sum()
        logger.info(f"# extracted annotations = {len(acc_to_annotation.keys())}")
        df["annotation"].fillna(value=acc_to_annotation, inplace=True)
        new_missing_annotations_num = df["annotation"].isna().sum()
        logger.info(
            f"# of added cds from pid {os.getpid()} = {old_missing_annotations_num - new_missing_annotations_num}"
        )

        old_missing_kws_num = df["keywords"].isna().sum()
        logger.info(f"# extracted keywords = {len(acc_to_keywords.keys())}")
        df["keywords"].fillna(value=acc_to_keywords, inplace=True)
        new_missing_kws_num = df["keywords"].isna().sum()
        logger.info(f"# of added KWS from pid {os.getpid()} = {old_missing_kws_num - new_missing_kws_num}")

        df["category"] = df["annotation"].apply(
            lambda x: "genome" if type(x) is str and ("complete genome" in x or "complete sequence" in x) else np.nan
        )

        df.reset_index(inplace=True)

    @staticmethod
    def fill_missing_data_by_acc(index_field_name: str, sequence_type: SequenceType, df: pd.DataFrame) -> str:

        df_path = f"{os.getcwd()}/df_{SequenceCollectingUtils.fill_missing_data_by_acc.__name__}_pid_{os.getpid()}.csv"

        if not os.path.exists(df_path):
            df.to_csv(df_path)

        accessions = [
            s.replace(" ", "").replace("*", "") for s in list(df.loc[df.source != "gi", "accession"].dropna().unique())
        ]
        if len(accessions) > 0:
            logger.info(
                f"performing efetch query to ncbi on {len(accessions)} genbank and refseq accessions from pid {os.getpid()}"
            )
            ncbi_raw_data = SequenceCollectingUtils.do_ncbi_batch_fetch_query(
                accessions=accessions, sequence_type=sequence_type
            )
            parsed_data = SequenceCollectingUtils.parse_ncbi_sequence_raw_data_by_unique_acc(
                ncbi_raw_data=ncbi_raw_data
            )
            SequenceCollectingUtils.fill_ncbi_data_by_unique_acc(df=df, parsed_data=parsed_data)

        df["category"] = df["annotation"].apply(
            lambda x: "genome" if pd.notna(x) and "complete genome" in x else np.nan
        )

        if sequence_type == SequenceType.GENOME:
            SequenceCollectingUtils.annotate_segmented_accessions(df=df, index_field_name=index_field_name)
            df = SequenceCollectingUtils.collapse_segmented_data(df=df, index_field_name=index_field_name)

        df.to_csv(df_path, index=False)
        return df_path

    @staticmethod
    def flatten_sequence_data(df: pd.DataFrame, data_prefix: str = "virus",) -> pd.DataFrame:
        """
        :param df: dataframe to flatten
        :param data_prefix: data prefix, for all column names
        :return: flattened dataframe
        """

        # remove data prefix
        flattened_df = df.rename(
            columns={col: col.replace(f"{data_prefix}{'_' if len(data_prefix) > 0 else ''}", "") for col in df.columns}
        )

        # set source by difference accession fields
        flattened_df["source"] = flattened_df[["genbank_accession", "gi_accession"]].apply(
            lambda x: "genbank" if pd.notna(x.genbank_accession) else ("gi" if pd.notna(x.gi_accession) else np.nan),
            axis=1,
        )

        # combine all the accession fields into a single accession field
        flattened_df.rename(columns={"genbank_accession": "accession"}, inplace=True)
        flattened_df["accession"].fillna(flattened_df["gi_accession"], inplace=True)
        flattened_df.drop("gi_accession", axis=1, inplace=True)
        flattened_df["accession"] = flattened_df["accession"].apply(
            lambda x: x.replace(",\s*", ";") if pd.notna(x) else x
        )

        # melt df by accession
        flattened_df = flattened_df.assign(accession=flattened_df.accession.str.split(";")).explode("accession")
        flattened_df = flattened_df.set_index(flattened_df.groupby(level=0).cumcount(), append=True)
        flattened_df.index.rename(["index", "accession_genome_index"], inplace=True)
        flattened_df.reset_index(inplace=True)

        # add fields to fill
        flattened_df["sequence"] = np.nan
        flattened_df["cds"] = np.nan
        flattened_df["annotation"] = np.nan
        flattened_df["keywords"] = np.nan
        flattened_df["category"] = np.nan

        flattened_df.drop_duplicates(inplace=True)

        return flattened_df

    @staticmethod
    def do_ncbi_batch_fetch_query(
        accessions: t.List[str], sequence_type: SequenceType = SequenceType.GENOME
    ) -> t.List[t.Dict]:
        """
        :param accessions: list of accessions to batch query on
        :param sequence_type: type of sequence data that should be fetched
        :return: list of ncbi records corresponding to the accessions
        """
        ncbi_raw_records = []
        if len(accessions) < ENTREZ_RETMAX:
            accessions_batches = [accessions]
        else:
            accessions_batches = [accessions[i : i + ENTREZ_RETMAX] for i in range(0, len(accessions), ENTREZ_RETMAX)]
        if len(accessions) == 0:
            return ncbi_raw_records
        for b in range(len(accessions_batches)):
            logger.info(
                f"submitting efetch query for batch {b} of pid {os.getpid()} out of {len(accessions_batches)} batches"
            )
            accessions_batch = accessions_batches[b]
            retry = True
            while retry:
                try:
                    ncbi_raw_records += list(
                        Entrez.parse(
                            Entrez.efetch(
                                db="nucleotide"
                                if sequence_type in [SequenceType.GENOME, SequenceType.CDS]
                                else "protein",
                                id=",".join([str(acc) for acc in accessions_batch]),
                                retmode="xml",
                                api_key=get_settings().ENTREZ_API_KEY,
                            )
                        )
                    )
                    retry = False
                except http.client.IncompleteRead as e:
                    logger.error(
                        f"Failed Entrez query on {len(accessions)} accessions due to error {e}. will retry after a second"
                    )
                    sleep(1)
                except HTTPError as e:
                    if e.code == 400:
                        logger.error(f"Enrez query failed due to error {e}. will not retry")
                        retry = False
                    if e.code == 429:
                        logger.info(f"Entrez query failed due to error {e}. will retry after a minute")
                        sleep(60)
                except Exception as e:
                    logger.error(
                        f"Failed Entrez query on {len(accessions)} accessions due to error {e}. will retry after a minute"
                    )
                    sleep(60)
            logger.info(f"{len(ncbi_raw_records)} out of {len(accessions)} records collected...")

        logger.info(f"collected {len(ncbi_raw_records)} records based on {len(accessions)} accessions")
        return ncbi_raw_records

    @staticmethod
    def do_ncbi_search_queries(
        organisms: t.List[str],
        text_conditions: t.Tuple[str] = tuple(["complete genome", "complete sequence"]),
        do_via_genome_db: bool = False,
        sequence_type: SequenceType = SequenceType.GENOME,
    ) -> t.Dict[str, t.List[str]]:
        """
        :param organisms: list of organisms names to search
        :param text_conditions: additional text conditions to search by
        :param do_via_genome_db: indicator weather queries through the genome ncbi db should also be performed
        :param sequence_type: type of sequence data for which accessions should be collected
        :return: map of organisms to their accessions
        """

        # perform direct search within the ncbi nucleotide databases (genbank and refseq)
        logger.info(
            f"performing {len(organisms)} esearch queries on [Organism] and text conditions {' OR '.join(text_conditions)}"
        )

        organism_to_accessions = defaultdict(list)

        logger.info(
            f"performing direct search within ncbi nucleotide databases for {len(organisms)} organism {' OR '.join(text_conditions)} accessions"
        )
        i = 0
        while i < len(organisms):
            if i % 50 == 0:
                logger.info(f"reached organism {i} out of {len(organisms)} within process {os.getpid()}")
            organism = organisms[i]
            query = (
                f"({organisms[i]}[Organism]) AND ({text_conditions[0]}[Text Word] OR "
                + " OR ".join([f"{text_condition}[Text Word]" for text_condition in text_conditions[1:]])
                + ")"
            )
            try:
                raw_data = Entrez.read(
                    Entrez.esearch(
                        db="nucleotide" if sequence_type in [SequenceType.GENOME, SequenceType.CDS] else "protein",
                        term=query,
                        retmode="xml",
                        idtype="acc",
                        api_key=get_settings().ENTREZ_API_KEY,
                    )
                )
                organism_to_accessions[organism] = organism_to_accessions[organism] + raw_data["IdList"]
                i += 1
                sleep(1)  # use 1 second interval to avoid more than 10 requests per second
            except HTTPError as e:
                if e.code == 429:
                    logger.info(
                        f"{os.getpid()} failed api request with error {e} and thus will sleep for a minute before trying again"
                    )
                    sleep(60)
                else:
                    logger.error(f"{os.getpid()} failed api request for tax {organisms[i]} with error {e}")
                    sleep(1)  # use 1 second interval to avoid more than 10 requests per second
                    i += 1
            except Exception as e:
                print(f"failed to perfrom query {query} due to error {e}")
                exit(1)

        # complement additional data based on each in genome db
        if do_via_genome_db:
            logger.info(
                f"performing indirect search within ncbi genome databases for {len(organisms)} organism {' OR '.join(text_conditions)} accessions"
            )
            i = 0
            while i < len(organisms):
                if i % 50 == 0:
                    logger.info(f"reached organism {i} out of {len(organisms)} within process {os.getpid()}")
                organism = organisms[i]
                cmd = f'esearch -db genome -query "{organism} complete genome" | epost -db genome | elink -target nuccore | efetch -format acc'
                ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output = ps.communicate()[0]
                if ps.returncode == 0:
                    accession_regex = re.compile(r"[a-zA-Z]+\_*\d*\.*\d")
                    output_str = output.decode("utf-8")
                    accessions = [item for item in output_str.split("\n") if accession_regex.match(item)]
                    organism_to_accessions[organism] = organism_to_accessions[organism] + accessions
                    i += 1
                    sleep(1)  # sleep 1 second in between requests
                elif ps.returncode == 429 or "too many requests" in output.decode("utf-8"):
                    logger.error(f"exceeded number of requests to ncbi. will sleep for a minute")
                    sleep(60)
                else:
                    logger.error(f"failed to obtain accessions for {organism} due to error {ps.returncode}")
                    sleep(1)  # sleep 1 second in between requests
                    i += 1

        return organism_to_accessions

    @staticmethod
    def fill_missing_data_by_organism(
        index_field_name: str, sequence_type: SequenceType, sequence_annotations: t.Tuple[str], df: pd.DataFrame
    ) -> str:
        """
        :param df: dataframe with sequence data to fill be taxa names based on their search in the genome db
        :param index_field_name: field name to extract query values from
        :param sequence_type: sequence type to collect
        :param sequence_annotations: text conditions (combined by OR) for searching sequence data by organism
        :return: path to filled dataframe
        """

        df_path = (
            f"{os.getcwd()}/df_{SequenceCollectingUtils.fill_missing_data_by_organism.__name__}_pid_{os.getpid()}.csv"
        )

        # find gi accessions for the given organism names
        organisms = list(df[index_field_name])
        if len(organisms) > 0:
            taxon_name_to_accessions = SequenceCollectingUtils.do_ncbi_search_queries(
                organisms=organisms, sequence_type=sequence_type, text_conditions=sequence_annotations
            )
            num_accessions = np.sum(len([taxon_name_to_accessions[taxname] for taxname in taxon_name_to_accessions]))
            logger.info(
                f"{num_accessions} accessions extracted for {len(taxon_name_to_accessions.keys())} out of {len(organisms)} taxa"
            )
            df.set_index(index_field_name, inplace=True)
            if "accession" not in df.columns:
                df["accession"] = np.nan
            df["accession"].fillna(value=taxon_name_to_accessions, inplace=True)
            df = df.explode(column="accession")
            df.reset_index(inplace=True)

            df.to_csv(df_path)

            # extract data based on the obtained gi accessions
            accessions = [str(item).replace(" ", "").replace("*", "") for item in list(df.accession.dropna().unique())]
            if len(accessions) > 0:
                logger.info(f"performing efetch query to ncbi on {len(accessions)} accessions from pid {os.getpid()}")
                ncbi_raw_data = SequenceCollectingUtils.do_ncbi_batch_fetch_query(
                    accessions=accessions, sequence_type=sequence_type
                )
                parsed_data = SequenceCollectingUtils.parse_ncbi_sequence_raw_data_by_unique_acc(
                    ncbi_raw_data=ncbi_raw_data
                )
                SequenceCollectingUtils.fill_ncbi_data_by_unique_acc(
                    df=df, parsed_data=parsed_data, index_field_name=index_field_name
                )

        df.to_csv(df_path, index=False)
        return df_path

    @staticmethod
    def collapse_segmented_data(df: pd.DataFrame, index_field_name: str) -> pd.DataFrame:
        """
        :param df: dataframe with segmented data inside
        :param index_field_name: name of field to be used as index
        :return: dataframe were the successive segmented records are concatenated
        """
        segmented = df.loc[df.annotation.str.contains("segment", na=False)]
        non_segmented = df.loc[df.annotation.str.contains("segment", na=False)]

        def agg_id(ids):
            return ";".join([str(item) for item in ids.dropna().unique()])

        def agg_seq(seqs):
            return "".join(seqs.dropna())

        if "accession_prefix" not in segmented:
            segmented["accession_prefix"] = segmented["accession"].apply(lambda acc: acc[:-2] if pd.notna(acc) else acc)
        segmented_collapsed = (
            segmented.sort_values(["species_name", "accession", "accession_genome_index"])
            .groupby([index_field_name, "accession_prefix"])
            .agg(
                {
                    col: agg_id if col != "sequence" else agg_seq
                    for col in segmented.columns
                    if col not in [index_field_name, "accession_prefix"]
                }
            )
            .reset_index()
        )
        segmented_collapsed.head()

        df = pd.concat([segmented, non_segmented])
        return df

    @staticmethod
    def annotate_segmented_accessions(df: pd.DataFrame, index_field_name: str = "taxon_name"):
        """
        :param df: dataframe holding some accessions of segmented genome records
        :param index_field_name: field name to index by
        :return: none, changes the dataframe inplace
        """
        df.drop(labels=[col for col in df.columns if "Unnamed" in col], axis=1, inplace=True)
        segmented_records = df.loc[
            (df.annotation.str.contains("DNA-", na=False, case=False))
            | (df.annotation.str.contains("segment", na=False, case=False))
        ]
        segmented_records.sort_values([index_field_name, "accession"], inplace=True)
        segmented_records["accession_prefix"] = segmented_records["accession"].apply(lambda acc: acc[:-1])
        segmented_records_by_full_genome = segmented_records.groupby([index_field_name, "accession_prefix"])
        index_to_genome_index_map = dict()
        for group_combo in segmented_records_by_full_genome.groups.keys():
            sorted_group_df = segmented_records_by_full_genome.get_group(group_combo).sort_values("accession_prefix")
            indices = sorted_group_df.index
            index_to_genome_index_map.update({indices[i]: i for i in range(len(indices))})
        segmented_records["accession_genome_index"] = np.nan
        segmented_records["accession_genome_index"].fillna(value=index_to_genome_index_map, inplace=True)
        segmented_records.drop("accession_prefix", axis=1, inplace=True)
        df.update(segmented_records)


class GenomeBiasCollectingService:
    @staticmethod
    def get_dinucleotides_by_range(coding_sequence: str, seq_range: range):
        """
        :param coding_sequence: coding sequence
        :param seq_range: range for sequence window
        :return: a sequence of bridge / non-bridge dinucleotides depending on requested range
        """
        dinuc_sequence = "".join([coding_sequence[i : i + 2] for i in seq_range])
        return dinuc_sequence

    @staticmethod
    def compute_dinucleotide_bias(
        sequence: str, computation_type: DinucleotidePositionType = DinucleotidePositionType.BRIDGE,
    ) -> t.Dict[str, float]:
        """
        :param sequence: a single coding sequences
        :param computation_type: can be either regular, or limited to bridge or non-bridge positions
        :return: dinucleotide bias dictionary
        dinculeotide bias computed according to https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        computation_type options:
            BRIDGE - consider only dinucleotide positions corresponding to bridges between codons (one is the last pos of a codon and the next is the first of another)
            NONBRIDGE - consider only dinucleotide positions do not correspond to bridges between codons
            REGULAR - consider all dinucleotide positions"""
        dinuc_sequence = sequence
        if computation_type == DinucleotidePositionType.BRIDGE:  # limit the sequence to bridge positions only
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
        dinucleotide_total_count = len(sequence) / 2
        dinucleotide_biases = dict()
        if nucleotide_total_count > 0 and dinucleotide_total_count > 0:
            for nuc_i in nucleotide_count.keys():
                for nuc_j in nucleotide_count.keys():
                    dinucleotide = nuc_i + nuc_j
                    try:
                        dinucleotide_biases[f"{computation_type.name}_{nuc_i}p{nuc_j}_bias"] = (
                            sequence.count(dinucleotide) / dinucleotide_total_count
                        ) / (
                            nucleotide_count[nuc_i]
                            / nucleotide_total_count
                            * nucleotide_count[nuc_j]
                            / nucleotide_total_count
                        )
                    except Exception as e:
                        logger.error(
                            f"failed to compute dinucleotide bias for {dinucleotide} due to error {e} and will thus set it to nan"
                        )
                        dinucleotide_biases[f"{computation_type.name}_{nuc_i}p{nuc_j}_bias"] = np.nan
        else:
            logger.error(
                f"dinucleotide sequence is of length {nucleotide_total_count} with {dinucleotide_total_count} dinucleotides in it, and thus dinucleotide bias cannot be computed"
            )

        return dinucleotide_biases

    @staticmethod
    def compute_codon_bias(coding_sequence: str) -> t.Dict[str, float]:
        """
        :param coding_sequence: a single coding sequences
        :return: the codon bias computation described in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        """
        codon_biases = dict()
        for codon in CODONS:
            if codon not in STOP_CODONS:
                aa = Bio.Data.CodonTable.standard_dna_table.forward_table[codon]
                other_codons = [
                    codon
                    for codon in CODONS
                    if codon not in STOP_CODONS and Bio.Data.CodonTable.standard_dna_table.forward_table[codon] == aa
                ]
                codon_biases[codon + "_bias"] = coding_sequence.count(codon) / np.sum(
                    [coding_sequence.count(c) for c in other_codons]
                )
        return codon_biases

    @staticmethod
    def compute_diaa_bias(coding_sequence: str) -> t.Dict[str, float]:
        """
        :param coding_sequence: a single coding sequences
        :return: the diaa biases, similar to compute_dinucleotide_bias
        """
        sequence = str(Seq(coding_sequence).translate())
        diaa_biases = dict()
        aa_frequencies = {
            aa: sequence.count(aa) + 0.0001 for aa in AMINO_ACIDS
        }  # 0.0001 was added to avoid division by 0
        total_diaa_count = len(sequence) / 2
        total_aa_count = len(sequence)
        for aa_i in AMINO_ACIDS:
            for aa_j in AMINO_ACIDS:
                diaa = aa_i + aa_j
                diaa_biases[f"{aa_i}p{aa_j}_bias"] = (sequence.count(diaa) / total_diaa_count) / (
                    aa_frequencies[aa_i] / total_aa_count * aa_frequencies[aa_j] / total_aa_count
                )
                if diaa_biases[f"{aa_i}p{aa_j}_bias"] == 0:
                    diaa_biases[f"{aa_i}p{aa_j}_bias"] += 0.0001
        return diaa_biases

    @staticmethod
    def compute_codon_pair_bias(coding_sequence: str, diaa_bias: t.Dict[str, float]) -> t.Dict[str, float]:
        """
        :param coding_sequence: a single coding sequences
        :param diaa_bias: dictionary mapping diaa to its bias
        :return: dictionary mapping each dicodon to its bias
        codon pair bias measured by the codon pair score (CPS) as shown in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        the denominator is obtained by multiplying the count od each codon with the bias of the respective amino acid pair
        """
        codon_count = dict()
        for codon in CODONS:
            codon_count[codon] = (
                coding_sequence.count(codon) + 0.0001
            )  # the 0.0001 addition prevents division by zero error
        codon_pair_scores = dict()
        for codon_i in CODONS:
            for codon_j in CODONS:
                if codon_i not in STOP_CODONS and codon_j not in STOP_CODONS:
                    codon_pair = codon_i + codon_j
                    codon_pair_count = coding_sequence.count(codon_pair)
                    denominator = (
                        codon_count[codon_i]
                        * codon_count[codon_j]
                        * diaa_bias[f"{str(Seq(codon_i).translate())}{str(Seq(codon_j).translate())}_bias"]
                    )
                    if denominator == 0:
                        diaa_bias_value = diaa_bias[
                            f"{str(Seq(codon_i).translate())}{str(Seq(codon_j).translate())}_bias"
                        ]
                        logger.error(
                            f"denominator is 0 due to components being: codon_count[{codon_i}]={codon_count[codon_i]}, codon_count[{codon_j}]={codon_count[codon_j]}, diaa_bias={diaa_bias_value}"
                        )
                        pass
                    else:
                        codon_pair_scores[f"{codon_i}p{codon_j}_bias"] = float(np.log(codon_pair_count / denominator))
        return codon_pair_scores

    @staticmethod
    def compute_mean_across_sequences(sequences: t.List[str], func: callable) -> t.Dict[str, float]:
        """
        :param sequences: list of sequences to compute measures across
        :param func: function to use for computing measures
        :return: dictionary with the mean measures values across sequences
        """
        sequences_measures = [func(sequence) for sequence in sequences]
        measures_names = list(sequences_measures[0].keys())
        final_measures = {
            measure: np.sum([d[measure] for d in sequences_measures]) / len(sequences_measures)
            for measure in measures_names
        }
        return final_measures

    @staticmethod
    def collect_genomic_bias_features(genome_sequence: str, coding_sequences: t.List[str]):
        """
        :param genome_sequence: genomic sequence
        :param coding_sequences: coding sequence (if available)
        :return: dictionary with genomic features to be added as a record to a dataframe
        """
        genome_sequence = genome_sequence.upper()
        if len(coding_sequences) > 0:
            upper_coding_sequences = [coding_sequence.upper() for coding_sequence in coding_sequences]
            coding_sequences = upper_coding_sequences
        logger.info(f"genomic sequence length={len(genome_sequence)} and {len(coding_sequences)} coding sequences")
        dinucleotide_biases = GenomeBiasCollectingService.compute_dinucleotide_bias(
            sequence=genome_sequence, computation_type=DinucleotidePositionType.REGULAR,
        )
        id_genomic_traits = dict(dinucleotide_biases)

        if len(coding_sequences) > 0:
            id_genomic_traits.update(
                GenomeBiasCollectingService.compute_mean_across_sequences(
                    sequences=coding_sequences,
                    func=partial(
                        GenomeBiasCollectingService.compute_dinucleotide_bias,
                        computation_type=DinucleotidePositionType.BRIDGE,
                    ),
                )
            )

        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                sequence=genome_sequence, computation_type=DinucleotidePositionType.NONBRIDGE,
            )
        )

        if len(coding_sequences) > 0:
            id_genomic_traits.update(
                GenomeBiasCollectingService.compute_mean_across_sequences(
                    sequences=coding_sequences, func=GenomeBiasCollectingService.compute_diaa_bias,
                )
            )
            id_genomic_traits.update(
                GenomeBiasCollectingService.compute_mean_across_sequences(
                    sequences=coding_sequences,
                    func=partial(GenomeBiasCollectingService.compute_codon_pair_bias, diaa_bias=id_genomic_traits,),
                )
            )
        return id_genomic_traits

    @staticmethod
    def extract_coding_sequences(genomic_sequence: str, coding_regions: t.Union[float, str]) -> t.List[str]:
        """
        :param genomic_sequence: genomic sequence
        :param coding_regions: list of coding sequence regions in the form of join(a..c,c..d,...), seperated by ";", or none if not available
        :return: the coding sequence
        """
        coding_region_regex = re.compile("(\d*)\.\.(\d*)")
        coding_sequences = []
        if pd.notna(coding_regions):
            for cds in coding_regions.split(";"):
                coding_sequence = ""
                for match in coding_region_regex.finditer(cds):
                    start = int(match.group(1))
                    try:
                        end = int(match.group(2))
                    except:
                        end = len(genomic_sequence)
                    coding_sequence += genomic_sequence[start - 1 : end]
                if len(coding_sequence) % 3 == 0 and len(coding_sequence) > 0:  # ignore illegal coding sequences
                    coding_sequences.append(coding_sequence)
        return coding_sequences


class SequenceAnnotationUtils:
    @staticmethod
    def get_annotation_type(annotation_type_str: str) -> AnnotationType:
        annotation_type_str = annotation_type_str.lower()
        if "cds" in annotation_type_str:
            return AnnotationType.CDS
        elif "gene" in annotation_type_str:
            return AnnotationType.GENE
        elif "utr" in annotation_type_str:
            if "3" in annotation_type_str:
                return AnnotationType.UTR3
            else:
                return AnnotationType.UTR5
        elif "peptide" in annotation_type_str:
            return AnnotationType.PEPTIDE
        else:
            return AnnotationType.UNDEFINED

    @staticmethod
    def exec_vadr(sequence_data_path, workdir, vadr_model_name: str = "flavi") -> str:
        """
        :param sequence_data_path: path to dataframe with sequence data per accession
        :param workdir: directory for vadr execution
        :param vadr_model_name: name of vadr model to use. see options at: https://github.com/ncbi/vadr/wiki/Available-VADR-model-files
        return: output path
        """
        vadr_output_path = f"{workdir}/vadr.ftr"
        if not os.path.exists(vadr_output_path):
            os.chdir(workdir)
            vadr_commands = f"v-annotate.pl --mkey {vadr_model_name} --mdir $VADRMODELDIR/vadr-models-{vadr_model_name}-1.2-1 {sequence_data_path} vadr"
            res = os.system(vadr_commands)
            if res != 0:
                error_msg = f"failed to executr vadr on {sequence_data_path} using {vadr_model_name} vadr model"
                logger.error(error_msg)
                raise ValueError(error_msg)
        return vadr_output_path

    @staticmethod
    def parse_vadr_output(vadr_output_path: str) -> pd.DataFrame:
        """
        :param vadr_output_path: .ftr path with annotated features
        :return: dataframe of annotations
        """
        annotation_data = pd.read_table(vadr_output_path, delim_whitespace=True, header=1)  # , sep="\t", header=[0,1])
        annotation_data.reset_index(inplace=True)
        annotation_data.drop(0, inplace=True)
        annotation_data = annotation_data.loc[annotation_data["p/f"] == "PASS"]
        old_to_new_colname = {
            "name": "accession",
            "name.1": "annotation_name",
            "type": "annotation_type",
            "coords.1": "coordinate",
        }
        annotation_data.rename(columns=old_to_new_colname, inplace=True)
        annotation_data = annotation_data[old_to_new_colname.values()]
        annotation_data["annotation_name"] = annotation_data["annotation_name"].str.lower()
        annotation_data["annotation_type"] = annotation_data["annotation_type"].str.lower()
        return annotation_data

    @staticmethod
    def get_vadr_annotations(
        accessions: t.List[str],
        sequence_data_path: str,
        workdir: str,
        acc_to_sp: t.Dict[str, str],
        vadr_model_name: str = "flavi",
    ) -> pd.DataFrame:
        """
        :param accessions: accessions to get vadr annotations for
        :param sequence_data_path: path to dataframe with sequence data per accession
        :param workdir: directory for vadr execution
        :param acc_to_sp: map of accession to species name
        :param vadr_model_name: name of vadr model to use. see options at: https://github.com/ncbi/vadr/wiki/Available-VADR-model-files
        :return: dataframe with vadr annotations
        """
        os.makedirs(workdir)
        sequence_data = pd.read_csv(sequence_data_path, usecols=["accession", "sequence", "seqlen"])
        relevant_sequence_data = sequence_data.loc[sequence_data.accession.isin(accessions)]
        sequence_data_path = f"{workdir}/vadr_input.fasta"
        if not os.path.exists(sequence_data_path):
            sequence_records = []
            for i, row in relevant_sequence_data.iterrows():
                sequence_records.append(
                    SeqRecord(id=row.accession, name=row.accession, description=row.accession, seq=Seq(row.sequence))
                )
            SeqIO.wrtie(sequence_records, sequence_data_path, format="fasta")
        vadr_output_path = SequenceAnnotationUtils.exec_vadr(
            sequence_data_path=sequence_data_path, vadr_model_name=vadr_model_name,
        )
        df = SequenceAnnotationUtils.parse_vadr_output(vadr_output_path=vadr_output_path)
        df["source"] = "vadr"
        df["species_name"] = df["accession"].apply(lambda acc: acc_to_sp[acc])
        return df

    @staticmethod
    def get_ncbi_annotations(accessions: t.List[str],) -> t.Dict[str, t.Dict[t.Tuple[str, str], t.Tuple[int, int]]]:
        """
        :param accessions: nucleotide accessions
        :return: dictionary mapping each accession to a dictionary mapping each annotation within the accession to its range
        """
        accession_to_annotations = defaultdict(dict)
        logger.info(f"processing {len(accessions)} accessions for ncbi annotation")
        ncbi_data = SequenceCollectingUtils.do_ncbi_batch_fetch_query(
            accessions=accessions, sequence_type=SequenceType.GENOME
        )
        for record in ncbi_data:
            accession = record["GBSeq_locus"]
            for feature in record["GBSeq_feature-table"]:
                feature_type = SequenceAnnotationUtils.get_annotation_type(feature["GBFeature_key"])
                if feature_type == AnnotationType.UNDEFINED:
                    continue
                feature_range = [
                    (int(interval["GBInterval_from"]), int(interval["GBInterval_to"]))
                    for interval in feature["GBFeature_intervals"]
                ]
                feature_annotation = feature_type.name
                if feature_type in [AnnotationType.GENE, AnnotationType.CDS]:
                    feature_annotation_lst = [
                        qualifier["GBQualifier_value"]
                        for qualifier in feature["GBFeature_quals"]
                        if qualifier["GBQualifier_name"] in ["gene", "product"]
                    ]
                    if len(feature_annotation_lst) > 0:
                        feature_annotation = feature_annotation_lst[0].lower().replace("protein", "")
                    else:
                        logger.info(
                            f"could not find annotation for feature of type {feature_type.name} with content {feature_annotation_lst}"
                        )
                        feature_annotation = np.nan
                if pd.notna(feature_annotation):
                    accession_to_annotations[accession][(feature_annotation, feature_type.name)] = feature_range

                if (
                    feature_annotation == "poly"
                ):  # in the case of a polyprotein, continue looking for qualifier of protein_id and then add more annotations for its products
                    product_accession_lst = [
                        qualifier["GBQualifier_value"]
                        for qualifier in feature["GBFeature_quals"]
                        if qualifier["GBQualifier_name"] == "protein_id"
                    ]
                    if len(product_accession_lst) > 0:
                        product_accession = product_accession_lst[0]
                        product_record = list(
                            Entrez.parse(Entrez.efetch(db="protein", id=product_accession, retmode="xml"))
                        )[0]
                        for product_feature in product_record["GBSeq_feature-table"]:
                            if product_feature["GBFeature_key"] == "Region":
                                product_feature_type = AnnotationType.CDS
                                product_feature_range = [
                                    (int(interval["GBInterval_from"]) * 3, int(interval["GBInterval_to"]) * 3)
                                    for interval in product_feature["GBFeature_intervals"]
                                ]
                                product_feature_annotation_components = (
                                    [
                                        qualifier["GBQualifier_value"]
                                        for qualifier in product_feature["GBFeature_quals"]
                                        if qualifier["GBQualifier_name"] == "region_name"
                                    ][0]
                                    .lower()
                                    .replace("protein", "")
                                    .split("_")
                                )
                                component_index = 0
                                if (
                                    len(product_feature_annotation_components) > 1
                                    and "like" not in product_feature_annotation_components
                                ):
                                    component_index = 1
                                product_feature_annotation = "_".join(
                                    product_feature_annotation_components[component_index:]
                                )
                                accession_to_annotations[accession][
                                    (product_feature_annotation, product_feature_type.name)
                                ] = product_feature_range

        return accession_to_annotations

    @staticmethod
    def parse_ncbi_annotations(accessions: t.List[str], acc_to_sp: t.Dict[str, str]) -> pd.DataFrame:
        """
        :param accessions: list of accessions to get ncbi annotations for
        :param acc_to_sp: map of accessions to species names
        :return: a dataframe with annotations of the given accessions
        """
        df = pd.DataFrame(columns=["species_name", "accession", "annotation_name", "annotation_type", "coordinate"])
        accession_to_annotations = SequenceAnnotationUtils.get_ncbi_annotations(accessions=accessions)
        for acc in accessions:
            values = {"species_name": acc_to_sp[acc], "accession": acc}
            annotations = accession_to_annotations[acc]
            for annotation in annotations:
                values["annotation_name"] = annotation[0].lower()
                values["annotation_type"] = annotation[1].lower()
                values["coordinate"] = ";".join(
                    [
                        f"{annotations[annotation][i][0]}..{annotations[annotation][i][1]}:+"
                        for i in range(len(annotations[annotation]))
                    ]
                )
                df = df.append(values, ignore_index=True)
        df["source"] = "manual"
        return df

    @staticmethod
    def get_largest_spanning_coordinate_range(coordinate_values) -> str:
        coord_to_start = []
        coord_to_end = []
        max_len_index = 0
        max_len = 0
        for i in range(len(coordinate_values)):
            coord = coordinate_values[i]
            coord_content = [item.replace(":", "").replace("+", "") for item in coord.split("..")]
            coord_to_start.append(int(coord_content[0]))
            coord_to_end.append(int(coord_content[-1]))
            total_len = coord_to_end[-1] - coord_to_start[-1]
            if total_len > max_len:
                max_len = total_len
                max_len_index = i
        return coordinate_values[max_len_index]

    @staticmethod
    def unite_flaviviridae_annotations(
        annotation_data: pd.DataFrame, acc_to_sp: t.Dict[str, str], acc_to_seqlen: t.Dict[str, int]
    ) -> pd.DataFrame:
        """
        :param annotation_data: dataframe of annotations
        :param acc_to_sp: map of accession to its species name
        :param acc_to_seqlen: map of accession to its sequence length
        :return: dataframe with union annotations
        """
        union_annotation_categories = {
            "C": ["core", "nucleocapsid", "capsid", "c_protein", "protein_c"],
            "prM": [
                "prm",
                "precursor_glycoprotein",
                "glycoprotein_precursor",
                "prem",
                "premembrane",
                "pre-membrane",
                "protein_pr",
                "glycoprotein_precursor",
            ],
            "M": ["transmembrane", "_m_protein", "protein_m", "membrane_protein", "membrane_glycoprotein_m", "matrix"],
            "E": ["envelope", "env" "spike_glycoprotein", "putative_glycoprotein", "glycoprot", "glycop_c"],
            "Erns": ["erns", "e-rns"],
            "E1": ["e1"],
            "E2": ["e2"],
            "NS1": ["nonstructural_protein_1", "non-structural_protein_1", "ns1"],
            "NS2": ["nonstructural_protein_2", "non-structural_protein_2", "ns2"],
            "NS2A": ["nonstructural_protein_2a", "non-structural_protein_2a", "ns2a"],
            "NS2B": ["nonstructural_protein_2b", "non-structural_protein_2b", "ns2b"],
            "NS3": ["nonstructural_protein_3", "non-structural_protein_3", "ns3", "peptidase"],
            "NS4A": ["nonstructural_protein_4a", "non-structural_protein_4a", "ns4a"],
            "NS4B": ["nonstructural_protein_4b", "non-structural_protein_4b", "ns4b"],
            "NS5": [
                "nonstructural_protein_5",
                "non-structural_protein_5",
                "ns5",
                "rna_polymerase",
                "polymerase",
                "rnrp",
                "rnap",
                "rt",
            ],
            "Npro": ["npro"],
            "Polyprotein": ["poly", "polyprotein"],
            "3UTR": ["utr3", "3utr", "3_utr", "3'utr", "3'_utr", "3ncr", "3_ncr", "3'ncr", "3'_ncr"],
            "5UTR": ["utr5", "5utr", "5_utr", "5'utr", "5'_utr", "5ncr", "5_ncr", "5'ncr", "5'_ncr"],
            "FIFO": ["fifo"],
            "P7": ["p7"],
            "2K": ["2k"],
        }

        def get_union_annotation(annotation_name):
            possible_categories = []
            for category in union_annotation_categories:
                patterns = union_annotation_categories[category]
                for pattern in patterns:
                    if pattern in annotation_name or pattern == annotation_name:
                        possible_categories.append(category)
                if category.lower() == annotation_name:
                    possible_categories.append(category)
            possible_categories = list(set(possible_categories))
            # process
            if annotation_name == "env":
                possible_categories = ["E"]
            if annotation_name == "rna_dep_rnap":
                possible_categories = ["NS5"]
            if len(possible_categories) > 1 and "Polyprotein" in possible_categories:
                possible_categories.remove("Polyprotein")
            if "E" in possible_categories and ("E1" in possible_categories or "E2" in possible_categories):
                possible_categories.remove("E")
            if "FIFO" in possible_categories:
                possible_categories = ["FIFO"]
            if "prM" in possible_categories and "E" in possible_categories:
                possible_categories.remove("E")
            if (
                "M" in possible_categories
                and ("E" in possible_categories)
                or ("prM" in possible_categories)
                and "membrane" in annotation_name
            ):
                possible_categories = ["M"]
            if "NS2A" in possible_categories or "NS2B" in possible_categories:
                possible_categories.remove("NS2")
            if "P7" in possible_categories:
                possible_categories = ["P7"]
            if "2K" in possible_categories:
                possible_categories = ["2K"]
            if len(possible_categories) == 0:
                # print(f"no categories were matched with {annotation_name}")
                return np.nan
            if len(possible_categories) > 1:
                # print(f"assigned categories for annotation {annotation_name} are {','.join(possible_categories)}")
                return np.nan
            elif len(possible_categories) == 1:
                return possible_categories[0]

        # step 1: group by annotation_type
        annotation_data["annotation_union_name"] = np.nan

        # step 2: within each annotation type, categorize to one of the annotation union categories specified above
        for annotation_type in annotation_data.annotation_type.unique():
            annotation_data.loc[
                annotation_data.annotation_type == annotation_type, "annotation_union_name"
            ] = annotation_data.loc[annotation_data.annotation_type == annotation_type, "annotation_name"].apply(
                get_union_annotation
            )
        logger.info(
            f"%missing union annotations={round(annotation_data.loc[annotation_data.annotation_union_name.isna()].shape[0] / annotation_data.shape[0] * 100, 2)}"
        )

        # step 3: throw away records without union annotation
        relevant_annotation_data = annotation_data.loc[annotation_data.annotation_union_name.notna()]
        annotation_data_by_acc_and_annot = relevant_annotation_data.groupby(
            ["accession", "annotation_union_name", "annotation_type"]
        )

        # step 4: collapse duplicate records by setting the largest spanning coordinate across them as the final one
        groups_data = []
        for group in annotation_data_by_acc_and_annot.groups.keys():
            group_data = annotation_data_by_acc_and_annot.get_group(group)
            coordinate_values = group_data.coordinate.values
            group_data["union_coordinate"] = SequenceAnnotationUtils.get_largest_spanning_coordinate_range(
                coordinate_values=coordinate_values
            )
            groups_data.append(group_data)
        filled_annotation_data = pd.concat(groups_data)

        # step 5: add 5UTR and 5UTR  annotations
        acc_to_poly_coord = (
            filled_annotation_data.loc[filled_annotation_data.annotation_union_name == "Polyprotein"]
            .set_index("accession")["union_coordinate"]
            .to_dict()
        )
        acc_to_poly_start = {
            acc: int(acc_to_poly_coord[acc].split("..")[0].replace(":+", "")) for acc in acc_to_poly_coord
        }
        acc_to_poly_end = {
            acc: int(acc_to_poly_coord[acc].split("..")[-1].replace(":+", "")) for acc in acc_to_poly_coord
        }
        accessions_with_poly_annotation = list(
            filled_annotation_data.loc[filled_annotation_data.annotation_union_name == "Polyprotein"].accession.unique()
        )

        annotations_by_accession = (
            filled_annotation_data.groupby("accession")["annotation_union_name"].apply(lambda x: list(set(x))).to_dict()
        )
        accessions_without_5utr_annotation = [
            acc
            for acc in annotations_by_accession
            if "5UTR" not in annotations_by_accession[acc] and acc in accessions_with_poly_annotation
        ]
        accessions_without_3utr_annotation = [
            acc
            for acc in annotations_by_accession
            if "3UTR" not in annotations_by_accession[acc] and acc in accessions_with_poly_annotation
        ]
        logger.info(
            f"complementing {len(accessions_without_5utr_annotation)} 5UTR annotations and {len(accessions_without_3utr_annotation)} 3UTR annotations"
        )

        complementary_5utr_data = pd.DataFrame(columns=filled_annotation_data.columns)
        for acc in accessions_without_5utr_annotation:
            complementary_5utr_data = complementary_5utr_data.append(
                SequenceAnnotationUtils.get_flavi_5utr_annotation(
                    acc=acc, acc_to_sp=acc_to_sp, acc_to_poly_start=acc_to_poly_start
                ),
                ignore_index=True,
            )

        complementary_3utr_data = pd.DataFrame(columns=filled_annotation_data.columns)
        for acc in accessions_without_3utr_annotation:
            complementary_3utr_data = complementary_3utr_data.append(
                SequenceAnnotationUtils.get_flavi_3utr_annotation(
                    acc=acc, acc_to_sp=acc_to_sp, acc_to_len=acc_to_seqlen, acc_to_poly_end=acc_to_poly_end
                ),
                ignore_index=True,
            )
        annotation_data = pd.concat([filled_annotation_data, complementary_5utr_data, complementary_3utr_data])
        annotation_data.drop(labels=[col for col in annotation_data.columns if "Unnamed" in col], axis=1, inplace=True)
        return annotation_data

    @staticmethod
    def get_flavi_5utr_annotation(
        acc: str, acc_to_sp: t.Dict[str, str], acc_to_poly_start: t.Dict[str, int]
    ) -> t.Dict[str, str]:
        """
        :param acc: accession
        :param acc_to_sp: map of accessions to species names
        :param acc_to_poly_start: map of accessions to polyprotein start position
        :return: dictionary of an entry of a 5UTR annotation of the accession
        """
        species_name = acc_to_sp[acc]
        annotation_type = "utr5"
        annotation_name = "5UTR"
        union_coordinate = f"0..{int(acc_to_poly_start[acc])}:+"
        annotation_dict = {
            "accession": acc,
            "annotation_union_name": annotation_name,
            "union_coordinate": union_coordinate,
            "species_name": species_name,
            "annotation_type": annotation_type,
        }
        return annotation_dict

    @staticmethod
    def get_flavi_3utr_annotation(
        acc: str, acc_to_sp: t.Dict[str, str], acc_to_len: t.Dict[str, int], acc_to_poly_end: t.Dict[str, int]
    ) -> t.Dict[str, str]:
        """
        :param acc: accession
        :param acc_to_sp: map of accessions to species names
        :param acc_to_len: map of accession to sequence length
        :param acc_to_poly_end: map of accessions to polyprotein end position
        :return: dictionary of an entry of a 3UTR annotation of the accession
        """
        annotation_name = "3UTR"
        union_coordinate = f"{int(acc_to_poly_end[acc])}..{int(acc_to_len[acc])}:+"
        species_name = acc_to_sp[acc]
        annotation_type = "utr3"
        annotation_dict = {
            "accession": acc,
            "annotation_union_name": annotation_name,
            "union_coordinate": union_coordinate,
            "species_name": species_name,
            "annotation_type": annotation_type,
        }
        return annotation_dict

    @staticmethod
    def get_annotations_frequencies(annotation_data: pd.DataFrame) -> pd.DataFrame:
        """
        :param annotation_data: annotation data by accessions
        :return: dataframe with frequencies all all the unique union annotations
        """

        # for each annotation, compute its frequency across the annotated accessions
        accession_to_annotations = (
            annotation_data.groupby("accession")["annotation_union_name"].apply(lambda x: list(set(x))).to_dict()
        )
        annotations = []
        for acc in accession_to_annotations:
            annotations += accession_to_annotations[acc]
        annotations_frequencies_dict = {
            annotation: float(
                len([acc for acc in accession_to_annotations if annotation in accession_to_annotations[acc]])
            )
            / len(accession_to_annotations.keys())
            for annotation in annotations
        }
        annotations_frequencies = (
            pd.DataFrame.from_dict(annotations_frequencies_dict, orient="index", columns=["frequency"])
            .reset_index()
            .rename(columns={"index": "annotation"})
        )
        annotations_frequencies.sort_values("frequency", ascending=False, inplace=True)
        return annotations_frequencies
