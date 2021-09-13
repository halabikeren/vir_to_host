import json
import logging
import os
import shutil
import typing as t
import re
from enum import Enum
from time import sleep
from tqdm import tqdm
from urllib.error import HTTPError
import sys

sys.path.append("..")
from settings import get_settings

tqdm.pandas()

import Bio
import numpy as np
import pandas as pd
from Bio import SeqIO, Entrez
from Bio.Seq import Seq

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
    def parse_ncbi_sequence_raw_data_by_unique_acc(
        ncbi_raw_data: t.List[t.Dict[str, str]], is_gi_acc: bool = False
    ) -> t.List[t.Dict[str, str]]:
        """
        :param ncbi_raw_data: raw data from api efetch call to ncbi api
        :param is_gi_acc: indicator weather the accession is gi accession and should thus be converted or not
        :return: parsed ncbi data
        """
        gi_conversion = []
        if is_gi_acc:
            gi_conversion = SequenceCollectingUtils.get_gi_accession_conversion(
                ncbi_raw_data=ncbi_raw_data
            )

        acc_to_seq = {
            record["GBSeq_locus"]: record["GBSeq_sequence"]
            for record in ncbi_raw_data
            if "GBSeq_sequence" in record
        }
        acc_to_cds = {
            record["GBSeq_locus"]: ";".join(
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
            record["GBSeq_locus"]: record["GBSeq_definition"]
            for record in ncbi_raw_data
            if "GBSeq_definition" in record
        }
        acc_to_keywords = {
            record["GBSeq_locus"]: record["GBSeq_keywords"]
            for record in ncbi_raw_data
            if "GBSeq_keywords" in record
        }
        parsed_data = [acc_to_seq, acc_to_cds, acc_to_annotation, acc_to_keywords]
        parsed_data = gi_conversion + parsed_data

        return parsed_data

    @staticmethod
    def fill_ncbi_data_by_unique_acc(
        df: pd.DataFrame, parsed_data: t.List[t.Dict[str, str]], is_gi_acc: bool = False
    ):
        """
        :param df: dataframe to fill
        :param parsed_data: parsed data to fill df with
        :param is_gi_acc: indicator if the accessions to fill data for are gi accessions
        :return: nothing. changes the df inplace
        """
        addition = 0
        gi_acc_to_actual_acc, gi_acc_to_source = dict(), dict()
        if is_gi_acc:
            gi_acc_to_actual_acc = parsed_data[0]
            gi_acc_to_source = parsed_data[1]
            addition = 2

        acc_to_seq = parsed_data[0 + addition]
        acc_to_cds = parsed_data[1 + addition]
        acc_to_annotation = parsed_data[2 + addition]
        acc_to_keywords = parsed_data[3 + addition]

        for col in ["sequence", "cds", "annotation", "keywords", "category"]:
            if col not in df.columns:
                df[col] = np.nan

        # replace values in acc field to exclude version number
        df["accession"] = df["accession"].apply(
            lambda x: str(x).split(".")[0] if pd.notna(x) else x
        )

        df.set_index("accession", inplace=True)
        if is_gi_acc:
            df["source"].update(gi_acc_to_source)
        df.reset_index(inplace=True)
        df["accession"] = df["accession"].replace(gi_acc_to_actual_acc)

        df.set_index("accession", inplace=True)
        old_missing_seq_num = df["sequence"].isna().sum()
        logger.info(f"# extracted sequences = {len(acc_to_seq.keys())}")
        df["sequence"].fillna(value=acc_to_seq, inplace=True)
        new_missing_seq_num = df["sequence"].isna().sum()

        old_missing_cds_num = df["cds"].isna().sum()
        logger.info(f"# extracted cds = {len(acc_to_cds.keys())}")
        df["cds"].fillna(value=acc_to_cds, inplace=True)
        new_missing_cds_num = df["cds"].isna().sum()

        old_missing_annotations_num = df["annotation"].isna().sum()
        logger.info(f"# extracted annotations = {len(acc_to_annotation.keys())}")
        df["annotation"].fillna(value=acc_to_annotation, inplace=True)
        new_missing_annotations_num = df["annotation"].isna().sum()

        old_missing_kws_num = df["keywords"].isna().sum()
        logger.info(f"# extracted keywords = {len(acc_to_keywords.keys())}")
        df["keywords"].fillna(value=acc_to_keywords, inplace=True)
        new_missing_kws_num = df["keywords"].isna().sum()

        df["category"] = df["annotation"].apply(
            lambda x: "genome" if type(x) is str and "complete genome" in x else np.nan
        )
        df.reset_index(inplace=True)

        logger.info(
            f"dataframe filling is complete in pid {os.getpid()}, with {old_missing_seq_num-new_missing_seq_num} sequences filled, {old_missing_cds_num-new_missing_cds_num} cds regions filled, {old_missing_annotations_num-new_missing_annotations_num} annotations filled and {old_missing_kws_num-new_missing_kws_num} keywords filled"
        )

    @staticmethod
    def fill_missing_data_by_acc(df: pd.DataFrame) -> str:

        df_path = f"{os.getcwd()}/df_{SequenceCollectingUtils.fill_missing_data_by_acc.__name__}_pid_{os.getpid()}.csv"

        # first, handle non gi accessions
        accessions = list(df.loc[df.source != "gi", "accession"].dropna().unique())
        if len(accessions) > 0:
            logger.info(
                f"performing efetch query to ncbi on {len(accessions)} genbank and refseq accessions"
            )
            ncbi_raw_data = SequenceCollectingUtils.do_ncbi_batch_fetch_query(
                accessions=accessions
            )
            parsed_data = (
                SequenceCollectingUtils.parse_ncbi_sequence_raw_data_by_unique_acc(
                    ncbi_raw_data=ncbi_raw_data
                )
            )
            SequenceCollectingUtils.fill_ncbi_data_by_unique_acc(
                df=df, parsed_data=parsed_data
            )

        # now, handle gi accessions
        accessions = list(df.loc[df.source == "gi", "accession"].dropna().unique())
        if len(accessions) > 0:
            logger.info(
                f"performing efetch query to ncbi on {len(accessions)} gi accessions"
            )
            logger.info(
                f"performing efetch query to ncbi on {len(accessions)} gi accessions"
            )
            ncbi_raw_data = SequenceCollectingUtils.do_ncbi_batch_fetch_query(
                accessions=accessions
            )
            parsed_data = (
                SequenceCollectingUtils.parse_ncbi_sequence_raw_data_by_unique_acc(
                    ncbi_raw_data=ncbi_raw_data, is_gi_acc=True
                )
            )
            SequenceCollectingUtils.fill_ncbi_data_by_unique_acc(
                df=df, parsed_data=parsed_data, is_gi_acc=True
            )

        df["category"] = df["annotation"].apply(
            lambda x: "genome" if pd.notna(x) and "complete genome" in x else np.nan
        )

        df.to_csv(df_path, index=False)
        return df_path

    @staticmethod
    def get_gi_accession_conversion(
        ncbi_raw_data: t.List[t.Dict[str, str]]
    ) -> t.List[t.Dict[str, str]]:
        """
        :param ncbi_raw_data: raw records from ncbi efetch request result
        :return: path to dataframe with translated accessions
        """

        gi_acc_to_actual_acc, gi_acc_to_source = dict(), dict()
        for record in ncbi_raw_data:
            gi_acc_found = False
            for acc_data in record["GBSeq_other-seqids"]:
                if "gi" in acc_data:
                    gi_acc = acc_data.split("|")[-1]
                    actual_acc = record["GBSeq_locus"]
                    source = (
                        "refseq"
                        if "ref" in " ".join(record["GBSeq_other-seqids"])
                        else "genbank"
                    )
                    gi_acc_to_actual_acc[gi_acc] = actual_acc
                    gi_acc_to_source[gi_acc] = source
                    gi_acc_found = True
            if not gi_acc_found:
                logger.info(f"no gi accession was found for record {record}")

        return [gi_acc_to_actual_acc, gi_acc_to_source]

    @staticmethod
    def flatten_sequence_data(
        df: pd.DataFrame,
        data_prefix: str = "virus",
    ) -> pd.DataFrame:
        """
        :param df: dataframe to flatten
        :param data_prefix: data prefix, for all column names
        :return: flattened dataframe
        """

        # remove data prefix
        flattened_df = df.rename(
            columns={
                col: col.replace(
                    f"{data_prefix}{'_' if len(data_prefix)>0 else ''}", ""
                )
                for col in df.columns
            }
        )

        # set source by differente accession fields
        flattened_df["source"] = flattened_df[
            ["genbank_accession", "gi_accession"]
        ].apply(
            lambda x: "genbank"
            if pd.notna(x.genbank_accession)
            else ("gi" if pd.notna(x.gi_accession) else np.nan),
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
        flattened_df = flattened_df.assign(
            accession=flattened_df.accession.str.split(";")
        ).explode("accession")
        flattened_df = flattened_df.set_index(
            flattened_df.groupby(level=0).cumcount(), append=True
        )
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
    def do_ncbi_batch_fetch_query(accessions: t.List[str]) -> t.List[t.Dict[str, str]]:
        """
        :param accessions: list of accessions to batch query on
        :return: list of ncbi records corresponding to the accessions
        """
        ncbi_raw_records = []
        if len(accessions) == 0:
            return ncbi_raw_records
        retry = True
        while retry:
            try:
                ncbi_raw_records = list(
                    Entrez.parse(
                        Entrez.efetch(
                            db="nucleotide",
                            id=",".join(accessions),
                            retmode="xml",
                            api_key=get_settings().ENTREZ_API_KEY,
                        )
                    )
                )
                retry = False
            except HTTPError as e:
                if e.code == 429:
                    logger.info(f"Entrez query failed due to error {e}. retrying...")
                    sleep(3)
                else:
                    logger.error(f"Failed Entrez query due to error {e}")
                    exit(1)
        logger.info(
            f"collected {len(ncbi_raw_records)} records based on {len(accessions)} accessions"
        )
        return ncbi_raw_records

    @staticmethod
    def do_ncbi_search_queries(
        organisms: t.List[str], text_condition: str = "complete genome"
    ) -> t.Dict[str, str]:
        """
        :param organisms: list of organisms names to search
        :param text_condition: additional text condition to search by
        :return: map of organisms to their gi accessions
        """
        logger.info(
            f"performing {len(organisms)} esearch queries on [Organism] and text condition {text_condition}"
        )
        organism_to_raw_data = dict()
        i = 0
        while i < len(organisms):
            try:
                organism_to_raw_data[organisms[i]] = Entrez.read(
                    Entrez.esearch(
                        db="nucleotide",
                        term=f"({organisms[i]}[Organism]) AND {text_condition}[Text Word]",
                        retmode="xml",
                        api_key=get_settings().ENTREZ_API_KEY,
                    )
                )
                i += 1
            except HTTPError as e:
                if e.code == 429:
                    print(
                        f"{os.getpid()} failed api request with error {e} and thus will sleep for 3 seconds before trying again"
                    )
                    sleep(3)
                else:
                    print(f"{os.getpid()} failed api request with error {e}")
                    exit(1)
        organism_to_gi_acc = {
            name: organism_to_raw_data[name]["IdList"][0]
            for name in organism_to_raw_data
            if len(organism_to_raw_data[name]["IdList"]) > 0
        }
        return organism_to_gi_acc

    @staticmethod
    def fill_missing_data_by_organism(df: pd.DataFrame) -> str:

        df_path = f"{os.getcwd()}/df_{SequenceCollectingUtils.fill_missing_data_by_organism.__name__}_pid_{os.getpid()}.csv"

        # find gi accessions for the given organism names
        organisms = list(df.taxon_name.unique())
        taxon_name_to_gi_accession = SequenceCollectingUtils.do_ncbi_search_queries(
            organisms=organisms
        )
        logger.info(
            f"gi accessions extracted for {len(taxon_name_to_gi_accession.keys())} out of {len(organisms)} records"
        )
        df.set_index("taxon_name", inplace=True)
        df["accession"].fillna(value=taxon_name_to_gi_accession, inplace=True)
        df.reset_index(inplace=True)

        # extract data based on the obtained gi accessions
        accessions = [str(item) for item in list(df.accession.dropna().unique())]
        if len(accessions) > 0:
            logger.info(
                f"performing efetch query to ncbi on {len(accessions)} gi accessions"
            )
            ncbi_raw_data = SequenceCollectingUtils.do_ncbi_batch_fetch_query(
                accessions=accessions
            )
            parsed_data = (
                SequenceCollectingUtils.parse_ncbi_sequence_raw_data_by_unique_acc(
                    ncbi_raw_data=ncbi_raw_data, is_gi_acc=True
                )
            )
            SequenceCollectingUtils.fill_ncbi_data_by_unique_acc(
                df=df, parsed_data=parsed_data, is_gi_acc=True
            )

        df.to_csv(df_path, index=False)
        return df_path


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
        coding_sequence: str,
        computation_type: DinucleotidePositionType = DinucleotidePositionType.BRIDGE,
    ) -> t.Dict[str, float]:
        """
        :param coding_sequence: a single coding sequences
        :param computation_type: can be either regular, or limited to bridge or non-bridge positions
        :return: dinucleotide bias dictionary
        dinculeotide bias computed according to https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        computation_type options:
            BRIDGE - consider only dinucleotide positions corresponding to bridges between codons (one is the last pos of a codon and the next is the first of another)
            NONBRIDGE - consider only dinucleotide positions do not correspond to bridges between codons
            REGULAR - consider all dinucleotide positions"""
        dinuc_sequence = coding_sequence
        if (
            computation_type == DinucleotidePositionType.BRIDGE
        ):  # limit the sequence to bridge positions only
            dinuc_sequence = GenomeBiasCollectingService.get_dinucleotides_by_range(
                coding_sequence, range(2, len(coding_sequence) - 2, 3)
            )
        elif computation_type == DinucleotidePositionType.NONBRIDGE:
            dinuc_sequence = GenomeBiasCollectingService.get_dinucleotides_by_range(
                coding_sequence, range(0, len(coding_sequence) - 2, 3)
            )
        nucleotide_count = {
            "A": dinuc_sequence.count("A"),
            "C": dinuc_sequence.count("C"),
            "G": dinuc_sequence.count("G"),
            "T": dinuc_sequence.count("T"),
        }
        nucleotide_total_count = len(dinuc_sequence)
        assert nucleotide_total_count > 0
        dinucleotide_total_count = len(coding_sequence) / 2
        assert dinucleotide_total_count > 0
        dinucleotide_biases = dict()
        for nuc_i in nucleotide_count.keys():
            for nuc_j in nucleotide_count.keys():
                dinucleotide = nuc_i + "p" + nuc_j
                dinucleotide_biases[
                    computation_type.name + "_" + dinucleotide + "_bias"
                ] = (coding_sequence.count(dinucleotide) / dinucleotide_total_count) / (
                    nucleotide_count[nuc_i]
                    / nucleotide_total_count
                    * nucleotide_count[nuc_j]
                    / nucleotide_total_count
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
                    if codon not in STOP_CODONS
                    and Bio.Data.CodonTable.standard_dna_table.forward_table[codon]
                    == aa
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
        coding_sequence: str, diaa_bias: t.Dict[str, float]
    ) -> t.Dict[str, float]:
        """
        :param coding_sequence: a single coding sequences
        :param diaa_bias: dictionary mapping diaa to its bias
        :return: dictionary mapping each dicodon to its bias
        codon pair bias measured by the codon pair score (CPS) as shown in https://science.sciencemag.org/content/sci/suppl/2018/10/31/362.6414.577.DC1/aap9072_Babayan_SM.pdf
        the denominator is obtained by multiplying the count od each codon with the bias of the respective amino acid pair
        """
        codon_count = dict()
        for codon in CODONS:
            codon_count[codon] = coding_sequence.count(codon)
        codon_pair_scores = dict()
        for codon_i in CODONS:
            for codon_j in CODONS:
                if codon_i not in STOP_CODONS and codon_j not in STOP_CODONS:
                    codon_pair = codon_i + codon_j
                    codon_pair_count = coding_sequence.count(codon_pair)
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
    def collect_genomic_bias_features(genome_sequence: str, coding_sequence: str):
        """
        :param genome_sequence: list of genomic sequences
        :param coding_sequence: list of coding sequences
        :return: dictionary with genomic features to be added as a record to a dataframe
        """
        dinucleotide_biases = GenomeBiasCollectingService.compute_dinucleotide_bias(
            coding_sequence=genome_sequence,
            computation_type=DinucleotidePositionType.REGULAR,
        )
        id_genomic_traits = dict(dinucleotide_biases)
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                coding_sequence=coding_sequence,
                computation_type=DinucleotidePositionType.BRIDGE,
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_dinucleotide_bias(
                coding_sequence=genome_sequence,
                computation_type=DinucleotidePositionType.NONBRIDGE,
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_codon_bias(
                coding_sequence=coding_sequence
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_diaa_bias(
                coding_sequence=coding_sequence
            )
        )
        id_genomic_traits.update(
            GenomeBiasCollectingService.compute_codon_pair_bias(
                coding_sequence=coding_sequence, diaa_bias=id_genomic_traits
            )
        )
        return id_genomic_traits

    @staticmethod
    def extract_coding_sequence(genomic_sequence: str, coding_regions: str) -> str:
        """
        :param genomic_sequence: genomic sequence
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
                coding_sequence += genomic_sequence[start - 1 : end]
            assert len(coding_sequence) % 3 == 0
            coding_sequences.append(coding_sequence)
        return ",".join(coding_sequences)

    @staticmethod
    def compute_genome_bias_features(df: pd.DataFrame) -> str:
        """
        :param df: dataframe with sequence data to compute genome bias over
        :return: path ot the output df with the computed genomic biases
        """
        genomic_bias_df_path = f"{os.getcwd()}/{GenomeBiasCollectingService.collect_genomic_bias_features.__name__}_pid_{os.getpid()}.csv"
        genomic_bias_df = pd.DataFrame()

        # collect genomic bias features
        for index, row in df.iterrows():
            record = {"taxon_name": row.taxon_name}
            genomic_sequence = row.sequence
            coding_sequence = GenomeBiasCollectingService.extract_coding_sequence(
                genomic_sequence=row.sequence, coding_regions=row.cds
            )
            genomic_features = (
                GenomeBiasCollectingService.collect_genomic_bias_features(
                    genome_sequence=genomic_sequence,
                    coding_sequence=coding_sequence,
                )
            )
            record.update(genomic_features)
            genomic_bias_df = genomic_bias_df.append(record, ignore_index=True)

        genomic_bias_df.to_csv(genomic_bias_df_path, index=False)
        return genomic_bias_df_path
