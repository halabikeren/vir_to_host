import re
from enum import Enum
from functools import partial

import Bio
import numpy as np
import typing as t

import logging

import pandas as pd
from Bio.Data import CodonTable
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


class DinucleotidePositionType(Enum):
    REGULAR = 1
    BRIDGE = 2
    NONBRIDGE = 3


class GenomeType(Enum):
    RNA = 0
    DNA = 1
    UNKNOWN = np.nan


class GenomeBiasComputationUtils:
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
            dinuc_sequence = GenomeBiasComputationUtils.get_dinucleotides_by_range(
                sequence, range(2, len(sequence) - 2, 3)
            )
        elif computation_type == DinucleotidePositionType.NONBRIDGE:
            dinuc_sequence = GenomeBiasComputationUtils.get_dinucleotides_by_range(
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
        dinucleotide_biases = GenomeBiasComputationUtils.compute_dinucleotide_bias(
            sequence=genome_sequence, computation_type=DinucleotidePositionType.REGULAR,
        )
        id_genomic_traits = dict(dinucleotide_biases)

        if len(coding_sequences) > 0:
            id_genomic_traits.update(
                GenomeBiasComputationUtils.compute_mean_across_sequences(
                    sequences=coding_sequences,
                    func=partial(
                        GenomeBiasComputationUtils.compute_dinucleotide_bias,
                        computation_type=DinucleotidePositionType.BRIDGE,
                    ),
                )
            )

        id_genomic_traits.update(
            GenomeBiasComputationUtils.compute_dinucleotide_bias(
                sequence=genome_sequence, computation_type=DinucleotidePositionType.NONBRIDGE,
            )
        )

        if len(coding_sequences) > 0:
            id_genomic_traits.update(
                GenomeBiasComputationUtils.compute_mean_across_sequences(
                    sequences=coding_sequences, func=GenomeBiasComputationUtils.compute_diaa_bias,
                )
            )
            id_genomic_traits.update(
                GenomeBiasComputationUtils.compute_mean_across_sequences(
                    sequences=coding_sequences,
                    func=partial(GenomeBiasComputationUtils.compute_codon_pair_bias, diaa_bias=id_genomic_traits,),
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
