import os
import typing as t
import logging

import re

from Bio import SeqIO

logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class PRFSite:
    slippery_site_start_position: int
    slippery_site_sequence: str
    support_structure_start_position: int
    support_structure_end_position: int
    support_structure_sequence: str
    knotted_structure: str
    knotted_structure_mfe: float
    nested_structure: str
    nested_structure_mfe: float
    length: int
    significant: bool  # deem as significant only if Deltarel > 0.1 based on the recommendation in the paper:
    # https://academic.oup.com/nar/article/36/18/6013/1073753#92964095
    rank: float  # measure between 1-10. the lower it is, the higher the certainty that the candidate indeed represents a true PRF.


class PRFPredUtils:
    @staticmethod
    def parse_knotinframe_output(results_path: str) -> t.List[PRFSite]:
        """
        :param results_path: path to knotinframe execution output
        :return: list of detected PRF sites
        """
        prf_summary_regex = re.compile(
            "Rank\:\s*(\d*).*?Slippery sequence\:\s*([^\n]*).*?Slippery position\:\s*(\d*).*?Substring length\:\s*(\d*).*?Deltarel\:\s*(\d*\.?\d*)\n\s*(\d*)\s*(\w*)\s*(\d*)\n\s*(-?\d*\.?\d*)\s*([^\s]*)\s*knotted structure\n\s*(-?\d*\.?\d*)\s*([^\s]*)\s*nested structure",
            re.MULTILINE | re.DOTALL,
        )
        with open(results_path, "r") as outfile:
            result = outfile.read()
        if "No suitable slippery sites have been detected" in result:
            return []
        prf_summaries = result.split("\n\n")
        prf_sites = []
        for prf_summary in prf_summaries:
            match = prf_summary_regex.search(prf_summary)
            if match:
                rank = int(match.group(1))
                slippery_site_sequence = match.group(2)
                slippery_site_start_position = int(match.group(3))
                length = int(match.group(4))
                significant = True if float(match.group(5)) > 0.1 else False
                support_structure_start_position = int(match.group(6))
                support_structure_sequence = match.group(7)
                support_structure_end_position = int(match.group(8))
                knotted_structure_mfe = float(match.group(9))
                knotted_structure = match.group(10)
                nested_structure_mfe = float(match.group(11))
                nested_structure = match.group(12)
                prf_site = PRFSite(
                    rank=rank,
                    slippery_site_sequence=slippery_site_sequence,
                    slippery_site_start_position=slippery_site_start_position,
                    length=length,
                    significant=significant,
                    support_structure_start_position=support_structure_start_position,
                    support_structure_sequence=support_structure_sequence,
                    support_structure_end_position=support_structure_end_position,
                    knotted_structure_mfe=knotted_structure_mfe,
                    knotted_structure=knotted_structure,
                    nested_structure_mfe=nested_structure_mfe,
                    nested_structure=nested_structure,
                )
                prf_sites.append(prf_site)
        return prf_sites

    @staticmethod
    def exec_knotinframe(input_path: str, aligned_input_path: str, output_path: str) -> t.List[PRFSite]:
        """
        :param input_path: a path to sequence data to execute knotinframe on, in a fasta format
        :param aligned_input_path: path to the aligned sequence data, based on which common -1 PRF sites will be selected
        :param output_path: file name to write knotinframe output to
        :return: list of significant prf sites
        """
        if not os.path.exists(input_path):
            logger.error("the provided input path does not exist")
            raise ValueError("the provided input path does not exist")
        sequence_records = list(SeqIO.parse(input_path, format="fasta"))
        seq_to_prf_sites = dict()
        for record in sequence_records:
            res = os.system(
                f"singularity run /groups/itay_mayrose/halabikeren/programs/knotinframe/knotinframe.simg {record.seq} > {output_path}"
            )
            seq_to_prf_sites[record.id] = PRFPredUtils.parse_knotinframe_output(output_path)

        # in the case of multiple sequences, there might be repetitive prf sites across sequences - what should I do?
        if len(seq_to_prf_sites.keys()) == 1:
            prf_sites = seq_to_prf_sites[list(seq_to_prf_sites.keys())[0]]
        else:
            prf_sites = PRFPredUtils.get_intersection_prf_sites(
                seq_to_sites=seq_to_prf_sites, aligned_sequences_path=aligned_input_path
            )
        significant_prf_sites = [prf_site for prf_site in prf_sites if prf_site.significant]
        return significant_prf_sites

    @staticmethod
    def get_intersection_prf_sites(
        seq_to_sites: t.Dict[str, t.List[PRFSite]], aligned_sequences_path: str,
    ) -> t.List[PRFSite]:
        """
        :param seq_to_sites: map of sequence identifier to its detected -1 PRF sites
        :param aligned_sequences_path: path to aligned sequences file, based on which common -1 PRF sites will be selected
        :return: list of common -1 PRF sites across sequences
        """
        # filter data to consist only of PRF sites that
        # 1. appear in > 50% of the sequences, with sequence homology of at least 85%
        # 2. appear within similar range across the sequences (while considering their alignment), with a shift of no more than 100 bp
        # there is a lot of disagreement here, so there us the option of using gentack as an alternative: https://www.worldscientific.com/doi/epdf/10.1142/S0219720010004847
        # in the latter case, I need to check if the predicted genes are annotated
        return []
