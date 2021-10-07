import logging
import os
import re
import typing as t
from dataclasses import dataclass

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

logger = logging.getLogger(__name__)

RNAALIFOLD_NUM_SEQ_LIMIT = 3000
RNAALIFOLD_SEQLEN_LIMIT = 30000


@dataclass
class RNASecondaryStruct:
    alignment_path: str
    consensus_representation: str  # ~structure classification - need to make sure that this
    consensus_sequence: str
    is_significant: bool
    mean_pairwise_identity: float
    shannon_entropy: float
    gc_content: float
    mean_single_sequence_mfe: float
    consensus_mfe: float
    mean_zscore: float
    structure_conservation_index: float
    svm_rna_probability: float


class RNAPredUtils:
    @staticmethod
    def partition_rnalalifold_input(
        input_path: str, output_dir: str, window_size: int = 1000, jump_size: int = 500
    ) -> t.List[str]:
        """
        :param input_path: path with the complete alignment
        :param output_dir: directory to hold the alignment components, partitioned according to a sliding window
        :param window_size: size of partition
        :param jump_size: size of jump (overlap is allowed)
        :return: list of input paths
        """
        os.makedirs(output_dir, exist_ok=True)
        input_records = list(SeqIO.parse(input_path, format="fasta"))
        input_length = len(input_records[0].seq)
        input_windows = [
            (i, i + window_size) for i in range(0, input_length, jump_size)
        ]
        output_paths = []
        for window in input_windows:
            start = window[0]
            end = window[1]
            window_records = [
                SeqRecord(id=record.id, seq=Seq(str(record.seq)[start:end]))
                for record in input_records
            ]
            output_path = f"{output_dir}/window_{start}_{end}.fasta"
            SeqIO.write(window_records, output_path, "fasta")
            output_paths.append(output_path)
        return output_paths

    @staticmethod
    def exec_rnalalifold(input_path: str, output_dir: str) -> int:
        """
        :param input_path: directory of a multiple sequence alignment corresponding ot genomic sequences in fasta format
        :param output_dir: directory of the output files of RNAAlifold
        :return: none
        """

        # if the output path exists, do nothing
        output_paths = [
            f"{output_dir}/{path}/RNALalifold_results.stk"
            for path in os.listdir(output_dir)
        ]
        if len(output_paths) > 0 and np.all(
            [os.path.exists(path) for path in output_paths]
        ):
            return 0

        # to do: based on input size limitations of the program, determine weather sliding window is required
        input_content = list(SeqIO.parse(input_path, format="fasta"))
        if (
            len(input_content) > RNAALIFOLD_NUM_SEQ_LIMIT
            or len(input_content[0].seq) > RNAALIFOLD_SEQLEN_LIMIT
        ):
            input_dir = f"{os.path.dirname(input_path)}partitions_{os.path.basename(input_path).split('.')[0]}/"
            input_paths = RNAPredUtils.partition_rnalalifold_input(
                input_path=input_path, output_dir=input_dir
            )
        else:
            input_paths = [input_path]

        os.makedirs(output_dir, exist_ok=True)
        for i in range(len(input_paths)):
            exec_output_dir = f"{output_dir}/{i}/"
            os.makedirs(exec_output_dir, exist_ok=True)
            output_path = f"{exec_output_dir}/RNALalifold_results.stk"
            if not os.path.exists(output_path):
                log_path = f"{output_dir}/RNALalifold.log"
                cmd = f"RNALalifold {input_path} --input-format=F --csv --aln > {log_path}"
                res = os.system(cmd)
                if res != 0 or not os.path.exists(output_path):
                    logger.error(
                        f"failed to execute RMALalifold properly on {input_path} due to error. Additional info can be found in {log_path}"
                    )
                    exit(1)
            # remove redundant output files
            for path in os.listdir(exec_output_dir):
                if f"{exec_output_dir}{path}" != output_path:
                    os.remove(f"{output_dir}{path}")

        return 0

    @staticmethod
    def parse_rnaalifold_output(rnaalifold_output_dir: str, mlocarna_input_dir: str):
        """
        :param rnaalifold_output_dir: directory holding the output of RNAAliFold execution
        :param mlocarna_input_dir: directory to hold the input for mLocARNA executions
        :return: none. parses RNAAliFold output and creastes inputs for mlocARNA based on it
        """
        os.makedirs(mlocarna_input_dir, exist_ok=True)
        structure_segment_regex = re.compile(
            "# STOCKHOLM 1.0(.*?)\/\/", re.DOTALL | re.MULTILINE
        )
        output_path = f"{rnaalifold_output_dir}/RNALalifold_results.stk"
        with open(output_path, "r") as output_file:
            output_content = output_file.read()
        structures_segments = [
            match.group(1) for match in structure_segment_regex.finditer(output_content)
        ]
        local_region_regex = re.compile("#=GF ID aln_(\d*)_(\d*)\n.*?\n(.*?)#")
        for structure_segment in structures_segments:
            match = local_region_regex.search(structure_segment)
            start_pos = match.group(1)
            end_pos = match.group(2)
            local_sequences_data = match.group(3).split(" ")
            local_seq_records = []
            for i in range(len(local_sequences_data) // 2):
                acc = local_sequences_data[i]
                seq = Seq(local_sequences_data[i + 1].replace("-", ""))
                if len(seq) > 0:
                    record = SeqRecord(id=acc, seq=seq)
                    local_seq_records.append(record)

            # write a fasta file of the local region as input from mLocARNA
            output_path = f"{mlocarna_input_dir}/{start_pos}_{end_pos}.fasta"
            SeqIO.write(local_seq_records, output_path, "fasta")

    @staticmethod
    def exec_mlocarna(input_path: str, output_path: str) -> int:
        if not os.path.exists(output_path):
            cmd = f"mlocarna {input_path} --probabilistic --consistency-transform --it-reliable-structure > {output_path}"
            res = os.system(cmd)
            if res != 0 or not os.path.exists(output_path):
                logger.error(
                    f"failed to execute RMALalifold properly on {input_path} due to error. Additional info can be found in {output_path}"
                )
                exit(1)
        return 0

    @staticmethod
    def parse_mlocarna_output(mlocarna_output_path: str, rnaz_input_path: str):
        """
        :param mlocarna_output_path: path to the output of mLocARNA program
        :param rnaz_input_path: path to which the input alignment with its structural data will be written to in clustal-w format
        :return:
        """
        with open(mlocarna_output_path, "r") as out_file:
            out_content = out_file.read()
        aligned_data_regex = re.compile(
            "Perform progressive alignment ...\n*(.*?)\n{2,}.*\*(.*)",
            re.MULTILINE | re.DOTALL,
        )
        aligned_data = aligned_data_regex.search(out_content)
        aligned_seq_data = aligned_data.group(1).replace("\s+", "\n").split("\n")
        acc_to_seq = {
            aligned_seq_data[i]: aligned_seq_data[i + 1]
            for i in range(0, len(aligned_seq_data), 2)
        }
        records = []
        for acc in acc_to_seq:
            records.append(SeqRecord(id=acc, seq=acc_to_seq[acc]))
        SeqIO.write(records, rnaz_input_path, format="clustal")

    @staticmethod
    def exec_rnaz(input_path: str, output_path: str) -> int:
        """
        :param input_path:
        :param output_path:
        :return:
        """
        if not os.path.exists(output_path):
            cmd = f"RNAz {input_path} > {output_path}"
            res = os.system(cmd)
            if res != 0 or not os.path.exists(output_path):
                logger.error(
                    f"failed to execute RNAz on {input_path}. For error details, see {output_path}"
                )
                exit(1)
        return 0

    @staticmethod
    def parse_rnaz_output(
        rnaz_output_path, significance_score_cutoff=0.9
    ) -> RNASecondaryStruct:
        """
        :param rnaz_output_path:
        :param significance_score_cutoff:
        :return:
        """
        rnaz_output_regex = re.compile(
            "Mean pairwise identity\:\s*(\d*\.?\d*).*?Shannon entropy\:\s*(-?\d*\.?\d*).*?G\+C content\:\s*(\d*\.?\d*).*?Mean single sequence MFE\:\s*(-?\d*\.?\d*).*?Consensus MFE\:\s*(-?\d*\.?\d*).*?Mean z-score\:\s*(-?\d*\.?\d*).*?Structure conservation index\:\s*(-?\d*\.?\d*).*?SVM RNA-class probability\:\s*(\d*\.?\d*).*?>consensus\n([A-Za-z]*)\n(\D*)\s",
            re.MULTILINE | re.DOTALL,
        )
        with open(rnaz_output_path, "r") as rnaz_output_file:
            rnaz_output = rnaz_output_file.read()
        rnaz_output_match = rnaz_output_regex.search(rnaz_output)
        alignment_path = rnaz_output_path
        consensus_representation = rnaz_output_match.group(10)
        consensus_sequence = rnaz_output_match.group(9)
        mean_pairwise_identity = float(rnaz_output_match.group(1))
        shannon_entropy = float(rnaz_output_match.group(2))
        gc_content = float(rnaz_output_match.group(3))
        mean_single_sequence_mfe = float(rnaz_output_match.group(4))
        consensus_mfe = float(rnaz_output_match.group(5))
        mean_zscore = float(rnaz_output_match.group(6))
        structure_conservation_index = float(rnaz_output_match.group(7))
        svm_rna_probability = float(rnaz_output_match.group(8))
        significant = (
            True if svm_rna_probability >= significance_score_cutoff else False
        )
        structure_instance = RNASecondaryStruct(
            alignment_path=alignment_path,
            consensus_representation=consensus_representation,
            consensus_sequence=consensus_sequence,
            mean_pairwise_identity=mean_pairwise_identity,
            shannon_entropy=shannon_entropy,
            gc_content=gc_content,
            mean_single_sequence_mfe=mean_single_sequence_mfe,
            consensus_mfe=consensus_mfe,
            mean_zscore=mean_zscore,
            structure_conservation_index=structure_conservation_index,
            svm_rna_probability=svm_rna_probability,
            significant=significant,
        )
        return structure_instance
