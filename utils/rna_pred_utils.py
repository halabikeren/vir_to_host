import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
import typing as t
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

logger = logging.getLogger(__name__)

RNAALIFOLD_NUM_SEQ_LIMIT = 3000
RNAALIFOLD_SEQLEN_LIMIT = 30000


@dataclass
class RNASecondaryStruct:
    alignment_path: str
    consensus_representation: str  # ~structure classification - will be used to represent the secondary structure
    consensus_sequence: str
    mean_single_sequence_mfe: float
    consensus_mfe: float
    mean_zscore: float
    is_functional_structure: t.Optional[bool] = None
    is_significant: t.Optional[bool] = None
    mean_pairwise_identity: t.Optional[float] = None
    shannon_entropy: t.Optional[float] = None
    gc_content: t.Optional[float] = None
    structure_conservation_index: t.Optional[float] = None
    svm_rna_probability: t.Optional[float] = None

class RNAPredUtils:

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
                    raise ValueError(
                        f"failed to execute RMALalifold properly on {input_path} due to error. Additional info can be found in {log_path}"
                    )
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
        """
        :param input_path: path to alignment in a fasta format
        :param output_path: structure-guided alignment in a clustal format
        :return:
        """
        output_dir = f"{os.path.dirname(output_path)}/{output_path.split('.')[0]}/"
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_path):
            cmd = f"mlocarna {input_path} --probabilistic --consistency-transform --it-reliable-structure=10 --tgtdir {output_dir} > {output_dir}mlocarna.log"
            # cmd = f"mlocarna {input_path} --tgtdir {output_dir} > {output_dir}mlocarna.log"
            res = os.system(cmd)
            indir_output_path = f"{output_dir}/results/result.aln"
            if res != 0 or not os.path.exists(indir_output_path):
                logger.error(
                    f"failed to execute MLocaRNA properly on {input_path} due to error. Additional info can be found in {output_path}"
                )
                raise ValueError(
                    f"failed to execute MLocaRNA properly on {input_path} due to error. Additional info can be found in {output_path}"
                )
            os.rename(indir_output_path, output_path)
            shutil.rmtree(output_dir)
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
        :param input_path: path to a multiple sequence alignment of a suspected rna secondary structure
        :param output_path: path to rnaz result
        :return: execution code
        """
        if not os.path.exists(output_path):
            cmd = f"RNAz --locarnate --outfile {output_path} {input_path}"
            res = os.system(cmd)
            if res != 0 or not os.path.exists(output_path):
                logger.error(
                    f"failed to execute RNAz on {input_path}. For error details, see {output_path}"
                )
                raise ValueError(
                    f"failed to execute RNAz on {input_path}. For error details, see {output_path}"
                )
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
            "Mean pairwise identity\:\s*(\d*\.?\d*)\n\sShannon entropy\:\s*(-?\d*\.?\d*)\n\sG\+C content\:\s*(\d*\.?\d*)\n\sMean single sequence MFE\:\s*(-?\d*\.?\d*)\n\sConsensus MFE\:\s*(-?\d*\.?\d*).*\n\sMean z-score\:\s*(-?\d*\.?\d*)\n\sStructure conservation index\:\s*(-?\d*\.?\d*).*\n\sSVM RNA-class probability\:\s*(\d*\.?\d*)\n\sPrediction\:\s*(.*?)\n.*>consensus\n([A-Za-z|_]*)\n(\D*)\s",
            re.MULTILINE | re.DOTALL,
        )
        with open(rnaz_output_path, "r") as rnaz_output_file:
            rnaz_output = rnaz_output_file.read()
        rnaz_output_match = rnaz_output_regex.search(rnaz_output)
        alignment_path = rnaz_output_path
        consensus_representation = rnaz_output_match.group(11)
        prediction = rnaz_output_match.group(9)
        consensus_sequence = rnaz_output_match.group(10)
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
            is_functional_structure = False if prediction == "OTHER" else True,
            is_significant=significant,
        )
        return structure_instance

    @staticmethod
    def exec_rnaz_window(
            input_path: str,
            output_path: str,
    ):
        """
        :param input_path: path to alignment in fasta format
        :param output_path: path to hold the sub-alignments of relevant windows, that need to be parsed and given as input to rnaz cluster
        :return:
        """
        if not os.path.exists(output_path):
            # convert alignment to capital letters and clustal format
            input_clustal_path = input_path.replace(".fasta", ".clustal")
            records = list(SeqIO.parse(input_path, format="fasta"))
            for record in records:
                record.seq = Seq(str(record.seq).upper())
            SeqIO.write(records, input_clustal_path, format="clustal")

            # execute rnaz
            cmd = f"rnazWindow.pl {input_clustal_path} --min-seqs=2 --no-reference &> {output_path}"
            res = os.system(cmd)

            # delete the clustal file
            os.remove(input_clustal_path)

        return 0

    @staticmethod
    def exec_rnaz_cluster(input_path: str, output_path: str) -> int:
        """
        :param input_path:
        :param output_path:
        :return:
        """
        if not os.path.exists(output_path):
            cmd = f"rnazCluster.pl --window --header {input_path} > {output_path}"
            res = os.system(cmd)
            return res
        return 0

    @staticmethod
    def parse_candidates(candidates_info_path: str, sequence_data_path: str, output_dir: str, windows_aligned: bool = False):
        """
        :param candidates_info_path: output path of rnazCluster that lists the relevant windows for downstream analysis
        :param sequence_data_path: file with rthe window alignments given by rnazWindow
        :param output_dir: directory holding the candidates sequence data (either aligned in clustal format or unaligned in fasta format)
        :param windows_aligned: boolean indicating weather output windows data should be aligned or not
        :return: none
        """
        # parse windows seq data
        with open(sequence_data_path, "r") as infile:
            windows_content = infile.read()
        delim = "CLUSTAL W(1.81) multiple sequence alignment"
        windows =  [delim+item for item in windows_content.split(delim) if item]
        coordinates_regex = re.compile("\/(\d*)-(\d*)")
        coordinates_to_window = {
            (int(coordinates_regex.search(window).group(1)), int(coordinates_regex.search(window).group(2))): window for
            window in windows}

        # extract relevant windows
        relevant_windows_df = pd.read_csv(candidates_info_path, sep="\t", index_col=False)
        relevant_windows_df["coordinate"] = relevant_windows_df.apply(lambda row: (int(row['start']), int(row['end'])),
                                                                      axis=1)
        relevant_coordinates = list(relevant_windows_df["coordinate"])
        relevant_windows = {coord: coordinates_to_window[coord] for coord in relevant_coordinates}

        # write unaligned windows seq data
        os.makedirs(output_dir, exist_ok=True)
        for window_coord in relevant_windows:
            seq_path = f"{output_dir}{window_coord[0]}_{window_coord[1]}.fasta"
            with open(seq_path, "w") as outfile:
                outfile.write(relevant_windows[window_coord])
            if not windows_aligned:
                records = list(SeqIO.parse(seq_path, format="clustal"))
                for record in records:
                    record.seq = Seq(str(record.seq).replace('-', ''))
                SeqIO.write(records, seq_path, format="fasta")

    @staticmethod
    def exec_rnalfold(input_path: str, output_path: str) -> int:
        """

        :param input_path: path to a fasta file with a single genome sequence
        :param output_path: path to rnalfold result
        :return: execution code
        """
        if not os.path.exists(output_path):
            cmd = f"RNALfold -z-0.001 −−zscore−report−subsumed --infile {input_path} > {output_path}" # use insignificant z-score to assume documentation of all the solutions
            res = os.system(cmd)
            return res
        return 0

    @staticmethod
    def parse_rnalfold_result(rnalfold_path: str, sequence_data_path: str) -> t.List[RNASecondaryStruct]:
        """
        :param rnalfold_path: path to rnalfold result
        :param sequence_data_path: path to a fasta file with the complete sequence
        :return: list of predicted rna secondary structures
        """
        complete_sequence = str(list(SeqIO.parse(sequence_data_path, format="fasta"))[0].seq)
        with open(rnalfold_path, "r") as infile:
            rnalfold_struct_content = infile.readlines()[1:-2]
        struct_regex = re.compile("([\.|\(|\)]*)\s*\((-?\d*\.?\d*)\)\s*(\d*)\s*z=\s*(-?\d*\.?\d*)")
        secondary_structure_instances = []
        for structure in rnalfold_struct_content:
            match = struct_regex.search(structure)
            struct_representation = match.group(1)
            mfe = float(match.group(2))
            start_pos = int(match.group(3))
            zscore = float(match.group(4))
            struct_sequence = complete_sequence[start_pos:start_pos+len(struct_representation)]
            sec_struct_instance = RNASecondaryStruct(alignment_path=sequence_data_path, consensus_representation=struct_representation,
                                                     consensus_sequence=struct_sequence, mean_single_sequence_mfe=mfe, consensus_mfe=mfe,
                                                     mean_zscore=zscore)
            secondary_structure_instances.append(sec_struct_instance)
        return secondary_structure_instances




