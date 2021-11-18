import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

logger = logging.getLogger(__name__)

RNAALIFOLD_NUM_SEQ_LIMIT = 3000
RNAALIFOLD_SEQLEN_LIMIT = 30000

logger = logging.getLogger(__name__)

@dataclass
class RNASecondaryStruct:
    alignment_path: str
    consensus_representation: str  # ~structure classification - will be used to represent the seocndary structure
    consensus_sequence: str
    is_functional_structure: bool
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
                    f"failed to execute RMALalifold properly on {input_path} due to error. Additional info can be found in {output_path}"
                )
                raise ValueError(
                    f"failed to execute RMALalifold properly on {input_path} due to error. Additional info can be found in {output_path}"
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
        :param input_path:
        :param output_path:
        :return:
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
            "Mean pairwise identity\:\s*(\d*\.?\d*).*?Shannon entropy\:\s*(-?\d*\.?\d*).*?G\+C content\:\s*(\d*\.?\d*).*?Mean single sequence MFE\:\s*(-?\d*\.?\d*).*?Consensus MFE\:\s*(-?\d*\.?\d*).*?Mean z-score\:\s*(-?\d*\.?\d*).*?Structure conservation index\:\s*(-?\d*\.?\d*).*?SVM RNA-class probability\:\s*(\d*\.?\d*).*?Prediction\:\s*(.*?)\n.*?>consensus\n([A-Za-z]*)\n(\D*)\s",
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
            is_structure = False if prediction == "OTHER" else True,
            significant=significant,
        )
        return structure_instance

    @staticmethod
    def execute_rnaz_window(
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
            cmd = f"/groups/itay_mayrose/halabikeren/miniconda3/pkgs/rnaz-2.1-h2d50403_2/share/RNAz/perl/rnazWindow.pl {input_clustal_path} --no-rangecheck --min-seqs=2 --no-reference &> {output_path}"
            res = os.system(cmd)

            # delete the clustal file
            os.remove(input_clustal_path)

        return 0

    @staticmethod
    def parse_rnaz_window_output(input_path: str, output_dir: str):
        """
        :param input_path: path with rnaz window output
        :param output_dir: directory that will hold fasta files with the aligned windows
        :return: none
        """
        window_seq_regex = re.compile("([^\n]*)\/(\d*-\d*)\s*([ACTG-]*)\n", re.MULTILINE | re.DOTALL)
        with open(input_path, "r") as input_file:
            input_content = input_file.read()
        window_to_records = defaultdict(dict)
        for match in window_seq_regex.finditer(input_content):
            if match:
                try:
                    accession = match.group(1)
                    window = match.group(2)
                    seq = match.group(3).replace("-", "")
                    record = SeqRecord(id=accession, description="", name="", seq=Seq(seq))
                    if record.id not in window_to_records[window]:
                        window_to_records[window][record.id] = record
                    else:
                        window_to_records[window][record.id].seq = Seq(str(window_to_records[window][record.id].seq) + str(record.seq))
                except Exception as e:
                    logger.error(f"failed to parse match {match.group(0)} into a sequence record due to error {e}")

        os.makedirs(output_dir, exist_ok=True)
        for window in window_to_records:
            output_path = f"{output_dir}{window}.fasta"
            try:
                SeqIO.write(list(window_to_records[window].values()), output_path, format="fasta")
            except Exception as e:
                logger.error(f"invalid window {window} due to error {e}. check output in {input_path}")



if __name__ == '__main__':

    # initialize logger
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # declare input paths
    msa_path = "/groups/itay_mayrose/halabikeren/frog_virus_3_aligned.fasta"
    rnaz_window_output_path = "/groups/itay_mayrose/halabikeren/frog_virus_3_rnaz_window.out"
    rnaz_window_output_dir = "/groups/itay_mayrose/halabikeren/frog_virus_3_rnaz_windows/"
    mlocarna_output_dir = "/groups/itay_mayrose/halabikeren/frog_virus_3_mlocarna/"
    rnaz_output_dir = "/groups/itay_mayrose/halabikeren/frog_virus_3_rnaz/"

    # execute pipeline
    logger.info(f"computing rnaz reliable windows for prediction")
    RNAPredUtils.execute_rnaz_window(input_path=msa_path, output_path=rnaz_window_output_path)
    RNAPredUtils.parse_rnaz_window_output(input_path=rnaz_window_output_path, output_dir=rnaz_window_output_dir)
    logger.info(f"refining reliable windows alignments using mlocarna for {len(os.listdir(rnaz_window_output_dir))} reliable windows")
    for path in os.listdir(rnaz_window_output_dir):
        RNAPredUtils.exec_mlocarna(input_path=f"{rnaz_window_output_dir}{path}", output_path=f"{mlocarna_output_dir}{path.replace('.fasta', '.clustal')}")
        logger.info(f"refinement of {path} is complete")
    logger.info(f"executing rnaz predictor on refined window alignments")
    for path in os.listdir(mlocarna_output_dir):
        RNAPredUtils.exec_rnaz(input_path=f"{mlocarna_output_dir}{path}", output_path=f"{rnaz_output_dir}{path.replace('.clustal', '_rnaz.out')}")
    secondary_structures = []
    logger.info(f"parsing rnaz output")
    for path in os.listdir(rnaz_window_output_dir):
        secondary_structures.append(RNAPredUtils.parse_rnaz_output(rnaz_output_path=f"{rnaz_output_dir}{path}"))
    significant_rna_structures = [struct for struct in secondary_structures if struct.is_significant]
    functional_rna_structures = [struct for struct in significant_rna_structures if struct.is_functional_structure]
    logger.info(f"{len(functional_rna_structures)} out of {len(significant_rna_structures)} significant structures has been annotated as functional")