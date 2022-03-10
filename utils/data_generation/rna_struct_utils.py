import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
import typing as t

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Levenshtein import distance as lev
from PIL import Image
from pandarallel import pandarallel

pandarallel.initialize()

logger = logging.getLogger(__name__)

RNAALIFOLD_NUM_SEQ_LIMIT = 3000
RNAALIFOLD_SEQLEN_LIMIT = 30000


@dataclass
class RNASecondaryStruct:
    alignment_path: str
    consensus_representation: str  # ~structure classification - will be used to represent the secondary structure
    consensus_sequence: str
    start_position: int
    end_position: int
    mean_single_sequence_mfe: float
    consensus_mfe: float
    mean_zscore: float
    structure_prediction_tool: str
    is_functional_structure: t.Optional[bool] = None
    is_significant: t.Optional[bool] = None
    mean_pairwise_identity: t.Optional[float] = None
    shannon_entropy: t.Optional[float] = None
    gc_content: t.Optional[float] = None
    structure_conservation_index: t.Optional[float] = None
    svm_rna_probability: t.Optional[float] = None


class RNAStructPredictionUtils:
    @staticmethod
    def exec_rnalalifold(input_path: str, output_dir: str) -> int:
        """
        :param input_path: directory of a multiple sequence alignment corresponding ot genomic sequences in fasta format
        :param output_dir: directory of the output files of RNAAlifold
        :return: none
        """

        os.makedirs(output_dir, exist_ok=True)
        old_dir = os.getcwd()
        os.chdir(output_dir)
        output_path = f"{output_dir}/RNALalifold_results.stk"
        if not os.path.exists(output_path):
            cmd = f"RNALalifold {input_path} --input-format=F --aln"
            res = os.system(cmd)
            if res != 0 or not os.path.exists(output_path):
                logger.error(f"failed to execute RNALalifold properly on {input_path} due to error\nused cmd={cmd}")
                return 1
        os.chdir(old_dir)
        for path in os.listdir(output_dir):
            if path.endswith(".eps"):
                full_path = f"{output_dir}/{path}"
                if path.startswith("ss"):
                    try:
                        img = Image.open(full_path)
                        img.convert("RGB").save(f"{full_path.replace('.eps', '.jpeg')}")
                    except Exception as e:
                        logger.error(f"failed to convert {full_path} to pdf file due to error {e}")
                os.remove(full_path)
        return 0

    @staticmethod
    def parse_rnalalifold_output(rnalalifold_output_dir: str, mlocarna_input_dir: str):
        """
        :param rnalalifold_output_dir: directory holding the output of RNAAliFold execution
        :param mlocarna_input_dir: directory to hold the input for mLocARNA executions
        :return: none. parses RNAAliFold output and creastes inputs for mlocARNA based on it
        """
        os.makedirs(mlocarna_input_dir, exist_ok=True)
        structure_segment_regex = re.compile("# STOCKHOLM 1.0(.*?)\/\/", re.DOTALL | re.MULTILINE)
        output_path = f"{rnalalifold_output_dir}/RNALalifold_results.stk"
        with open(output_path, "r") as output_file:
            output_content = output_file.read()
        structures_segments = [match.group(1) for match in structure_segment_regex.finditer(output_content)]
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
        output_dir = f"{os.path.dirname(output_path)}/{os.path.basename(output_path).split('.')[0]}/mlocarna_aux/"
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(output_path):
            # cmd = f"mlocarna {input_path} --probabilistic --consistency-transform --it-reliable-structure=10 --tgtdir {output_dir} > {output_dir}mlocarna.log"
            cmd = f"mlocarna {input_path} --it-reliable-structure=10 --tgtdir {output_dir} > {output_dir}mlocarna.log"
            res = os.system(cmd)
            indir_output_path = f"{output_dir}/results/result.aln"
            if res != 0 or not os.path.exists(indir_output_path):
                logger.error(
                    f"failed to execute MLocaRNA properly on {input_path} due to error. Additional info can be found in {output_path}"
                )
                return 1
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
            "Perform progressive alignment ...\n*(.*?)\n{2,}.*\*(.*)", re.MULTILINE | re.DOTALL,
        )
        aligned_data = aligned_data_regex.search(out_content)
        aligned_seq_data = aligned_data.group(1).replace("\s+", "\n").split("\n")
        acc_to_seq = {aligned_seq_data[i]: aligned_seq_data[i + 1] for i in range(0, len(aligned_seq_data), 2)}
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
            cmd = f"RNAz --locarnate --both-strands --outfile {output_path} {input_path}"
            res = os.system(cmd)
            if res != 0 or not os.path.exists(output_path):
                logger.error(f"failed to execute RNAz on {input_path}. For error details, see {output_path}")
                return 1
        return 0

    @staticmethod
    def parse_rnaz_output(rnaz_output_path, significance_score_cutoff=0.9) -> RNASecondaryStruct:
        """
        :param rnaz_output_path:
        :param significance_score_cutoff:
        :return:
        """
        rnaz_output_regex = re.compile(
            "Mean pairwise identity\:\s*(\d*\.?\d*)\n\sShannon entropy\:\s*(-?\d*\.?\d*)\n\sG\+C content\:\s*(\d*\.?\d*)\n\sMean single sequence MFE\:\s*(-?\d*\.?\d*)\n\sConsensus MFE\:\s*(-?\d*\.?\d*).*\n\sMean z-score\:\s*(-?\d*\.?\d*)\n\sStructure conservation index\:\s*(-?\d*\.?\d*).*\n\sSVM RNA-class probability\:\s*(\d*\.?\d*)\n\sPrediction\:\s*(.*?)\n.*>.*?\/(\d*)\-(\d*)\n.*>consensus\n([A-Za-z|_]*)\n(\D*)\s",
            re.MULTILINE | re.DOTALL,
        )
        with open(rnaz_output_path, "r") as rnaz_output_file:
            rnaz_output = rnaz_output_file.read()
        rnaz_output_match = rnaz_output_regex.search(rnaz_output)
        alignment_path = rnaz_output_path
        start_position = int(rnaz_output_match.group(10))
        end_position = int(rnaz_output_match.group(11))
        consensus_representation = rnaz_output_match.group(13)
        prediction = rnaz_output_match.group(9)
        consensus_sequence = rnaz_output_match.group(12)
        mean_pairwise_identity = float(rnaz_output_match.group(1))
        shannon_entropy = float(rnaz_output_match.group(2))
        gc_content = float(rnaz_output_match.group(3))
        mean_single_sequence_mfe = float(rnaz_output_match.group(4))
        consensus_mfe = float(rnaz_output_match.group(5))
        mean_zscore = float(rnaz_output_match.group(6))
        structure_conservation_index = float(rnaz_output_match.group(7))
        svm_rna_probability = float(rnaz_output_match.group(8))
        significant = True if svm_rna_probability >= significance_score_cutoff else False
        structure_instance = RNASecondaryStruct(
            alignment_path=alignment_path,
            consensus_representation=consensus_representation,
            consensus_sequence=consensus_sequence,
            start_position=start_position,
            end_position=end_position,
            mean_pairwise_identity=mean_pairwise_identity,
            shannon_entropy=shannon_entropy,
            gc_content=gc_content,
            mean_single_sequence_mfe=mean_single_sequence_mfe,
            consensus_mfe=consensus_mfe,
            mean_zscore=mean_zscore,
            structure_prediction_tool="rnaz",
            structure_conservation_index=structure_conservation_index,
            svm_rna_probability=svm_rna_probability,
            is_functional_structure=False if prediction == "OTHER" else True,
            is_significant=significant,
        )
        return structure_instance

    @staticmethod
    def exec_rnaz_window(
        input_path: str, output_path: str,
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

            # correct output file, if needed
            with open(output_path, "r") as infile:
                result = infile.read()
            result = re.sub("substr outside of string at.*?\n", "", result)
            result = re.sub("Use of uninitialized value.*?\n", "", result)
            with open(output_path, "w") as outfile:
                outfile.write(result)

            # delete the clustal file
            os.remove(input_clustal_path)

            if os.stat(output_path).st_size == 0:
                return 1

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
    def parse_candidates(candidates_info_path: str, sequence_data_path: str, output_dir: str):
        """
        :param candidates_info_path: output path of rnazCluster that lists the relevant windows for downstream analysis
        :param sequence_data_path: file with rthe window alignments given by rnazWindow
        :param output_dir: directory holding the candidates sequence data (either aligned in clustal format or unaligned in fasta format)
        :return: none
        """
        # parse windows seq data
        with open(sequence_data_path, "r") as infile:
            windows_content = infile.read()
        delim = "CLUSTAL W(1.81) multiple sequence alignment"
        windows = [delim + item for item in windows_content.split(delim) if item]
        coordinates_regex = re.compile("\/(\d*)-(\d*)")
        coordinates_to_window = dict()
        for window in windows:
            match = coordinates_regex.search(window)
            if match is not None:
                coordinates_to_window[(int(match.group(1)), int(match.group(2)))] = window

        # extract relevant windows
        relevant_windows_df = pd.read_csv(candidates_info_path, sep="\t", index_col=False)
        relevant_windows_df[
            "window_seq_path"
        ] = f"{sequence_data_path}/{relevant_windows_df.start}_{relevant_windows_df.end}.fasta"
        clustered_relevant_windows_df = relevant_windows_df.groupby("clusterID")
        relevant_windows = {}
        if relevant_windows_df.shape[0] > 0:
            # write unaligned windows seq data
            os.makedirs(output_dir, exist_ok=True)

            for cluster in clustered_relevant_windows_df.groups.keys():
                cluster_windows_data = clustered_relevant_windows_df.get_group(cluster)
                cluster_start, cluster_end = cluster_windows_data.start.min(), cluster_windows_data.end.max()
                windows_paths = cluster_windows_data.window_seq_path.values
                seq_path = f"{output_dir}{cluster_start}_{cluster_end}.fasta"
                records = list(SeqIO.parse(windows_paths[0], format="fasta"))
                record_acc_regex = re.compile("(.*?)/(\d*)-(\d*)")
                record_acc_to_record = dict()
                for record in records:
                    acc = record_acc_regex.search(record.id).group(1)
                    record.id = record.name = record.description = f"{acc}/{cluster_start}_{cluster_end}"
                    record.seq = Seq(str(record.seq).replace("-", ""))
                    record_acc_to_record[acc] = record
                for window_path in windows_paths[1:]:
                    window_records = list(SeqIO.parse(window_path), format="fasta")
                    window_accs_to_records = {
                        record_acc_regex.search(record.id).group(1): record for record in window_records
                    }
                    for acc in record_acc_to_record:
                        if acc in window_accs_to_records:
                            record_acc_to_record[acc].seq = Seq(
                                str(record_acc_to_record[acc].seq)
                                + str(window_accs_to_records[acc].seq).replace("-", "")
                            )
                concatenated_records = list(record_acc_to_record.values())
                SeqIO.write(concatenated_records, seq_path, format="fasta")
                SeqIO.write(records, seq_path, format="fasta")

    @staticmethod
    def exec_rnalfold(input_path: str, output_path: str) -> int:
        """

        :param input_path: path to a fasta file with a single genome sequence
        :param output_path: path to rnalfold result
        :return: execution code
        """
        if not os.path.exists(output_path):
            cmd = f"RNALfold -z-0.001 −−zscore−report−subsumed --infile {input_path} > {output_path}"  # use insignificant z-score to assume documentation of all the solutions
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
        struct_regex = re.compile("([\.|\(|\)]*)\s*\(\s*(-?\d*\.?\d*)\)\s*(\d*)\s*z=\s*(-?\d*\.?\d*)")
        secondary_structure_instances = []
        for structure in rnalfold_struct_content:
            match = struct_regex.search(structure)
            struct_representation = match.group(1)
            mfe = float(match.group(2))
            start_pos = int(match.group(3))
            zscore = float(match.group(4))
            struct_sequence = complete_sequence[start_pos : start_pos + len(struct_representation)]
            end_pos = start_pos + len(struct_sequence)
            sec_struct_instance = RNASecondaryStruct(
                alignment_path=sequence_data_path,
                consensus_representation=struct_representation,
                consensus_sequence=struct_sequence,
                start_position=start_pos,
                end_position=end_pos,
                mean_single_sequence_mfe=mfe,
                consensus_mfe=mfe,
                mean_zscore=zscore,
                structure_prediction_tool="rnalfold",
            )
            secondary_structure_instances.append(sec_struct_instance)
        return secondary_structure_instances

    @staticmethod
    def exec_rnaplot(input_sequence: str, input_structure: str, output_path: str) -> int:
        """
        :param input_sequence: nucleotide sequence of a structure
        :param input_structure: dot bracket representation of the structure
        :param output_path: path to write the plot to, in svg format
        :return:
        """
        input_str = f"{input_sequence}\n{input_structure}\n"
        cmd = f'(printf "{input_str}") | RNAplot --output-format=svg'
        output_dir = os.path.dirname(output_path)
        original_dir = os.getcwd()
        os.chdir(output_dir)
        res = os.system(cmd)
        if os.path.exists(f"{output_dir}/rna.svg"):
            os.rename(f"{output_dir}/rna.svg", output_path)
        os.chdir(original_dir)
        return res

    @staticmethod
    def exec_rnadistance(
        ref_struct: str,
        ref_struct_index: int,
        structs_path: str,
        workdir: str,
        alignment_path: str,
        output_path: str,
        batch_size: int = 800,
    ) -> int:
        """
        :param ref_struct: the dot bracket structure representation of the reference structure, to which all distances from other structures should be computed
        :param ref_struct_index: index from which computation should begin to avoid duplicate computations
        :param structs_path: path to a fasta file with dot bracket structures representations of structures to compute their distance from the reference structure
        :param workdir: directory to hold partial outputs in
        :param alignment_path: path to which the structures alignment should be written
        :param output_path: path to which the distances between structures should be written
        :param batch_size: number of structures to run rnadistance against in each batch
        :return: result code
        """
        struct_regex = re.compile(">(.*?)\n([\.|\(|\)]*)")
        with open(structs_path, "r") as infile:
            other_structs = [match.group(2) for match in struct_regex.finditer(infile.read())][ref_struct_index + 1 :]
        other_structs_batches = [other_structs[i : i + batch_size] for i in range(0, len(other_structs), batch_size)]
        logger.info(f"will execute RNADistance on {len(other_structs_batches)} batches of size {batch_size}")

        alignment_paths = []
        output_paths = []
        os.makedirs(workdir, exist_ok=True)
        curr_dir = os.getcwd()
        os.chdir(workdir)
        for i in range(len(other_structs_batches)):
            other_structs_batch = other_structs_batches[i]
            temporary_alignment_path = f"./batch_{i}_{os.path.basename(alignment_path)}"
            alignment_paths.append(temporary_alignment_path)
            temporary_output_path = f"./batch_{i}_{os.path.basename(output_path)}"
            output_paths.append(temporary_output_path)
            if not os.path.exists(temporary_alignment_path) or not os.path.exists(temporary_output_path):
                other_structs_str = "\\n".join(other_structs_batch)
                input_str = f"\\n{ref_struct}\\n{other_structs_str}\\n@\\n"
                cmd = f'(printf "{input_str}") | RNAdistance --backtrack={temporary_alignment_path} -Xf --distance=FHWCP > {temporary_output_path}'
                res = os.system(cmd)
                if res != 0:
                    logger.error(
                        f"error upon executing commands for reference structure {ref_struct} against {structs_path} wirth batch number {i} for structures {i*batch_size}-{i*batch_size+batch_size}. code = {res}"
                    )

        # concat all the sub-outputs to a single output
        complete_alignment = ""
        for partial_alignment_path in alignment_paths:
            with open(partial_alignment_path, "r") as infile:
                complete_alignment += infile.read()
        with open(alignment_path, "w") as outfile:
            outfile.write(complete_alignment)

        complete_output = ""
        for partial_output_path in output_paths:
            with open(partial_output_path, "r") as infile:
                complete_output += infile.read()
        with open(output_path, "w") as outfile:
            outfile.write(complete_output)
        os.chdir(curr_dir)
        shutil.rmtree(workdir, ignore_errors=True)

        return 0

    @staticmethod
    def exec_rnadistance_all_vs_all(structs_path: str, workdir: str, alignment_path: str, output_path: str) -> int:
        """
        :param structs_path: path to a fasta file with dot bracket structures representations of structures to compute their distance from the reference structure
        :param workdir: directory to hold partial outputs in
        :param alignment_path: path to which the structures alignment should be written
        :param output_path: path to which the distances between structures should be written
        :return: result code
        """
        struct_regex = re.compile(">(.*?)\n([\.|\(|\)]*)")
        with open(structs_path, "r") as infile:
            structs = [match.group(2) for match in struct_regex.finditer(infile.read())]
        logger.info(f"will execute RNADistance on {len(structs)} structures")

        os.makedirs(workdir, exist_ok=True)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(alignment_path), exist_ok=True)
        if not os.path.exists(alignment_path) or not os.path.exists(output_path):
            structs_str = "\\n".join(structs)
            input_str = f"\\n{structs_str}\\n@\\n"
            cmd = f'(printf "{input_str}") | RNAdistance --backtrack={alignment_path} −Xm --distance=f > {output_path}'
            res = os.system(cmd)
            if res != 0:
                logger.error(f"error upon executing commands for structures. code = {res}")
                return res

        if workdir != os.path.dirname(output_path) and workdir != os.path.dirname(alignment_path):
            shutil.rmtree(workdir, ignore_errors=True)

        return 0

    @staticmethod
    def parse_rnadistance_result(rnadistance_path: str, struct_alignment_path: str) -> t.Dict[str, t.List[float]]:
        """
        :param rnadistance_path: path to RNAdistance result over two structures, with the distances according to different metrics
        :param struct_alignment_path: path to the pairwise alignment between the two structures
        :return: the distance between the two structures, based on all the measures at the same time
        """
        distance_regex = re.compile("([F|H|W|C|P|f|h|w|c|p])\:\s(\d*\.?\d*)")
        with open(rnadistance_path, "r") as outfile:
            rnadistance_result = outfile.readlines()
        distances_to_rest = defaultdict(list)
        for i in range(len(rnadistance_result)):
            result = rnadistance_result[i]
            for match in distance_regex.finditer(result):
                dist_type = match.group(1)
                dist_value = float(match.group(2))
                distances_to_rest[dist_type].append(dist_value)
        with open(struct_alignment_path, "r") as infile:
            alignment_content = infile.read().split("\n\n")
            alignments = alignment_content[
                0:-1:4
            ]  # get only the first representation corresponding to coarse grained approach (https://link.springer.com/content/pdf/10.1007/BF00818163.pdf)
        for i in range(len(alignments)):
            aligned_sequences = alignments[i].split("\n")
            if len(aligned_sequences) > 2:  # in case of additional newline
                aligned_sequences = aligned_sequences[1:]
            # the normalized lev distance (by computation lev / len(aln) doesn't hold the triangle inequality: https://stackoverflow.com/questions/18910524/levenshtein-distance-and-triangle-inequality
            unaligned_seq1 = aligned_sequences[0].replace("\n", "").replace("_", "")
            unaligned_seq2 = aligned_sequences[1].replace("\n", "").replace("_", "")
            lev_dist = lev(unaligned_seq1, unaligned_seq2)
            alpha = 1  # the lev distance computed as per function lev of python penalize any edit by 1
            normalized_lev_dist = 1 - (2 * lev_dist) / (
                alpha * (len(unaligned_seq1) + len(unaligned_seq2)) + lev_dist
            )  # follows from https://ieeexplore.ieee.org/abstract/document/4160958?casa_token=dljP-khqCpYAAAAA:H7qszwA4oja-tYLAwYOO0z77j4Jerk5PHk6Ph2hwFNxlkDjiDl_qyygoEheRTa2XjXwoi__UTw, definition 3
            # the lev distance holds the triangle inequality only for unaligned structures, otherwise a structure may have more than one "aligned" representation depending on with whom it is pairwise aligned...
            distances_to_rest["edit_distance"].append(lev_dist)
        return distances_to_rest

    @staticmethod
    def infer_structural_regions(alignment_path: str, workdir: str) -> str:
        """
        :param alignment_path: path ot alignment file
        :param workdir: directory to apply the pipeline in
        :return: directory of suspected structural regions
        """
        logger.info(f"computing rnaz reliable windows for prediction")
        rnaz_window_output_path = f"{workdir}/rnaz_window.out"
        RNAStructPredictionUtils.exec_rnaz_window(input_path=alignment_path, output_path=rnaz_window_output_path)
        if os.stat(rnaz_window_output_path).st_size == 0:
            logger.info(
                f"not reliable alignment windows for structural region inference were found in {alignment_path}"
            )
            return None
        logger.info(f"executing RNAz predictor on initial window {rnaz_window_output_path}")
        rnaz_output_path = f"{workdir}/rnaz_initial.out"
        res = RNAStructPredictionUtils.exec_rnaz(input_path=rnaz_window_output_path, output_path=rnaz_output_path)
        if res != 0:
            error_msg = f"failed rnaz execution on suspected structural window"
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.info(f"clustering RNAz hits of overlapping windows")
        rnaz_cluster_output_path = f"{workdir}/rnaz_cluster.dat"
        res = RNAStructPredictionUtils.exec_rnaz_cluster(
            input_path=rnaz_output_path, output_path=rnaz_cluster_output_path
        )
        if res != 0:
            error_msg = f"failed rnaz clustering execution on candidate {rnaz_output_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            # consult with itay about this part
            logger.info(f"clustering windows of similar structures by concatenating them into a longer alignment")
            logger.info(f"extracting sequence data per selected window for mlocarna refinement")
            rnaz_candidates_output_dir = f"{workdir}/rnaz_candidates_sequence_data/"
            RNAStructPredictionUtils.parse_candidates(
                candidates_info_path=rnaz_cluster_output_path,
                sequence_data_path=rnaz_window_output_path,
                output_dir=rnaz_candidates_output_dir,
            )
            logger.info(f"creating refined alignments of candidates with mlocarna")
            mlocarna_output_dir = f"{workdir}/rnaz_candidates_mlocarna_aligned/"
            os.makedirs(mlocarna_output_dir, exist_ok=True)
            for path in os.listdir(rnaz_candidates_output_dir):
                input_path = f"{rnaz_candidates_output_dir}{path}"
                output_path = f"{mlocarna_output_dir}{path.replace('.fasta', '.clustal')}"
                res = RNAStructPredictionUtils.exec_mlocarna(input_path=input_path, output_path=output_path)
                if res != 0:
                    error_msg = f"failed mlocarna execution on candidate region {input_path}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            for path in os.listdir(mlocarna_output_dir):
                if os.path.isdir(f"{mlocarna_output_dir}{path}"):
                    shutil.rmtree(f"{mlocarna_output_dir}{path}")
            return rnaz_candidates_output_dir
