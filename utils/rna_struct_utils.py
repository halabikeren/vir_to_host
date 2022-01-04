import logging
import os
import pickle
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

import utils

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

class RNAStructUtils:

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
                        f"failed to execute RNALalifold properly on {input_path} due to error. Additional info can be found in {log_path}"
                    )
                    return 1
            # remove redundant output files
            for path in os.listdir(exec_output_dir):
                if f"{exec_output_dir}{path}" != output_path:
                    os.remove(f"{output_dir}{path}")

        return 0

    @staticmethod
    def parse_rnalalifold_output(rnalalifold_output_dir: str, mlocarna_input_dir: str):
        """
        :param rnalalifold_output_dir: directory holding the output of RNAAliFold execution
        :param mlocarna_input_dir: directory to hold the input for mLocARNA executions
        :return: none. parses RNAAliFold output and creastes inputs for mlocARNA based on it
        """
        os.makedirs(mlocarna_input_dir, exist_ok=True)
        structure_segment_regex = re.compile(
            "# STOCKHOLM 1.0(.*?)\/\/", re.DOTALL | re.MULTILINE
        )
        output_path = f"{rnalalifold_output_dir}/RNALalifold_results.stk"
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
                return 1
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
        significant = (
            True if svm_rna_probability >= significance_score_cutoff else False
        )
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
        coordinates_to_window = dict()
        for window in windows:
            match = coordinates_regex.search(window)
            if match is not None:
                coordinates_to_window[(int(match.group(1)), int(match.group(2)))]= window

        # extract relevant windows
        relevant_windows_df = pd.read_csv(candidates_info_path, sep="\t", index_col=False)
        relevant_windows = {}
        if relevant_windows_df.shape[0] > 0:
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
        struct_regex = re.compile("([\.|\(|\)]*)\s*\(\s*(-?\d*\.?\d*)\)\s*(\d*)\s*z=\s*(-?\d*\.?\d*)")
        secondary_structure_instances = []
        for structure in rnalfold_struct_content:
            match = struct_regex.search(structure)
            struct_representation = match.group(1)
            mfe = float(match.group(2))
            start_pos = int(match.group(3))
            zscore = float(match.group(4))
            struct_sequence = complete_sequence[start_pos:start_pos+len(struct_representation)]
            end_pos = start_pos + len(struct_sequence)
            sec_struct_instance = RNASecondaryStruct(alignment_path=sequence_data_path, consensus_representation=struct_representation,
                                                     consensus_sequence=struct_sequence, start_position=start_pos, end_position=end_pos, mean_single_sequence_mfe=mfe, consensus_mfe=mfe,
                                                     mean_zscore=zscore, structure_prediction_tool="rnalfold")
            secondary_structure_instances.append(sec_struct_instance)
        return secondary_structure_instances

    @staticmethod
    def exec_rnadistance(ref_struct:str, ref_struct_index: int, structs_path: str, workdir: str, alignment_path: str, output_path: str, batch_size: int = 800) -> int:
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
            other_structs = [match.group(2) for match in struct_regex.finditer(infile.read())][ref_struct_index+1:]
        other_structs_batches = [other_structs[i:i+batch_size] for i in range(0, len(other_structs), batch_size)]
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
                cmd = f'(printf "{input_str}") | RNAdistance --backtrack={temporary_alignment_path} --shapiro -Xf --distance=FHWCP > {temporary_output_path}'
                res = os.system(cmd)
                if res != 0:
                    logger.error(f"error upon executing commands for reference structure {ref_struct} against {structs_path} wirth batch number {i} for structures {i*batch_size}-{i*batch_size+batch_size}. code = {res}")

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
    def parse_rnadistance_result(rnadistance_path: str, struct_alignment_path: str) -> t.Dict[str, t.List[float]]:
        """
        :param rnadistance_path: path to RNAdistance result over two structures, with the distances according to different metrics
        :param struct_alignment_path: path to the pairwise alignment between the two structures
        :return: the distance between the two structures, based on all the measures at the same time
        """
        distance_regex = re.compile("([F|H|W|C|P])\:\s(\d*\.?\d*)")
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
            alignments = infile.read().split("\n\n")[0:-1:4] # get only the first representation corresponding to coarse grained approach (https://link.springer.com/content/pdf/10.1007/BF00818163.pdf)
        for i in range(len(alignments)):
            aligned_sequences = alignments[i].split("\n")
            if len(aligned_sequences) > 2: # in case of additional newline
                aligned_sequences = aligned_sequences[1:]
            distances_to_rest["edit_distance"].append(lev(aligned_sequences[0], aligned_sequences[1]) / float(len(aligned_sequences[0])))
        return distances_to_rest

    @staticmethod
    def create_group_wise_alignment(df: pd.DataFrame, seq_data_dir: str, group_wise_seq_path: str,
                                    group_wise_msa_path: str, representative_acc_to_sp_path: str) -> pd.DataFrame:
        """
        :param df: dataframe of species to align
        :param seq_data_dir: directory of species sequence data
        :param group_wise_seq_path: path to write the representative sequences of the species within the group
        :param group_wise_msa_path: path to write the aligned representative sequences of the species within the group
        :param representative_acc_to_sp_path: path to pickled dictionary mapping accessions to the species represented by them
        :return: None
        """
        representative_id_to_sp = dict()
        representative_records = []
        logger.info(f"selecting representative per virus species")
        df["accession"] = np.nan
        if not os.path.exists(group_wise_seq_path) or not os.path.exists(representative_acc_to_sp_path):
            species = df.virus_species_name.dropna().unique()
            for sp in species:
                sp_filename = re.sub('[^0-9a-zA-Z]+', '_', sp)
                representative_record = utils.ClusteringUtils.get_representative_by_msa(sequence_df=None,
                                                                                  unaligned_seq_data_path=f"{seq_data_dir}{sp_filename}.fasta",
                                                                                  aligned_seq_data_path=f"{seq_data_dir}{sp_filename}_aligned.fasta",
                                                                                  similarities_data_path=f"{seq_data_dir}{sp_filename}_similarity_values.csv")
                if pd.notna(representative_record):
                    df.loc[df.virus_species_name == sp, 'accession'] = representative_record.id
                    representative_id_to_sp[representative_record.id] = sp
                    representative_records.append(representative_record)
            unique_representative_records = []
            for record in representative_records:
                if record.id not in [item.id for item in unique_representative_records]:
                    unique_representative_records.append(record)
            if len(unique_representative_records) < 3:
                logger.error(f"insufficient number of sequences found - {len(unique_representative_records)}")
                return df
            with open(representative_acc_to_sp_path, "wb") as outfile:
                pickle.dump(representative_id_to_sp, file=outfile)
            SeqIO.write(unique_representative_records, group_wise_seq_path, format="fasta")

        else:
            # parse accession to species data and write it to the dataframe
            with open(representative_acc_to_sp_path, "rb") as infile:
                representative_id_to_sp = pickle.load(infile)
            sp_to_representative_id = {representative_id_to_sp[record_id]: record_id for record_id in
                                       representative_id_to_sp}
            df.set_index("virus_species_name", inplace=True)
            df["accession"].fillna(value=sp_to_representative_id, inplace=True)
            df.reset_index(inplace=True)

        # align written sequence data
        if not os.path.exists(group_wise_msa_path):
            logger.info(f"creating alignment from {group_wise_msa_path} at {group_wise_seq_path}")
            res = utils.ClusteringUtils.exec_mafft(input_path=group_wise_seq_path, output_path=group_wise_msa_path)
            if res != 0:
                exit(1)

        return df

    @staticmethod
    def get_unaligned_pos(aligned_pos: int, aligned_seq: Seq) -> int:
        unaligned_pos = aligned_pos - str(aligned_seq)[:aligned_pos].count("-")
        return unaligned_pos

    @staticmethod
    def get_aligned_pos(unaligned_pos: int, aligned_seq: Seq) -> int:
        num_gaps = 0
        for pos in range(len(aligned_seq)):
            if str(aligned_seq)[pos] == "-":
                num_gaps += 1
            respective_unaligned_pos = pos - num_gaps
            if respective_unaligned_pos == unaligned_pos:
                return pos
        return len(aligned_seq)

    @staticmethod
    def get_group_wise_positions(species_wise_start_pos: int, species_wise_end_pos: int,
                                 group_wise_msa_records: t.List[SeqRecord], species_wise_msa_records: t.List[SeqRecord],
                                 species_accession: str) -> t.Tuple[int, ...]:
        """
        :param species_wise_start_pos: start position in the species wise alignment
        :param species_wise_end_pos: end position in the species wise alignment
        :param group_wise_msa_records:
        :param species_wise_msa_records:
        :param species_accession:
        :return:
        """
        seq_from_group_wise_msa = [record for record in group_wise_msa_records if record.id == species_accession][0].seq
        seq_from_species_wise_msa = [record for record in species_wise_msa_records if record.id == species_accession][
            0].seq
        if str(seq_from_group_wise_msa).replace("-", "").lower() != str(seq_from_species_wise_msa).replace("-",
                                                                                                           "").lower():
            error_msg = f"sequence {species_accession} is inconsistent across the group-wise msa and species-wise msa"
            logger.error(error_msg)
            raise ValueError(error_msg)

        unaligned_start_pos = RNAStructUtils.get_unaligned_pos(aligned_pos=species_wise_start_pos,
                                                aligned_seq=seq_from_species_wise_msa)
        unaligned_end_pos = RNAStructUtils.get_unaligned_pos(aligned_pos=species_wise_end_pos, aligned_seq=seq_from_species_wise_msa)

        group_wise_start_pos = RNAStructUtils.get_aligned_pos(unaligned_pos=unaligned_start_pos, aligned_seq=seq_from_group_wise_msa)
        group_wise_end_pos = RNAStructUtils.get_aligned_pos(unaligned_pos=unaligned_end_pos, aligned_seq=seq_from_group_wise_msa)

        return tuple([unaligned_start_pos, group_wise_end_pos, group_wise_start_pos, group_wise_end_pos])

    @staticmethod
    def map_species_wise_pos_to_group_wise_pos(df: pd.DataFrame, seq_data_dir: str, species_wise_msa_dir: str,
                                               workdir: str) -> pd.DataFrame:
        """
        :param df: dataframe with secondary structures whose positions should be mapped to family-wise positions: struct_start_pos -> family_wise_struct_start_pos, struct_end_pos -> family_wise_struct_end_pos
        :param seq_data_dir: directory of species-wise genome alignments
        :param species_wise_msa_dir: directory to the species wise multiple sequence alignments
        :param workdir: directory to write files to
        :return: dataframe with the mapped structures ranges
        """

        # select a representative sequence per species, and write representatives to a fasta file
        os.makedirs(workdir, exist_ok=True)
        group_wise_seq_path = f"{workdir}/unaligned.fasta"
        group_wise_msa_path = f"{workdir}/aligned.fasta"
        representative_acc_to_sp_path = f"{workdir}/acc_to_species.pickle"
        intermediate_df_path = f"{workdir}/intermediate_structures_df.csv"

        if not os.path.exists(intermediate_df_path) or not os.path.exists(
                representative_acc_to_sp_path) or not os.path.exists(group_wise_msa_path):
            df = RNAStructUtils.create_group_wise_alignment(df=df,
                                             seq_data_dir=seq_data_dir,
                                             group_wise_seq_path=group_wise_seq_path,
                                             group_wise_msa_path=group_wise_msa_path,
                                             representative_acc_to_sp_path=representative_acc_to_sp_path)
            df.to_csv(intermediate_df_path)
        else:
            df = pd.read_csv(intermediate_df_path)
        with open(representative_acc_to_sp_path, "rb") as map_file:
            representative_acc_to_sp = pickle.load(map_file)
        sp_to_acc = {representative_acc_to_sp[acc]: acc for acc in representative_acc_to_sp}
        group_wise_msa_records = list(SeqIO.parse(group_wise_msa_path, format="fasta"))

        # for each species, map the species-wise start and end positions ot group wise start and end positions
        cols_to_add = ["unaligned_struct_start_pos",
                       "unaligned_struct_end_pos",
                       "group_wise_struct_start_pos",
                       "group_wise_struct_end_pos"]
        if not np.all([col in df.columns for col in cols_to_add]):
            df.rename(columns={"struct_start_pos": "species_wise_struct_start_pos",
                               "struct_end_pos": "species_wise_struct_end_pos"}, inplace=True)
            df = df.loc[(df.species_wise_struct_start_pos.notna()) & (df.species_wise_struct_end_pos.notna())]
            df[cols_to_add] = df[["virus_species_name",
                                  "species_wise_struct_start_pos",
                                  "species_wise_struct_end_pos"]].parallel_apply(
                lambda row: RNAStructUtils.get_group_wise_positions(species_wise_start_pos=int(row.species_wise_struct_start_pos),
                                                     species_wise_end_pos=int(row.species_wise_struct_end_pos),
                                                     group_wise_msa_records=group_wise_msa_records,
                                                     species_wise_msa_records=list(SeqIO.parse(
                                                         f"{species_wise_msa_dir}/{re.sub('[^0-9a-zA-Z]+', '_', row.virus_species_name)}_aligned.fasta",
                                                         format="fasta")),
                                                     species_accession=sp_to_acc[row.virus_species_name]),
                axis=1, result_type="expand")
            df.to_csv(intermediate_df_path)
        return df

    @staticmethod
    def assign_partition_by_size(df: pd.DataFrame, partition_size: int) -> pd.DataFrame:
        """
        :param df: dataframe of secondary structures
        :param partition_size: size of required partitions
        :return: dataframe with the assigned partitions of each secondary structure
        """
        max_group_wise_pos = np.max(df.group_wise_struct_end_pos)
        if partition_size > max_group_wise_pos:
            error_msg = f"partition size {partition_size} is larger than the total number of positions {max_group_wise_pos}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        max_struct_size = np.max(df.group_wise_struct_end_pos - df.group_wise_struct_start_pos)
        if partition_size < max_struct_size:
            logger.warning(
                f"selected partition size {partition_size} is smaller than the maximal structure size {max_struct_size} and thus will be changed to {max_struct_size}")
            partition_size = max_struct_size

        partitions = [(i, i + partition_size) for i in range(0, max_group_wise_pos, partition_size)]

        def get_assigned_partitions(start_pos: int, end_pos: int, used_partitions: t.List[t.Tuple[int, int]]) -> str:
            partition_of_start_pos = [partition for partition in used_partitions if
                                      partition[0] <= start_pos <= partition[1]][0]
            partition_of_end_pos = [partition for partition in used_partitions if
                                    partition[0] <= end_pos <= partition[1]][0]
            assigned_partitions = list({str(partition_of_start_pos), str(partition_of_end_pos)})
            return ";".join(assigned_partitions).replace(",", "-").replace(" ", "")

        df["assigned_partition"] = df[["group_wise_struct_start_pos", "group_wise_struct_end_pos"]].parallel_apply(
            lambda row: get_assigned_partitions(start_pos=row.group_wise_struct_start_pos,
                                                end_pos=row.group_wise_struct_end_pos,
                                                used_partitions=partitions),
            axis=1)

        return df

    @staticmethod
    def get_assigned_annotations(struct_start_pos: int, struct_end_pos: int,
                                 accession_annotations: t.Dict[t.Tuple[str, str], t.Tuple[int, int]],
                                 intersection_annotations: t.List[t.Tuple[str, str]]) -> t.Tuple[str, str]:
        """
        :param struct_start_pos: start position of the structure
        :param struct_end_pos: end position of the structure
        :param accession_annotations: annotations that belong to the accession
        :param intersection_annotations: annotations that appear both in the structure's accession and in ones of other species
        :return: the annotations that the structure belongs ot in the accession (accession specific and intersection ones)
        """
        accession_assigned_annotations = []
        for annotation in accession_annotations:
            annotation_start_pos = accession_annotations[annotation][0][0]
            annotation_end_pos = accession_annotations[annotation][-1][-1]
            if struct_start_pos >= annotation_start_pos and struct_end_pos <= annotation_end_pos:
                accession_assigned_annotations.append(annotation)

        assigned_annotations = []
        for annotation in accession_assigned_annotations:
            if annotation in intersection_annotations:
                assigned_annotations.append(annotation)

        accession_assigned_annotations = str(accession_assigned_annotations).replace("), (", "); (").replace(",",
                                                                                                             "-").replace(
            " ", "")
        assigned_annotations = str(assigned_annotations).replace("), (", "); (").replace(",", "-").replace(" ", "")
        return (accession_assigned_annotations, assigned_annotations)

    @staticmethod
    def assign_partition_by_annotation(df: pd.DataFrame) -> pd.DataFrame:
        """
        :param df: dataframe of secondary structures to partition by annotations
        :return: dataframe with the assigned partition of each structure
        """

        accessions = list(df.accession.dropna().unique())
        accession_to_annotations = utils.SequenceCollectingUtils.get_annotations(accessions=accessions)
        intersection_annotations = []  # maybe switch to relaxed annotation of: an intersection annotation is an annotation that appears in move than 80% of the accessions (.i.e., samples)
        all_annotations = []
        for accession in accessions:
            all_annotations += accession_to_annotations[accession]
        all_annotations = list(set(all_annotations))
        for annotation in all_annotations:
            is_intersection_annotation = True if np.sum(
                [annotation in accession_to_annotations[acc] for acc in accessions]) / len(
                accession_to_annotations.keys()) > 0.9 else False
            if is_intersection_annotation:
                intersection_annotations.append(annotation)

        # assign annotations to each secondary structure based on its accession and annotation (be mindful of un-aligning the start and end positions when computing its location within the original accession)
        df[["assigned_accession_partitions", "assigned_partitions"]] = df[
            ["accession", "unaligned_struct_start_pos", "unaligned_struct_end_pos"]].apply(
            lambda row: RNAStructUtils.get_assigned_annotations(struct_start_pos=int(row.unaligned_struct_start_pos),
                                                 struct_end_pos=int(row.unaligned_struct_end_pos),
                                                 accession_annotations=accession_to_annotations[row.accession],
                                                 intersection_annotations=intersection_annotations),
            axis=1,
            result_type="expand")  # ,nb_workers=multiprocessing.cpu_count()-1)

        # remove records with no assigned intersection annotations as there cannot be compared across species

        # name partitions according to intersection annotations, namely, ones that appear in all the accessions

        return df
