import itertools
import logging
import os
import pickle
import re
import subprocess
import typing as t
from enum import Enum
from Bio import pairwise2
from scipy.spatial import distance
import pandas as pd
import numpy as np
import psutil
from Bio import SeqIO
from Levenshtein import distance as lev

logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    CDHIT = 1


class ClusteringUtils:
    @staticmethod
    def get_sequence_similarity_with_multiple_alignment(
        sequence_data_path: str,
    ) -> t.List[float]:

        mean_sim, min_sim, max_sim, med_sim = np.nan, np.nan, np.nan, np.nan

        if not os.path.exists(sequence_data_path):
            logger.info(f"input path {sequence_data_path} does not exist")
            return [mean_sim, min_sim, max_sim, med_sim]

        output_path = sequence_data_path.replace(".", "_aligned.")
        log_path = sequence_data_path.replace(".fasta", ".log")

        if not os.path.exists(output_path):
            num_sequences = len(list(SeqIO.parse(sequence_data_path, format="fasta")))
            logger.info(
                f"executing mafft on {num_sequences} sequences from {sequence_data_path}"
            )
            cmd = (
                f"mafft --retree 1 --maxiterate 0 {sequence_data_path} > {output_path}"
            )
            res = os.system(cmd)
            if res != 0:
                logger.error(
                    f"failed to execute mafft on input path {sequence_data_path}"
                )
                return [mean_sim, min_sim, max_sim, med_sim]
            if not os.path.exists(output_path):
                raise RuntimeError(f"failed to execute mafft on {sequence_data_path}")
            logger.info(
                f"aligned {num_sequences} sequences with mafft, in {output_path}"
            )
            if os.path.exists(log_path):
                os.remove(log_path)
        aligned_sequences = list(SeqIO.parse(output_path, format="fasta"))
        seq_map = {
            "A": 0,
            "a": 0,
            "C": 1,
            "c": 1,
            "G": 2,
            "g": 2,
            "T": 3,
            "t": 3,
            "-": 4,
        }
        try:
            seq_id_to_array = {
                s.id: np.asarray([seq_map[s] for s in str(s.seq)])
                for s in aligned_sequences
            }
        except Exception as e:
            raise ValueError(
                f"failed to convert sequences  in {output_path} to arrays of integers due to error {e}"
            )
        sequences_pairs = list(itertools.combinations(list(seq_id_to_array.keys()), 2))
        pair_to_similarity = dict()
        for pair in sequences_pairs:
            pair_to_similarity[(pair[0], pair[1])] = 1 - distance.hamming(
                seq_id_to_array[pair[0]], seq_id_to_array[pair[1]]
            )
        similarities = list(pair_to_similarity.values())
        if len(similarities) > 0:
            mean_sim = float(np.mean(similarities))
            min_sim = float(np.min(similarities))
            max_sim = float(np.max(similarities))
            med_sim = float(np.median(similarities))
            logger.info(
                f"computed similarities across {len(similarities)} sequence pairs, yielding mean similarity of {mean_sim}"
            )
        return [
            mean_sim,
            min_sim,
            max_sim,
            med_sim,
        ]

    @staticmethod
    def get_sequences_similarity_with_pairwise_alignments(
        sequence_data_path: str,
    ) -> t.List[float]:
        """
        :param sequence_data_path: path for sequences to compute similarity for
        :return: similarity measure between 0 and 1, corresponding to the mean pairwise alignment score based distance across sequences
        """
        if not os.path.exists(sequence_data_path):
            return [np.nan, np.nan, np.nan, np.nan]

        sequences = list(SeqIO.parse(sequence_data_path, format="fasta"))
        logger.info(
            f"computing pairwise similarities across {len(sequences)} sequences, meaning, {int(len(sequences)**2/2)} comparisons"
        )
        sequences_pairs = list(itertools.combinations(sequences, 2))
        sequences_pair_to_pairwise_alignment = {
            (pair[0].id, pair[1].id): pairwise2.align.globalxx(pair[0].seq, pair[1].seq)
            for pair in sequences_pairs
        }
        sequences_pair_to_pairwise_similarity = {
            (pair[0].id, pair[1].id): (
                sequences_pair_to_pairwise_alignment[pair].score
                / len(sequences_pair_to_pairwise_alignment[pair].seqA)
            )
            for pair in sequences_pairs
        }
        pickle_path = sequence_data_path.replace(
            ".fasta", "_sequences_similarity.pickle"
        )
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(obj=sequences_pair_to_pairwise_similarity, file=pickle_file)

        similarities = list(sequences_pair_to_pairwise_similarity.values())
        mean_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))
        med_sim = float(np.median(similarities))
        logger.info(
            f"mean similarity = {min_sim}, min similarity = {min_sim}, max similarity = {max_sim} \n median similarity = {med_sim}"
        )
        return [
            mean_sim,
            min_sim,
            max_sim,
            med_sim,
        ]

    @staticmethod
    def get_sequences_similarity_with_cdhit(
        sequence_data_path: str,
        mem_limit: int = 4000,
        threshold: float = 0.5,
    ) -> t.List[float]:
        """
        :param sequence_data_path: path for sequences to compute similarity for
        :param mem_limit: memory limitation for cdhit
        :param threshold: similarity threshold to use
        :return: similarity measure between 0 and 1, corresponding to the
        lowest sequence homology between any member of the largest cluster
        (usually the only one, if the threshold is 0.5) to the cluster's representative
        """

        if not os.path.exists(sequence_data_path):
            return [np.nan, np.nan, np.nan, np.nan]

        threshold_range_to_wordlen = {
            (0.7, 1.0): 5,
            (0.6, 0.7): 4,
            (0.5, 0.6): 3,
            (0.4, 0.5): 2,
        }  # based on https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#CDHITEST
        aux_dir = f"{os.getcwd()}/cdhit_aux/"
        os.makedirs(aux_dir, exist_ok=True)
        cdhit_input_path = sequence_data_path
        cdhit_output_path = (
            f"{aux_dir}/cdhit_group_out_{os.path.basename(cdhit_input_path)}"
        )
        cdhit_log_path = (
            f"{aux_dir}/cdhit_group_out_{os.path.basename(cdhit_input_path)}.log"
        )
        if not os.path.exists(cdhit_output_path):
            num_sequences = len(list(SeqIO.parse(sequence_data_path, format="fasta")))
            if num_sequences < 3:
                return (
                    ClusteringUtils.get_sequences_similarity_with_pairwise_alignments(
                        sequence_data_path
                    )
                )
            logger.info(
                f"executing cdhit on {num_sequences} sequences from {sequence_data_path}"
            )

            word_len = [
                threshold_range_to_wordlen[key]
                for key in threshold_range_to_wordlen.keys()
                if key[0] <= threshold <= key[1]
            ][0]
            cmd = f"cd-hit -M {mem_limit} -i {cdhit_input_path} -o {cdhit_output_path} -c {threshold} -n {word_len} > {cdhit_log_path}"
            res = os.system(cmd)
            if res != 0:
                logger.error(
                    f"CD-HIT failed to properly execute and provide an output file on {sequence_data_path}"
                )

        similarity_regex = re.compile("(\d+\.\d*)%")
        with open(f"{cdhit_output_path}.clstr", "r") as clusters_file:
            similarities = [
                float(match.group(1)) / 100
                for match in similarity_regex.finditer(clusters_file.read())
            ]
        if len(similarities) == 0:
            return [np.nan, np.nan, np.nan, np.nan]

        res = os.system(f"rm -r {cdhit_output_path}")
        if res != 0:
            raise RuntimeError(f"failed to remove {cdhit_output_path}")
        if os.path.exists(cdhit_log_path):
            res = os.system(f"rm -r {cdhit_log_path}")
            if res != 0:
                raise RuntimeError(f"failed to remove {cdhit_log_path}")

        mean_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))
        med_sim = float(np.median(similarities))
        return [
            mean_sim,
            min_sim,
            max_sim,
            med_sim,
        ]

    @staticmethod
    def get_cdhit_clusters(
        elements: pd.DataFrame,
        homology_threshold: float = 0.99,
        memory_limit: int = 6000,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
    ) -> t.Dict[t.Union[np.int64, str], np.int64]:
        """
        :param elements: elements to cluster using kmeans
        :param homology_threshold: cdhit threshold in clustering
        :param memory_limit: memory limit in MB
        :param aux_dir: directory ot write output files of cdhit to
        :return: a list of element ids corresponding the the representatives of the cdhit clusters
        """

        os.makedirs(aux_dir, exist_ok=True)

        cdhit_input_path = f"{aux_dir}/sequences.fasta"
        names_translator_path = f"{aux_dir}/names_translator.pickle"
        logger.info(f"creating input files for cdhit clustering at {aux_dir}")
        elm_to_seq = dict()
        elm_to_fake_name = dict()
        fake_name_to_elm = dict()
        i = 0
        if not os.path.exists(cdhit_input_path) or not os.path.exists(
            names_translator_path
        ):
            logger.info(
                f"either the input path {cdhit_input_path} or the aux path {names_translator_path} does not exist, so will create them"
            )
            for (
                index,
                row,
            ) in elements.iterrows():
                elm = f"{row.accession}_{row.taxon_name}"
                seq = row["sequence"]
                elm_to_fake_name[elm] = f"S{i}"
                fake_name_to_elm[f"S{i}"] = elm
                elm_to_seq[elm] = seq
                i += 1

            with open(cdhit_input_path, "w") as infile:
                infile.write(
                    "\n".join(
                        [
                            f">{elm_to_fake_name[elm]}\n{elm_to_seq[elm]}"
                            for elm in elm_to_seq
                        ]
                    )
                )

            with open(names_translator_path, "wb") as infile:
                pickle.dump(obj=fake_name_to_elm, file=infile)

        logger.info(f"cdhit input paths created at {aux_dir}")
        cdhit_output_file = f"{aux_dir}/cdhit_out_thr_{homology_threshold}"
        cdhit_log_file = f"{aux_dir}/cdhit.log"
        if not os.path.exists(cdhit_output_file):
            word_len = (
                (8 if homology_threshold > 0.7 else 4)
                if homology_threshold > 0.6
                else (3 if homology_threshold > 0.5 else 2)
            )
            logger.info(
                f"executing cdhit on {cdhit_input_path} with homology threshold of {homology_threshold} and word length {word_len}"
            )
            cmd = f"cd-hit-est -i {cdhit_input_path} -o {cdhit_output_file} -c {homology_threshold} -n {word_len} -M {memory_limit} > {cdhit_log_file}"
            res = os.system(cmd)
            if res != 0:
                raise RuntimeError(
                    f"CD-HIT failed to properly execute and provide an output file with error"
                )

        elm_to_cluster = dict()
        clusters_data_path = f"{cdhit_output_file}.clstr"
        member_regex = re.compile(">(.*?)\.\.\.", re.MULTILINE | re.DOTALL)

        logger.info(
            f"parsing cdhit output using the auxiliary file {names_translator_path}"
        )
        with open(names_translator_path, "rb") as infile:
            fake_name_to_elm = pickle.load(file=infile)

        with open(clusters_data_path, "r") as outfile:
            clusters = outfile.read().split(">Cluster")[1:]
            for cluster in clusters:
                data = cluster.split("\n")
                cluster_id = np.int64(data[0])
                cluster_members = []
                for member_data in data[1:]:
                    if len(member_data) > 0:
                        member_fake_name = member_regex.search(member_data).group(1)
                        member = fake_name_to_elm[
                            member_fake_name
                        ]  # ?? tried to rename sequences? if so, save map as pickle
                        cluster_members.append(member)
                elm_to_cluster.update(
                    {member: cluster_id for member in cluster_members}
                )

        return elm_to_cluster

    @staticmethod
    def compute_clusters_representatives(
        elements: pd.DataFrame,
        clustering_method: ClusteringMethod = ClusteringMethod.CDHIT,
        homology_threshold: t.Optional[float] = 0.99,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
    ):
        """
        :param elements: elements to cluster using cdhit
        :param clustering_method: either cdhit or kmeans
        :param homology_threshold: cdhit threshold in clustering
        :param aux_dir: directory to write cdhit output files to
        :return: none, adds cluster_id and cluster_representative columns to the existing elements dataframe
        """
        logger.info(
            f"computing clusters based on method {clustering_method} for {elements.shape[0]} elements"
        )
        if clustering_method == ClusteringMethod.CDHIT:
            elm_to_cluster = ClusteringUtils.get_cdhit_clusters(
                elements=elements,
                homology_threshold=homology_threshold,
                aux_dir=aux_dir,
            )
        else:
            logger.error(f"clustering method {clustering_method} is not implemented")
            raise ValueError(
                f"clustering method {clustering_method} is not implemented"
            )
        elements["cluster_id"] = np.nan
        elements.set_index("taxon_name", inplace=True)
        elements["cluster_id"].fillna(value=elm_to_cluster, inplace=True)
        elements.reset_index(inplace=True)

        clusters = list(set(elm_to_cluster.values()))
        cluster_to_representative = dict()
        for cluster in clusters:
            cluster_members = elements.loc[elements.cluster_id == cluster]
            if cluster_members.shape[0] == 1:
                cluster_representative = cluster_members.iloc[0]["taxon_name"]
            else:
                elements_distances = (
                    ClusteringUtils.compute_pairwise_sequence_distances(
                        elements=cluster_members,
                    )
                )
                cluster_representative = ClusteringUtils.get_centroid(
                    elements_distances
                )
            cluster_to_representative[cluster] = cluster_representative

        elements["cluster_representative"] = np.nan
        elements.set_index("cluster_id", inplace=True)
        elements["cluster_representative"].fillna(
            value=cluster_to_representative, inplace=True
        )
        elements.reset_index(inplace=True)

    @staticmethod
    def get_pairwise_alignment_distance(seq1: str, seq2: str) -> float:
        """
        :param seq1: sequence 1
        :param seq2: sequence 2
        :return: a float between 0 and 1 representing the distance between the two sequences based on their pairwise alignment
        """
        try:
            dist = float(lev(seq1, seq2) / np.max([len(seq1), len(seq2)]))
            return dist
        except Exception as e:
            logger.error(f"failed to compute distance due to error: {e}")
            logger.error(f"len(seq1)={len(seq1)}, len(seq2)={len(seq2)}")
            process = psutil.Process(os.getpid())
            logger.error(process.memory_info().rss)  # in bytes
        return np.nan

    @staticmethod
    def get_distance(x: pd.Series, elements: pd.DataFrame):
        elm1 = x["element_1"]
        elm2 = x["element_2"]
        try:
            elm1_seq = (
                elements.loc[elements["taxon_name"] == elm1]["sequence"]
                .dropna(axis=1)
                .values[0]
            )
            elm2_seq = (
                elements.loc[elements["taxon_name"] == elm2]["sequence"]
                .dropna(axis=1)
                .values[0]
            )
            return ClusteringUtils.get_pairwise_alignment_distance(elm1_seq, elm2_seq)
        except Exception as e:
            logger.error(f"failed to compute pairwise distance between {elm1} and {elm2} due to error {e}")
            return np.nan

    @staticmethod
    def compute_pairwise_sequence_distances(
        elements: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        :param elements: elements to compute pairwise distances for
        :return: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        """
        elements_distances = pd.DataFrame(
            [
                (elm1, elm2)
                for elm1 in elements["taxon_name"]
                for elm2 in elements["taxon_name"]
            ],
            columns=["element_1", "element_2"],
        )

        elements_distances["distance"] = np.nan
        elements_distances["distance"] = elements_distances.apply(
            lambda x: ClusteringUtils.get_distance(x, elements),
            axis=1,
        )

        return elements_distances

    @staticmethod
    def get_centroid(elements_distances: pd.DataFrame) -> t.Union[np.int64, str]:
        """
        :param elements_distances: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        :return: the element id of the centroid
        """
        elements_sum_distances = (
            elements_distances.groupby("element_1")["distance"].sum().reset_index()
        )
        centroid = elements_sum_distances.iloc[elements_distances["distance"].argmin()][
            "element_1"
        ]
        return centroid
