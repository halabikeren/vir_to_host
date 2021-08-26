import logging
import os
import re
import subprocess
import time
import typing as t
from enum import Enum
from random import random

import pandas as pd
import numpy as np
import psutil
from Levenshtein import distance as lev

logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    CDHIT = 1


class ClusteringUtils:
    @staticmethod
    def get_sequences_similarity(members_names: str, sequence_data: pd.DataFrame, mem_limit: int = 4000) -> float:
        """
        :param members_names: names of viruses, separated by comma
        :param sequence_data: dataframe holding sequences of ny viruses names
        :param mem_limit: memory limitation for cdhit
        :return: similarity measure between 0 and 1 for the sequences of thew viruses, if available, induced by the number of cd-hit clusters at threshold 0.8 for the set of respective sequences
        """
        viruses_names_lst = members_names.split(",")
        relevant_virus_seq_data = list(sequence_data.loc[sequence_data.virus_taxon_name.isin(viruses_names_lst)][
                                           [col for col in sequence_data.columns if "sequence" in col]].dropna(
            axis=1).values.flatten())
        if len(relevant_virus_seq_data) == 0:
            return np.nan
        elif len(relevant_virus_seq_data) == 1:
            return 1
        aux_dir = f"{os.getcwd()}/cdhit_aux/"
        os.makedirs(aux_dir, exist_ok=True)
        rand_id = random.random()
        cdhit_input_path = f"{aux_dir}/sequences_{time.time()}_{os.getpid()}_{rand_id}.fasta"
        with open(cdhit_input_path, "w") as infile:
            infile.write(
                "\n".join([f">S{i}\n{relevant_virus_seq_data[i]}" for i in range(len(relevant_virus_seq_data))]))
        cdhit_output_path = f"{aux_dir}/cdhit_group_out_{time.time()}_{os.getpid()}_{rand_id}"
        cdhit_log_path = f"{aux_dir}/cdhit_group_out_{time.time()}_{os.getpid()}_{rand_id}.log"
        cmd = f"cd-hit-est -M {mem_limit} -i {cdhit_input_path} -o {cdhit_output_path} -c 0.99 -n 5 > {cdhit_log_path}"
        res = os.system(cmd)
        if res != 0:
            logger.error("CD-HIT failed to properly execute and provide an output file")
            res = os.system(f"rm -r {cdhit_input_path}")
            if res != 0:
                raise RuntimeError(f"failed to remove {cdhit_input_path}")
            res = os.system(f"rm -r {cdhit_log_path}")
            if res != 0:
                raise RuntimeError(f"failed to remove {cdhit_log_path}")
            return np.nan

        with open(f"{cdhit_output_path}.clstr", "r") as clusters_path:
            dist = (clusters_path.read().count(">Cluster") - 1) / len(relevant_virus_seq_data)
            similarity = 1 - dist
        res = os.system(f"rm -r {cdhit_input_path}")
        if res != 0:
            raise RuntimeError(f"failed to remove {cdhit_input_path}")
        res = os.system(f"rm -r {cdhit_output_path}")
        if res != 0:
            raise RuntimeError(f"failed to remove {cdhit_output_path}")
        res = os.system(f"rm -r {cdhit_output_path}.clstr")
        if res != 0:
            raise RuntimeError(f"failed to remove {cdhit_output_path}.clstr")
        res = os.system(f"rm -r {cdhit_log_path}")
        if res != 0:
            raise RuntimeError(f"failed to remove {cdhit_log_path}")

        return similarity

    @staticmethod
    def get_cdhit_clusters(
            elements: pd.DataFrame,
            id_colname: str,
            seq_colnames: t.List[str],
            homology_threshold: float = 0.99,
    ) -> t.Dict[t.Union[np.int64, str], np.int64]:
        """
        :param elements: elements to cluster using kmeans
        :param id_colname: column holding the id of the elements
        :param seq_colnames: names of columns holding the sequences of the elements
        :param homology_threshold: cdhit threshold in clustering
        :return: a list of element ids corresponding the the representatives of the cdhit clusters
        """

        aux_dir = f"{os.getcwd()}/cdhit_aux/"
        os.makedirs(aux_dir, exist_ok=True)

        elm_to_seq = dict()
        elm_to_fake_name = dict()
        fake_name_to_elm = dict()
        for index, row in elements.iterrows():
            elm = row[id_colname]
            seq = row[seq_colnames].dropna().values[0]
            elm_to_fake_name[elm] = f"S{index}"
            fake_name_to_elm[f"S{index}"] = elm
            elm_to_seq[elm] = seq

        cdhit_input_path = f"{aux_dir}/sequences.fasta"
        with open(cdhit_input_path, "w") as infile:
            infile.write(
                "\n".join(
                    [
                        f">{elm_to_fake_name[elm]}\n{elm_to_seq[elm]}"
                        for elm in elm_to_seq
                    ]
                )
            )

        cdhit_output_file = f"{aux_dir}/cdhit_out_thr_{homology_threshold}"
        if not os.path.exists(cdhit_output_file):
            word_len = (
                (5 if homology_threshold > 0.7 else 4)
                if homology_threshold > 0.6
                else (3 if homology_threshold > 0.5 else 2)
            )
            cmd = f"cd-hit-est -i {cdhit_input_path} -o {cdhit_output_file} -c {homology_threshold} -n {word_len}"
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if len(process.stderr.read()) > 0:
                raise RuntimeError(
                    f"CD-HIT failed to properly execute and provide an output file with error {process.stderr.read()} and output is {process.stdout.read()}"
                )

        elm_to_cluster = dict()
        clusters_data_path = f"{cdhit_output_file}.clstr"
        member_regex = re.compile(">(.*?)\.\.\.", re.MULTILINE | re.DOTALL)
        with open(clusters_data_path, "r") as outfile:
            clusters = outfile.read().split(">Cluster")[1:]
            for cluster in clusters:
                data = cluster.split("\n")
                cluster_id = np.int64(data[0])
                cluster_members = []
                for member_data in data[1:]:
                    if len(member_data) > 0:
                        member_fake_name = member_regex.search(member_data).group(1)
                        member = fake_name_to_elm[member_fake_name]
                        cluster_members.append(member)
                elm_to_cluster.update(
                    {member: cluster_id for member in cluster_members}
                )

        return elm_to_cluster

    @staticmethod
    def compute_clusters_representatives(
            elements: pd.DataFrame,
            id_colname: str,
            seq_colnames: t.List[str],
            clustering_method: ClusteringMethod = ClusteringMethod.CDHIT,
            homology_threshold: t.Optional[float] = 0.99,
    ):
        """
        :param elements: elements to cluster using cdhit
        :param id_colname: column holding the id of the elements
        :param seq_colnames: names of columns holding the sequences of the elements
        :param clustering_method: either cdhit or kmeans
        :param homology_threshold: cdhit threshold in clustering
        :return: none, adds cluster_id and cluster_representative columns to the existing elements dataframe
        """
        if clustering_method == ClusteringMethod.CDHIT:
            elm_to_cluster = ClusteringUtils.get_cdhit_clusters(
                elements=elements,
                id_colname=id_colname,
                seq_colnames=seq_colnames,
                homology_threshold=homology_threshold,
            )
        else:
            logger.error(f"clustering method {clustering_method} is not implemented")
            raise ValueError(
                f"clustering method {clustering_method} is not implemented"
            )
        elements["cluster_id"] = np.nan
        elements.set_index(id_colname, inplace=True)
        elements["cluster_id"].fillna(value=elm_to_cluster, inplace=True)
        elements.reset_index(inplace=True)

        clusters = list(set(elm_to_cluster.values()))
        cluster_to_representative = dict()
        for cluster in clusters:
            cluster_members = elements.loc[elements.cluster_id == cluster]
            if cluster_members.shape[0] == 1:
                cluster_representative = cluster_members.iloc[0][id_colname]
            else:
                elements_distances = ClusteringUtils.compute_pairwise_sequence_distances(
                    elements=cluster_members,
                    id_colname=id_colname,
                    seq_colnames=seq_colnames,
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
            dist = lev(seq1, seq2) / np.max([len(seq1), len(seq2)])
            return dist
        except Exception as e:
            logger.error(f"failed to compute distance due to error: {e}")
            logger.error(f"len(seq1)={len(seq1)}, len(seq2)={len(seq2)}")
            process = psutil.Process(os.getpid())
            logger.error(process.memory_info().rss)  # in bytes
            return np.nan

    @staticmethod
    def get_distance(x, id_colname, seq_colnames, elements):
        elm1 = x["element_1"]
        elm2 = x["element_2"]
        elm1_seq = (
            elements.loc[elements[id_colname] == elm1][seq_colnames]
                .dropna(axis=1)
                .values[0][0]
        )
        elm2_seq = (
            elements.loc[elements[id_colname] == elm2][seq_colnames]
                .dropna(axis=1)
                .values[0][0]
        )
        return ClusteringUtils.get_pairwise_alignment_distance(elm1_seq, elm2_seq)

    @staticmethod
    def compute_pairwise_sequence_distances(
            elements: pd.DataFrame,
            id_colname: str,
            seq_colnames: t.List[str],
    ) -> pd.DataFrame:
        """
        :param elements: elements to compute pairwise distances for
        :param id_colname: column holding the id of the elements
        :param seq_colnames: names of columns holding the sequences of the elements
        :return: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        """
        elements_distances = pd.DataFrame(
            [
                (elm1, elm2)
                for elm1 in elements[id_colname]
                for elm2 in elements[id_colname]
            ],
            columns=["element_1", "element_2"],
        )

        elements_distances["distance"] = elements_distances.apply(
            lambda x: ClusteringUtils.get_distance(
                x, id_colname, seq_colnames, elements
            ),
            axis=1,
        )

        return elements_distances

    @staticmethod
    def get_centroid(elements_distances: pd.DataFrame) -> t.Union[np.int64, str]:
        """
        :param elements_distances: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        :return: the element id of the centroid
        """
        elements_sum_distances = elements_distances.groupby("element_1")["distance"].sum().reset_index()
        centroid = elements_sum_distances.iloc[
            elements_distances["distance"].argmin()
        ]["element_1"]
        return centroid