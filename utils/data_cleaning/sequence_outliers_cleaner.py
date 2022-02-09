import os
import random

import numpy as np
import pandas as pd
from Bio import Phylo, SeqIO
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio.Seq import Seq
from ete3 import Tree
import typing as t

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

import logging

import utils
from utils.data_clustering.sequence_clustering_utils import SequenceClusteringUtils
from utils.programs.cdhit import CdHit
from utils.programs.mafft import Mafft

logger = logging.getLogger(__name__)


class SequenceOutliersCleaner:
    @staticmethod
    def get_upgma_tree(distances: np.ndarray, names: t.List[str], tree_path: str) -> Tree:
        """
        :param distances: distances matrix in the form of np.ndarray
        :param names: names of items included in the distance matrix(len(names) == distances.shape[0])
        :param tree_path: path ot upgma tree in newick format
        :return: upgma ete tree
        """
        distances_lst = distances.tolist()
        for i in range(distances.shape[0]):  # turn matrix into a lower triangle one, as biopython requires
            distances_lst[i] = distances_lst[i][: i + 1]
        distance_matrix = DistanceMatrix(names=names, matrix=distances_lst)
        constructor = DistanceTreeConstructor()
        if not os.path.exists(tree_path):
            upgma_structures_tree = constructor.upgma(distance_matrix)
            tree_leaves = upgma_structures_tree.get_terminals()
            for leaf in tree_leaves:
                leaf.name = leaf.name.replace(";", "&")
            Phylo.write(upgma_structures_tree, tree_path, "newick")
        upgma_tree = Tree(tree_path, format=1)
        return upgma_tree

    @staticmethod
    def get_distances(similarities: np.ndarray) -> np.ndarray:
        distances = 1.0 - similarities
        if np.any(pd.isna(distances)):
            distances[pd.isna(distances)] = 0
            distances = np.maximum(distances, distances.transpose())
        return distances

    @staticmethod
    def get_largest_legal_cluster(
        tree: Tree, similarity_cutoff: float, distances: np.ndarray, samples: t.List[str]
    ) -> t.Optional[t.List[int]]:
        clusters = []
        for node in tree.traverse("levelorder"):
            node.add_feature(pr_name="confers_cluster", pr_value=False)
        for node in tree.traverse("levelorder"):
            if node.up is None or node.up.confers_cluster:
                continue
            leaves = [leaf_name.replace("&", ";") for leaf_name in node.get_leaf_names()]
            if len(leaves) > np.min([3, int(distances.shape[0] * 0.1)]):  # do not accept cluster of too small of sizes
                leaves_idx = np.argwhere(np.isin(samples, leaves)).ravel()
                leaves_distances = distances[leaves_idx, :][:, leaves_idx]
                max_leaves_distance = np.nanmax(leaves_distances)
                if max_leaves_distance <= 1 - similarity_cutoff:
                    clusters.append(leaves_idx)
                    node.confers_cluster = True

        largest_cluster = None
        if len(clusters) > 0:
            largest_cluster = max(clusters, key=lambda cluster: cluster.shape[0])
        return largest_cluster

    @staticmethod
    def compute_outliers_based_on_similarities(
        data: pd.DataFrame,
        data_dist_plot_path: str,
        tree_path: str,
        similarity_cutoff: t.Optional[float] = None,
        plot_space: bool = False,
    ) -> t.Tuple[t.List[str], t.List[str]]:

        similarities = data.to_numpy()
        distances = SequenceOutliersCleaner.get_distances(similarities=similarities)

        upgma_tree = SequenceOutliersCleaner.get_upgma_tree(
            distances=distances, names=list(data.index), tree_path=tree_path
        )

        # find highest internal node for which the max distance across its children in < 1-similarity_threshold
        remaining_idx = random.sample(list(range(distances.shape[0])), 1)
        largest_cluster = SequenceOutliersCleaner.get_largest_legal_cluster(
            tree=upgma_tree, similarity_cutoff=similarity_cutoff, distances=distances, samples=list(data.index)
        )
        if largest_cluster is not None:
            remaining_idx = largest_cluster
        outlier_idx = [i for i in range(distances.shape[0]) if i not in remaining_idx]
        remaining_accessions = list(data.index[remaining_idx])
        outlier_accessions = list(data.index[outlier_idx])

        # plot records distribution - this is projection of the first 2 dimensions only and is thus not as reliable
        if plot_space:
            SequenceOutliersCleaner.plot_sequence_data_in_space(
                similarities=similarities, output_path=data_dist_plot_path
            )

        logger.info(
            f"mean similarity across remaining sequences = {np.nanmean(similarities[remaining_idx][:, remaining_idx].tolist())}"
        )
        logger.info(
            f"min similarity across remaining sequences = {np.nanmin(similarities[remaining_idx][:, remaining_idx].tolist())}"
        )
        return remaining_accessions, outlier_accessions

    @staticmethod
    def get_relevant_accessions_using_sequence_data_directly(
        data_path: str, similarity_cutoff: t.Optional[float] = None
    ) -> t.Union[str, int]:
        """
        :param data_path: an alignment of sequences
        :param similarity_cutoff: similarity cutoff to filter out sequence data according to
        :return: string of the list of relevant accessions that were not identified as outliers, separated by ";;"
        """
        if not os.path.exists(data_path):
            logger.info(f"alignment fie {data_path} does not exist")
            return np.nan
        sequence_records = list(SeqIO.parse(data_path, format="fasta"))

        logger.info(
            f"original alignment consists of {len(sequence_records)} sequences and {len(sequence_records[0].seq)} positions"
        )

        if len(sequence_records) < 20:
            return ";;".join([record.description for record in sequence_records])
        if len(sequence_records) > 1000:
            accessions_to_keep = CdHit.get_largest_cdhit_cluster(
                sequence_records, workdir=f"{data_path.replace('_aligned.fasta', '_cdhit_aux/')}"
            )
            return ";;".join(accessions_to_keep)

        similarities_data_path = data_path.replace(".fasta", "_similarity_values.csv")
        if not os.path.exists(similarities_data_path):
            logger.info(f"similarities matrix between items in {data_path} does not exist. will create it now")
            SequenceClusteringUtils.compute_msa_based_similarity_values(
                alignment_path=data_path, similarities_output_path=similarities_data_path
            )
        pairwise_similarities_df = SequenceClusteringUtils.get_pairwise_similarities_df(
            input_path=similarities_data_path
        )
        outliers_accessions, accessions_to_keep = [], list(pairwise_similarities_df.index)
        logger.info(f"computing outlier accessions based on similarities values")
        if pairwise_similarities_df.shape[0] > 1:
            accessions_to_keep, outliers_accessions = SequenceOutliersCleaner.compute_outliers_based_on_similarities(
                data=pairwise_similarities_df,
                data_dist_plot_path=data_path.replace(".fasta", "clusters.png"),
                tree_path=data_path.replace(".fasta", "_upgma.nwk"),
                similarity_cutoff=similarity_cutoff,
            )
            logger.info(f"{len(outliers_accessions)} out of {len(sequence_records)} are outliers")
        logger.info(
            f"{len(accessions_to_keep)} accessions remain after removing {len(outliers_accessions)} outliers\naccessions {','.join(outliers_accessions)} were determined as outliers"
        )
        return ";;".join(accessions_to_keep)

    @staticmethod
    def remove_sequence_outliers(
        alignment_path: str, unaligned_output_path: str, aligned_output_path: str, similarity_cutoff: float
    ):
        """
        :param alignment_path: path to alignment sequences
        :param unaligned_output_path: path in which filtered un-aligned sequence data would be written
        :param aligned_output_path: path in which filtered aligned sequence data would be written
        :param similarity_cutoff: similarity cutoff between sequences to filter by
        :return: None
        """
        aligned_sequence_records = list(SeqIO.parse(alignment_path, format="fasta"))
        print(f"filtering outliers of the {len(aligned_sequence_records)} sequences in {alignment_path}")
        accessions_to_keep = SequenceOutliersCleaner.get_relevant_accessions_using_sequence_data_directly(
            data_path=alignment_path, similarity_cutoff=similarity_cutoff
        ).split(";;")
        unaligned_relevant_records = [record for record in aligned_sequence_records if record.id in accessions_to_keep]
        for record in unaligned_relevant_records:
            record.seq = Seq(str(record.seq).replace("-", ""))
        logger.info(f"after filtering, {len(unaligned_relevant_records)} sequences remain")
        SeqIO.write(unaligned_relevant_records, unaligned_output_path, format="fasta")
        logger.info(f"unaligned filtered data written to {unaligned_output_path}")
        res = Mafft.exec_mafft(input_path=unaligned_output_path, output_path=aligned_output_path)
        logger.info(f"aligned filtered data written to {aligned_output_path}")

    @staticmethod
    def plot_sequence_data_in_space(
        similarities: np.ndarray, output_path: str, outlier_idx: t.List[int], non_outlier_idx: t.List[int]
    ):

        # extract first two PCs
        pca = PCA(n_components=2)
        distances = 1 - similarities
        if np.any(pd.isna(distances)):
            distances[pd.isna(distances)] = 0
            distances = np.maximum(distances, distances.transpose())
        pca.fit(np.stack(distances))
        # translate each coordinate to its reduced representation based on the two PCs
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        reduced_coordinates = pd.DataFrame(columns=["x", "y"])
        for i in range(distances.shape[0]):
            reduced_coordinates.at[i, "x"] = np.dot(pc1, distances[i])
            reduced_coordinates.at[i, "y"] = np.dot(pc2, distances[i])
        plt.xlabel("pc1")
        plt.ylabel("pc2")
        plt.scatter(reduced_coordinates.iloc[outlier_idx].x, reduced_coordinates.iloc[outlier_idx].y, color="r")
        plt.scatter(reduced_coordinates.iloc[non_outlier_idx].x, reduced_coordinates.iloc[non_outlier_idx].y, color="b")
        plt.savefig(output_path)

    @staticmethod
    def get_relevant_accessions_using_pairwise_distances(data_path: str,) -> str:
        """
        :param data_path: path to a dataframe matching a similarity value to each pair of accessions
        :return: string of the list of relevant accessions that were not identified as outliers, separated by ";;"
        """

        accessions_data = SequenceClusteringUtils.get_pairwise_similarities_df(input_path=data_path)

        outliers_idx = []
        if accessions_data.shape[0] > 2:
            outliers_idx = SequenceOutliersCleaner.compute_outliers_based_on_similarities(
                data=accessions_data[[col for col in accessions_data.columns if "similarity_to" in col]],
                data_dist_plot_path=data_path.replace(".csv", ".png"),
                tree_path=data_path.replace(".csv", ".nwk"),
            )

        accessions = list(accessions_data.accession)
        accessions_to_keep = [accessions[idx] for idx in range(len(accessions)) if idx not in outliers_idx]
        logger.info(
            f"{len(accessions_to_keep)} accessions remain after removing {len(outliers_idx)} outliers\naccessions {[acc for acc in accessions if acc not in accessions_to_keep]} were determined as outliers"
        )
        return ";;".join(accessions_to_keep)
