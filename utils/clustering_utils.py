import logging
import os
import pickle
import re
import typing as t
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from ete3 import Tree
from Bio import Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from matplotlib import pyplot as plt
from scipy.spatial import distance
from Levenshtein import distance as lev
import pandas as pd
import numpy as np
from Bio import SeqIO
from copkmeans.cop_kmeans import *
from sklearn.decomposition import PCA
from settings import get_settings

logger = logging.getLogger(__name__)


class ClusteringUtils:

    # TO DO: this seems to me like: https://towardsdatascience.com/multidimensional-scaling-d84c2a998f72 - try to find a more elegant way to do this
    @staticmethod
    def map_items_to_plane_by_distance(items: t.List[str], distances_df: t.Optional[pd.DataFrame]) -> t.List[np.array]:
        """
        :param items: list of structures that need to be mapped to a plane
        :param distances_df: distances between structures. Should be provided in case of choosing to vectorize structures using the relative method
        :return: vectors representing the structures, in the same order of the given structures
        using relative method: uses an approach inspired by the second conversion of scores to euclidean space in: https://doi.org/10.1016/j.jmb.2018.03.019
        """
        vectorized_structures = []

        # take the 100 structures with the largest distances from ome another - each of these will represent an axis
        if len(items) > 300:
            logger.warning(
                f"the total number of structures is {len(items)} > 300 and thus, distance-based vectorization will consider only 300 structures and not all"
            )
        max_num_axes = np.min([300, len(items)])
        s = distances_df.sum()
        distances_df = distances_df[s.sort_values(ascending=False).index]
        axes_items = distances_df.index[:max_num_axes]
        for i in range(len(items)):
            vectorized_items = np.array([distances_df[items[i]][axes_items[j]] for j in range(len(axes_items))])
            vectorized_structures.append(vectorized_items)

        return vectorized_structures

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
            Phylo.write(upgma_structures_tree, tree_path, "newick")
        upgma_tree = Tree(tree_path, format=1)
        return upgma_tree

    @staticmethod
    def compute_outliers_based_on_similarities(
        data: pd.DataFrame,
        data_dist_plot_path: str,
        tree_path: str,
        similarity_cutoff: t.Optional[float] = None,
        plot_space: bool = False,
    ) -> t.Tuple[t.List[str], t.List[str]]:

        similarities = data.to_numpy()
        distances = 1.0 - similarities
        if np.any(pd.isna(distances)):
            distances[pd.isna(distances)] = 0
            distances = np.maximum(distances, distances.transpose())

        upgma_tree = ClusteringUtils.get_upgma_tree(distances=distances, names=list(data.index), tree_path=tree_path)
        # find highest internal node for which the max distance across its children in < 1-similarity_threshold
        clusters = []
        for node in upgma_tree.traverse("levelorder"):
            node.add_feature(pr_name="confers_cluster", pr_value=False)
        for node in upgma_tree.traverse("levelorder"):
            if node.up is None or node.up.confers_cluster:
                continue
            leaves = node.get_leaf_names()
            if len(leaves) > np.min([10, int(distances.shape[0] * 0.1)]):  # do not accept cluster of too small of sizes
                leaves_idx = np.argwhere(np.isin(list(data.index), leaves)).ravel()
                leaves_distances = distances[leaves_idx, :][:, leaves_idx]
                max_leaves_distance = np.nanmax(leaves_distances)
                if max_leaves_distance <= 1 - similarity_cutoff:
                    clusters.append(leaves_idx)
                    node.confers_cluster = True

        largest_cluster = max(clusters, key=lambda cluster: cluster.shape[0])
        remaining_idx = largest_cluster
        outlier_idx = [i for i in range(distances.shape[0]) if i not in remaining_idx]
        remaining_accessions = list(data.index[remaining_idx])
        outlier_accessions = list(data.index[outlier_idx])

        # plot records distribution - this is projection of the first 2 dimensions only and is thus not as reliable
        if plot_space:
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
            plt.scatter(reduced_coordinates.iloc[remaining_idx].x, reduced_coordinates.iloc[remaining_idx].y, color="b")
            plt.savefig(data_dist_plot_path)

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

        if len(sequence_records) < 3:
            return ";;".join([record.description for record in sequence_records])
        if len(sequence_records) > 1000:
            accessions_to_keep = ClusteringUtils.get_largest_cdhit_cluster(
                sequence_records, workdir=f"{data_path.replace('_aligned.fasta', '_cdhit_aux/')}"
            )

        similarities_data_path = data_path.replace(".fasta", "_similarity_values.csv")
        if not os.path.exists(similarities_data_path):
            logger.info(f"similarities matrix between items in {data_path} does not exist. will create it now")
            ClusteringUtils.compute_msa_based_similarity_values(
                alignment_path=data_path, similarities_output_path=similarities_data_path
            )
        pairwise_similarities_df = ClusteringUtils.get_pairwise_similarities_df(input_path=similarities_data_path)
        outliers_accessions, accessions_to_keep = [], list(pairwise_similarities_df.index)
        logger.info(f"computing outlier accessions based on similarities values")
        if pairwise_similarities_df.shape[0] > 1:
            accessions_to_keep, outliers_accessions = ClusteringUtils.compute_outliers_based_on_similarities(
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
    def get_pairwise_similarities_df(input_path: str) -> pd.DataFrame:

        similarities_df = pd.read_csv(input_path)
        filler = (
            similarities_df.pivot_table(
                values="similarity", index="accession_1", columns="accession_2", aggfunc="first",
            )
            .reset_index()
            .rename(columns={"accession_1": "accession"})
        ).set_index("accession")

        accessions = list(similarities_df.accession_1.unique())
        accessions_data = pd.DataFrame(index=accessions, columns=accessions)
        accessions_data.update(filler)

        logger.info(f"computed similarities table across {accessions_data.shape[0]} accessions")
        return accessions_data

    @staticmethod
    def get_relevant_accessions_using_pairwise_distances(data_path: str,) -> str:
        """
        :param data_path: path to a dataframe matching a similarity value to each pair of accessions
        :return: string of the list of relevant accessions that were not identified as outliers, separated by ";;"
        """

        accessions_data = ClusteringUtils.get_pairwise_similarities_df(input_path=data_path)

        outliers_idx = []
        if accessions_data.shape[0] > 2:
            outliers_idx = ClusteringUtils.compute_outliers_based_on_similarities(
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

    @staticmethod
    def compute_similarity_across_aligned_sequences(
        record: pd.Series, seq_to_token: t.Dict[str, np.array], gap_code: int
    ) -> float:
        """
        :param record: record that consists of two accessions to be compared
        :param seq_to_token: map of accessions to their tokenized aligned sequences
        :param gap_code: gap code - based on which the length of pairwise aligned sequences to standardize the distance between will be computed
        :return:
        """
        if record.accession_1 == record.accession_2:
            return 1
        seq_1 = seq_to_token[record.accession_1]
        seq_2 = seq_to_token[record.accession_2]

        # compute hamming distance
        non_gap_pos = [pos for pos in range(len(seq_1)) if not (seq_1[pos] == seq_2[pos] == gap_code)]
        standardized_edit_dist = distance.hamming(seq_1, seq_2) * len(seq_1) / len(non_gap_pos)
        similarity = 1 - standardized_edit_dist

        return similarity

    @staticmethod
    def exec_mafft(input_path: str, output_path: str, threads_num: int = 1) -> int:
        """
        :param input_path: unaligned sequence data path
        :param output_path: aligned sequence data path
        :param threads_num: number of threads to use with mafft
        :return: return code
        """
        cmd = f"mafft --retree 1 --maxiterate 0 --thread {threads_num} {input_path} > {output_path}"
        res = os.system(cmd)
        if not os.path.exists(output_path):
            raise RuntimeError(f"failed to execute mafft on {input_path}")
        if res != 0:
            with open(output_path, "r") as outfile:
                outcontent = outfile.read()
            logger.error(f"failed mafft execution on {input_path} sequences from due to error {outcontent}")
        return res

    @staticmethod
    def compute_msa_based_similarity_values(alignment_path: str, similarities_output_path: str) -> pd.DataFrame:
        aligned_sequence_records = list(SeqIO.parse(alignment_path, format="fasta"))
        for record in aligned_sequence_records:
            record.seq = Seq(str(record.seq).lower())
        num_seq = len(aligned_sequence_records)
        aln_len = len(aligned_sequence_records[0].seq)

        logger.info(f"computing tokenized sequences for {num_seq} sequences of aligned length {aln_len}")
        unique_chars = list(set("".join([str(record.seq) for record in aligned_sequence_records])))
        char_to_int = {unique_chars[i]: i for i in range(len(unique_chars))}
        seq_id_to_array = {
            record.id: np.asarray([char_to_int[s] for s in str(record.seq)]) for record in aligned_sequence_records
        }
        gap_code = char_to_int["-"] if "-" in char_to_int else np.nan

        logger.info(f"computing msa-based pairwise similarities across {num_seq} sequences of aligned length {aln_len}")
        accessions = list(seq_id_to_array.keys())
        pair_to_similarity = pd.DataFrame(
            [(accessions[i], accessions[j]) for i in range(len(accessions)) for j in range(i + 1, len(accessions))],
            columns=["accession_1", "accession_2"],
        )
        pair_to_similarity["similarity"] = pair_to_similarity.apply(
            lambda x: ClusteringUtils.compute_similarity_across_aligned_sequences(
                record=x, seq_to_token=seq_id_to_array, gap_code=gap_code,
            ),
            axis=1,
        )
        pair_to_similarity.to_csv(similarities_output_path)
        return pair_to_similarity

    @staticmethod
    def get_sequence_similarity_with_multiple_alignment(sequence_data_path: str,) -> t.List[float]:

        mean_sim, min_sim, max_sim, med_sim = np.nan, np.nan, np.nan, np.nan

        if not os.path.exists(sequence_data_path):
            logger.info(f"input path {sequence_data_path} does not exist")
            return [mean_sim, min_sim, max_sim, med_sim]

        output_path = sequence_data_path.replace(".", "_aligned.")
        log_path = sequence_data_path.replace(".fasta", ".log")

        if not os.path.exists(output_path):
            num_sequences = len(list(SeqIO.parse(sequence_data_path, format="fasta")))
            if num_sequences > 8000:
                logger.info(
                    f"number of sequences = {num_sequences} is larger than 1000 and so the pipeline will be halted"
                )
                return [mean_sim, min_sim, max_sim, med_sim]

            logger.info(f"executing mafft on {num_sequences} sequences from {sequence_data_path}")
            res = ClusteringUtils.exec_mafft(input_path=sequence_data_path, output_path=output_path)
            if res != 0:
                return [mean_sim, min_sim, max_sim, med_sim]
            logger.info(f"aligned {num_sequences} sequences with mafft, in {output_path}")
            if os.path.exists(log_path):
                os.remove(log_path)
        similarities_output_path = sequence_data_path.replace(".fasta", "_similarity_values.csv")
        if not os.path.exists(similarities_output_path):
            pair_to_similarity = ClusteringUtils.compute_msa_based_similarity_values(
                alignment_path=output_path, similarities_output_path=similarities_output_path
            )
        else:
            pair_to_similarity = pd.read_csv(similarities_output_path)

        similarities = pair_to_similarity["similarity"]
        if pair_to_similarity.shape[0] > 0:
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
    def exec_cdhit(input_path: str, output_dir: str, homology_threshold: float = 0.95) -> str:
        """
        :param input_path: path to unaligned sequences in fasta format
        :param output_dir: directory to create output in
        :param homology_threshold: threshold for cdhit
        :return:
        """

        logger.info(f"cdhit input paths created at {output_dir}")
        cdhit_output_prefix = f"{output_dir}/cdhit_out_thr_{homology_threshold}"
        cdhit_log_file = f"{output_dir}/cdhit.log"
        if not os.path.exists(f"{cdhit_output_prefix}.clstr"):
            word_len = (
                (8 if homology_threshold > 0.7 else 4)
                if homology_threshold > 0.6
                else (3 if homology_threshold > 0.5 else 2)
            )
            logger.info(
                f"executing cdhit on {input_path} with homology threshold of {homology_threshold} and word length {word_len}"
            )
            cmd = f"{get_settings().CDHIT_DIR}cd-hit-est -i {input_path} -o {cdhit_output_prefix} -c {homology_threshold} -n {word_len} -M {output_dir} > {cdhit_log_file}"
            res = os.system(cmd)
            if res != 0:
                raise RuntimeError(f"CD-HIT failed to properly execute and provide an output file with error")

        return cdhit_output_prefix

    @staticmethod
    def get_cdhit_clusters(
        elements: pd.DataFrame,
        homology_threshold: float = 0.99,
        memory_limit: int = 6000,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
        return_cdhit_cluster_representative: bool = False,
    ) -> t.Dict[t.Union[np.int64, str], np.int64]:
        """
        :param elements: elements to cluster using cdhit
        :param homology_threshold: cdhit threshold in clustering
        :param memory_limit: memory limit in MB
        :param aux_dir: directory ot write output files of cdhit to
        :param return_cdhit_cluster_representative: indicator weather mapping
        to cluster id should be return or to the accession corresponding to
        the cluster representative chosen bt cdhit
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
        if not os.path.exists(cdhit_input_path) or not os.path.exists(names_translator_path):
            logger.info(
                f"either the input path {cdhit_input_path} or the aux path {names_translator_path} does not exist, so will create them"
            )
            for (index, row,) in elements.iterrows():
                elm = f"{row.accession}_{row.taxon_name}"
                seq = row["sequence"]
                elm_to_fake_name[elm] = f"S{i}"
                fake_name_to_elm[f"S{i}"] = elm
                elm_to_seq[elm] = seq
                i += 1

            with open(cdhit_input_path, "w") as infile:
                infile.write("\n".join([f">{elm_to_fake_name[elm]}\n{elm_to_seq[elm]}" for elm in elm_to_seq]))

            with open(names_translator_path, "wb") as infile:
                pickle.dump(obj=fake_name_to_elm, file=infile)

        cdhit_output_prefix = ClusteringUtils.exec_cdhit(
            input_path=cdhit_input_path, output_dir=aux_dir, homology_threshold=homology_threshold
        )

        elm_to_cluster = dict()
        clusters_data_path = f"{cdhit_output_prefix}.clstr"
        member_regex = re.compile(">(.*?)\.\.\.", re.MULTILINE | re.DOTALL)

        logger.info(f"parsing cdhit output using the auxiliary file {names_translator_path}")
        with open(names_translator_path, "rb") as infile:
            fake_name_to_elm = pickle.load(file=infile)

        logger.info(f"extracting cdhit clusters from {clusters_data_path}")
        accession_regex = re.compile("(.*?)_\D")
        with open(clusters_data_path, "r") as outfile:
            clusters = outfile.read().split(">Cluster")[1:]
            logger.info(f"{len(clusters)} clusters detected")
            for cluster in clusters:
                data = cluster.split("\n")
                cluster_id = np.int64(data[0])
                cluster_members = []
                for member_data in data[1:]:
                    if len(member_data) > 0:
                        member_fake_name = member_regex.search(member_data).group(1)
                        member = fake_name_to_elm[member_fake_name]
                        cluster_members.append(member)
                if return_cdhit_cluster_representative:
                    cluster_representative_full_name = cluster_members[0]
                    cluster_representative_accession = accession_regex.search(cluster_representative_full_name).group(1)
                    cluster_id = cluster_representative_accession
                elm_to_cluster.update({member: cluster_id for member in cluster_members})
                logger.info(f"cluster {clusters.index(cluster)} added to list with {len(cluster_members)} members")

        return elm_to_cluster

    @staticmethod
    def get_representative_by_msa(
        sequence_df: t.Optional[pd.DataFrame],
        unaligned_seq_data_path: str,
        aligned_seq_data_path: str,
        similarities_data_path: str,
    ) -> SeqRecord:
        """

        :param sequence_df: dataframe with sequence data of the element to get representative for
        :param unaligned_seq_data_path: path of unaligned sequence data file
        :param aligned_seq_data_path: path of aligned sequence data file
        :param similarities_data_path: oath of similarities values dataframe
        :return:
        """
        representative_record = np.nan

        if sequence_df is None and not os.path.exists(aligned_seq_data_path):
            logger.error(
                f"either data to compute similarities based on and nor computed similarity values were provided"
            )
            raise ValueError(
                f"either data to compute similarities based on and nor computed similarity values were provided"
            )

        if sequence_df is not None and sequence_df.shape[0] == 0:
            logger.error(f"no sequences in df to select representative from")
            return representative_record

        # write unaligned sequence data
        if sequence_df is not None and not os.path.exists(similarities_data_path):
            if not os.path.exists(unaligned_seq_data_path):
                sequence_data = [
                    SeqRecord(id=row.accession, name=row.accession, description=row.accession, seq=Seq(row.sequence))
                    for i, row in sequence_df.iterrows()
                    if pd.notna(row.sequence)
                ]
                if len(sequence_data) == 0:
                    return representative_record
                SeqIO.write(sequence_data, unaligned_seq_data_path, format="fasta")

            # align seq data
            if not os.path.exists(aligned_seq_data_path):
                res = ClusteringUtils.exec_mafft(input_path=unaligned_seq_data_path, output_path=aligned_seq_data_path)
                if res != 0:
                    return representative_record

            # compute similarity scores
            pairwise_similarities_df = ClusteringUtils.compute_msa_based_similarity_values(
                alignment_path=aligned_seq_data_path, similarities_output_path=similarities_data_path
            )
        else:
            pairwise_similarities_df = pd.read_csv(similarities_data_path)

        similarities_values_data = (
            pairwise_similarities_df.pivot_table(
                values="similarity", index="accession_1", columns="accession_2", aggfunc="first"
            )
            .reset_index()
            .rename(columns={"accession_1": "accession"})
        )
        representative_accession = similarities_values_data.set_index("accession").sum(axis=1).idxmax()
        representative_record = [
            record
            for record in list(SeqIO.parse(unaligned_seq_data_path, format="fasta"))
            if record.id == representative_accession
        ][0]
        return representative_record

    @staticmethod
    def collapse_redundant_sequences(
        elements: pd.DataFrame,
        homology_threshold: t.Optional[float] = 0.99,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
        mem_limit: int = 4000,
    ):
        """
        :param elements: elements to cluster using cdhit for the purpose of removing redundancy using cdhit
        :param homology_threshold: cdhit threshold in removing redundant sequences
        :param aux_dir: directory to write cdhit output files to
        :param mem_limit: memory allocation for cdhit
        :return: none, adds a column of "sequence_representative" to each column, with the accession selected by cdhit as the cluster representative
        as the sequences within each cluster are at least 99% similar, the choice of the cluster representative doesn't have to be wise
        """
        logger.info(
            f"removing redundancy across {elements.shape[0]} elements using cd-hit with a threshold of {homology_threshold}"
        )

        elm_to_cluster = ClusteringUtils.get_cdhit_clusters(
            elements=elements,
            homology_threshold=homology_threshold,
            aux_dir=aux_dir,
            memory_limit=mem_limit,
            return_cdhit_cluster_representative=True,
        )

        accession_regex = re.compile("(.*?)_\D")
        elements["sequence_representative"] = np.nan
        accession_to_cluster = {accession_regex.search(elm).group(1): elm_to_cluster[elm] for elm in elm_to_cluster}
        elements.set_index("accession", inplace=True)
        elements["sequence_representative"].fillna(value=accession_to_cluster, inplace=True)
        elements.reset_index(inplace=True)
        logger.info(f"representative of redundant sequences have been recorded")

    @staticmethod
    def compute_clusters_representatives(
        elements: pd.DataFrame,
        homology_threshold: t.Optional[float] = 0.99,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
        mem_limit: int = 4000,
    ):
        """
        :param elements: elements to cluster using cdhit
        :param homology_threshold: cdhit threshold in clustering
        :param aux_dir: directory to write cdhit output files to
        :param mem_limit: memory allocation for cdhit
        :return: none, adds cluster_id and cluster_representative columns to the existing elements dataframe
        """
        logger.info(f"computing clusters based on cd-hit for {elements.shape[0]} elements")

        elm_to_cluster = ClusteringUtils.get_cdhit_clusters(
            elements=elements, homology_threshold=homology_threshold, aux_dir=aux_dir, memory_limit=mem_limit,
        )
        logger.info("collected clusters data successfully, now merging ito associations data")
        accession_regex = re.compile("(.*?)_\D")
        elements["cluster_id"] = np.nan
        accession_to_cluster = {accession_regex.search(elm).group(1): elm_to_cluster[elm] for elm in elm_to_cluster}
        elements.set_index("accession", inplace=True)
        elements["cluster_id"].fillna(value=accession_to_cluster, inplace=True)
        elements.reset_index(inplace=True)
        logger.info(f"cluster ids synced")

        clusters = list(set(elm_to_cluster.values()))
        cluster_to_representative = dict()
        logger.info(f"extracting accession per cluster using centroid method")
        for cluster in clusters:
            cluster_members = elements.loc[elements.cluster_id == cluster]
            logger.info(f"extracting centroid for cluster {clusters.index(cluster)} of size {cluster_members.shape[0]}")
            if cluster_members.shape[0] == 0:
                logger.error(
                    f"cluster {cluster} has no taxa assigned to it\naccession_to_cluster={accession_to_cluster}\nelm_to_cluster={elm_to_cluster}"
                )
                exit(1)

            if cluster_members.shape[0] == 1:
                cluster_representative = cluster_members.iloc[0]["accession"]
            else:
                elements_distances = ClusteringUtils.compute_pairwise_sequence_distances(elements=cluster_members,)
                cluster_representative = ClusteringUtils.get_centroid(elements_distances)
            cluster_to_representative[cluster] = cluster_representative

        logger.info(f"cluster representatives extracted synced")

        elements["cluster_representative"] = np.nan
        elements.set_index("cluster_id", inplace=True)
        elements["cluster_representative"].fillna(value=cluster_to_representative, inplace=True)
        elements.reset_index(inplace=True)
        logger.info("cluster representatives synced")

    @staticmethod
    def get_centroid(elements_distances: pd.DataFrame) -> t.Union[np.int64, str]:
        """
        :param elements_distances: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        :return: the element id of the centroid
        """
        elements_sum_distances = elements_distances.groupby("element_1")["distance"].sum().reset_index()
        centroid = elements_sum_distances.iloc[elements_distances["distance"].argmin()]["element_1"]
        return centroid

    @staticmethod
    def cop_kmeans_with_initial_centers(
        dataset: np.ndarray,
        k: int,
        ml: t.List[t.Tuple[int]] = [],
        cl: t.List[t.Tuple[int]] = [],
        initial_centers: t.List[np.array] = [],
        initialization="kmpp",
        max_iter=300,
        tol=1e-4,
        write: bool = False,
        output_dir: str = os.getcwd(),
    ):
        """
        minor modification of the already package implemented cop_kmeans that enables providing a set of initial centers
        """

        ml, cl = transitive_closure(ml, cl, len(dataset))
        ml_info = get_ml_info(ml, dataset)
        tol = tolerance(tol, dataset)

        centers = initial_centers
        if len(centers) < k:
            try:
                centers = initialize_centers(dataset, k, initialization)
            except Exception as e:
                logger.warning(
                    f"failed to initialize centers for clustering with k={k} using initialization method {initialization} due to error {e}. will now attempt random initialization"
                )
                centers = initialize_centers(dataset, k, "random")

        clusters_, centers_ = np.nan, np.nan
        for _ in range(max_iter):
            clusters_ = [-1] * len(dataset)
            for i, d in enumerate(dataset):
                indices, _ = closest_clusters(centers, d)
                counter = 0
                if clusters_[i] == -1:
                    found_cluster = False
                    while (not found_cluster) and counter < len(indices):
                        index = indices[counter]
                        if not violate_constraints(i, index, clusters_, ml, cl):
                            found_cluster = True
                            clusters_[i] = index
                            for j in ml[i]:
                                clusters_[j] = index
                        counter += 1

                    if not found_cluster:
                        return None, None

            clusters_, centers_ = compute_centers(clusters_, dataset, k, ml_info)
            shift = sum(l2_distance(centers[i], centers_[i]) for i in range(k))
            if shift <= tol:
                break

            centers = centers_

        if write:
            clusters_output_path = f"{output_dir}/clusters.pickle"
            with open(clusters_output_path, "wb") as clusters_output_file:
                pickle.dump(obj=clusters_, file=clusters_output_file)
            centers_output_path = f"{output_dir}/centers.pickle"
            with open(centers_output_path, "wb") as centers_output_file:
                pickle.dump(obj=centers_, file=centers_output_file)

        return clusters_, centers_

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
        accessions_to_keep = ClusteringUtils.get_relevant_accessions_using_sequence_data_directly(
            data_path=alignment_path, similarity_cutoff=similarity_cutoff
        ).split(";;")
        unaligned_relevant_records = [record for record in aligned_sequence_records if record.id in accessions_to_keep]
        for record in unaligned_relevant_records:
            record.seq = Seq(str(record.seq).replace("-", ""))
        logger.info(f"after filtering, {len(unaligned_relevant_records)} sequences remain")
        SeqIO.write(unaligned_relevant_records, unaligned_output_path, format="fasta")
        logger.info(f"unaligned filtered data written to {unaligned_output_path}")
        res = ClusteringUtils.exec_mafft(input_path=unaligned_output_path, output_path=aligned_output_path)
        logger.info(f"aligned filtered data written to {aligned_output_path}")

    @staticmethod
    def compute_pairwise_sequence_distances(elements: pd.DataFrame,) -> pd.DataFrame:
        """
        :param elements: elements to compute pairwise distances for
        :return: a dataframe with row1 as element id, row 2 as element id and row3 ad the pairwise distance between the two elements correspond to ids in row1 and row2
        """

        elements_distances = pd.DataFrame(
            [(elm1, elm2) for elm1 in elements["accession"] for elm2 in elements["accession"]],
            columns=["element_1", "element_2"],
        )

        def get_distance(record: pd.Series, records_data: pd.DataFrame):
            seq1 = records_data.loc[records_data["accession"] == record["element_1"]]["sequence"].dropna().values[0]
            seq2 = records_data.loc[records_data["accession"] == record["element_2"]]["sequence"].dropna().values[0]
            return float(lev(seq1, seq2) / np.max([len(seq1), len(seq2)]))

        elements_distances["distance"] = elements_distances.apply(
            lambda x: get_distance(record=x, records_data=elements), axis=1,
        )

        return elements_distances

    @staticmethod
    def get_cdhit_cluster_members(clusters_path: str) -> t.List[t.List[str]]:
        """
        :param clusters_path: oath of cdhit clustering output
        :return: a list of cluster members within each cluster id (which corresponds to the list index)
        """
        with open(clusters_path, "r") as infile:
            clusters_data = [item.split("\n") for item in infile.read().split(">Cluster ")[1:]]
        cluster_member_regex = re.compile(">(\w*).")
        clusters = []
        for data in clusters_data:
            cluster_members = [cluster_member_regex.search(item).group(1) for item in data[1:]]
            clusters.append(cluster_members)
        return clusters

    @staticmethod
    def get_largest_cdhit_cluster(sequence_records: t.List[SeqRecord], workdir: str, homology_threshold: float = 0.95):
        """
        :param sequence_records: aligned sequence records
        :param workdir: directory to execute cdhit on unaligned records in, and select the ones in the largest cluster
        :param homology_threshold: threshold for cdhit execution
        :return: the accessions of the records within the largest cluster
        """
        os.makedirs(workdir, exist_ok=True)
        cdhit_input_path = f"{workdir}/cdhit_input.fasta"
        unaligned_sequence_records = sequence_records
        for record in unaligned_sequence_records:
            record.seq = Seq(str(record.seq).replace("-", ""))
        SeqIO.write(unaligned_sequence_records, cdhit_input_path, format="fasta")
        cdhit_output_prefix = ClusteringUtils.exec_cdhit(
            input_path=cdhit_input_path, output_dir=workdir, homology_threshold=homology_threshold
        )
        clusters = ClusteringUtils.get_cdhit_cluster_members(clusters_path=f"{cdhit_output_prefix}.clstr")
        return max(clusters, key=len)
