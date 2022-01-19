import itertools
import logging
import os
import pickle
import re
import typing as t
from enum import Enum

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from matplotlib import patches, pyplot as plt
from scipy.spatial import distance
import pandas as pd
import numpy as np
import psutil
from Bio import SeqIO
from scipy.stats import chi2
from Levenshtein import distance as lev
from copkmeans.cop_kmeans import *

from settings import get_settings

logger = logging.getLogger(__name__)

from Bio.Data import CodonTable
NUCLEOTIDES = ["A", "C", "G", "T"]
AMINO_ACIDS = list(set(CodonTable.standard_dna_table.forward_table.values())) + ["O","S","U","T","W","Y","V","B","Z","X","J"]


class ClusteringMethod(Enum):
    CDHIT = 1


class ClusteringUtils:
    @staticmethod
    def compute_outliers_with_mahalanobis_dist(
        data: pd.DataFrame, data_dist_plot_path: str
    ) -> t.Union[t.List[int], float]:
        """
        :param data: numeric dataframe with features based on which outliers should be removed
        :param data_dist_plot_path: path to write to a plot with the distribution of the data points
        :return: list of the indices of the outlier data points
        taken from https://towardsdatascience.com/multivariate-outlier-detection-in-python-e946cfc843b3
        """
        data = data.to_numpy()
        try:
            det = np.linalg.det(data)
            if det == 0:
                logger.error(
                    f"unable to compute outliers due data matrix with zero determinant, returning nan"
                )
                return np.nan
        except Exception as e:  # data is not squared
            pass
        distances = []
        centroid = np.mean(data, axis=0)
        covariance = np.cov(data, rowvar=False)
        covariance_pm1 = np.linalg.pinv(covariance)
        for i, val in enumerate(data):
            if type(val) != str:
                p1 = np.float64(val)
                p2 = np.float64(centroid)
                dist = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
                distances.append(dist)
        distances = np.array(distances)
        # Cutoff (threshold) value from Chi-Square Distribution for detecting outliers
        cutoff = chi2.ppf(0.95, data.shape[1])
        # Index of outliers
        outlier_indexes = list(np.where(distances > cutoff)[0])

        # compute statistics
        pearson = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        lambda_, v = np.linalg.eig(covariance)
        lambda_ = np.sqrt(lambda_)

        # report data
        logger.info(
            f"centroid={centroid}\ncutoff={cutoff}\noutlier_indexes={outlier_indexes}\nell_radius=({ell_radius_x},{ell_radius_y})"
        )

        # plot records distribution
        ellipse = patches.Ellipse(
            xy=(centroid[0], centroid[1]),
            width=lambda_[0] * np.sqrt(cutoff) * 2,
            height=lambda_[1] * np.sqrt(cutoff) * 2,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            edgecolor="#fab1a0",
        )
        ellipse.set_facecolor("#0984e3")
        ellipse.set_alpha(0.5)
        fig = plt.figure()
        ax = plt.subplot()
        ax.add_artist(ellipse)
        plt.scatter(data[:, 0], data[:, 1])
        plt.xlabel("similarity to accession 1", fontsize=16)
        plt.ylabel("similarity to accession 2", fontsize=16)
        fig.savefig(data_dist_plot_path, transparent=True)

        return outlier_indexes

    @staticmethod
    def compute_outliers_with_euclidean_dist(
        data: pd.DataFrame, data_dist_plot_path: str
    ) -> t.Union[t.List[int], float]:
        similarities = data.to_numpy()
        distances = np.mean(1 - similarities, axis=1)
        cutoff = np.max([np.percentile(distances, 95), 0.15])
        outlier_indexes = list(np.where(distances > cutoff)[0])

        # plot records distribution - this is projection of the first 2 dimensions only and is thus not as reliable
        circle = patches.Circle(
            xy=(1, 1),
            radius=np.max(cutoff),
            edgecolor="#fab1a0",
        )
        circle.set_facecolor("#0984e3")
        circle.set_alpha(0.5)
        fig = plt.figure()
        ax = plt.subplot()
        ax.add_artist(circle)
        plt.scatter(similarities[:, 0], similarities[:, 1])
        fig.savefig(data_dist_plot_path, transparent=True)

        return outlier_indexes

    @staticmethod
    def get_relevant_accessions_using_sequence_data_directly(
        data_path: str,
    ) -> t.Union[str, int]:
        """
        :param data_path: an alignment of sequences
        :return: string of the list of relevant accessions that were not identified as outliers, separated by ";;"
        """
        if not os.path.exists(data_path):
            logger.info(f"alignment fie {data_path} does not exist")
            return np.nan
        sequence_records = list(SeqIO.parse(data_path, format="fasta"))
        if len(sequence_records) < 3:
            return ";;".join([record.description for record in sequence_records])

        nuc_regex = re.compile("[ACGT-]*")
        if len(str(sequence_records[0].seq)) == len(nuc_regex.match(str(sequence_records[0].seq)).group(0)):
            chars = NUCLEOTIDES
        else:
            chars = AMINO_ACIDS
        char_to_int = {chars[i].upper(): i for i in range(len(chars))}
        char_to_int.update({chars[i].lower(): i for i in range(len(chars))})
        char_to_int.update({"-": len(chars), 'X': len(chars)+1, 'x': len(chars)+1})

        acc_to_seq = {
            record.description: [char_to_int[char] for char in record.seq]
            for record in sequence_records
        }
        data = pd.DataFrame({"accession": list(acc_to_seq.keys())})
        data["sequence"] = data["accession"].apply(func=lambda acc: acc_to_seq[acc])
        data[
            [f"pos_{pos}" for pos in range(len(sequence_records[0].seq))]
        ] = pd.DataFrame(data.sequence.tolist(), index=data.index)

        use_alternative_metric = False
        outliers_idx = []
        try:
            outliers_idx = ClusteringUtils.compute_outliers_with_mahalanobis_dist(
                data=data[
                    [f"pos_{pos}" for pos in range(len(sequence_records[0].seq))]
                ],
                data_dist_plot_path=data_path.replace(
                    "_aligned.fasta", "_mahalanobis.png"
                ),
            )
            if pd.isna(outliers_idx):
                use_alternative_metric = True
        except Exception as e:
            logger.info(
                f"unable to compute mahalanobis distance based outliers indices due to error {e}, will attempt computation using euclidean distance over pairwise similarities"
            )
            use_alternative_metric = True

        if use_alternative_metric:
            logger.info(
                "unable to compute mahalanobis distance based outliers indices, will attempt computation using euclidean distance over pairwise similarities"
            )
            pairwise_similarities_df = ClusteringUtils.get_pairwise_similarities_df(
                input_path=data_path.replace("_aligned.fasta", "_similarity_values.csv")
            )
            outliers_idx = []
            if pairwise_similarities_df.shape[0] > 1:
                outliers_idx = ClusteringUtils.compute_outliers_with_euclidean_dist(
                    data=pairwise_similarities_df[
                        [
                            col
                            for col in pairwise_similarities_df.columns
                            if "similarity_to" in col
                        ]
                    ],
                    data_dist_plot_path=data_path.replace(
                        "_aligned.fasta", "_euclidean.png"
                    ),
                )
        accessions = list(data.accession)
        accessions_to_keep = [
            accessions[idx] for idx in range(len(accessions)) if idx not in outliers_idx
        ]
        logger.info(
            f"{len(accessions_to_keep)} accessions remain after removing {len(outliers_idx)} outliers\naccessions {','.join([acc for acc in accessions if acc not in accessions_to_keep])} were determined as outliers"
        )
        return ";;".join(accessions_to_keep)

    @staticmethod
    def get_pairwise_similarities_df(input_path: str) -> pd.DataFrame:
        similarities_df = pd.read_csv(input_path)

        accessions_data = (
            similarities_df.pivot_table(
                values="similarity",
                index="accession_1",
                columns="accession_2",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"accession_1": "accession"})
        )
        accessions_data.rename(
            columns={
                col: f"similarity_to_{col}"
                for col in accessions_data.columns
                if col != "accession"
            },
            inplace=True,
        )
        accessions_data["mean_similarity_from_rest"] = accessions_data[
            [col for col in accessions_data.columns if col != "accession"]
        ].apply(lambda x: np.mean(x), axis=1)

        logger.info(
            f"computed similarities table across {accessions_data.shape[0]} accessions"
        )
        return accessions_data

    @staticmethod
    def get_relevant_accessions_using_pairwise_distances(
        data_path: str,
    ) -> str:
        """
        :param data_path: path to a dataframe matching a similarity value to each pair of accessions
        :return: string of the list of relevant accessions that were not identified as outliers, separated by ";;"
        """

        accessions_data = ClusteringUtils.get_pairwise_similarities_df(
            input_path=data_path
        )

        outliers_idx = []
        if accessions_data.shape[0] > 2:
            outliers_idx = ClusteringUtils.compute_outliers_with_euclidean_dist(
                data=accessions_data[
                    [col for col in accessions_data.columns if "similarity_to" in col]
                ],
                data_dist_plot_path=data_path.replace(".csv", "_euclidean.png"),
            )

        accessions = list(accessions_data.accession)
        accessions_to_keep = [
            accessions[idx] for idx in range(len(accessions)) if idx not in outliers_idx
        ]
        logger.info(
            f"{len(accessions_to_keep)} accessions remain after removing {len(outliers_idx)} outliers\naccessions {[acc for acc in accessions if acc not in accessions_to_keep]} were determined as outliers"
        )
        return ";;".join(accessions_to_keep)

    @staticmethod
    def compute_similarity_across_aligned_sequences(
        record: pd.Series, seq_to_token: t.Dict[str, np.array]
    ) -> float:
        if record.accession_1 == record.accession_2:
            return 1
        seq_1 = seq_to_token[record.accession_1]
        seq_2 = seq_to_token[record.accession_2]
        similarity = 1 - distance.hamming(seq_1, seq_2)
        logger.info(
            f"similarity({record.accession_1}, {record.accession_2})={similarity}"
        )
        return similarity

    @staticmethod
    def exec_mafft(input_path: str, output_path: str, nthreads: int = 1) -> int:
        """
        :param input_path: unaligned sequence data path
        :param output_path: aligned sequence data path
        :param nthreads: number of threads to use with mafft
        :return: return code
        """
        cmd = (
            f"mafft --retree 1 --maxiterate 0 --thread {nthreads} {input_path} > {output_path}"
        )
        res = os.system(cmd)
        if not os.path.exists(output_path):
            raise RuntimeError(f"failed to execute mafft on {input_path}")
        if res != 0:
            with open(output_path, "r") as outfile:
                outcontent = outfile.read()
            logger.error(
                f"failed mafft execution on {input_path} sequences from due to error {outcontent}"
            )
        return res

    @staticmethod
    def compute_pairwise_similarity_values(alignment_path: str, similarities_output_path: str) -> pd.DataFrame:
        aligned_sequences = list(SeqIO.parse(alignment_path, format="fasta"))
        nuc_regex = re.compile("[ACGT-]*")
        if len(str(aligned_sequences[0].seq)) == len(nuc_regex.match(str(aligned_sequences[0].seq)).group(0)):
            chars = NUCLEOTIDES
        else:
            chars = AMINO_ACIDS
        char_to_int = {chars[i].upper(): i for i in range(len(chars))}
        char_to_int.update({chars[i].lower(): i for i in range(len(chars))})
        char_to_int.update({"-": len(chars), 'X': len(chars)+1, 'x': len(chars)+1})
        logger.info(
            f"computing tokenized sequences for {len(aligned_sequences)} sequences of aligned length {len(aligned_sequences[0].seq)}"
        )
        seq_id_to_array = dict()
        for record in aligned_sequences:
            try:
                seq = str(record.seq)
                numerical_seq = np.asarray([char_to_int[s] for s in seq])
                seq_id_to_array[record.id] = numerical_seq
            except Exception as e:
                logger.error(
                    f"failed to convert sequence {record.id} due to error {e} and so it will be ignored"
                )
                continue
        logger.info(
            f"computing pairwise similarities across {len(aligned_sequences)} sequences of aligned length {len(aligned_sequences[0].seq)}"
        )
        pair_to_similarity = pd.DataFrame(
            [
                (acc1, acc2)
                for acc1 in seq_id_to_array.keys()
                for acc2 in seq_id_to_array.keys()
            ],
            columns=["accession_1", "accession_2"],
        )
        pair_to_similarity["similarity"] = pair_to_similarity.apply(
            lambda x: ClusteringUtils.compute_similarity_across_aligned_sequences(
                record=x, seq_to_token=seq_id_to_array
            ),
            axis=1,
        )
        pair_to_similarity.to_csv(similarities_output_path, index=False)
        return pair_to_similarity

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
            if num_sequences > 8000:
                logger.info(f"number of sequences = {num_sequences} is larger than 1000 and so the pipeline will be halted")
                return [mean_sim, min_sim, max_sim, med_sim]

            logger.info(
                f"executing mafft on {num_sequences} sequences from {sequence_data_path}"
            )
            res = ClusteringUtils.exec_mafft(input_path=sequence_data_path, output_path=output_path)
            if res != 0:
                return [mean_sim, min_sim, max_sim, med_sim]
            logger.info(
                f"aligned {num_sequences} sequences with mafft, in {output_path}"
            )
            if os.path.exists(log_path):
                os.remove(log_path)
        similarities_output_path = sequence_data_path.replace(
            ".fasta", "_similarity_values.csv"
        )
        if not os.path.exists(similarities_output_path):
            pair_to_similarity = ClusteringUtils.compute_pairwise_similarity_values(alignment_path=output_path, similarities_output_path=similarities_output_path)
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
        if len(sequences) > 2060:
            logger.info(f"number of sequences = {len(sequences)} is larger than 1000 and so the pipeline will be halted")
            return [np.nan, np.nan, np.nan, np.nan]
        logger.info(
            f"computing pairwise similarities across {len(sequences)} sequences, meaning, {int(len(sequences) ** 2 / 2)} comparisons"
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
        return_cdhit_cluster_representative: bool = False,
    ) -> t.Dict[t.Union[np.int64, str], np.int64]:
        """
        :param elements: elements to cluster using kmeans
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
            cmd = f"{get_settings().CDHIT_DIR}cd-hit-est -i {cdhit_input_path} -o {cdhit_output_file} -c {homology_threshold} -n {word_len} -M {memory_limit} > {cdhit_log_file}"
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
                    cluster_representative_accession = accession_regex.search(
                        cluster_representative_full_name
                    ).group(1)
                    cluster_id = cluster_representative_accession
                elm_to_cluster.update(
                    {member: cluster_id for member in cluster_members}
                )
                logger.info(
                    f"cluster {clusters.index(cluster)} added to list with {len(cluster_members)} members"
                )

        return elm_to_cluster

    @staticmethod
    def get_representative_by_msa(sequence_df: t.Optional[pd.DataFrame], unaligned_seq_data_path: str, aligned_seq_data_path: str, similarities_data_path: str) -> SeqRecord:
        """

        :param sequence_df: dataframe with sequence data of the element to get representative for
        :param unaligned_seq_data_path: path of unaligned sequence data file
        :param aligned_seq_data_path: path of aligned sequence data file
        :param similarities_data_path: oath of similarities values dataframe
        :return:
        """
        representative_record = np.nan

        if sequence_df is None and not os.path.exists(aligned_seq_data_path):
            logger.error(f"either data to compute similarities based on and nor computed similarity values were provided")
            raise ValueError(f"either data to compute similarities based on and nor computed similarity values were provided")

        if sequence_df is not None and sequence_df.shape[0] == 0:
            logger.error(f"no sequences in df to select representative from")
            return representative_record


        # write unaligned sequence data
        if not os.path.exists(similarities_data_path):
            if not os.path.exists(unaligned_seq_data_path):
                sequence_data = [SeqRecord(id=row.accession, name=row.accession, description=row.accession, seq=Seq(row.sequence)) for i, row in sequence_df.iterrows() if pd.notna(row.sequence)]
                if len(sequence_data) == 0:
                    return representative_record
                SeqIO.write(sequence_data, unaligned_seq_data_path, format="fasta")

            # align seq data
            if not os.path.exists(aligned_seq_data_path):
                res = ClusteringUtils.exec_mafft(input_path=unaligned_seq_data_path, output_path=aligned_seq_data_path)
                if res != 0:
                    return representative_record

            # compute similarity scores
            pairwise_similarities_df = ClusteringUtils.compute_pairwise_similarity_values(
                alignment_path=aligned_seq_data_path, similarities_output_path=similarities_data_path)
        else:
            pairwise_similarities_df = pd.read_csv(similarities_data_path)


        similarities_values_data = pairwise_similarities_df.pivot_table(
            values="similarity",
            index="accession_1",
            columns="accession_2",
            aggfunc="first").reset_index().rename(columns={"accession_1": "accession"})
        representative_accession = similarities_values_data.set_index("accession").sum(axis=1).idxmax()
        representative_record = [record for record in list(SeqIO.parse(unaligned_seq_data_path, format="fasta")) if record.id == representative_accession][0]
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
        accession_to_cluster = {
            accession_regex.search(elm).group(1): elm_to_cluster[elm]
            for elm in elm_to_cluster
        }
        elements.set_index("accession", inplace=True)
        elements["sequence_representative"].fillna(
            value=accession_to_cluster, inplace=True
        )
        elements.reset_index(inplace=True)
        logger.info(f"representative of redundant sequences have been recorded")

    @staticmethod
    def compute_clusters_representatives(
        elements: pd.DataFrame,
        clustering_method: ClusteringMethod = ClusteringMethod.CDHIT,
        homology_threshold: t.Optional[float] = 0.99,
        aux_dir: str = f"{os.getcwd()}/cdhit_aux/",
        mem_limit: int = 4000,
    ):
        """
        :param elements: elements to cluster using cdhit
        :param clustering_method: either cdhit or kmeans
        :param homology_threshold: cdhit threshold in clustering
        :param aux_dir: directory to write cdhit output files to
        :param mem_limit: memory allocation for cdhit
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
                memory_limit=mem_limit,
            )
        else:
            logger.error(f"clustering method {clustering_method} is not implemented")
            raise ValueError(
                f"clustering method {clustering_method} is not implemented"
            )
        logger.info(
            "collected clusters data successfully, now merging ito associations data"
        )
        accession_regex = re.compile("(.*?)_\D")
        elements["cluster_id"] = np.nan
        accession_to_cluster = {
            accession_regex.search(elm).group(1): elm_to_cluster[elm]
            for elm in elm_to_cluster
        }
        elements.set_index("accession", inplace=True)
        elements["cluster_id"].fillna(value=accession_to_cluster, inplace=True)
        elements.reset_index(inplace=True)
        logger.info(f"cluster ids synced")

        clusters = list(set(elm_to_cluster.values()))
        cluster_to_representative = dict()
        logger.info(f"extracting accession per cluster using centroid method")
        for cluster in clusters:
            cluster_members = elements.loc[elements.cluster_id == cluster]
            logger.info(
                f"extracting centroid for cluster {clusters.index(cluster)} of size {cluster_members.shape[0]}"
            )
            if cluster_members.shape[0] == 0:
                logger.error(
                    f"cluster {cluster} has no taxa assigned to it\naccession_to_cluster={accession_to_cluster}\nelm_to_cluster={elm_to_cluster}"
                )
                exit(1)

            if cluster_members.shape[0] == 1:
                cluster_representative = cluster_members.iloc[0]["accession"]
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

        logger.info(f"cluster representatives extracted synced")

        elements["cluster_representative"] = np.nan
        elements.set_index("cluster_id", inplace=True)
        elements["cluster_representative"].fillna(
            value=cluster_to_representative, inplace=True
        )
        elements.reset_index(inplace=True)
        logger.info("cluster representatives synced")

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
    def get_distance(record: pd.Series, records_data: pd.DataFrame):
        elm1 = record["element_1"]
        elm2 = record["element_2"]
        try:
            elm1_seq = (
                records_data.loc[records_data["accession"] == elm1]["sequence"]
                .dropna()
                .values[0]
            )
            elm2_seq = (
                records_data.loc[records_data["accession"] == elm2]["sequence"]
                .dropna()
                .values[0]
            )
            return ClusteringUtils.get_pairwise_alignment_distance(elm1_seq, elm2_seq)
        except Exception as e:
            logger.error(
                f"failed to compute pairwise distance between {elm1} and {elm2} due to error {e}"
            )
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
                for elm1 in elements["accession"]
                for elm2 in elements["accession"]
            ],
            columns=["element_1", "element_2"],
        )

        elements_distances["distance"] = elements_distances.apply(
            lambda x: ClusteringUtils.get_distance(record=x, records_data=elements),
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

    @staticmethod
    def cop_kmeans_with_initial_centers(dataset: np.ndarray, k: int, ml: t.List[t.Tuple[int]] = [], cl: t.List[t.Tuple[int]] =[],
                                        initial_centers: t.List[np.array] = [], initialization='kmpp',
                                        max_iter=300, tol=1e-4, write: bool = False, output_dir: str = os.getcwd()):
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
                logger.warning(f"failed to initalize centers for clustering with k={k} using intialization method {initialization} due to error {e}. will now attempt random initialization")
                centers = initialize_centers(dataset, k, 'random')

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
