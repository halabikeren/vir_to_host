import os
import re

import pandas as pd
import numpy as np
import typing as t

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.spatial import distance
from Levenshtein import distance as lev

from utils.programs.cdhit import CdHit
from utils.programs.mafft import Mafft

import logging

logger = logging.getLogger(__name__)


class SequenceClusteringUtils:
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
            func=(
                lambda x: SequenceClusteringUtils.compute_similarity_across_aligned_sequences(
                    record=x, seq_to_token=seq_id_to_array, gap_code=gap_code,
                )
            ),
            axis=1,
        )
        pair_to_similarity.to_csv(similarities_output_path)
        return pair_to_similarity

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
            res = Mafft.exec_mafft(input_path=sequence_data_path, output_path=output_path)
            if res != 0:
                return [mean_sim, min_sim, max_sim, med_sim]
            logger.info(f"aligned {num_sequences} sequences with mafft, in {output_path}")
            if os.path.exists(log_path):
                os.remove(log_path)
        similarities_output_path = sequence_data_path.replace(".fasta", "_similarity_values.csv")
        if not os.path.exists(similarities_output_path):
            pair_to_similarity = SequenceClusteringUtils.compute_msa_based_similarity_values(
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
    def get_mean_similarity_across_elements(elements: pd.DataFrame,) -> pd.DataFrame:
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

        if os.path.exists(unaligned_seq_data_path):
            sequence_records = list(SeqIO.parse(unaligned_seq_data_path, format="fasta"))
            if len(sequence_records) == 1:
                return sequence_records[0]
            elif len(sequence_records) == 2:
                if len(sequence_records[0].seq) > len(sequence_records[1].seq):
                    return sequence_records[0]
                return sequence_records[1]

        representative_record = np.nan
        if (sequence_df is None or not os.path.exists(unaligned_seq_data_path)) and not os.path.exists(
            aligned_seq_data_path
        ):
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
                res = Mafft.exec_mafft(input_path=unaligned_seq_data_path, output_path=aligned_seq_data_path)
                if res != 0:
                    return representative_record

            # compute similarity scores
            pairwise_similarities_df = SequenceClusteringUtils.compute_msa_based_similarity_values(
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

        elm_to_cluster = CdHit.get_cdhit_clusters(
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
                elements_distances = SequenceClusteringUtils.get_mean_similarity_across_elements(
                    elements=cluster_members,
                )
                cluster_representative = SequenceClusteringUtils.get_centroid(elements_distances)
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
