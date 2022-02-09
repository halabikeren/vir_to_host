import os
import pickle
import re
import typing as t
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from copkmeans.cop_kmeans import *

import logging

from utils.programs.mafft import Mafft

logger = logging.getLogger(__name__)


class RNAStructsClusteringUtils:

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
    def assign_partition_by_size(df: pd.DataFrame, partition_size: int) -> pd.DataFrame:
        """
        :param df: dataframe of secondary structures
        :param partition_size: size of required partitions
        :return: dataframe with the assigned partitions of each secondary structure
        """
        partition_size = int(partition_size)
        max_group_wise_pos = int(np.max(df.group_wise_struct_end_pos))
        if partition_size > max_group_wise_pos:
            error_msg = (
                f"partition size {partition_size} is larger than the total number of positions {max_group_wise_pos}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        aligned_max_struct_size = int(np.max(df.group_wise_struct_end_pos - df.group_wise_struct_start_pos))
        if partition_size < aligned_max_struct_size:
            logger.warning(
                f"selected partition size {partition_size} is smaller than the maximal structure size {aligned_max_struct_size} and thus will be changed to {aligned_max_struct_size}"
            )
            partition_size = aligned_max_struct_size

        # partition by group-wise alignment while considering the amount of gaps in the alignment within the partition size
        if max_group_wise_pos < partition_size * 2:  # there are no two complete partitions across the data
            logger.warning(
                f"selected partition size doesn't fit to 2 or more windows within the alignment and so a single partition will be used"
            )
            partitions = [(0, max_group_wise_pos)]
            df["assigned_partition"] = f"(0, {max_group_wise_pos})"
        else:
            partitions = [
                (i, np.min([i + partition_size, max_group_wise_pos]))
                for i in range(0, max_group_wise_pos, partition_size)
            ]

            def get_assigned_partitions(
                start_pos: int, end_pos: int, used_partitions: t.List[t.Tuple[int, int]]
            ) -> t.List[str]:
                partition_of_start_pos = [
                    partition for partition in used_partitions if partition[0] <= start_pos <= partition[1]
                ][0]
                partition_of_end_pos = [
                    partition for partition in used_partitions if partition[0] <= end_pos <= partition[1]
                ][0]
                assigned_partitions = list({str(partition_of_start_pos), str(partition_of_end_pos)})
                return [partition.replace(",", "-").replace(" ", "") for partition in assigned_partitions]

            df["assigned_partition"] = df[["group_wise_struct_start_pos", "group_wise_struct_end_pos"]].apply(
                lambda row: get_assigned_partitions(
                    start_pos=row.group_wise_struct_start_pos,
                    end_pos=row.group_wise_struct_end_pos,
                    used_partitions=partitions,
                ),
                axis=1,
            )

            df = df.explode(
                "assigned_partition"
            )  # for a record assigned to two partitions, duplicate its corresponding row

        return df

    @staticmethod
    def get_assigned_annotations(
        structure_alignment_path: str, species_alignment_path: str, species_annotation_data: pd.DataFrame,
    ) -> t.Tuple[str, str]:
        """
        :param structure_alignment_path: path to the mlocarna alignment based on which the structure was predicted
        :param species_alignment_path: path to species-wise genomic alignment
        :param species_annotation_data: dataframe with the annotations assigned to all annotated accessions of the species
        :return: two lists of the relevant accessions and the structure's annotations within these accessions - to be exploded later outside the scope of this function
        """
        # get the accessions based on which the structure was predicted
        structure_sequence_records = list(SeqIO.parse(structure_alignment_path, format="fasta"))
        structure_accessions = [record.id.split("/")[0] for record in structure_sequence_records]

        # get the annotation for these accessions, if available
        structure_accessions_annotation_data = species_annotation_data.loc[
            species_annotation_data.accession.isin(structure_accessions)
        ]
        if structure_accessions_annotation_data.shape[0] == 0:
            return np.nan, np.nan
        annotated_structure_accessions = list(structure_accessions_annotation_data.accession.unique())

        # compute the unaligned start and end positions of the structure within the respective accessions, by mapping the aligned position in the structure_alignment_path to the unaligned position of each respective accession, based on the species-wise MSA
        species_aln_sequence_records = list(SeqIO.parse(species_alignment_path, format="fasta"))
        accession_to_struct_annotations = dict()
        accession_to_struct_sequence = {
            record.id.split("/")[0]: re.search("(\w*)", str(record.seq)).group(1).lower().replace("u", "t")
            for record in structure_sequence_records
        }
        accession_to_complete_sequence = {record.id: str(record.seq).lower() for record in species_aln_sequence_records}
        for accession in annotated_structure_accessions:
            if accession in accession_to_struct_sequence and accession in accession_to_complete_sequence:
                structure_aln_sequence = accession_to_struct_sequence[accession]
                unaligned_structure_sequence = structure_aln_sequence.replace("-", "")
                species_aln_sequence = accession_to_complete_sequence[accession]
                unaligned_complete_sequence = species_aln_sequence.replace("-", "")
                unaligned_start_position = unaligned_complete_sequence.find(unaligned_structure_sequence)
                unaligned_end_position = unaligned_start_position + len(unaligned_structure_sequence)

                # find all the annotations that the structure falls within their range
                accession_annotations = species_annotation_data.loc[species_annotation_data.accession == accession]

                def extract_coord_pos(coord):
                    coord_pos_regex = re.compile("\d+")
                    coord_pos = [int(item) for item in coord_pos_regex.findall(coord.values[0])]
                    return tuple([coord_pos[0], coord_pos[-1]])

                structure_annotations = []
                accession_annotations = (
                    accession_annotations.groupby(["annotation_union_name", "annotation_type"])["union_coordinate"]
                    .apply(extract_coord_pos)
                    .to_dict()
                )
                for annotation in accession_annotations:
                    if (
                        unaligned_start_position >= accession_annotations[annotation][0]
                        and unaligned_end_position <= accession_annotations[annotation][-1]
                    ):
                        structure_annotations.append(annotation)
                    if (
                        accession_annotations[annotation][0]
                        <= unaligned_start_position
                        <= accession_annotations[annotation][-1]
                    ):
                        logger.warning(
                            f"structure within range ({unaligned_start_position}, {unaligned_end_position}) starts at {annotation} of range ({accession_annotations[annotation][0]}, {accession_annotations[annotation][1]}) but ends outside its scope"
                        )
                        structure_annotations.append(annotation)
                    elif (
                        accession_annotations[annotation][-1]
                        >= unaligned_end_position
                        >= accession_annotations[annotation][0]
                    ):
                        logger.warning(
                            f"structure within range ({unaligned_start_position}, {unaligned_end_position}) ends at {annotation} of range ({accession_annotations[annotation][0]}, {accession_annotations[annotation][1]}) but starts outside its scope"
                        )
                        structure_annotations.append(annotation)
                if len(structure_annotations) > 0:
                    accession_to_struct_annotations[accession] = structure_annotations

        relevant_accessions = list(accession_to_struct_annotations.keys())
        relevant_annotations = []
        for acc in relevant_accessions:
            relevant_annotations += accession_to_struct_annotations[acc]
        if len(relevant_annotations) == 0:
            logger.info(
                f"structure within range ({unaligned_start_position}, {unaligned_end_position}) has no assigned annotation"
            )
        return (relevant_accessions, relevant_annotations)

    @staticmethod
    def assign_partition_by_annotation(
        df: pd.DataFrame, annotation_data_path: str, alignments_dir: str,
    ) -> t.Dict[t.Tuple[str, str], pd.DataFrame]:
        """
        :param df: dataframe of secondary structures to partition by annotations
        :param annotation_data_path: path to annotation data in csv format
        :param alignments_dir: directory of the aligned species-wise sequence data
        :return: dictionary mapping each annotation (represented by a tuple of name and type) to a dataframe with the records whose assigned partition correspond to it
        """
        annotation_data = pd.read_csv(annotation_data_path)

        # assign annotations to each secondary structure based on its accession and annotation (be mindful of un-aligning the start and end positions when computing its location within the original accession)
        intersection_annotations = list(
            annotation_data.groupby(["annotation_union_name", "annotation_type"]).groups.keys()
        )

        df[["assigned_accession_partitions", "assigned_partitions"]] = df.parallel_apply(
            lambda row: RNAStructsClusteringUtils.get_assigned_annotations(
                structure_alignment_path=row.struct_src_aln_path,
                species_alignment_path=f"{alignments_dir}/{re.sub('[^0-9a-zA-Z]+', '_', row.virus_species_name)}_aligned.fasta",
                species_annotation_data=annotation_data.loc[annotation_data.species_name == row.virus_species_name],
            ),
            axis=1,
            result_type="expand",
        )

        #  create a dictionary of dataframes - one per annotation, with all the records corresponding to it
        annotation_to_structural_data = dict()
        structures_annotations = list(df["assigned_partitions"])
        for annotation in intersection_annotations:
            relevant_indices = [
                i
                for i in range(len(structures_annotations))
                if np.all(pd.notna(structures_annotations[i])) and annotation in structures_annotations[i]
            ]
            if len(relevant_indices) > 0:
                annotation_to_structural_data[annotation] = df.iloc[relevant_indices]

        return annotation_to_structural_data
