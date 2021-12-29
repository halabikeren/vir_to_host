# create family-wise alignment and use it to map the start position of each secondary structure in the species alignment to a start position in the family alignment,
# so that later on structures from different species within the family could be compared
import os
import pickle
import re
import typing as t

import click
import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

import sys

from Bio import SeqIO, Seq
from Bio.SeqRecord import SeqRecord

sys.path.append("..")
from utils.clustering_utils import ClusteringUtils
from utils.sequence_utils import SequenceCollectingUtils, AnnotationType

logger = logging.getLogger(__name__)


def create_group_wise_alignment(df: pd.DataFrame, seq_data_dir: str, group_wise_seq_path: str, group_wise_msa_path: str, representative_acc_to_sp_path: str) -> pd.DataFrame:
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
    if not os.path.exists(group_wise_seq_path):
        df["accession"] = np.nan
        species = df.virus_species_name.dropna().unique()
        for sp in species:
            sp_filename = re.sub('[^0-9a-zA-Z]+', '_', sp)
            representative_record = ClusteringUtils.get_representative_by_msa(sequence_df=None,
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

    # align written sequence data
    if not os.path.exists(group_wise_msa_path):
        logger.info(f"creating alignment from {group_wise_msa_path} at {group_wise_seq_path}")
        res = ClusteringUtils.exec_mafft(input_path=group_wise_seq_path, output_path=group_wise_msa_path)
        if res != 0:
            exit(1)

    return df


def get_unaligned_pos(aligned_pos: int, aligned_seq: Seq) -> int:
    unaligned_pos = aligned_pos - str(aligned_seq)[:aligned_pos].count("-")
    return unaligned_pos


def get_aligned_pos(unaligned_pos: int, aligned_seq: Seq) -> int:
    num_gaps = 0
    for pos in range(len(aligned_seq)):
        if str(aligned_seq)[pos] == "-":
            num_gaps += 1
        respective_unaligned_pos = pos - num_gaps
        if respective_unaligned_pos == unaligned_pos:
            return pos


def get_group_wise_positions(species_wise_start_pos: int, species_wise_end_pos: int, group_wise_msa_records: t.List[SeqRecord], species_wise_msa_records: t.List[SeqRecord], species_accession: str) -> t.Tuple[int, ...]:
    """
    :param species_wise_start_pos: start position in the species wise alignment
    :param species_wise_end_pos: end position in the species wise alignment
    :param group_wise_msa_records:
    :param species_wise_msa_records:
    :param species_accession:
    :return:
    """
    seq_from_group_wise_msa = [record for record in group_wise_msa_records if record.id == species_accession][0].seq
    seq_from_species_wise_msa = [record for record in species_wise_msa_records if record.id == species_accession][0].seq
    if str(seq_from_group_wise_msa).replace("-","").lower() != str(seq_from_species_wise_msa).replace("-","").lower():
        error_msg = f"sequence {species_accession} is inconsistent across the group-wise msa and species-wise msa"
        logger.error(error_msg)
        raise ValueError(error_msg)

    unaligned_start_pos = get_unaligned_pos(aligned_pos=species_wise_start_pos, aligned_seq=seq_from_species_wise_msa)
    unaligned_end_pos = get_unaligned_pos(aligned_pos=species_wise_end_pos, aligned_seq=seq_from_species_wise_msa)

    group_wise_start_pos = get_aligned_pos(unaligned_pos=unaligned_start_pos, aligned_seq=seq_from_group_wise_msa)
    group_wise_end_pos = get_aligned_pos(unaligned_pos=unaligned_end_pos, aligned_seq=seq_from_group_wise_msa)

    return tuple([unaligned_start_pos, group_wise_end_pos, group_wise_start_pos, group_wise_end_pos])


def map_species_wise_pos_to_group_wise_pos(df: pd.DataFrame, seq_data_dir: str, species_wise_msa_dir: str, workdir: str) -> pd.DataFrame:
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
    df = create_group_wise_alignment(df=df,
                                seq_data_dir=seq_data_dir,
                                group_wise_seq_path=group_wise_seq_path,
                                group_wise_msa_path=group_wise_msa_path,
                                representative_acc_to_sp_path=representative_acc_to_sp_path)
    group_wise_msa_records = list(SeqIO.parse(group_wise_msa_path, format="fasta"))
    with open(representative_acc_to_sp_path, "rb") as map_file:
        representative_acc_to_sp = pickle.load(map_file)
    sp_to_acc = {representative_acc_to_sp[acc]:acc for acc in representative_acc_to_sp}

    # for each species, map the species-wise start and end positions ot group wise start and end positions
    df.rename(columns={"struct_start_pos": "species_wise_struct_start_pos", "struct_end_pos": "species_wise_struct_end_pos"}, inplace=True)
    df[["unaligned_struct_start_pos",
        "unaligned_struct_end_pos",
        "group_wise_struct_start_pos",
        "group_wise_struct_end_pos"]] = df[["virus_species_name",
                                            "species_wise_struct_start_pos",
                                            "species_wise_struct_end_pos"]].apply(lambda row: get_group_wise_positions(species_wise_start_pos=row.species_wise_struct_start_pos,
                                                                                                                         species_wise_end_pos=row.species_wise_struct_end_pos,
                                                                                                                         group_wise_msa_records=group_wise_msa_records,
                                                                                                                         species_wise_msa_records=list(SeqIO.parse(f"{species_wise_msa_dir}/{re.sub('[^0-9a-zA-Z]+', '_', row.virus_species_name.values[0])}_aligned.fasta", format="fasta")),
                                                                                                                         species_accession = sp_to_acc[row.virus_species_name]),
                                                                                 axis=1, result_type="expand")
    return df


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

    max_struct_size = np.max(df.group_wise_struct_end_pos-df.group_wise_struct_start_pos)
    if partition_size < max_struct_size:
        logger.warning(f"selected partition size {partition_size} is smaller than the maximal structure size {max_struct_size} and thus will be changed to {max_struct_size}")
        partition_size = max_struct_size

    partitions = [(i, i+partition_size) for i in range(0, max_group_wise_pos, partition_size)]
    def get_assigned_partitions(start_pos: int, end_pos: int, used_partitions: t.List[t.Tuple[int, int]]) -> str:
        partition_of_start_pos = [partition for partition in used_partitions if
                                  partition[0] <= start_pos <= partition[1]][0]
        partition_of_end_pos = [partition for partition in used_partitions if
                                  partition[0] <= end_pos <= partition[1]][0]
        assigned_partitions = list({str(partition_of_start_pos), str(partition_of_end_pos)})
        return ";".join(assigned_partitions).replace(",","-").replace(" ","")
    df["assigned_partition"] = df[["group_wise_struct_start_pos", "group_wise_struct_end_pos"]].apply(lambda row: get_assigned_partitions(start_pos=row.group_wise_struct_start_pos,
                                                                                                                                        end_pos=row.group_wise_struct_end_pos,
                                                                                                                                          used_partitions=partitions),
                                                                                                      axis=1)

    return df


def get_assigned_annotations(struct_start_pos: int, struct_end_pos: int, accession_annotations: t.Dict[t.Tuple[str, AnnotationType], t.List[t.Tuple[int, int]]], intersection_annotations: t.Tuple[str, AnnotationType]) -> t.Tuple[str, str]:
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

    accession_assigned_annotations = str(accession_assigned_annotations).replace("), (","); (").replace(",","-").replace(" ","")
    assigned_annotations = str(assigned_annotations).replace("), (","); (").replace(",","-").replace(" ","")
    return [accession_assigned_annotations, assigned_annotations]


def assign_partition_by_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe of secondary structures to partition by annotations
    :return: dataframe with the assigned partition of each structure
    """

    accessions = list(df.accession.drpna().unique())
    accession_to_annotations = SequenceCollectingUtils.get_annotations(accessions=accessions)
    intersection_annotations = []
    for annotation in accession_to_annotations[accessions[0]]:
        is_intersection_annotation = True
        for accession in accessions[1:]:
            if accession in accession_to_annotations and annotation not in accession_to_annotations[accession]:
                is_intersection_annotation = False
                break
        if is_intersection_annotation:
            intersection_annotations.append(annotation)



    # assign annotations to each secondary structure based on its accession and annotation (be mindful of un-aligning the start and end positions when computing its location within the original accession)
    df[["assigned_accession_partitions", "assigned_partitions"]] = df[["accession", "unaligned_struct_start_pos", "unaligned_struct_end_pos"]].apply(lambda row: get_assigned_annotations(struct_start_pos=row.unaligned_struct_start_pos,
                                                                                                                                                                                         struct_end_pos=row.unaligned_struct_end_pos,
                                                                                                                                                                                         accession_annotations=accession_to_annotations[row.accession],
                                                                                                                                                                                         intersection_annotations=intersection_annotations),
                                                                                                                                                     axis=1,
                                                                                                                                                     result_type="expand")

    # name partitions according to intersection annotations, namely, ones that appear in all the accessions

    return df


@click.command()
@click.option(
    "--secondary_structures_df_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to dataframe holding the names of species for which data should be collected. this dataframe is  expected to hold both the virus_species_name and the grouping_field in it",
)
@click.option(
    "--grouping_field",
    type=str,
    help="column name to group data by",
    required=False,
    default="virus_family_name",
)
@click.option(
    "--sequence_data_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory holding the original sequence data before filtering out outliers. this directory should also hold similarity values tables per species",
)
@click.option(
    "--species_wise_msa_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory holding the aligned genomes per species after filtering out outliers, which were used in the inference process of the secondary structures",
)
@click.option(
    "--partition_by",
    type=click.Choice(["range", "annotation"]),
    help="indicator weather tp partition the data by constant ranges across the genome or by annotations (UTRs / CDS / ect.)",
    required=False,
    default="annotation"
)
@click.option(
    "--partition_size",
    type=click.IntRange(100, float("inf")),
    help="value to use as partition size, in case that partition_by=range",
    required=False,
    default=120,
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory to write family sequence data and align it in",
)
@click.option(
    "--log_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the logging of the script",
)
@click.option(
    "--df_output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output dataframe to write",
)
def partition_secondary_structures(
    secondary_structures_df_path: click.Path,
    grouping_field: str,
    sequence_data_dir: str,
    species_wise_msa_dir: str,
    partition_by: str,
    partition_size: int,
    workdir: str,
    log_path: click.Path,
    df_output_path: click.Path,
):

    # initialize the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path)),
        ],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    # create a family alignment by selecting genome representative per species within the family and aligning the representatives
    secondary_structures_df = pd.read_csv(secondary_structures_df_path)
    logger.info(f"applying partitioning pipeline for {secondary_structures_df.shape[0]} structures")
    if grouping_field not in secondary_structures_df.columns:
        error = f"grouping field {grouping_field} is absent from the dataframe columns at {secondary_structures_df_path}"
        logger.error(error)
        raise ValueError(error)
    if "virus_species_name" not in secondary_structures_df.columns:
        error_msg = f"virus_species_name is absent from the dataframe columns at {secondary_structures_df_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    secondary_structures_groups = secondary_structures_df.groupby(grouping_field)
    logger.info(f"dataframe consists of {len(secondary_structures_groups.groups.keys())} groups of {grouping_field}")
    output_dfs = []
    for g in secondary_structures_groups.groups.keys():
        if g == "flaviviridae": # remove after analysis to generalize to all viral families

            logger.info(f"handling group {g}")
            df = secondary_structures_groups.get_group(g)

            # map the start position of each secondary structure from its original value (determined by the species-wise alignment) to its new value (determined by the family-wise alignment) in the output df
            logger.info(f"mapping species-wise structures positions to group-wise positions")
            df = map_species_wise_pos_to_group_wise_pos(df=df,
                                                   seq_data_dir=sequence_data_dir,
                                                   species_wise_msa_dir=species_wise_msa_dir,
                                                   workdir=f"{workdir}/{g}/")

            # obtain the annotations of each species in the family, and the range of each annotation within the representative genome (throw away annotations that do not appear in all the species within the family)
            logger.info(f"assigning partitions to structures based on {partition_by}")
            if partition_by == "range":
                df = assign_partition_by_size(df=df, partition_size=partition_size)
            elif partition_by == "annotation":
                df = assign_partition_by_annotation(df=df)
            output_dfs.append(df)

    output_df = pd.concat(output_dfs)
    output_df.to_csv(df_output_path, index=False)


if __name__ == '__main__':
    partition_secondary_structures()