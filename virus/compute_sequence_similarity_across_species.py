import logging
import os
import re
import sys
from enum import Enum

import click
from tqdm import tqdm

tqdm.pandas()
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

sys.path.append("..")
from utils.clustering_utils import ClusteringUtils


class SimilarityComputationMethod(Enum):
    CDHIT = 0
    MSA = 1
    PAIRWISE = 2


def compute_sequence_similarities_across_species(
    associations_by_virus_species: pd.DataFrame,
    species_info: pd.DataFrame,
    seq_data_dir: str,
    output_path: str,
    keep_threshold: float = 0.8,
):
    """
    :param associations_by_virus_species: df to add sequence similarity measures to
    :param species_info: data with the names of viruses corresponding to each viral species and the number of available sequences
    :param seq_data_dir: directory holding fasta files of collected sequences per species to compute similarity based on
    :param output_path: path to write the output dataframe to
    :param keep_threshold: parameter for determining the similarity percentile threshold of accessions that are selected to be included in a cluster
    :return:
    """
    relevant_species_info = species_info.loc[
        species_info.virus_species_name.isin(
            associations_by_virus_species.virus_species_name.unique()
        )
    ]
    relevant_species_info = compute_entries_sequence_similarities(
        df=relevant_species_info,
        seq_data_dir=seq_data_dir,
        output_path=output_path.replace(".", "_intermediate."),
    )

    relevant_species_info = remove_outliers(
        df=relevant_species_info,
        similarities_data_dir=seq_data_dir,
        output_path=output_path.replace(".", "_intermediate."),
        keep_threshold=keep_threshold,
    )

    associations_by_virus_species.set_index("virus_species_name", inplace=True)
    sequence_similarity_fields = [
        "#sequences",
        "mean_sequence_similarity",
        "min_sequence_similarity",
        "max_sequence_similarity",
        "med_sequence_similarity",
        "relevant_genome_accessions",
    ]
    for field in sequence_similarity_fields:
        associations_by_virus_species[field] = np.nan
        associations_by_virus_species[field].fillna(
            value=relevant_species_info.set_index("virus_species_name")[
                field
            ].to_dict(),
            inplace=True,
        )

    associations_by_virus_species.reset_index(inplace=True)
    associations_by_virus_species.to_csv(output_path, index=False)
    logger.info(f"wrote associations data clustered by virus species to {output_path}")


def compute_entries_sequence_similarities(
    df: pd.DataFrame,
    seq_data_dir: str,
    output_path: str,
    similarity_computation_method: SimilarityComputationMethod = SimilarityComputationMethod.MSA,
) -> pd.DataFrame:
    """
    :param df: dataframe with association entries
    :param seq_data_dir: directory with fasta file corresponding ot each species with its corresponding collected sequences
    :param output_path: path to write the intermediate result to
    :param similarity_computation_method: indicator of the method that should be employed to compute the similarity values
    :return:
    """
    pid = os.getpid()
    tqdm.pandas(desc="worker #{}".format(pid), position=pid)

    new_df = df
    new_df[
        [
            "mean_sequence_similarity",
            "min_sequence_similarity",
            "max_sequence_similarity",
            "med_sequence_similarity",
        ]
    ] = np.nan
    if new_df.shape[0] > 0:
        logger.info(f"computing sequence similarity across {new_df.shape[0]} species")

        func = (
            ClusteringUtils.get_sequences_similarity_with_pairwise_alignments
            if similarity_computation_method == SimilarityComputationMethod.PAIRWISE
            else (
                ClusteringUtils.get_sequences_similarity_with_cdhit
                if similarity_computation_method == SimilarityComputationMethod.CDHIT
                else ClusteringUtils.get_sequence_similarity_with_multiple_alignment
            )
        )
        new_df[
            [
                "mean_sequence_similarity",
                "min_sequence_similarity",
                "max_sequence_similarity",
                "med_sequence_similarity",
            ]
        ] = new_df.progress_apply(
            lambda x: func(
                sequence_data_path=f"{seq_data_dir}/{re.sub('[^0-9a-zA-Z]+', '_', x.virus_species_name)}.fasta",
            ),
            axis=1,
            result_type="expand",
        )

    new_df.to_csv(output_path, index=False)
    return new_df


def remove_outliers(
    df: pd.DataFrame,
    similarities_data_dir: str,
    output_path: str,
    keep_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    :param df: dataframe with association entries
    :param similarities_data_dir: directory with similarity dataframes corresponding ot each species with its corresponding collected sequences
    :param output_path: path to write the intermediate result to
    :param keep_threshold: similarity threshold for keeping accessions in a cluster
    :return:
    """
    pid = os.getpid()
    tqdm.pandas(desc="worker #{}".format(pid), position=pid)

    new_df = df
    new_df[
        [
            "mean_sequence_similarity",
            "min_sequence_similarity",
            "max_sequence_similarity",
            "med_sequence_similarity",
        ]
    ] = np.nan
    if new_df.shape[0] > 0:
        logger.info(f"computing sequence similarity across {new_df.shape[0]} species")

        func = ClusteringUtils.get_relevant_accessions_from_multiple_alignment
        new_df["relevant_genome_accessions"] = new_df[
            "virus_species_name"
        ].progress_apply(
            lambda x: func(
                similarities_data_path=f"{similarities_data_dir}/{re.sub('[^0-9a-zA-Z]+', '_', x)}_similarity_values.csv",
                keep_threshold=keep_threshold,
            )
        )

    new_df.to_csv(output_path, index=False)
    return new_df


@click.command()
@click.option(
    "--associations_by_species_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="input path, holding associations grouped by viral species",
)
@click.option(
    "--species_info_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to dataframe holding the names of taxa under each viral species",
)
@click.option(
    "--sequence_data_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory holding sequence data files per species with their collected sequences",
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
@click.option(
    "--keep_threshold",
    type=click.FloatRange(0, 1),
    help="percentile similarity threshold for keeping accessions in a cluster",
    required=False,
    default=0.8,
)
def compute_seq_similarities(
    associations_by_species_path: click.Path,
    species_info_path: click.Path,
    sequence_data_dir: click.Path,
    log_path: click.Path,
    df_output_path: click.Path,
    keep_threshold: float,
):

    # initialize the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

    # process input data
    associations_by_virus_species = pd.read_csv(associations_by_species_path)
    species_info = pd.read_csv(species_info_path)

    # compute sequence similarities
    compute_sequence_similarities_across_species(
        associations_by_virus_species=associations_by_virus_species,
        species_info=species_info,
        seq_data_dir=str(sequence_data_dir),
        output_path=str(df_output_path),
        keep_threshold=keep_threshold,
    )


if __name__ == "__main__":
    compute_seq_similarities()
