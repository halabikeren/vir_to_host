import logging
import os
import re
import sys
from enum import Enum

import click
from Bio import SeqIO
from Bio.Seq import Seq
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


def clean_sequence_data_from_outliers(
    record: pd.Series, input_path: str, output_path: str
):
    """
    :param record: pandas row representative of a cluster of species sequences
    :param input_path: path to the aligned sequences that include outliers
    :param output_path: path to create in aligned sequences without the outliers
    (without re-aligning - just removing outliers and then cleainnig the induced alignment from only gap positions)
    :return:
    """
    if pd.notna(record.relevant_genome_accessions):
        selected_accessions = record.relevant_genome_accessions.split(";;")
        input_sequences = list(SeqIO.parse(input_path, format="fasta"))
        relevant_sequences = [
            seq for seq in input_sequences if seq.id in selected_accessions
        ]
        pos_number = len(str(relevant_sequences[0].seq))
        pos = 0
        while pos < pos_number:
            pos_components = list(
                set(
                    [
                        str(relevant_sequences[i].seq)[pos]
                        for i in range(len(relevant_sequences))
                    ]
                )
            )
            if len(pos_components) == 1 and pos_components[0] == "-":
                for i in range(len(relevant_sequences)):
                    relevant_sequences[i].seq = Seq(
                        "".join(
                            [
                                str(relevant_sequences[i].seq)[p]
                                for p in range(pos_number)
                                if p != pos
                            ]
                        )
                    )
                    pos_number -= 1
            else:
                pos += 1
        SeqIO.write(relevant_sequences, output_path, format="fasta")


def compute_sequence_similarities_across_species(
    species_info: pd.DataFrame,
    seq_data_dir: str,
    output_path: str,
    use_sequence_directly: bool = True,
):
    """
    :param species_info: data with the names of viruses corresponding to each viral species and the number of available sequences
    :param seq_data_dir: directory holding fasta files of collected sequences per species to compute similarity based on
    :param output_path: path to write the output dataframe to
    :param use_sequence_directly: indicator weather outliers should be removed based on the sequence data directly or based on their pairwise distances
    :return:
    """
    relevant_species_info = species_info.loc[
        species_info.virus_species_name.isin(
            species_info.virus_species_name.unique()
        )
    ]
    if (
        relevant_species_info.shape[0] > 0
        and relevant_species_info["#sequences"].values[0] > 0
    ):
        logger.info(
            f"computing sequence similarities across {len(species_info.virus_species_name)} species"
        )

        intermediate_output_path = output_path.replace(".", "_intermediate.")
        if (
            os.path.exists(intermediate_output_path)
            and relevant_species_info["#sequences"].values[0] > 2
        ):
            relevant_species_info = pd.read_csv(intermediate_output_path)
        else:
            if relevant_species_info.shape[0] > 0:
                logger.info(
                    f"computing sequence similarity value for species {','.join(relevant_species_info.virus_species_name.unique())}"
                )
                relevant_species_info = compute_entries_sequence_similarities(
                    df=relevant_species_info,
                    seq_data_dir=seq_data_dir,
                    output_path=output_path.replace(".", "_intermediate."),
                )
        if (
            "relevant_genome_accessions" not in relevant_species_info.columns
            or "#relevant_sequences" not in relevant_species_info.columns
        ) or (
            relevant_species_info.loc[
                relevant_species_info.relevant_genome_accessions.isna()
            ].shape[0]
            > 0
        ):
            logger.info(
                f"computing outlier sequences for species {relevant_species_info.virus_species_name.unique()}"
            )
            relevant_species_info = remove_outliers(
                df=relevant_species_info,
                similarities_data_dir=seq_data_dir,
                output_path=output_path.replace(".", "_intermediate."),
                use_sequence_directly=use_sequence_directly,
            )

        # create new alignments without the outliers
        new_seq_data_dir = f"{seq_data_dir}/no_outliers/"
        os.makedirs(new_seq_data_dir, exist_ok=True)

        relevant_species_info.loc[relevant_species_info["#sequences"] > 1].apply(
            lambda record: clean_sequence_data_from_outliers(
                record=record,
                input_path=f"{seq_data_dir}/{re.sub('[^0-9a-zA-Z]+', '_', record.virus_species_name)}_aligned.fasta",
                output_path=f"{new_seq_data_dir}/{re.sub('[^0-9a-zA-Z]+', '_', record.virus_species_name)}_aligned.fasta",
            ),
            axis=1,
        )
        sequence_similarity_fields = [
            "#sequences",
            "mean_sequence_similarity",
            "min_sequence_similarity",
            "max_sequence_similarity",
            "med_sequence_similarity",
            "relevant_genome_accessions",
            "#relevant_sequences",
        ]
        relevant_species_info["#relevant_sequences"] = relevant_species_info[
            "relevant_genome_accessions"
        ].apply(lambda x: x.count(";") + 1 if pd.notna(x) else np.nan)
        species_info.set_index("virus_species_name", inplace=True)
        for field in sequence_similarity_fields:
            species_info[field] = np.nan
            species_info[field].fillna(
                value=relevant_species_info.set_index("virus_species_name")[
                    field
                ].to_dict(),
                inplace=True,
            )

        species_info.reset_index(inplace=True)
    else:
        species_info["#sequences"] = 0
        species_info["#relevant_sequences"] = 0

    species_info.to_csv(output_path, index=False)
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
    :param mem_limit: RAM in MB that should be allocated to cdhit
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
        logger.info(
            f"computing sequence similarities for #species {len(new_df.virus_species_name.values)} that consists of {new_df['#sequences'].values} sequences respectively"
        )

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
            lambda x: [1, 1, 1, 1]
            if x["#sequences"] == 1
            else func(
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
    use_sequence_directly: bool = False,
) -> pd.DataFrame:
    """
    :param df: dataframe with association entries
    :param similarities_data_dir: directory with similarity dataframes corresponding ot each species with its corresponding collected sequences
    :param output_path: path to write the intermediate result to
    :param use_sequence_directly: indicator weather outlier detection should use the sequence data directly or use the pairwise distances etween sequences as features
    :return:
    """
    pid = os.getpid()
    tqdm.pandas(desc="worker #{}".format(pid), position=pid)

    if not os.path.exists(output_path) or (
        os.path.exists(output_path)
        and "relevant_genome_accessions" not in pd.read_csv(output_path).columns
    ):
        new_df = df
        new_df["relevant_genome_accessions"] = np.nan
        if new_df.shape[0] > 0:
            logger.info(
                f"computing sequence outliers for for species {list(new_df.virus_species_name.unique())[0]} that consists of {','.join(list([str(i) for i in new_df['#sequences'].values]))} sequences respectively"
            )

            func = (
                ClusteringUtils.get_relevant_accessions_using_sequence_data_directly
                if use_sequence_directly
                else ClusteringUtils.get_relevant_accessions_using_pairwise_distances
            )
            input_path_suffix = (
                "_aligned.fasta" if use_sequence_directly else "_similarity_values.csv"
            )
            new_df.loc[
                new_df["#sequences"] > 1, "relevant_genome_accessions"
            ] = new_df.loc[
                new_df["#sequences"] > 1, "virus_species_name"
            ].progress_apply(
                lambda x: func(
                    data_path=f"{similarities_data_dir}/{re.sub('[^0-9a-zA-Z]+', '_', x)}{input_path_suffix}"
                )
            )
            new_df["#relevant_sequences"] = new_df["relevant_genome_accessions"].apply(
                lambda x: x.count(";") + 1 if pd.notna(x) else np.nan
            )

        new_df.to_csv(output_path, index=False)
    else:
        new_df = pd.read_csv(output_path)
    return new_df


@click.command()
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
    "--use_sequence_directly",
    type=click.BOOL,
    help="indicator weather outliers should be removed based on sequence data directly or based on pairwise distances",
    required=False,
    default=False,
)
def compute_seq_similarities(
    species_info_path: click.Path,
    sequence_data_dir: click.Path,
    log_path: click.Path,
    df_output_path: click.Path,
    use_sequence_directly: bool,
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

    # process input data
    species_info = pd.read_csv(species_info_path)

    # compute sequence similarities
    compute_sequence_similarities_across_species(
        species_info=species_info,
        seq_data_dir=str(sequence_data_dir),
        output_path=str(df_output_path),
        use_sequence_directly=use_sequence_directly,
    )


if __name__ == "__main__":
    compute_seq_similarities()
