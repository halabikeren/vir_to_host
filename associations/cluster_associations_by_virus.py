import logging
import multiprocessing
import os
import sys
from functools import partial
from multiprocessing import current_process
from tqdm import tqdm

tqdm.pandas()
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

sys.path.append("..")
from utils.clustering_utils import ClusteringUtils
from utils.parallelization_service import ParallelizationService

workdir = "/groups/itay_mayrose/halabikeren/vir_to_host/data/"
logger_path = f"{workdir}/cluster_associations_by_virus.log"
debug_mode = logging.DEBUG
associations_data_path = f"{workdir}/associations_united.csv"
viral_sequence_data_path = f"{workdir}/virus_sequence_data.csv"
associations_by_virus_species_path = f"{workdir}/associations_by_virus_species.csv"
associations_by_virus_cluster_path = (
    f"{workdir}/associations_by_virus_cluster_0.8_seq_homology.csv"
)
clustering_threshold = 0.8
mem_limit = 4000  # in MB


def concat(x):
    return ",".join(list(set([str(val) for val in x.dropna().values])))


def compute_entries_sequence_similarities(
    df: pd.DataFrame, seq_data: pd.DataFrame
) -> str:
    """
    :param df: dataframe with association entries
    :param seq_data: dataframe with sequences
    :return:
    """
    pid = int(current_process().name.split("-")[1])
    df_path = f"{os.getcwd()}/df_{compute_entries_sequence_similarities.__name__}_pid_{pid}.csv"
    tqdm.pandas(desc="worker #{}".format(pid), position=pid)

    new_df = df
    new_df["sequence_similarity"] = new_df.progress_apply(
        lambda x: ClusteringUtils.get_sequences_similarity(
            viruses_names=x["virus_taxon_name"],
            viral_seq_data=seq_data,
            mem_limit=mem_limit,
        ),
        axis=1,
    )
    new_df.to_csv(df_path)
    return df_path


def plot_seqlen_distribution(
    associations_df: pd.DataFrame, virus_sequence_df: pd.DataFrame, output_dir: str
):
    # plot dist of sequences lengths
    associations_vir_data = associations_df[
        [col for col in associations_df.columns if "virus_" in col and "_name" in col]
    ].drop_duplicates()
    taxonomic_units = [
        unit
        for unit in associations_vir_data.columns
        if unit not in ["virus_taxon_name", "virus_strain_name"]
    ]
    print("taxonomic_unit\t#values")
    for unit in taxonomic_units:
        print(f"{unit}\t{len(associations_df[unit].unique())}")

    taxonomic_unit_to_seqlen_df = dict()
    for unit in taxonomic_units:
        seqlen_df = pd.DataFrame(
            columns=[
                "taxonomic_unit_value",
                "#viruses",
                "#sequences",
                "mean_len",
                "min_len",
                "max_len",
                "median_len",
                "var_len",
                "cv_len",
            ]
        )
        for taxonomic_unit_value in associations_vir_data[unit].unique():
            viruses_names = list(
                associations_vir_data.loc[
                    associations_vir_data[unit] == taxonomic_unit_value,
                    "virus_taxon_name",
                ].unique()
            )
            seq_data_match = virus_sequence_df.loc[
                virus_sequence_df.taxon_name.isin(viruses_names)
            ][["sequence"]]
            sequences_data = seq_data_match.values.flatten()
            sequences_lengths = [len(s) for s in sequences_data if type(s) is str]
            record = {
                "taxonomic_unit_value": taxonomic_unit_value,
                "#viruses": len(viruses_names),
                "#sequences": len(sequences_lengths),
                "mean_len": np.mean(sequences_lengths)
                if len(sequences_lengths) > 0
                else np.nan,
                "min_len": np.min(sequences_lengths)
                if len(sequences_lengths) > 0
                else np.nan,
                "max_len": np.max(sequences_lengths)
                if len(sequences_lengths) > 0
                else np.nan,
                "median_len": np.median(sequences_lengths)
                if len(sequences_lengths) > 0
                else np.nan,
                "var_len": np.var(sequences_lengths)
                if len(sequences_lengths) > 0
                else np.nan,
                "cv_len": np.var(sequences_lengths) / np.mean(sequences_lengths)
                if len(sequences_lengths) > 0
                else np.nan,
            }
            seqlen_df = seqlen_df.append(record, ignore_index=True)
        taxonomic_unit_to_seqlen_df[unit] = seqlen_df

    # write dataframes to output dir
    for taxonomic_unit in taxonomic_unit_to_seqlen_df:
        taxonomic_unit_to_seqlen_df[taxonomic_unit].to_csv(
            f"{output_dir}/seqlen_dist_{taxonomic_unit}.csv",
            index=False,
        )


def cluster_by_species(
    associations_df: pd.DataFrame, virus_sequence_df: pd.DataFrame, output_path: str
):
    if not os.path.exists(output_path):
        associations_by_virus_species = (
            associations_df.groupby(["virus_species_name", "host_taxon_id"])
            .agg(
                {
                    col: concat
                    for col in associations_df.columns
                    if col not in ["virus_species_name", "host_taxon_id"]
                }
            )
            .reset_index()
        )
    else:
        associations_by_virus_species = pd.read_csv(output_path)
        for col in ["Unnamed: 0", "index", "df_index"]:
            if col in virus_sequence_df.columns:
                virus_sequence_df.drop(col, axis=1, inplace=True)

        # collect sequence similarity data
    species_info = associations_by_virus_species.drop_duplicates(
        subset=["virus_species_name"]
    )
    species_info = ParallelizationService.parallelize(
        df=species_info,
        func=partial(compute_entries_sequence_similarities, seq_data=virus_sequence_df),
        num_of_processes=multiprocessing.cpu_count() - 1,
    )
    associations_by_virus_species["sequence_similarity"] = np.nan
    associations_by_virus_species.set_index("virus_species_name", inplace=True)
    associations_by_virus_species["sequence_similarity"].fillna(
        value=species_info.set_index("virus_species_name")["sequence_similarity"],
        inplace=True,
    )
    associations_by_virus_species.reset_index(inplace=True)
    associations_by_virus_species.to_csv(output_path, index=False)
    logger.info(f"wrote associations data clustered by virus species to {output_path}")


def cluster_by_sequence_homology(
    associations_df: pd.DataFrame, virus_sequence_df: pd.DataFrame, output_path: str
):
    if not os.path.exists(output_path):
        logger.info("creating associations_by_virus_cluster")
        sequence_colnames = [
            "virus_refseq_sequence",
            "virus_genbank_sequence",
            "virus_gi_sequences",
        ]
        virus_sequence_df.dropna(subset=sequence_colnames, how="all", inplace=True)
        ClusteringUtils.compute_clusters_representatives(
            elements=virus_sequence_df,
            id_colname="virus_taxon_name",
            seq_colnames=sequence_colnames,
            homology_threshold=clustering_threshold,
        )
        virus_to_cluster_id = virus_sequence_df.set_index("virus_taxon_name")[
            "cluster_id"
        ].to_dict()
        virus_to_representative = virus_sequence_df.set_index("virus_taxon_name")[
            "cluster_representative"
        ].to_dict()
        associations_df.set_index("virus_taxon_name", inplace=True)
        associations_df["virus_cluster_id"] = np.nan
        associations_df["virus_cluster_id"].fillna(
            value=virus_to_cluster_id, inplace=True
        )
        associations_df["virus_cluster_representative"] = np.nan
        associations_df["virus_cluster_representative"].fillna(
            value=virus_to_representative, inplace=True
        )
        associations_df.reset_index(inplace=True)
        associations_by_virus_cluster = (
            associations_df.groupby(
                ["virus_cluster_id", "virus_cluster_representative", "host_taxon_name"]
            )
            .agg(
                {
                    c: concat
                    for c in associations_df.columns
                    if c
                    not in [
                        "virus_cluster_id",
                        "virus_cluster_representative",
                        "host_taxon_name",
                    ]
                }
            )
            .reset_index()
        )
        associations_by_virus_cluster.to_csv(output_path, index=False)
        logger.info(
            f"wrote associations data clustered by sequence homology at threshold 0.99 to {associations_by_virus_cluster_path}"
        )


if __name__ == "__main__":

    # initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logger_path),
        ],
    )

    associations = pd.read_csv(associations_data_path)
    for col in ["Unnamed: 0", "index", "df_index"]:
        if col in associations.columns:
            associations.drop(col, axis=1, inplace=True)
    virus_sequence_data = pd.read_csv(viral_sequence_data_path)
    for col in ["Unnamed: 0", "index", "df_index"]:
        if col in virus_sequence_data.columns:
            virus_sequence_data.drop(col, axis=1, inplace=True)
    # limit sequence data to genomes
    virus_sequence_data = virus_sequence_data.loc[
        virus_sequence_data.category == "genome"
    ]

    # remove from associations viruses with missing sequence data
    viruses_with_no_seq_data = virus_sequence_data.loc[
        (virus_sequence_data.sequence.isna()),
        "taxon_name",
    ].unique()
    associations = associations.loc[
        ~associations.virus_taxon_name.isin(viruses_with_no_seq_data)
    ]
    associations.to_csv(
        associations_data_path.replace(".csv", "_only_viruses_with_seq_data.csv"),
        index=False,
    )

    plot_seqlen_distribution(
        associations_df=associations,
        virus_sequence_df=virus_sequence_data,
        output_dir=os.path.dirname(associations_by_virus_species_path),
    )

    # group associations by virus_species_name
    cluster_by_species(
        associations_df=associations,
        virus_sequence_df=virus_sequence_data,
        output_path=associations_by_virus_species_path,
    )

    # group associations by sequence homology
    cluster_by_sequence_homology(
        associations_df=associations,
        virus_sequence_df=virus_sequence_data,
        output_path=associations_by_virus_species_path,
    )
