import logging
import multiprocessing
import os
import sys
from functools import partial
from multiprocessing import cpu_count, current_process
from tqdm import tqdm
#tqdm.pandas()
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

sys.path.append("..")
from utils.data_collecting_utils import ClusteringUtils
from utils.parallelization_service import ParallelizationService

logger_path = "./cluster_associations_by_virus.log"
debug_mode = logging.DEBUG
associations_data_path = "../data/associations_united.csv"
viral_sequence_data_path = "../data/virus_data.csv"
associations_by_virus_species_path = "../data/associations_by_virus_species.csv"
associations_by_virus_cluster_path = "../data/associations_by_virus_cluster_0.8_seq_homology.csv"
clustering_threshold = 0.8
mem_limit = 4000 # in MB

def concat(x):
    return ",".join(list(set([str(val) for val in x.dropna().values])))


def compute_entries_sequence_similarities(df: pd.DataFrame, seq_data: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe with association entries
    :param seq_data: dataframe with sequences
    :return:
    """
    pid = int(current_process().name.split('-')[1])
    tqdm.pandas(desc='worker #{}'.format(pid), position=pid)

    new_df = df
    new_df["sequence_similarity"] = new_df.progress_apply(
        lambda x: ClusteringUtils.get_sequences_similarity(
            viruses_names=x["virus_taxon_name"], viral_seq_data=seq_data, mem_limit = mem_limit
        ),
        axis=1,
    )
    return new_df

if __name__ == "__main__":

    # initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logger_path), ],
    )

    associations = pd.read_csv(associations_data_path)
    for col in ["Unnamed: 0", "index", "df_index"]:
        if col in associations.columns:
            associations.drop(col, axis=1, inplace=True)
    virus_sequence_data = pd.read_csv(viral_sequence_data_path)
    for col in ["Unnamed: 0", "index", "df_index"]:
        if col in virus_sequence_data.columns:
            virus_sequence_data.drop(col, axis=1, inplace=True)

    # group associations by virus_species_name
    if not os.path.exists(associations_by_virus_species_path):
        associations_by_virus_species = associations.groupby(
            ["virus_species_name", "host_taxon_id"]
        ).agg(
            {
                col: concat
                for col in associations.columns
                if col not in ["virus_species_name", "host_taxon_id"]
            }
        ).reset_index()
    else:
        associations_by_virus_species = pd.read_csv(associations_by_virus_species_path)
        for col in ["Unnamed: 0", "index", "df_index"]:
            if col in virus_sequence_data.columns:
                virus_sequence_data.drop(col, axis=1, inplace=True)

    # collect sequence similarity data
    species_info = associations_by_virus_species.drop_duplicates(subset=["virus_species_name"])
    species_info = ParallelizationService.parallelize(df=species_info, func=partial(compute_entries_sequence_similarities, seq_data=virus_sequence_data), num_of_processes=multiprocessi.cpu_count())
    associations_by_virus_species["sequence_similarity"] = np.nan
    associations_by_virus_species.set_index("virus_species_name", inplace=True)
    associations_by_virus_species["sequence_similarity"].fillna(value=species_info.set_index("virus_species_name")["sequence_similarity"], inplace=True)
    associations_by_virus_species.reset_index(inplace=True)
    associations_by_virus_species.to_csv(
        associations_by_virus_species_path, index=False
    )
    logger.info(f"wrote associations data clustered by virus species to {associations_by_virus_species_path}")

    # group associations by sequence homology
    if not os.path.exists(associations_by_virus_cluster_path):
        logger.info("creating associations_by_virus_cluster")
        sequence_colnames = [
            "virus_refseq_sequence",
            "virus_genbank_sequence",
            "virus_gi_sequences",
        ]
        virus_sequence_data.dropna(subset=sequence_colnames, how="all", inplace=True)
        ClusteringUtils.compute_clusters_representatives(
            elements=virus_sequence_data,
            id_colname="virus_taxon_name",
            seq_colnames=sequence_colnames,
            homology_threshold=clustering_threshold
        )
        virus_to_cluster_id = virus_sequence_data.set_index("virus_taxon_name")[
            "cluster_id"
        ].to_dict()
        virus_to_representative = virus_sequence_data.set_index("virus_taxon_name")[
            "cluster_representative"
        ].to_dict()
        associations.set_index("virus_taxon_name", inplace=True)
        associations["virus_cluster_id"] = np.nan
        associations["virus_cluster_id"].fillna(value=virus_to_cluster_id, inplace=True)
        associations["virus_cluster_representative"] = np.nan
        associations["virus_cluster_representative"].fillna(
            value=virus_to_representative, inplace=True
        )
        associations.reset_index(inplace=True)
        associations_by_virus_cluster = associations.groupby(
            ["virus_cluster_id", "virus_cluster_representative", "host_taxon_name"]
        ).agg(
            {
                col: concat
                for col in associations.columns
                if col not in ["virus_cluster_id", "virus_cluster_representative", "host_taxon_name"]
            }
        ).reset_index()
        associations_by_virus_cluster.to_csv(
            associations_by_virus_cluster_path, index=False
        )
        logger.info(
            f"wrote associations data clustered by sequence homology at threshold 0.99 to {associations_by_virus_cluster_path}")
