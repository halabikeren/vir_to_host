import logging
import os
import sys

import click
import numpy as np
import pandas as pd

sys.path.append("..")
from utils.clustering_utils import ClusteringUtils


@click.command()
@click.option(
    "--viral_sequence_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path holding the output dataframe to write",
    required=False,
    default="/groups/itay_mayrose/halabikeren/vir_to_host/data/virus_sequence_data.csv",
)
@click.option(
    "--workdir",
    type=click.Path(exists=False),
    help="directory for cdhit auxiliary files",
    required=False,
    default="/groups/itay_mayrose/halabikeren/vir_to_host/data/",
)
@click.option(
    "--output_path",
    type=click.Path(exists=False),
    help="path holding the output dataframe to write",
    required=False,
    default="/groups/itay_mayrose/halabikeren/vir_to_host/data/virus_sequence_data_clusters.csv",
)
@click.option(
    "--redundancy_threshold",
    type=click.FloatRange(min=0.5, max=1),
    help="cdhit redundancy removal threshold in case of > 5000 sequences",
    required=False,
    default=0.99,
)
@click.option(
    "--clustering_threshold",
    type=click.FloatRange(min=0.5, max=0.99),
    help="cdhit clustering threshold in case for clustering associations by cdhit viral sequence clusters",
    required=False,
    default=0.88,
)
@click.option(
    "--logger_path",
    type=click.Path(exists=False),
    help="path to the log of the script",
    required=True,
)
@click.option(
    "--mem_limit",
    type=click.INT,
    help="memory in MB to allocate to cdhit",
    required=False,
    default=4000,
)
def cluster_sequence_data(
    viral_sequence_data_path: click.Path,
    workdir: click.Path,
    output_path: click.Path,
    redundancy_threshold: float,
    clustering_threshold: float,
    logger_path: click.Path,
    mem_limit: int,
):
    # initialize the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(logger_path)),
        ],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    virus_sequence_subdf = pd.read_csv(viral_sequence_data_path)

    # if the dataframe is too large for centroid computation,
    # map redundant sequences to a single representative
    # and following the analysis, map them to the location of the analyzed representative
    if virus_sequence_subdf.shape[0] > 5000:
        redundancy_removal_aux_dir = f"{workdir}/redundancy_removal/"
        os.makedirs(redundancy_removal_aux_dir, exist_ok=True)
        ClusteringUtils.collapse_redundant_sequences(
            elements=virus_sequence_subdf,
            homology_threshold=redundancy_threshold,
            aux_dir=redundancy_removal_aux_dir,
            mem_limit=mem_limit,
        )

    else:
        # otherwise, each accession will represent itself
        virus_sequence_subdf["sequence_representative"] = virus_sequence_subdf[
            "accession"
        ]

    virus_sequence_non_redundant_subdf = virus_sequence_subdf.loc[
        virus_sequence_subdf.accession == virus_sequence_subdf.sequence_representative
    ]
    virus_sequence_redundant_subdf = virus_sequence_subdf.loc[
        virus_sequence_subdf.accession != virus_sequence_subdf.sequence_representative
    ]

    ClusteringUtils.compute_clusters_representatives(
        elements=virus_sequence_non_redundant_subdf,
        homology_threshold=clustering_threshold,
        aux_dir=str(workdir),
        mem_limit=mem_limit,
    )

    virus_sequence_redundant_subdf.set_index("sequence_representative", inplace=True)
    for col in virus_sequence_non_redundant_subdf.columns:
        if col not in virus_sequence_redundant_subdf.columns:
            virus_sequence_redundant_subdf[col] = np.nan
            virus_sequence_redundant_subdf[col].fillna(
                value=virus_sequence_non_redundant_subdf.set_index("accession")[col],
                inplace=True,
            )
    virus_sequence_redundant_subdf.reset_index(inplace=True)
    virus_sequence_subdf = pd.concat(
        [virus_sequence_non_redundant_subdf, virus_sequence_redundant_subdf]
    )

    virus_to_cluster = virus_sequence_subdf[
        [
            "taxon_name",
            "species_name",
            "accession",
            "cluster_id",
            "cluster_representative",
        ]
    ]
    virus_to_cluster.to_csv(output_path, index=False)


if __name__ == "__main__":
    cluster_sequence_data()
