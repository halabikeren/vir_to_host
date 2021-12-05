import os
import shutil
import sys
import typing as t
from time import sleep
import click
import logging
import numpy as np
import pandas as pd

sys.path.append("..")
from utils.rna_pred_utils import RNAPredUtils

from utils.pbs_utils import PBSUtils

logger = logging.getLogger(__name__)



def compute_pairwise_distances(ref_structures: pd.Series, other_structures: pd.Series, workdir: str, output_dir: str) -> pd.DataFrame:
    os.makedirs(workdir, exist_ok=True)

    # first, write a fasta file with all the structures representations
    other_structures_path = f"{workdir}/other_structures.fasta"
    logger.info(f"writing structures to fasta file")
    other_structures.drop_duplicates(inplace=True)
    if not os.path.exists(other_structures_path):
        with open(other_structures_path, "w") as infile:
            i = 0
            for struct in other_structures:
                infile.write(f">{i}\n{struct}")
                i += 1

    index_to_output = dict()
    i = 0
    batch_size = 1900
    ref_structures_batches = [ref_structures[i:i + batch_size] for i in
                              range(0, len(ref_structures), batch_size)]
    for ref_structures_batch in ref_structures_batches:
        output_to_wait_for = []
        jobs_paths = []
        starting_index = i
        # create jobs
        for ref_struct in ref_structures_batch:
            alignment_path = f"'{workdir}/alignment_{i}'"
            output_path = f"'{workdir}/rnadistance_{i}.out'"
            job_workdir = f"'{workdir}/rnadistance_{i}_aux/'"
            job_path = f"{workdir}/rnadistance_{i}.sh"
            job_output_dir = f"{workdir}/rnadistance_out_{i}"
            os.makedirs(job_output_dir, exist_ok=True)
            index_to_output[i] = (output_path, alignment_path)
            parent_path = f"'{os.path.dirname(os.getcwd())}'"
            ref_struct = f"'{ref_struct}'"
            structs_path = f"'{other_structures_path}'"
            cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_pred_utils import RNAPredUtils;RNAPredUtils.exec_rnadistance(ref_struct={ref_struct}, ref_struct_index={i}, structs_path={structs_path}, workdir={job_workdir}, alignment_path={alignment_path}, output_path={output_path})"'
            if not os.path.exists(output_path) or not os.path.exists(alignment_path):
                if not os.path.exists(job_path):
                    PBSUtils.create_job_file(job_path=job_path, job_name=f"rnadistance_{i}",
                                             job_output_dir=job_output_dir, commands=[cmd], ram_gb_size=10)
                output_to_wait_for.append(output_path)
                jobs_paths.append(job_path)
            i += 1
        finishing_index = i

        # submit jobs
        logger.info(
            f"submitting {batch_size} jobs for missing RNAdistance outputs from index {starting_index} to {finishing_index}")
        for job_path in jobs_paths:
            # check how many jobs are running
            curr_jobs_num = PBSUtils.compute_curr_jobs_num()
            while curr_jobs_num > 1990:
                logger.info(f"current job number is {curr_jobs_num}. will wait a minute before checking again")
                sleep(60)
                curr_jobs_num = PBSUtils.compute_curr_jobs_num()
            res = os.system(f"qsub {job_path}")

        # wait for jobs to finish
        paths_exist = [len(os.listdir(f"{workdir}/rnadistance_out_{i}")) for i in range(starting_index, finishing_index)]
        logger.info(f"{len([item for item in paths_exist if item])} out of {len(paths_exist)} jobs are completed")
        complete = np.all(paths_exist)
        while not complete:
            sleep(2 * 60)
            paths_exist = [len(os.listdir(f"{workdir}/rnadistance_out_{i}")) for i in
                           range(starting_index, finishing_index)]
            logger.info(f"{len([item for item in paths_exist if item])} out of {len(paths_exist)} jobs are completed. will wait 2 minutes before checking again")
            complete = np.all(paths_exist)
            for j in range(len(paths_exist)):
                if paths_exist[j]:
                    if os.path.exists(f"{workdir}/rnadistance_{i}.sh"):
                        os.remove(f"{workdir}/rnadistance_{i}.sh")
                    shutil.rmtree(f"{workdir}/rnadistance_{i}_aux/", ignore_errors=True)
                    shutil.rmtree(f"{workdir}/rnadistance_out_{i}", ignore_errors=True)

    # now, parse the distances and save them into a matrix
    distances_dfs = {dist_type: pd.DataFrame(index=ref_structures, columns=other_structures) for dist_type in
                     ["F", "H", "W", "C", "P"]}
    for i in range(len(ref_structures)):
        distances_from_i = RNAPredUtils.parse_rnadistance_result(rnadistance_path=index_to_output[i][0],
                                                                 struct_alignment_path=index_to_output[i][1])
        for dist_type in distances_dfs:
            distances_from_i[dist_type][str(i)] = 0
            distances_dfs[dist_type].iloc[ref_structures[i]] = distances_from_i[dist_type]

    # derive a single distances df of integrated, standardized, measures to use
    num_dist_metrics = len(list(distances_dfs.keys()))
    integrated_distances_df = distances_dfs["edit_distance"] * 1. / num_dist_metrics
    for dist_type in distances_dfs:
        if dist_type != "edit_distance":
            distances_matrix = (distances_dfs[dist_type] - distances_dfs[dist_type].min()) / (
                    distances_dfs[dist_type].max() - distances_dfs[dist_type].min())
            integrated_distances_df += distances_matrix * 1. / num_dist_metrics
    distances_dfs["integrated"] = integrated_distances_df

    # write dfs to output
    os.makedirs(output_dir, exist_ok=True)
    for dist_type in distances_dfs:
        output_path = f"{output_dir}/{dist_type}_distances.csv"
        distances_dfs[dist_type].to_csv(output_path)

    return distances_dfs["integrated"]

def get_distances_from_ref_structures(ref_structures: pd.Series, other_structures: pd.Series, workdir: str, distances_df: t.Optional[pd.DataFrame] = None) -> float:
    """
    :param ref_structures:
    :param other_structures:
    :param workdir:
    :param distances_df:
    :return:
    """

    if not distances_df:
        distances_df = compute_pairwise_distances(ref_structures=ref_structures, other_structures=other_structures, workdir=f"{workdir}/rnadistance_aux/", output_dir=f"{workdir}/distances_matrices/")
    return float(np.mean(np.mean(distances_df, axis=0)))

def get_inter_cluster_distance(cluster_1_structures: pd.Series, cluster_2_structures: pd.Series, workdir: str, distances_df: t.Optional[pd.DataFrame] = None) -> float:
    """
    :param cluster_1_structures: structures to compute their distances from cluster_2_structures
    :param cluster_2_structures: structures to compute their distances from cluster_1_structures
    :param workdir: directory to write rnadistance output to
    :param distances_df: dataframe with pairwise structures differences
    :return: mean distance across structures of cluster 1 and cluster 2
    """
    return get_distances_from_ref_structures(ref_structures=cluster_1_structures, other_structures=cluster_2_structures, workdir=workdir, distances_df=distances_df)

def get_intra_cluster_distance(cluster_structures: pd.Series, workdir: str, distances_df: t.Optional[pd.DataFrame] = None) -> float:
    """
    :param cluster_structures: structures to compute their distances
    :param workdir: directory to write rnadistance output to
    :param distances_df: dataframe with pairwise structures distances
    :return: mean distance across structures
    """
    return get_distances_from_ref_structures(ref_structures=cluster_structures, other_structures=cluster_structures, workdir=workdir, distances_df=distances_df)

def compute_clusters_distances(clusters_data: pd.core.groupby.GroupBy, distances_df: pd.DataFrame, workdir: str, output_path :str):
    """
    :param clusters_data:
    :param workdir:
    :param distances_df:
    :param output_path:
    :return:
    """
    df = pd.DataFrame(index=clusters_data.groups.keys(), columns=["intra_cluster_distance"] + [f"distance_from_{cluster}" for cluster in clusters_data.groups.keys()])

    max_cluster_size = np.max([clusters_data.get_group(cluster).shape[0] for cluster in clusters_data.groups.keys()])
    logger.info(f"computing intra-cluster distances for {len(clusters_data.groups.keys())} clusters of maximum size {max_cluster_size}")
    intra_cluster_distances_workdir = f"{workdir}/intra_cluster_distances/"
    df["intra_cluster_distance"] = df.index.map(lambda cluster: get_intra_cluster_distance(cluster_structures=clusters_data.get_group(cluster).struct_representation, workdir=f"{intra_cluster_distances_workdir}/{cluster}", distances_df=distances_df))

    logger.info(f"computing inter-cluster distances across {len(clusters_data.groups.keys())**2/2} combinations of clusters")
    inter_cluster_distances_workdir = f"{workdir}/inter_cluster_distances/"
    for cluster2 in clusters_data.groups.keys():
        logger.info(f"computing inter-cluster distances from {cluster2}")
        df[f"distance_from_{cluster2}"] = df.index.map(lambda cluster1: get_inter_cluster_distance(cluster_1_structures=clusters_data.get_group(cluster1).struct_representation, cluster_2_structures=clusters_data.get_group(cluster2).struct_representation, workdir=f"{inter_cluster_distances_workdir}/{cluster1}_{cluster2}_distances/",  distances_df=distances_df))

    logger.info(f"writing output to {output_path}")
    df.to_csv(output_path)


@click.command()
@click.option(
    "--structures_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path of dataframe of secondary structures of viral species, with their hosts assignment and viral families assignment",
)
@click.option(
    "--partition_by_column",
    type=str,
    help="column to partition data for analysis by",
    required=False,
    default="virus_family_name",
)
@click.option(
    "--host_partition_to_use",
    type = click.Choice(["species", "genus", "family", "order", "class", "kingdom"]),
    help="host taxonomic hierarchy to cluster by",
    required=False,
    default="class"
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory to hold the RNA prediction pipeline files in",
    required=False,
    default=None
)
@click.option(
    "--log_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the logging of the script",
)
@click.option(
    "--df_output_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output dataframe to write",
)
def cluster_secondary_structures(structures_data_path: str,
                                 partition_by_column: str,
                                 host_partition_to_use: str,
                                 workdir: str,
                                 log_path: str,
                                 df_output_dir: str):
    """
    :param structures_data_path: path to dataframe with secondary structures
    :param partition_by_column: column to partition data by to partitions on which independent analyses will be conducted
    :param host_partition_to_use: host taxonomic hierarchy to cluster secondary structures by
    :param workdir: directory to write pipeline products in
    :param log_path: path to logger
    :param df_output_dir: directory to write the output files with the clusters distances analyses
    :return:
    """
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

    logger.info(f"loading viral rna secondary sourcetrees data from {structures_data_path}")
    structures_df = pd.read_csv(structures_data_path)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(df_output_dir, exist_ok=True)

    logger.info(f"partitioning data by {partition_by_column} into {len(structures_df[partition_by_column].dropna().unique())} groups")

    structures_df_partitions = structures_df.groupby(partition_by_column)
    for group_name in structures_df_partitions.groups.keys():
        if group_name == "hepadnaviridae": # only for the purpose of testing

            structures_df_partition = structures_df_partitions.get_group(group_name)
            logger.info(f"performing analysis on group {group_name} of size {structures_df_partition.shape[0]}")

            # compute distances across all structures for assessment of homogeneity across all structures
            ref_structures = structures_df_partition.struct_representation
            other_structures = structures_df_partition.struct_representation
            distances_df = compute_pairwise_distances(ref_structures=ref_structures, other_structures=other_structures, workdir=f"{workdir}/rnadistance_{group_name}", output_dir=f"{workdir}/distances_{group_name}/")
            logger.info(f"homogeneity across all structures, regardless of host classification, is {1-np.mean(np.mean(distances_df, axis=1))}")

            # cluster by decreasing the given host taxonomic hierarchy
            logger.info(f"clustering structures by host")
            structures_df[f"virus_hosts_{host_partition_to_use}_names"] = structures_df[f"virus_hosts_{host_partition_to_use}_names"].apply(lambda hosts: hosts.split(";"))
            structures_df = structures_df.explode(f"virus_hosts_{host_partition_to_use}_names")
            structures_df_by_hosts = structures_df.groupby("virus_hosts_names")
            logger.info(f"computing inter-cluster and intra-cluster distances across {len(structures_df.virus_hosts_names.unique())} clusters")
            compute_clusters_distances(clusters_data=structures_df_by_hosts, distances_df=distances_df, workdir=f"{workdir}/clusters_by_hosts/", output_path=f"{df_output_dir}/clusters_by_host_{host_partition_to_use}_distances.csv")

if __name__ == '__main__':
    cluster_secondary_structures()



