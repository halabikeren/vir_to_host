import os
import sys
from time import sleep
import typing as t
import click
import logging
import numpy as np
import pandas as pd

sys.path.append("..")
from utils.rna_pred_utils import RNAPredUtils

from utils.pbs_utils import PBSUtils

logger = logging.getLogger(__name__)

def compute_distances(df: pd.DataFrame, input_path: str, output_dir: str, workdir: str) -> t.Dict[str, pd.DataFrame]:
    """
    :param df: dataframe of structures for which distances should be computed
    :param input_path:
    :param output_dir: directory to write the distances dataframes to, as per the available distances metrics
    :param workdir: directory to write rnadistance output to
    :return: dataframe of the distances
    """
    os.makedirs(workdir, exist_ok=True)

    # first, write a fasta file with all the structures representations
    logger.info(f"writing structures to fasta file")
    if not os.path.exists(input_path):
        with open(input_path, "w") as infile:
            i = 0
            for index in df.index:
                infile.write(f">{i}\n{index}")
                i += 1

    # first, execute RNAdistance via pbs on each secondary structure as reference
    logger.info(f"executing RNAdistance on each reference structure against the rest to obtain a distance matrix")
    index_to_output = dict()
    batch_size = 1900
    index_batches = [df.index[i:i+batch_size] for i in range(0, len(df.index), batch_size)]
    i = 0
    for index_batch in index_batches:
        output_to_wait_for = []
        jobs_paths = []
        starting_index = i

        # create jobs
        for index in index_batch:
            alignment_path = f"'{workdir}/alignment_{i}'"
            output_path = f"'{workdir}/rnadistance_{i}.out'"
            job_path = f"{workdir}/rnadistance_{i}.sh"
            job_output_dir = f"{workdir}/rnadistance_out_{i}"
            os.makedirs(job_output_dir, exist_ok=True)
            index_to_output[i] = (output_path, alignment_path)
            parent_path = f"'{os.path.dirname(os.getcwd())}'"
            ref_struct = f"'{index}'"
            structs_path = f"'{input_path}'"
            cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_pred_utils import RNAPredUtils;RNAPredUtils.exec_rnadistance(ref_struct_index={ref_struct}, structs_path={structs_path}, alignment_path={alignment_path}, output_path={output_path})"'
            if not os.path.exists(output_path) or not os.path.exists(alignment_path):
                if not os.path.exists(job_path):
                    PBSUtils.create_job_file(job_path=job_path, job_name=f"rnadistance_{i}", job_output_dir=job_output_dir, commands=[cmd], ram_gb_size=20)
                output_to_wait_for.append(output_path)
                jobs_paths.append(job_path)
            i += 1
        finishing_index = i

        # submit jobs
        logger.info(f"submitting {batch_size} jobs for missing RNAdistance outputs from index {starting_index} to {finishing_index}")
        for job_path in jobs_paths:
            # check how many jobs are running
            curr_jobs_num = PBSUtils.compute_curr_jobs_num()
            while curr_jobs_num > 1990:
                sleep(120)
                curr_jobs_num = PBSUtils.compute_curr_jobs_num()
            res = os.system(f"qsub {job_path}")

        # wait for jobs to finish
        paths_exist = [os.path.exists(output_path) for output_path in output_to_wait_for]
        logger.info(f"{len([item for item in paths_exist if item])} out of {len(paths_exist)} jobs are completed")
        complete = np.all(paths_exist)
        while not complete:
            sleep(5*60)
            paths_exist = [os.path.exists(output_path) for output_path in output_to_wait_for]
            logger.info(f"{len([item for item in paths_exist if item])} out of {len(paths_exist)} jobs are completed")
            complete = np.all(paths_exist)
        for job_path in jobs_paths:
            os.remove(job_path)


    # now, parse the distances and save them into a matrix
    distances_dfs = {dist_type: pd.DataFrame(index=df.index, columns=df.index) for dist_type in ["F", "H", "W", "C", "P"]}
    for index in df.index:
        distances_from_i = RNAPredUtils.parse_rnadistance_result(rnadistance_path=index_to_output[index][0], struct_alignment_path=index_to_output[index][1])
        for dist_type in distances_dfs:
            distances_dfs[dist_type].iloc[index] = distances_from_i[dist_type]

    # derive a single distances df of integrated, standardized, measures to use
    num_dist_metrics = len(list(distances_dfs.keys()))
    integrated_distances_df = distances_dfs["edit_distance"] * 1./num_dist_metrics
    for dist_type in distances_dfs:
        if dist_type != "edit_distance":
            distances_matrix =(distances_dfs[dist_type]-distances_dfs[dist_type].min())/(distances_dfs[dist_type].max()-distances_dfs[dist_type].min())
            integrated_distances_df += distances_matrix * 1./num_dist_metrics
    distances_dfs["integrated"] = integrated_distances_df

    # write dfs to output
    os.makedirs(output_dir, exist_ok=True)
    for dist_type in distances_dfs:
        output_path = f"{output_dir}/{dist_type}_distances.csv"
        distances_dfs[dist_type].to_csv(output_path)

    return distances_dfs


def get_intra_cluster_distance(cluster: str, distances_df: pd.DataFrame) -> float:
    pass

def get_inter_clusters_distance(cluster1: str, cluster2: str, distances_df: pd.DataFrame) -> float:
    pass

def compute_clusters_distances(clusters_data: pd.core.groupby.DataFrameGroupBy, integrated_distances_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    :param clusters_data:
    :param integrated_distances_df:
    :param output_path:
    :return:
    """
    df = pd.DataFrame(index=clusters_data.groups.keys(), columns=["intra_cluster_distance"] + [f"distance_from_{cluster}" for cluster in clusters_data.groups.keys()])

    max_cluster_size = np.max([clusters_data.get_group(cluster).shape[0] for cluster in clusters_data.groups.keys()])
    logger.info(f"computing intra-cluster distances for {len(clusters_data.groups.keys())} clusters of maximum size {max_cluster_size}")
    df["intra_cluster_distance"] = df.index.map(lambda cluster: get_intra_cluster_distance(cluster=cluster, distances_df=integrated_distances_df))

    logger.info(f"computing inter-cluster distances across {len(clusters_data.groups.keys())**2/2} combinations of clusters")
    for cluster2 in clusters_data.groups.keys():
        df[f"distance_from_{cluster2}"] = df.index.map(lambda cluster1: get_inter_clusters_distance(cluster1=cluster1, cluster2=cluster2, distances_df=integrated_distances_df))






@click.command()
@click.option(
    "--structures_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path of dataframe of secondary structures of viral species, with their hosts assignment and viral familires assignment",
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
                                 workdir: str,
                                 log_path: str,
                                 df_output_dir: str):
    """
    :param structures_data_path: path to dataframe with secondary structures
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

    structures_df = pd.read_csv(structures_data_path, index_col="struct_representation")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(df_output_dir, exist_ok=True)

    # compute pairwise distances between structures
    logger.info(f"computing pairwise distances across {structures_df.shape[0]} clusters")
    distances_dfs = compute_distances(df=structures_df, input_path=f"{workdir}/structures.fasta", output_dir=f"{workdir}/distances/", workdir=f"{workdir}/rnadistance_workdir/")
    integrated_distances_df = distances_dfs["integrated"]

    # cluster by viral family
    logger.info(f"clustering structures by viral family")
    structures_by_viral_families = structures_df.groupby("virus_family_name")
    logger.info(f"computing inter-cluster and intra-clusters distances across {len(structures_df.virus_family_name.unique())} clusters")
    compute_clusters_distances(clusters_data=structures_by_viral_families, integrated_distances_df=integrated_distances_df, output_path=f"{df_output_dir}/cluster_by_viral_family_intra_cluster_distances.csv")

    # cluster by hosts
    logger.info(f"clustering structures by host")
    structures_df["virus_hosts_names"] = structures_df["virus_hosts_names"].apply(lambda hosts: hosts.split(";"))
    structures_df = structures_df.explode("virus_hosts_names")
    structures_df_by_hosts = structures_df.groupby("virus_hosts_names")
    logger.info(f"computing inter-cluster and intra-cluster distances across {len(structures_df.virus_hosts_names.unique())} clusters")
    compute_clusters_distances(clusters_data=structures_df_by_hosts, integrated_distances_df=integrated_distances_df, output_path=f"{df_output_dir}/cluster_by_host_intra_cluster_distances.csv")

if __name__ == '__main__':
    cluster_secondary_structures()



