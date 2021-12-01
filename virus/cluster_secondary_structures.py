import os
import sys
from time import sleep
import click
import logging
import numpy as np
import pandas as pd

sys.path.append("..")
from utils.rna_pred_utils import RNAPredUtils

from utils.pbs_utils import PBSUtils

logger = logging.getLogger(__name__)

def get_distances_from_ref_structures(ref_structures: pd.Series, other_structures: pd.Series, workdir: str) -> float:

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
            job_path = f"{workdir}/rnadistance_{i}.sh"
            job_output_dir = f"{workdir}/rnadistance_out_{i}"
            os.makedirs(job_output_dir, exist_ok=True)
            index_to_output[i] = (output_path, alignment_path)
            parent_path = f"'{os.path.dirname(os.getcwd())}'"
            ref_struct = f"'{ref_struct}'"
            structs_path = f"'{other_structures_path}'"
            cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_pred_utils import RNAPredUtils;RNAPredUtils.exec_rnadistance(ref_struct={ref_struct}, structs_path={structs_path}, alignment_path={alignment_path}, output_path={output_path})"'
            if not os.path.exists(output_path) or not os.path.exists(alignment_path):
                if not os.path.exists(job_path):
                    PBSUtils.create_job_file(job_path=job_path, job_name=f"rnadistance_{ref_struct}",
                                             job_output_dir=job_output_dir, commands=[cmd], ram_gb_size=20)
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
                sleep(120)
                curr_jobs_num = PBSUtils.compute_curr_jobs_num()
            res = os.system(f"qsub {job_path}")

        # wait for jobs to finish
        paths_exist = [os.path.exists(output_path) for output_path in output_to_wait_for]
        logger.info(f"{len([item for item in paths_exist if item])} out of {len(paths_exist)} jobs are completed")
        complete = np.all(paths_exist)
        while not complete:
            sleep(5 * 60)
            paths_exist = [os.path.exists(output_path) for output_path in output_to_wait_for]
            logger.info(f"{len([item for item in paths_exist if item])} out of {len(paths_exist)} jobs are completed")
            complete = np.all(paths_exist)
        for job_path in jobs_paths:
            os.remove(job_path)

    # now, parse the distances and save them into a matrix
    distances_dfs = {dist_type: pd.DataFrame(index=ref_structures, columns=other_structures) for dist_type in
                     ["F", "H", "W", "C", "P"]}
    for i in range(len(ref_structures)):
        distances_from_i = RNAPredUtils.parse_rnadistance_result(rnadistance_path=index_to_output[i][0],
                                                                 struct_alignment_path=index_to_output[i][1])
        for dist_type in distances_dfs:
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
    output_dir = f"{workdir}/distances_matrices/"
    os.makedirs(output_dir, exist_ok=True)
    for dist_type in distances_dfs:
        output_path = f"{output_dir}/{dist_type}_distances.csv"
        distances_dfs[dist_type].to_csv(output_path)

    return float(np.mean(np.mean(integrated_distances_df, axis=0)))


def get_inter_cluster_distance(cluster_1_structures: pd.Series, cluster_2_structures: pd.Series, workdir: str) -> float:
    """
    :param cluster_1_structures: structures to compute their distances from cluster_2_structures
    :param cluster_2_structures: structures to compute their distances from cluster_1_structures
    :param workdir: directory to write rnadistance output to
    :return: mean distance across structures of cluster 1 and cluster 2
    """
    return get_distances_from_ref_structures(ref_structures=cluster_1_structures, other_structures=cluster_2_structures, workdir=workdir)


def get_intra_cluster_distance(cluster_structures: pd.Series, workdir: str) -> float:
    """
    :param cluster_structures: structures to compute their distances
    :param workdir: directory to write rnadistance output to
    :return: mean distance across structures
    """
    return get_distances_from_ref_structures(ref_structures=cluster_structures, other_structures=cluster_structures, workdir=workdir)


def compute_clusters_distances(clusters_data: pd.core.groupby.GroupBy, workdir: str, output_path :str):
    """
    :param clusters_data:
    :param workdir:
    :param output_path:
    :return:
    """
    df = pd.DataFrame(index=clusters_data.groups.keys(), columns=["intra_cluster_distance"] + [f"distance_from_{cluster}" for cluster in clusters_data.groups.keys()])

    max_cluster_size = np.max([clusters_data.get_group(cluster).shape[0] for cluster in clusters_data.groups.keys()])
    logger.info(f"computing intra-cluster distances for {len(clusters_data.groups.keys())} clusters of maximum size {max_cluster_size}")
    intra_cluster_distances_workdir = f"{workdir}/intra_cluster_distances/"
    df["intra_cluster_distance"] = df.index.map(lambda cluster: get_intra_cluster_distance(cluster_structures=clusters_data.get_group(cluster).struct_representation, workdir=f"{intra_cluster_distances_workdir}/{cluster}"))

    logger.info(f"computing inter-cluster distances across {len(clusters_data.groups.keys())**2/2} combinations of clusters")
    inter_cluster_distances_workdir = f"{workdir}/inter_cluster_distances/"
    for cluster2 in clusters_data.groups.keys():
        logger.info(f"computing inter-cluster distances from {cluster2}")
        df[f"distance_from_{cluster2}"] = df.index.map(lambda cluster1: get_inter_cluster_distance(cluster_1_structures=clusters_data.get_group(cluster1).struct_representation, cluster_2_structures=clusters_data.get_group(cluster2).struct_representation, workdir=f"{inter_cluster_distances_workdir}/{cluster1}_{cluster2}_distances/"))

    logger.info(f"writing output to {output_path}")
    df.to_csv(output_path)


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

    logger.info(f"loading viral rna secondary sourcetrees data from {structures_data_path}")
    structures_df = pd.read_csv(structures_data_path)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(df_output_dir, exist_ok=True)

    # cluster by viral family
    logger.info(f"clustering structures by viral family")
    structures_by_viral_families = structures_df.groupby("virus_family_name")
    logger.info(f"computing inter-cluster and intra-clusters distances across {len(structures_df.virus_family_name.unique())} clusters")
    compute_clusters_distances(clusters_data=structures_by_viral_families, workdir=f"{workdir}/clusters_by_viral_families/", output_path=f"{df_output_dir}/cluster_by_viral_family_intra_cluster_distances.csv")

    # cluster by hosts
    logger.info(f"clustering structures by host")
    structures_df["virus_hosts_names"] = structures_df["virus_hosts_names"].apply(lambda hosts: hosts.split(";"))
    structures_df = structures_df.explode("virus_hosts_names")
    structures_df_by_hosts = structures_df.groupby("virus_hosts_names")
    logger.info(f"computing inter-cluster and intra-cluster distances across {len(structures_df.virus_hosts_names.unique())} clusters")
    compute_clusters_distances(clusters_data=structures_df_by_hosts, workdir=f"{workdir}/clusters_by_hosts/", output_path=f"{df_output_dir}/cluster_by_host_intra_cluster_distances.csv")

if __name__ == '__main__':
    cluster_secondary_structures()



