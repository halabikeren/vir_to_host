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
    with open(input_path, "w") as infile:
        for index, row in df.iterrows():
            infile.write(f">{index}\n{row.struct_representation}")

    # first, execute RNAdistance via pbs on each secondary structure as reference
    index_to_output = dict()
    output_to_wait_for = []
    for index in df.index:
        alignment_path = f"{workdir}/alignment_{index}"
        output_path = f"{workdir}/rnadistance_{index}.out"
        output_to_wait_for.append(output_path)
        job_path = f"{workdir}/rnadistance_{index}.sh"
        job_output_dir = f"{workdir}/rnadistance_out_{index}"
        os.makedirs(job_output_dir, exist_ok=True)
        index_to_output[index] = (output_path, alignment_path)
        parent_path = f"'{os.path.dirname(os.getcwd())}'"
        cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_pred_utils import RNAPredUtils;RNAPredUtils.exec_rnadistance(ref_struct_index={index}, structs_path={input_path}, alignment_path={alignment_path}, output_path={output_path})"'
        if not os.path.exists(output_path) or not os.path.exists(alignment_path):
            PBSUtils.create_job_file(job_path=job_path, job_name=f"rnadistance_{index}", job_output_dir=job_output_dir, commands=[cmd])
            res = os.system(f"qsub {job_path}")
    complete = np.all([os.path.exists(output_path) for output_path in output_to_wait_for])
    while not complete:
        sleep(120)
        complete = np.all([os.path.exists(output_path) for output_path in output_to_wait_for])

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


def compute_intra_clusters_distance(clusters_data: pd.DataFrameGroupBy, integrated_distances_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    :param clusters_data:
    :param integrated_distances_df:
    :param output_path:
    :return:
    """
    pass

def compute_inter_clusters_distance(clusters_data: pd.DataFrameGroupBy, integrated_distances_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    :param clusters_data:
    :param integrated_distances_df:
    :param output_path:
    :return:
    """
    pass


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
    "--df_output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output dataframe to write",
)
def cluster_secondary_structures(structures_data_path: str,
                                 workdir: str,
                                 log_path: str,
                                 df_output_path: str):
    """
    :param structures_data_path: path to dataframe with secondary structures
    :param workdir: directory to write pipeline products in
    :param log_path: path to logger
    :param df_output_path: path to output file
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

    # compute pairwise distances between structures
    logger.info(f"computing pairwise distances across {structures_df.shape[0]}")
    distances_dfs = compute_distances(df=structures_df, input_path=f"{workdir}/structures.fasta", output_dir=f"{workdir}/distances/", workdir=f"{workdir}/rnadistance/")
    integrated_distances_df = distances_dfs["integrated"]

    # cluster by viral family
    logger.info(f"clustering structures by viral family")
    structures_by_viral_families = structures_df.groupby("virus_family_name")
    logger.info(f"computing inter-cluster distances across {len(structures_df.virus_family_name.unique())} clusters")
    compute_inter_clusters_distance(clusters_data=structures_by_viral_families, integrated_distances_df=integrated_distances_df, output_path=f"{workdir}/cluster_by_viral_family_intra_cluster_distances.csv")
    logger.info(f"computing intra-cluster distances across {len(structures_df.virus_family_name.unique())} clusters")
    compute_inter_clusters_distance(clusters_data=structures_by_viral_families, integrated_distances_df=integrated_distances_df, output_path=f"{workdir}/cluster_by_viral_family_inter_cluster_distances.csv")

    # cluster by hosts
    logger.info(f"clustering structures by host")
    structures_df["virus_hosts_names"] = structures_df["virus_hosts_names"].apply(lambda hosts: hosts.split(";"))
    structures_df = structures_df.explode("virus_hosts_names")
    structures_df_by_hosts = structures_df.groupby("virus_hosts_names")
    logger.info(f"computing inter-cluster distances across {len(structures_df.virus_hosts_names.unique())} clusters")
    compute_inter_clusters_distance(clusters_data=structures_df_by_hosts, integrated_distances_df=integrated_distances_df, output_path=f"{workdir}/cluster_by_host_intra_cluster_distances.csv")
    logger.info(f"computing intra-cluster distances across {len(structures_df.virus_hosts_names.unique())} clusters")
    compute_inter_clusters_distance(clusters_data=structures_df_by_hosts, integrated_distances_df=integrated_distances_df, output_path=f"{workdir}/cluster_by_host_inter_cluster_distances.csv")

if __name__ == '__main__':
    cluster_secondary_structures()