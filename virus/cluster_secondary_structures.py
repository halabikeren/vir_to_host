import itertools
import os
import shutil
import sys
import typing as t
from collections import defaultdict
from time import sleep
import click
import logging
import numpy as np
import pandas as pd
from copkmeans.cop_kmeans import cop_kmeans
from sklearn.metrics import silhouette_score
sys.path.append("..")
from utils.rna_struct_utils import RNAStructUtils

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
            index_to_output[i] = (output_path.replace("'",''), alignment_path.replace("'",''))
            parent_path = f"'{os.path.dirname(os.getcwd())}'"
            ref_struct = f"'{ref_struct}'"
            structs_path = f"'{other_structures_path}'"
            cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_pred_utils import RNAPredUtils;RNAPredUtils.exec_rnadistance(ref_struct={ref_struct}, ref_struct_index={i}, structs_path={structs_path}, workdir={job_workdir}, alignment_path={alignment_path}, output_path={output_path})"'
            if not os.path.exists(output_path.replace("'", "")) or not os.path.exists(alignment_path.replace("'", "")):
                if not os.path.exists(job_path):
                    PBSUtils.create_job_file(job_path=job_path, job_name=f"rnadistance_{i}",
                                             job_output_dir=job_output_dir, commands=[cmd], ram_gb_size=10)
                output_to_wait_for.append(output_path)
                jobs_paths.append(job_path)
            else:
                if os.path.exists(f"{workdir}/rnadistance_{i}.sh"):
                    os.remove(f"{workdir}/rnadistance_{i}.sh")
                shutil.rmtree(f"{workdir}/rnadistance_{i}_aux/", ignore_errors=True)
                shutil.rmtree(f"{workdir}/rnadistance_out_{i}", ignore_errors=True)
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
                     ["F", "H", "W", "C", "P", "edit_distance"]}
    for i in range(len(ref_structures)-1):
        distances_from_i = RNAStructUtils.parse_rnadistance_result(rnadistance_path=index_to_output[i][0],
                                                                   struct_alignment_path=index_to_output[i][1])
        for dist_type in distances_dfs:
            distances_dict = {list(ref_structures)[i+j+1]: distances_from_i[dist_type][j-1] for j in range(len(ref_structures)-(i+1))}
            distances_dict[list(ref_structures)[i]] = 0
            distances_dfs[dist_type].loc[list(ref_structures)[i]] = pd.Series(distances_dict)

    # derive a single distances df of integrated, standardized, measures to use
    num_dist_metrics = len(list(distances_dfs.keys()))
    integrated_distances_df = distances_dfs["edit_distance"] * 1. / num_dist_metrics
    for dist_type in distances_dfs:
        if dist_type != "edit_distance":
            distances_matrix = (distances_dfs[dist_type] - distances_dfs[dist_type].min()) / (
                    distances_dfs[dist_type].max() - distances_dfs[dist_type].min() + 0.0001) # the addition of 0.0001 is meant for avoiding division by 0
            integrated_distances_df += distances_matrix * (1. / num_dist_metrics)
    distances_dfs["integrated"] = integrated_distances_df

    # write dfs to output
    os.makedirs(output_dir, exist_ok=True)
    for dist_type in distances_dfs:
        output_path = f"{output_dir}/{dist_type}_distances.csv"
        distances_dfs[dist_type].to_csv(output_path)

    # clear workspace
    shutil.rmtree(workdir)

    return distances_dfs["integrated"]

def get_distances_from_ref_structures(ref_structures: pd.Series, other_structures: pd.Series, workdir: str, distances_df: t.Optional[pd.DataFrame] = None) -> float:
    """
    :param ref_structures:
    :param other_structures:
    :param workdir:
    :param distances_df:
    :return:
    """

    if type(distances_df) != pd.core.frame.DataFrame:
        distances_df = compute_pairwise_distances(ref_structures=ref_structures, other_structures=other_structures, workdir=f"{workdir}/rnadistance_aux/", output_dir=f"{workdir}/distances_matrices/")

    # limit distances df to the relevant structures
    relevant_distances_df_side1 = distances_df.loc[ref_structures][other_structures]
    relevant_distances_df_side2 = distances_df.loc[other_structures][ref_structures]

    mean_dist_1 = np.nanmean(np.nanmean(relevant_distances_df_side1, axis=0))
    mean_dist_2 = np.nanmean(np.nanmean(relevant_distances_df_side2, axis=0))
    final_dist = np.nanmean([mean_dist_1, mean_dist_2])

    return float(final_dist)

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

def get_gram_matrix(input_matrix: np.ndarray) -> np.ndarray:
    sqrt_dist_vec = np.power(input_matrix[0, :], 2)
    gram_component_2 = np.tile(sqrt_dist_vec, (input_matrix.shape[0], 1))
    gram_component_1 = gram_component_2.T
    gram_mat = (gram_component_1 + gram_component_2 - np.power(input_matrix, 2)) / 2
    return gram_mat

def map_distances_to_2d_plane(distances: np.ndarray) -> np.ndarray:
    """
    :param distances: numpy 2d array of the pairwise distances between records
    :return: 2d coordinates of the respective records, based on their pairwise distances,
             using: https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
    """
    gram_mat = get_gram_matrix(input_matrix=distances)
    eigenvalues, eigenvectors = np.linalg.eig(gram_mat) # get eigen decomposition components
    distance_based_coordinates = np.dot(eigenvectors, np.sqrt(np.diag(eigenvalues))) # row i = coordinate of item i
    return distance_based_coordinates

def get_mean_distance_from_rest(query: int, rest: pd.Series(int), coordinates: np.ndarray) -> np.float32:
    """
    :param query: query index
    :param rest: indices of items to compute their distance from the item corresponding to the query index
    :param coordinates: array in which each row i corresponds to the coordinates of the item corresponding to index i
    :return: the mean euclidean distance of the query to the rest
    """
    distances = [np.linalg.norm(coordinates[query]-coordinates[i]) for i in rest]
    return np.float32(np.mean(distances))

def assign_cluster_by_homology(df: pd.DataFrame, sequence_data_dir: str, species_wise_msa_dir: str, workdir: str, distances: np.ndarray, partition_size: int = 79) -> pd.DataFrame:
    """
    :param df: dataframe of structure records to assign to clusters
    :param sequence_data_dir: directory holding the original sequence data before filtering out outliers. this directory should also hold similarity values tables per species
    :param species_wise_msa_dir: directory holding the aligned genomes per species after filtering out outliers, which were used in the inference process of the secondary structures
    :param workdir: directory to write family sequence data and align it in
    :param distances: 2d array of the pairwise distances between structures
    :param partition_size: size to partition structures by their location prior to clustering. the default value correspond to the maximal structure size in RNASIV
    :return: same input df with an added column of the assigned_cluster. clusters are represented by a combination of window range and an index.
    """

    os.makedirs(workdir, exist_ok=True)

    # create mapping of structures start and end positions from species-wise alignments to family-wise alignments
    df = RNAStructUtils.map_species_wise_pos_to_group_wise_pos(df=df,
                                                               seq_data_dir=sequence_data_dir,
                                                               species_wise_msa_dir=species_wise_msa_dir,
                                                               workdir=workdir)

    # segment structures by locations across the family-wise alignment
    df = RNAStructUtils.assign_partition_by_size(df=df, partition_size=partition_size)

    # map the pairwise distance to a 2d plane and assign to each record a respective 2d coordination
    coordinates = map_distances_to_2d_plane(distances=distances)

    # for each segment, cluster structures falling in it according to their distances
    # project pairwise distances to points in space and add 2d coordination to each record in the dataframe accordingly
    df_by_partition = df.groupby("assigned_partition")
    df["cluster_index_within_partition"] = np.nan
    for partition in df_by_partition.groups.keys():
        sub_df = df_by_partition.get_group(partition)
        sub_coordinates = coordinates[list(sub_df.index)]

        # gather constraints on records that belong to the same species, and thus must appear in different clusters
        def get_pairwise_combos(iterable):
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)
        cannot_link = []
        data_by_cluster_and_sp = sub_df.groupy(["cluster_index_within_partition", "virus_species_name"])
        for group in data_by_cluster_and_sp.groups.keys():
            group_df = data_by_cluster_and_sp.get_group(group)
            cannot_link += list(get_pairwise_combos(group_df.index))

        # apply distance-based clustering using kmeans clustering with silhouette-based optimization of k
        # use COP-Kmeans (https://github.com/Behrouz-Babaki/COP-Kmeans) to constrain clusters to have at most one copy per species
        k_to_cluster_assignment = defaultdict(dict)
        k_to_sil_score = dict()
        kmax = len(df.virus_species_name.unique()) # do not allow more clusters than species
        for k in range(2, kmax + 1):
            clusters, centers = cop_kmeans(dataset=sub_coordinates, k=k, cl=cannot_link)
            k_to_cluster_assignment[k] = {i: clusters for i in sub_df.index}
            k_to_sil_score[k] = silhouette_score(sub_coordinates, clusters, metric='euclidean')
        best_clustering = k_to_cluster_assignment[max(k_to_cluster_assignment, key=k_to_cluster_assignment.get)]
        df["cluster_index_within_partition"].fillna(value=best_clustering, inplace=True)

    df["assigned_cluster"] = df[["assigned_partition", "cluster_index_within_partition"]].apply(lambda row: f"{row.assigned_partition}_{row.cluster_index_within_partition}")
    return df


@click.command()
@click.option(
    "--structures_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path of dataframe of secondary structures of viral species, "
         "with their hosts assignment and viral families assignment and the "
         "unaligned, species-wise aligned and family-wise aligned start and end positions",
)
@click.option(
    "--by",
    type=click.Choice(["homology", "column"]),
    help="path holding the output dataframe to write",
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
                                 by: str,
                                 sequence_data_dir: str,
                                 species_wise_msa_dir: str,
                                 host_partition_to_use: str,
                                 workdir: str,
                                 log_path: str,
                                 df_output_dir: str):

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

    logger.info(f"loading viral rna secondary structures data from {structures_data_path}")
    structures_df = pd.read_csv(structures_data_path)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(df_output_dir, exist_ok=True)

    # compute pairwise distances between structures
    logger.info(f"computing distances across structures")
    df_output_path = f"{workdir}/distances_all_structures/integrated_distances.csv"
    if os.path.exists(df_output_path):
        distances_df = pd.read_csv(df_output_path)
    else:
        ref_structures = structures_df.struct_representation
        other_structures = structures_df.struct_representation
        distances_df = compute_pairwise_distances(ref_structures=ref_structures, other_structures=other_structures,
                                                  workdir=f"{workdir}/rnadistance_all_structures/",
                                                  output_dir=f"{workdir}/distances_all_structures/")

    # assign structures to clusters
    if by != "column":
        logger.info(f"clustering structures by homology")
        structures_df = assign_cluster_by_homology(df=structures_df,
                                                   sequence_data_dir=sequence_data_dir,
                                                   species_wise_msa_dir=species_wise_msa_dir,
                                                   workdir=f"{workdir}/clustering_by_homology/",
                                                   distances=distances_df.to_numpy())
        structures_df.to_csv(f"{df_output_dir}/{os.path.basename(structures_data_path).replace('.csv', '_clustered_by_homology.csv')}", index=False)
        structures_df_by_clusters = structures_df.groupby("assigned_cluster")

    else:
        logger.info(f"clustering structures by host {host_partition_to_use}")
        clustering_col = f"virus_hosts_{host_partition_to_use}_names"
        logger.info(f"clustering data by {clustering_col} into {len(structures_df[clustering_col].dropna().unique())} groups")
        structures_df[f"virus_hosts_{host_partition_to_use}_names"] = structures_df[
            clustering_col].apply(
            lambda hosts: hosts.split(";") if pd.notna(hosts) else np.nan)
        structures_df_partition = structures_df.explode(clustering_col)
        structures_df_by_clusters = structures_df_partition.groupby(f"virus_hosts_{host_partition_to_use}_names")

    logger.info(
        f"computing inter-cluster and intra-cluster distances across {len(structures_df_partition[f'virus_hosts_{host_partition_to_use}_names'].unique())} clusters")
    compute_clusters_distances(clusters_data=structures_df_by_clusters, distances_df=distances_df, workdir=f"{workdir}/structures_clusters_by_hosts_{host_partition_to_use}/", output_path=f"{df_output_dir}/clusters_by_host_{host_partition_to_use}_distances_within_{group_name}.csv")

if __name__ == '__main__':
    cluster_secondary_structures()



