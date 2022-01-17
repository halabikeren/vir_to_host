import os
import pickle
import shutil
import typing as t
from collections import defaultdict
import random
from time import sleep
import logging

from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from ete3 import Tree

import click
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from itertools import combinations

import nltk
nltk.download('punkt')
import warnings
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec

import sys
sys.path.append("..")
from utils.rna_struct_utils import RNAStructUtils
from utils.pbs_utils import PBSUtils
from utils.clustering_utils import ClusteringUtils

logger = logging.getLogger(__name__)


def is_positive(input_matrix: np.ndarray) -> bool:
    diagonal_values_zeros = np.all([input_matrix[i,i] == 0 for i in range(input_matrix.shape[0])])
    non_diagonal_values_positive = np.all([input_matrix[i,j] > 0 for i in range(input_matrix.shape[0]) for j in range(input_matrix.shape[0]) if i != j])
    return diagonal_values_zeros and non_diagonal_values_positive

def is_symmetric(input_matrix: np.ndarray) -> bool:
    for i in range(input_matrix.shape[0]):
        for j in range(i, input_matrix.shape[0]):
            if input_matrix[i,j] != input_matrix[j,i]:
                return False
    return True

def is_triangle_inequality_held(input_matrix: np.ndarray) -> bool:
        for i in range(input_matrix.shape[0]):
            for j in range(i, input_matrix.shape[0]):
                for k in range(input_matrix.shape[0]):
                    if i != k and j != k:
                        if input_matrix[i,j] > (input_matrix[i,k] + input_matrix[k,j]):
                            return False
        return True

def is_legal_dist_metric(df: pd.DataFrame) -> bool:
    distances = df.to_numpy()
    # fill missing values in the symmetric matrix
    if np.any(pd.isna(distances)):
        distances[pd.isna(distances)] = 0
        distances = np.maximum(distances, distances.transpose())
    positive = is_positive(input_matrix=distances)
    symmetric = is_symmetric(input_matrix=distances)
    triangle_inequality_holds = is_triangle_inequality_held(input_matrix=distances)
    return positive and symmetric and triangle_inequality_holds

def compute_pairwise_distances(ref_structures: pd.Series, other_structures: pd.Series, workdir: str, output_dir: str) -> pd.DataFrame:
    """
    :param ref_structures: reference structures from which the distance to the other structures should be computed
    :param other_structures: structures to compute their distance from the reference structures
    :param workdir: directory to write auxiliary files to (e.g., rnadistance software unparsed result)
    :param output_dir: directory to write the distances dataframes to, by the different metrics provided by rnadistasnce software
    :return:
    """
    os.makedirs(workdir, exist_ok=True)

    # first, write a fasta file with all the structures representations
    other_structures_path = f"{workdir}/other_structures.fasta"
    logger.info(f"writing structures to fasta file")
    other_structures.drop_duplicates(inplace=True)
    if not os.path.exists(other_structures_path):
        with open(other_structures_path, "w") as infile:
            i = 0
            for struct in other_structures:
                infile.write(f">{i}\n{struct}\n")
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
            parent_path = f"'{os.path.dirname(os.getcwd())}'" if "pycharm" not in os.getcwd() else "'/groups/itay_mayrose/halabikeren/vir_to_host/'"
            ref_struct = f"'{ref_struct}'"
            structs_path = f"'{other_structures_path}'"
            cmd = f'python -c "import sys;sys.path.append({parent_path});from utils.rna_struct_utils import RNAStructUtils;RNAStructUtils.exec_rnadistance(ref_struct={ref_struct}, ref_struct_index={i}, structs_path={structs_path}, workdir={job_workdir}, alignment_path={alignment_path}, output_path={output_path})"'
            if not os.path.exists(output_path.replace("'", "")) or not os.path.exists(alignment_path.replace("'", "")):
                if not os.path.exists(job_path):
                    PBSUtils.create_job_file(job_path=job_path, job_name=f"rnadistance_{i}",
                                             job_output_dir=job_output_dir, commands=[cmd], cpus_num=4, ram_gb_size=10)
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
            for j in range(i, len(ref_structures)):
                d = distances_from_i[dist_type][j - i - 1]
                if j == i:
                    d = 0
                distances_dfs[dist_type].loc[list(ref_structures)[i]][list(ref_structures)[j]] = d

    # write dfs to output
    os.makedirs(output_dir, exist_ok=True)
    for dist_type in distances_dfs:
        output_path = f"{output_dir}/{dist_type}_distances.csv"
        distances_dfs[dist_type].to_csv(output_path)

    # clear workspace
    if output_dir != workdir:
        shutil.rmtree(workdir, ignore_errors=True)

    # assert that the return distance metric is legal
    chosen_return_metric = "F"
    assert(is_legal_dist_metric(distances_dfs[chosen_return_metric]))
    # out of the computed distances metrices, only the F and W distance do not violate any of the distance metric conditions: positivity, symmetric, and holding of the triangle inequality
    # as for the edit distance, while levinstein distance is a legal distance metric, because it is computed across already aligned pairwise sequences, and not the original structure sequences, it does not obey the triangle inequality
    return distances_dfs[chosen_return_metric] # use the distance computed based on the full (F) structure as it holds all 3 conditions of a distance metric (the HIT based distance also holds the triangle inequality

def get_distances_from_ref_structures(ref_structures: pd.Series, other_structures: pd.Series, workdir: str, distances_df: t.Optional[pd.DataFrame] = None) -> float:
    """
    :param ref_structures: reference structures from which the distance to the other structures should be computed
    :param other_structures: structures to compute their distance from the reference structures
    :param workdir: directory to write auxiliary rnadistance files to, in the case that the distances df is not provided
    :param distances_df: dataframe of pairwise distances. If not provided, will be created.
    :return: number representing the mean value of the pairwise distances
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
    if len(cluster_structures) == 1:
        return 0
    return get_distances_from_ref_structures(ref_structures=cluster_structures, other_structures=cluster_structures, workdir=workdir, distances_df=distances_df)

def compute_clusters_distances(clusters_data: pd.core.groupby.GroupBy, distances_df: pd.DataFrame, workdir: str, output_path :str):
    """
    :param clusters_data: dataframe with the structures grouped by clusters
    :param distances_df: dataframe of the pairwise distances between structures
    :param workdir: directory to write auxiliary files to
    :param output_path: path to write the results (clusters distances) to
    :return:
    """
    df = pd.DataFrame(index=clusters_data.groups.keys(), columns=["intra_cluster_distance"] + [f"distance_from_{cluster}" for cluster in clusters_data.groups.keys()])

    clusters_sizes = [clusters_data.get_group(cluster).shape[0] for cluster in clusters_data.groups.keys()]
    max_cluster_size = np.max(clusters_sizes)
    logger.info(f"mean cluster size = {np.mean(clusters_sizes)}")

    logger.info(f"computing intra-cluster distances for {len(clusters_data.groups.keys())} clusters of maximum size {max_cluster_size}")
    intra_cluster_distances_workdir = f"{workdir}/intra_cluster_distances/"
    df["intra_cluster_distance"] = df.index.map(lambda cluster: get_intra_cluster_distance(cluster_structures=clusters_data.get_group(cluster).struct_representation, workdir=f"{intra_cluster_distances_workdir}/{cluster}", distances_df=distances_df))
    logger.info(f"mean intra-cluster distance = {np.mean(df['intra_cluster_distance'])}")

    logger.info(f"computing inter-cluster distances across {int(len(clusters_data.groups.keys())**2/2)} combinations of clusters")
    inter_cluster_distances_workdir = f"{workdir}/inter_cluster_distances/"
    for cluster2 in clusters_data.groups.keys():
        df[f"distance_from_{cluster2}"] = df.index.map(lambda cluster1: get_inter_cluster_distance(cluster_1_structures=clusters_data.get_group(cluster1).struct_representation, cluster_2_structures=clusters_data.get_group(cluster2).struct_representation, workdir=f"{inter_cluster_distances_workdir}/{cluster1}_{cluster2}_distances/",  distances_df=distances_df))
    logger.info(f"mean inter-cluster distance = {np.mean(np.mean(df[[col for col in df.columns if 'distance_from_' in col]]))}")

    logger.info(f"writing output to {output_path}")
    df.to_csv(output_path)


def map_structures_to_plane(structures: t.List[str], distances_df: t.Optional[pd.DataFrame], method: str = "relative") -> t.List[np.array]:
    """
    :param structures: list of structures that need to be mapped to a plane
    :param distances_df: distances between structures. Should be provided in case of choosing to vectorize structures using the relative method
    :param method: method for vectorizing structures: either word2vec or relative trajectory based on a sample of structures
    :return: vectors representing the structures, in the same order of the given structures
    using word2vec method: employs CBOW algorithm, inspired by https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
    using relative method: uses an approach inspired by the second conversion of scores to euclidean space in: https://doi.org/10.1016/j.jmb.2018.03.019
    """
    vectorized_structures = []
    if method == "word2vec":
        data = [[struct] for struct in structures]
        model = gensim.models.Word2Vec(data, min_count=1, window=5, vector_size=np.max([len(struct) for struct in structures]))
        for struct in structures:
            vectorized_structures.append(model.wv.get_vector(struct))
    else:
        # take the 100 structures with the largest distances from ome another - each of these will represent an axis
        if len(structures) > 300:
            logger.warning(f"the total number of structures is {len(structures)} > 300 and thus, distance-based vectorization will consider only 300 structures and not all")
        max_num_axes = np.min([300, len(structures)])
        s = distances_df.sum()
        distances_df = distances_df[s.sort_values(ascending=False).index]
        axes_structures = distances_df.index[:max_num_axes]
        for i in range(len(structures)):
            vectorized_structure = np.array([distances_df[structures[i]][axes_structures[j]] for j in range(len(axes_structures))])
            vectorized_structures.append(vectorized_structure)

    return vectorized_structures


def get_gram_matrix(input_matrix: np.ndarray) -> np.ndarray:
    sqrt_dist_vec = np.power(input_matrix[0, :], 2)
    gram_component_2 = np.tile(sqrt_dist_vec, (input_matrix.shape[0], 1))
    gram_component_1 = gram_component_2.T
    gram_mat = (gram_component_1 + gram_component_2 - np.power(input_matrix, 2)) / 2
    return gram_mat

# deprecated - didn't work
def _map_distances_to_plane(distances: np.ndarray) -> np.ndarray:
    """
    :param distances: numpy 2d array of the pairwise distances between records
    :return: 2d coordinates of the respective records, based on their pairwise distances,
             using: https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix
                    theory in: https://link.springer.com/content/pdf/10.1007/BF02288916.pdf
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


def compute_lowest_nodes(parent, threshold, lowest):
    added_child = False
    for child in parent.get_children():
        if child.dist == threshold and child not in lowest:
            lowest.append(child)
            added_child = True
        elif child.dist < threshold:
            prev_lst_size = len(lowest)
            compute_lowest_nodes(child, threshold-child.dist, lowest)
            if len(lowest) > prev_lst_size:
                added_child = True
    if not added_child and parent not in lowest:
        lowest.append(parent)


def get_upgma_based_starting_points(tree: Tree) -> t.Dict[int, t.List[int]]:
    root = tree.get_tree_root()
    assert (len(root.get_children()) == 2)  # upgma is rooted and thus this assert is always expected to pass
    leaves_by_dist_from_root = root.get_leaf_names()
    leaves_by_dist_from_root.sort(key=lambda x: root.get_distance(x))
    k_to_starting_point_indices = defaultdict(list)

    # for each node, list all sibling nodes whose distance from the root is at most the distance of the respective node from the tree
    # each internal node induces a centroid, corresponding to a leaf its its subtree with minimal distance from the root
    # this can be obtained using BFS (Level-order traversal)
    for node in tree.traverse("levelorder"):
        lowest = [node]
        compute_lowest_nodes(root, node.get_distance(root), lowest)
        centroid_indices = []
        for candidate in lowest:
            centroid_indices.append(
                [int(leaf_name) for leaf_name in leaves_by_dist_from_root if leaf_name in candidate.get_leaf_names()][
                    0])
        k_to_starting_point_indices[len(lowest)] = centroid_indices

    return k_to_starting_point_indices



def get_optimal_clusters_num(kmin: int, kmax: int, clustering_df: pd.DataFrame, clustering_coordinates: t.List[np.array], cannot_link: t.List[t.Tuple[int]], distances_df: pd.DataFrame, workdir: str, use_upgma_based_starting_points: bool = True) -> t.Tuple[int, t.Dict[int, t.Dict], t.Dict[int, t.Dict]]:
    """
    :param kmin: minimal number of clusters
    :param kmax: maximal number of clusters
    :param clustering_df: dataframe with objects to cluster
    :param clustering_coordinates: coordinates of the objects to cluster
    :param cannot_link: constraints on indices of objects that cannot be clustered together
    :param distances_df: dataframe with the pairwise distances between structures
    :param workdir directory to save clustering output in
    :param use_upgma_based_starting_points: indicator weather upgma based starting points should be used
    :return: optimal number of clusters, and mapping of the clusters number of assignment amd centroids
    """
    os.makedirs(workdir, exist_ok=True)
    coordinates_vectors = np.stack([np.array(clustering_coordinates[i]) for i in range(len(clustering_coordinates))])
    logger.info(f"searching for optimal clusters number within range ({kmin}, {kmax})")

    k_to_clusters_path = f"{workdir}/k_to_clusters_assignment.pickle"
    k_to_centers_path = f"{workdir}/k_to_cluster_centroids.pickle"

    if not os.path.exists(k_to_clusters_path) or not os.path.exists(k_to_centers_path):

        # build upgma tree based on pairwise distances between structures, that will be used for initialization of centers in the k-means executions
        if use_upgma_based_starting_points:
            logger.info(f"building upgma tree to derive inital centroids for cop-kmeans clustering")
            constructor = DistanceTreeConstructor()
            distances = distances_df.to_numpy()
            distances_lst = distances.tolist()
            for i in range(distances.shape[0]): # turn matrix into a lower triangle one, as biopython requires
                distances_lst[i] = distances_lst[i][:i+1]
            distance_matrix = DistanceMatrix(names=[str(i) for i in distances_df.index], matrix=distances_lst)
            tree_path = f"{workdir}/structures_upgma_tree.nwk"
            if not os.path.exists(tree_path):
                upgma_structures_tree = constructor.upgma(distance_matrix) # 32.88 of the tree internal nodes have leaf children from the same species
                Phylo.write(upgma_structures_tree, tree_path, "newick")
            upgma_structures_tree = Tree(tree_path, format=1)
            starting_points_path = f"{workdir}/upgma_based_centers.pickle"
            if not os.path.exists(starting_points_path):
                cluster_size_to_starting_points = get_upgma_based_starting_points(tree=upgma_structures_tree)
                with open(starting_points_path, "wb") as outfile:
                    pickle.dump(obj=cluster_size_to_starting_points, file=outfile)
            else:
                with open(starting_points_path, "rb") as infile:
                    cluster_size_to_starting_points = pickle.load(file=infile)
            k_with_stating_points =list(cluster_size_to_starting_points.keys())
            k_with_stating_points.sort()

        k_to_clusters_assignment = dict()
        k_to_cluster_centroids = dict()
        starting_points_coordinates = []
        for k in range(kmin, kmax):  # switch with binary search , stop upn maximum of sl score
           if use_upgma_based_starting_points:
            if k in cluster_size_to_starting_points:
                starting_points_indices = cluster_size_to_starting_points[k]
            else: # choose the closest and remove some starting points
                i = len(k_with_stating_points)-1
                while i > 0 and k_with_stating_points[i] > k:
                    i -= 1
                closest_k = k_with_stating_points[i+1]
                starting_points_indices = random.sample(population=cluster_size_to_starting_points[closest_k], k=k)
            starting_points_coordinates = [np.array(clustering_coordinates[i]) for i in starting_points_indices]
            clusters, centers = ClusteringUtils.cop_kmeans_with_initial_centers(dataset=coordinates_vectors, k=k, cl=cannot_link, initial_centers=starting_points_coordinates)
            if clusters is not None:
                unique_clusters = list(set(clusters))
                cluster_to_structures_indices = {
                    cluster: [list(clustering_df.index)[i] for i in range(len(clustering_df.index)) if
                              clusters[i] == cluster] for cluster in
                    unique_clusters}
                k_to_clusters_assignment[k] = dict()
                for i in clustering_df.index:
                    k_to_clusters_assignment[k][i] = [cluster for cluster in cluster_to_structures_indices if
                                                      i in cluster_to_structures_indices[cluster]][0]
                k_to_cluster_centroids[k] = {i: centers[k_to_clusters_assignment[k][i]] for i in clustering_df.index}

        # save clustering for latest usage
        with open(k_to_clusters_path, "wb") as outfile:
            pickle.dump(obj=k_to_clusters_assignment, file=outfile)
        with open(k_to_centers_path, "wb") as outfile:
            pickle.dump(obj=k_to_cluster_centroids, file=outfile)

    else:
        with open(k_to_clusters_path, "rb") as infile:
            k_to_clusters_assignment = pickle.load(file=infile)
        with open(k_to_centers_path, "rb") as infile:
            k_to_cluster_centroids = pickle.load(file=infile)

    prev_score, curr_score = float("-inf"), float("-inf")
    k_to_score = dict()
    for k in range(kmin, kmax):
        clusters = [k_to_clusters_assignment[k][i] for i in clustering_df.index]
        if clusters is not None and k > 1:
            k_to_score[k] = silhouette_score(coordinates_vectors, clusters, metric='euclidean')
            curr_score = k_to_score[k]
        logger.info(f"k={k} yields silhouette score of {curr_score}")
        if curr_score < prev_score: # under assumption of global maximum, if we got deterioration - we should stop at the prev value
            logger.info(f"silhouette score has been reduced with increase to {k} - possibly reached a local maxima at {k-1}")
        prev_score = curr_score
    optimal_k = max(k_to_score, key=k_to_score.get)
    logger.info(f"the optimal number of clusters based on silhouette scores is {optimal_k} with a score of {k_to_score[optimal_k]}")

    return optimal_k, k_to_clusters_assignment, k_to_cluster_centroids


def assign_cluster_by_homology(df: pd.DataFrame, sequence_data_dir: str, species_wise_msa_dir: str, workdir: str, distances_df: pd.DataFrame, use_upgma_based_starting_points: bool = True) -> t.Tuple[pd.DataFrame, np.ndarray]:
    """
    :param df: dataframe of structure records to assign to clusters
    :param sequence_data_dir: directory holding the original sequence data before filtering out outliers. this directory should also hold similarity values tables per species
    :param species_wise_msa_dir: directory holding the aligned genomes per species after filtering out outliers, which were used in the inference process of the secondary structures
    :param workdir: directory to write family sequence data and align it in
    :param distances_df: dataframe of pairwise distances between structures
    :param use_upgma_based_starting_points: indicator weather upgma based starting points should be used
    :return: same input df with an added column of the assigned_cluster. clusters are represented by a combination of window range and an index.
    """

    os.makedirs(workdir, exist_ok=True)

    # create mapping of structures start and end positions from species-wise alignments to family-wise alignments
    df = RNAStructUtils.map_species_wise_pos_to_group_wise_pos(df=df,
                                                               seq_data_dir=sequence_data_dir,
                                                               species_wise_msa_dir=species_wise_msa_dir,
                                                               workdir=workdir)

    # map the pairwise distance to a 2d plane and assign to each record a respective 2d coordination
    structures = list(distances_df.index)
    logger.info(f"mapping structures to a place of dimension {np.min([len(structures), 300])} based on their distances")
    coordinates = map_structures_to_plane(structures=structures, distances_df=distances_df, method = "relative")
    clustering_coordinates = [coordinates[i] for i in df.index]

    # for each segment, cluster structures falling in it according to their distances
    # project pairwise distances to points in space and add 2d coordination to each record in the dataframe accordingly
    df["assigned_cluster"] = np.nan
    df["cluster_centroid"] = np.nan
    df.reset_index(inplace=True) # must be done to provide cop_kmeans indices which correspond to the sub-coordinates and not all the coordinates

    # gather constraints on records that belong to the same species, and thus must appear in different clusters
    def get_pairwise_combos(iterable): # not returning all the pairs
        return [tuple(map(int, comb)) for comb in combinations(iterable, 2)]
    cannot_link = []
    data_by_sp = df.groupby(["virus_species_name"])
    for group in data_by_sp.groups.keys():
        group_df = data_by_sp.get_group(group)
        cannot_link += list(get_pairwise_combos(group_df.index))

    # apply distance-based clustering using kmeans clustering with silhouette-based optimization of k
    # use COP-Kmeans (https://github.com/Behrouz-Babaki/COP-Kmeans) to constrain clusters to have at most one copy per species
    # theory in: https://link.springer.com/content/pdf/10.1007/BF02288916.pdf
    best_clustering = {list(df.index)[i]: i for i in range(len(list(df.index)))} # default clustering - cluster per structure
    best_clustering_centroids = {list(df.index)[i]: clustering_coordinates[i] for i in range(len(list(df.index)))}
    if df.shape[0] > 1:
        kmin = np.max([2, np.max([data_by_sp.get_group(sp).shape[0] for sp in data_by_sp.groups.keys()])]) # the number of cluster must equal at least the number of representatives per species, as two representatives of the same species cannot appear in the same cluster
        kmax = df.shape[0]-1 # at the worst case, each structure will have its own cluster. we do not want to account for such case, which is naturally optimal, so we end and this number deducted by 1

        best_k, k_to_clusters_assignment, k_to_cluster_centroids = get_optimal_clusters_num(kmin=kmin, kmax=kmax,
                                          clustering_df=df, clustering_coordinates=clustering_coordinates,
                                          cannot_link=cannot_link, distances_df=distances_df,
                                          workdir=f"{workdir}/cop_clustering/", use_upgma_based_starting_points=use_upgma_based_starting_points)
        best_clustering = k_to_clusters_assignment[best_k]
        best_clustering_centroids = {i: k_to_cluster_centroids[best_k][i] for i in df.index}

    df["assigned_cluster"].fillna(value=best_clustering, inplace=True)
    df["cluster_centroid"].fillna(value=best_clustering_centroids, inplace=True)
    return df, coordinates


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
@click.option(
    "--use_upgma",
    type=click.BOOL,
    help="indicator weather upgma tree should be used to derive initial starting points for cop kmeans clustering by homology",
    required=False,
    default=True
)
def cluster_secondary_structures(structures_data_path: str,
                                 by: str,
                                 sequence_data_dir: str,
                                 species_wise_msa_dir: str,
                                 host_partition_to_use: str,
                                 workdir: str,
                                 log_path: str,
                                 df_output_dir: str,
                                 use_upgma: bool):

    # initialize the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path)),
        ],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    logger.info(f"loading viral rna secondary structures data from {structures_data_path}")
    structures_df = pd.read_csv(structures_data_path)
    structures_df = structures_df.dropna(subset=["struct_representation"], axis=0)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(df_output_dir, exist_ok=True)

    # compute pairwise distances between structures
    logger.info(f"computing distances across structures")
    df_output_path = f"{workdir}/distances_all_structures/F_distances.csv"
    if os.path.exists(df_output_path):
        distances_df = pd.read_csv(df_output_path)
        if "struct_representation" in distances_df.columns:
            distances_df.set_index("struct_representation", inplace=True)
    else:
        ref_structures = structures_df.struct_representation
        other_structures = structures_df.struct_representation
        distances_df = compute_pairwise_distances(ref_structures=ref_structures, other_structures=other_structures,
                                                  workdir=f"{workdir}/rnadistance_all_structures/",
                                                  output_dir=f"{workdir}/distances_all_structures/")

    # assign structures to clusters
    if by != "column":
        logger.info(f"clustering structures by homology")
        distances = distances_df.to_numpy()
        # fill missing values in the symmetric matrix
        if np.any(pd.isna(distances)):
            distances[pd.isna(distances)] = 0
            distances = np.maximum(distances, distances.transpose())
        distances_clustering_df = pd.DataFrame(distances)
        structures_df = assign_cluster_by_homology(df=structures_df,
                                                   sequence_data_dir=sequence_data_dir,
                                                   species_wise_msa_dir=species_wise_msa_dir,
                                                   workdir=f"{workdir}/clustering_by_homology{'_using_upgma_sp' if use_upgma else ''}/",
                                                   distances_df=distances_clustering_df,
                                                   use_upgma_based_starting_points=use_upgma)

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

    compute_clusters_distances(clusters_data=structures_df_by_clusters, distances_df=distances_df, workdir=f"{workdir}/structures_clusters/", output_path=f"{df_output_dir}/clusters_by_host_{host_partition_to_use}_distances.csv")

    # scatter clusters in 2d space based on the first 2 pcs of the structures coordinates
    pca = PCA(n_components=2)

    pca.fit()

if __name__ == '__main__':
    cluster_secondary_structures()



