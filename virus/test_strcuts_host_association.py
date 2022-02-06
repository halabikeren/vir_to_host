import os
import re
import shutil
import tarfile
import sys
import typing as t
from time import sleep

import mysql.connector
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from ete3 import Tree
from statsmodels.stats.multitest import multipletests

sys.path.append("..")
from utils.data_collecting_utils import MySQLUtils
from serivces.pbs_service import PBSService

import click
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def get_virus_name_id_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection, output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
    :param output_path: path to write the output dataframe to
    :return: the output dataframe
    """

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    virus_id_name_query = 'SELECT ncbi_id, species FROM taxonomy WHERE tax_string LIKE "%Viruses%";'
    rfam_virus_name_to_virus_id_df = pd.read_sql(sql=virus_id_name_query, con=connection)
    rfam_virus_name_to_virus_id_df.to_csv(output_path, index=False)

    return rfam_virus_name_to_virus_id_df


def get_id_to_rfamseq_acc_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection,
    query_rfam_viruses_ids: t.List[str],
    query_batch_size: int,
    output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
    :param query_rfam_viruses_ids: viruses ids to query on
    :param query_batch_size: batch size for sqlite queries
    :param output_path: path to write the output dataframe to
    :return: the output dataframe
    :return:
    """

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    query_template = "SELECT DISTINCT tx.ncbi_id, rf.rfamseq_acc FROM taxonomy tx, rfamseq rf WHERE tx.ncbi_id IN ({}) AND rf.ncbi_id = tx.ncbi_id;"
    query_workdir = f"{os.path.dirname(output_path)}/rfam_acc_queries/"
    rfam_virus_ids_to_rfamseq_acc_df = MySQLUtils.do_batch_query(
        connection=connection,
        query_template=query_template,
        query_items=query_rfam_viruses_ids,
        workdir=query_workdir,
        batch_size=query_batch_size,
    )

    rfam_virus_ids_to_rfamseq_acc_df.to_csv(output_path, index=False)
    shutil.rmtree(query_workdir, ignore_errors=True)

    return rfam_virus_ids_to_rfamseq_acc_df


def get_rfamseq_acc_to_rfam_acc_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection,
    query_rfamseq_acc_ids: t.List[str],
    query_batch_size: int,
    output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
    :param query_rfamseq_acc_ids: rfamseq accessions to query on
    :param query_batch_size: batch size for sqlite queries
    :param output_path: path to write the output dataframe to
    :return: the output dataframe
    :return:
    """

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    query_template = "SELECT DISTINCT rf.rfamseq_acc, fr.rfam_acc FROM rfamseq rf, full_region fr WHERE rf.rfamseq_acc IN ({}) AND rf.rfamseq_acc = fr.rfamseq_acc AND fr.is_significant = 1;"
    query_workdir = f"{os.path.dirname(output_path)}/rfam_seq_acc_queries/"
    rfamseq_acc_to_rfam_acc_df = MySQLUtils.do_batch_query(
        connection=connection,
        query_template=query_template,
        query_items=query_rfamseq_acc_ids,
        workdir=query_workdir,
        batch_size=query_batch_size,
    )

    rfamseq_acc_to_rfam_acc_df.to_csv(output_path, index=False)
    shutil.rmtree(query_workdir, ignore_errors=True)

    return rfamseq_acc_to_rfam_acc_df


def get_rfam_acc_to_rfam_id_mapping(
    connection: mysql.connector.connection_cext.CMySQLConnection,
    query_rfam_acc_ids: t.List[str],
    query_batch_size: int,
    output_path: str,
) -> pd.DataFrame:
    """
    :param connection: db connection to rfam db
    :param query_rfam_acc_ids: rfam accessions to query on
    :param query_batch_size: batch size for sqlite queries
    :param output_path: path to write the output dataframe to
    :return: the output dataframe
    :return:
    """

    if os.path.exists(output_path):
        return pd.read_csv(output_path)

    query_template = "SELECT DISTINCT fr.rfam_acc, fm.rfam_id, fm.description FROM full_region fr, family fm WHERE fr.rfam_acc IN ({}) AND fr.rfam_acc = fm.rfam_acc;"
    query_workdir = f"{os.path.dirname(output_path)}/rfam_id_queries/"
    rfam_acc_to_rfam_id_df = MySQLUtils.do_batch_query(
        connection=connection,
        query_template=query_template,
        query_items=query_rfam_acc_ids,
        workdir=query_workdir,
        batch_size=query_batch_size,
    )

    rfam_acc_to_rfam_id_df.to_csv(output_path, index=False)
    shutil.rmtree(query_workdir, ignore_errors=True)

    return rfam_acc_to_rfam_id_df


def get_viral_rfam_data(output_path: str) -> pd.DataFrame:
    """
    :param output_path: path ot write the collected data to
    :return:
    """
    # connect to public rfam db using details in https://docs.rfam.org/en/latest/database.html
    connection = mysql.connector.connect(
        user="rfamro", host="mysql-rfam-public.ebi.ac.uk", port="4497", database="Rfam"
    )

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    rfam_virus_name_to_virus_id_df = get_virus_name_id_mapping(
        connection=connection, output_path=f"{output_dir}/rfam_virus_name_to_virus_id.csv",
    )

    rfam_virus_ids = [str(int(item)) for item in rfam_virus_name_to_virus_id_df["ncbi_id"].unique()]
    rfam_virus_ids_to_rfamseq_acc_df = get_id_to_rfamseq_acc_mapping(
        connection=connection,
        query_rfam_viruses_ids=rfam_virus_ids,
        output_path=f"{output_dir}/rfam_virus_ids_to_rfamseq_acc.csv",
        query_batch_size=1000,
    )

    df = rfam_virus_name_to_virus_id_df.merge(rfam_virus_ids_to_rfamseq_acc_df, on="ncbi_id", how="left")

    rfamseq_acc_ids = ['"' + item + '"' for item in rfam_virus_ids_to_rfamseq_acc_df["rfamseq_acc"].unique()]
    rfamseq_acc_to_rfam_acc_df = get_rfamseq_acc_to_rfam_acc_mapping(
        connection=connection,
        query_rfamseq_acc_ids=rfamseq_acc_ids,
        output_path=f"{output_dir}/rfamseq_acc_to_rfam_acc.csv",
        query_batch_size=1000,
    )

    df = df.merge(rfamseq_acc_to_rfam_acc_df, on="rfamseq_acc", how="left")

    rfam_acc_ids = ['"' + item + '"' for item in rfamseq_acc_to_rfam_acc_df["rfam_acc"].unique()]
    rfam_acc_to_rfam_id_df = get_rfam_acc_to_rfam_id_mapping(
        connection=connection,
        query_rfam_acc_ids=rfam_acc_ids,
        output_path=f"{output_dir}/rfam_acc_to_rfam_id.csv",
        query_batch_size=1000,
    )

    df = df.merge(rfam_acc_to_rfam_id_df, on="rfam_acc", how="left")

    relevant_df = df[["ncbi_id", "species", "rfam_acc"]].rename(
        columns={"ncbi_id": "species_id", "species": "species_name"}
    )

    relevant_df.dropna(subset=["rfam_acc"], how="any", axis=0, inplace=True)
    relevant_df.drop_duplicates(inplace=True)
    relevant_df.to_csv(f"{output_dir}/rfam_data.csv", index=False)

    connection.close()

    return relevant_df


def write_sequence_db(sequence_data: pd.DataFrame, seq_db_path: str):
    """
    :param sequence_data: dataframe with sequence data
    :param seq_db_path:path to write the database to
    :return: none
    """
    if not os.path.exists(seq_db_path):
        db_records = []
        for i, row in sequence_data.iterrows():
            db_records.append(
                SeqRecord(id=row.accession, name=row.accession, description=row.accession, seq=Seq(row.sequence))
            )
        SeqIO.write(db_records, seq_db_path, format="fasta")


def get_rfam_cm_models(required_rfam_ids: t.List[str], output_dir: str, wget_path: str):
    """
    :param required_rfam_ids: rfam ids to get alignments for
    :param output_dir: directory to write the alignments to
    :param wget_path: wget url of the cm models
    :return: none
    """
    os.makedirs(output_dir, exist_ok=True)
    output_paths = [f"{output_dir}/{rfam_id}.cm" for rfam_id in required_rfam_ids]
    cm_models_available = np.all([os.path.exists(output_path) for output_path in output_paths])
    if not cm_models_available:
        zipped_output_path = f"{os.getcwd()}/{os.path.basename(wget_path)}"
        if not os.path.exists(zipped_output_path):
            res = os.system(f"wget {wget_path}")
        # unzip output to output_dir
        with tarfile.open(zipped_output_path, "r:gz") as file:
            file.extractall(output_dir)
        for path in os.listdir(output_dir):
            if path.replace(".cm", "") not in required_rfam_ids:
                os.remove(f"{output_dir}/{path}")


def apply_infernal_search(cm_models_dir: str, workdir: str, output_dir: str, db_path: str):
    """
    :param cm_models_dir: directory of covariance models of relevant rfam ids
    :param workdir: path for write the jobs of the pipeline per alignment in
    :param output_dir: directory to write the pipeline outputs on the cm models in
    :param db_path: path to sequence db file
    :return: none
    """
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for path in os.listdir(cm_models_dir):
        rfam_id = path.replace(".cm", "")
        search_output_dir = f"{output_dir}/{rfam_id}/"
        os.makedirs(search_output_dir, exist_ok=True)
        if len(os.listdir(search_output_dir)) == 0:
            rfam_workdir = f"{workdir}/{rfam_id}/"
            os.makedirs(rfam_workdir, exist_ok=True)
            cov_model_path = f"{cm_models_dir}/{path}"
            job_name = f"cmsearch_{rfam_id}"
            job_path = f"{rfam_workdir}/{job_name}.sh"
            cmd = f"cmsearch -A {search_output_dir}aligned_hits.fasta --tblout {search_output_dir}hists.tsv {cov_model_path} {db_path} > {search_output_dir}cmsearch.out"
            PBSService.create_job_file(
                job_name=job_name,
                job_output_dir=rfam_workdir,
                job_path=job_path,
                commands=[cmd],
                cpus_num=2,
                ram_gb_size=10,
            )
            os.system(f"qsub {job_path}")
    complete = np.all(
        [len(os.listdir(f"{workdir}/{path.replace('.cm', '')}/")) > 1 for path in os.listdir(cm_models_dir)]
    )
    while not complete:
        sleep(2 * 60)
        complete = np.all(
            [len(os.listdir(f"{workdir}/{path.replace('.cm', '')}/")) > 1 for path in os.listdir(cm_models_dir)]
        )


def get_rfam_species_hits(
    rfam_ids: t.List[str], internal_results_dir: str, accession_to_species_map: str
) -> t.Dict[str, t.List[str]]:
    """
    :param rfam_ids: ids of rfam for which infernal search has been applied
    :param internal_results_dir: oath to the results of infernal search
    :param accession_to_species_map: map of accessions to their respective species
    :return: map fo rfam ids to their hit species
    """
    rfam_id_to_species = dict()
    for rfam_id in rfam_ids:
        hits_table_path = f"{internal_results_dir}/{rfam_id}/hits.tsv"
        hits_alignment_path = f"{internal_results_dir}/{rfam_id}/aligned_hits.fasta"
        if (
            os.path.exists(hits_table_path)
            and os.path.exists(hits_alignment_path)
            and os.stat(hits_alignment_path).st_size > 0
        ):
            rfam_hits = pd.read_csv(hits_alignment_path, sep="\s+", skiprows=[1])
            rfam_hits_accessions = rfam_hits.loc[rfam_hits["#target"] != "#", "#target"]
            rfam_hits_species = list(set([accession_to_species_map[acc] for acc in rfam_hits_accessions]))
            rfam_id_to_species[rfam_id] = rfam_hits_species
    return rfam_id_to_species


def get_rfam_pa_matrix(
    viral_species: t.List[str],
    rfam_ids: t.List[str],
    infernal_results_dir: str,
    accession_to_species_map: t.Dict[str, str],
    output_path: str,
):
    """
    :param viral_species: list of viral species names (corresponds to rows)
    :param rfam_ids: list of rfam ids of interest (corresponds to columns)
    :param infernal_results_dir: directory of infernal search results on the respective rfam ids
    :param accession_to_species_map: map of accessions to their species
    :param output_path: path to write the pa matrix to
    :return: pa matrix
    """

    if os.path.exists(output_path):
        pa_matrix = pd.read_csv(output_path).rename(columns={"Unnamed: 0": "species_name"})
        if "species_name" in pa_matrix.columns:
            pa_matrix.set_index("species_name", inplace=True)
        return pa_matrix

    rfam_id_to_species = get_rfam_species_hits(
        rfam_ids=rfam_ids, internal_results_dir=infernal_results_dir, accession_to_species_map=accession_to_species_map,
    )
    species_to_rfam_ids = {
        species: {
            rfam_id: 1 if rfam_id in rfam_id_to_species and species in rfam_id_to_species[rfam_id] else 0
            for rfam_id in rfam_ids
        }
        for species in viral_species
    }
    degenerate_rfam_pa_matrix = pd.DataFrame(species_to_rfam_ids).transpose()
    nondegenerate_rfam_pa_matrix = degenerate_rfam_pa_matrix.loc[:, (degenerate_rfam_pa_matrix != 0).any()]
    nondegenerate_rfam_pa_matrix.to_csv(output_path, index_label="species_name")

    return nondegenerate_rfam_pa_matrix


def compute_kinship_matrix(tree: Tree, samples_to_include: t.List[str]):
    """
    :param tree: tree with the species of interest
    :param samples_to_include: sample ids to include in the data
    :return: none
    """
    kinship_matrix = pd.DataFrame(columns=samples_to_include, index=samples_to_include)
    for i in range(len(samples_to_include)):
        for j in range(i, len(samples_to_include)):
            if i == j:
                kinship_matrix.loc[kinship_matrix.index == samples_to_include[i], samples_to_include[j]] = 0
            else:
                dist = tree.get_distance(samples_to_include[i], samples_to_include[j])
                kinship_matrix.loc[kinship_matrix.index == samples_to_include[i], samples_to_include[j]] = dist
                kinship_matrix.loc[kinship_matrix.index == samples_to_include[j], samples_to_include[i]] = dist
    # convert distance values to similarity values by reducing the original values from the maximal distance
    max_dist = np.max(np.max(kinship_matrix, axis=1))
    kinship_matrix = max_dist - kinship_matrix

    # standardize the matrix
    # (as the matrix is symmetric, standardizing rows should have them same effect as standardizing columns - check!)
    normalized_kinship_matrix = (kinship_matrix - kinship_matrix.mean()) / kinship_matrix.std()
    return normalized_kinship_matrix


def process_samples_data(pa_matrix: pd.DataFrame, samples_to_include: t.List[str]):
    """
    :param pa_matrix: test subjects presence absence matrix across samples
    :param samples_to_include: sample ids to include in the data
    :return: none
    """
    pa_matrix = pa_matrix[pa_matrix.index.isin(samples_to_include)]
    pa_matrix.sort_index(inplace=True)
    processed_pa_matrix = pa_matrix.T
    col_names = processed_pa_matrix.columns.to_list()
    processed_pa_matrix["demi_allele_1"] = "A"
    processed_pa_matrix["demi_allele_2"] = "T"
    processed_pa_matrix = processed_pa_matrix[["demi_allele_1", "demi_allele_2"] + col_names]
    logger.info(f"processed samples elements data for {pa_matrix.shape[0]} samples and {pa_matrix.shape[1]} elements")
    return processed_pa_matrix


def process_samples_trait(
    samples_trait_data: pd.DataFrame, sample_id_name: str, trait_name: str, samples_to_include: t.List[str]
) -> pd.DataFrame:
    """
    :param samples_trait_data: trait data of the samples
    :param sample_id_name: name of samples id column
    :param trait_name: name of column corresponding ot the trait of interest
    :param samples_to_include: sample ids to include in the data
    :return: the processed samples data
    """
    relevant_samples_trait_data = samples_trait_data.loc[samples_trait_data[sample_id_name].isin(samples_to_include)][
        [sample_id_name, trait_name]
    ].drop_duplicates()
    relevant_samples_trait_data.sort_values([sample_id_name, trait_name], inplace=True)
    processed_samples_trait_data = relevant_samples_trait_data.groupby(sample_id_name).agg(
        lambda x: ";".join(list(x.dropna()))
    )
    processed_samples_trait_data.sort_index(inplace=True)
    processed_samples_trait_data[trait_name] = pd.Categorical(processed_samples_trait_data[trait_name])
    processed_samples_trait_data[trait_name] = processed_samples_trait_data[trait_name].cat.codes
    processed_samples_trait_data.replace(np.nan, "NA")
    logger.info(f"processed trait data for {processed_samples_trait_data.shape[0]} samples")
    return processed_samples_trait_data


def apply_lmm_association_test(
    pa_matrix: pd.DataFrame,
    samples_trait_data: pd.DataFrame,
    sample_id_name: str,
    trait_name: str,
    tree_path: str,
    output_dir: str,
    multiple_test_correction_method: str,
) -> pd.DataFrame:
    """
    :param pa_matrix: presence absence matrix of elements to test
    :param samples_trait_data: dataframe with map of samples to trait value
    :param sample_id_name:name of sample identifier column
    :param trait_name:name of trait identifier column
    :param tree_path: path to tree based on which a kinship matrix will be computed
    :param output_dir: directory in which gemma output will be written
    :param multiple_test_correction_method method for correcting for multiple tests
    :return: dataframe of the association test results
    """

    os.makedirs(output_dir, exist_ok=True)
    kinship_matrix_path = f"{output_dir}/kinship_matrix.csv"
    samples_data_path = f"{output_dir}/samples_pa_data.csv"
    samples_trait_path = f"{output_dir}/samples_trait_data.csv"
    test_results_suffix = "gemma_test_result"
    results_path = f"{output_dir}output/{test_results_suffix}.assoc.txt"

    if not os.path.exists(results_path):

        if (
            not os.path.exists(kinship_matrix_path)
            or not os.path.exists(samples_data_path)
            or not os.path.exists(samples_trait_path)
        ):
            tree = Tree(tree_path)
            kinship_samples = tree.get_leaf_names()
            pa_samples = pa_matrix.index.values
            trait_samples = samples_trait_data[sample_id_name].values
            intersection_species = [
                sample for sample in kinship_samples if sample in pa_samples and sample in trait_samples
            ]
            if not os.path.exists(kinship_matrix_path):
                kinship_matrix = compute_kinship_matrix(tree=tree, samples_to_include=intersection_species)
                kinship_matrix.to_csv(kinship_matrix_path, index=False, header=False)

            if not os.path.exists(samples_trait_path):
                samples_traits = process_samples_trait(
                    samples_trait_data=samples_trait_data,
                    sample_id_name=sample_id_name,
                    trait_name=trait_name,
                    samples_to_include=intersection_species,
                )
                samples_traits.to_csv(samples_trait_path, index=False, header=False)

            if not os.path.exists(samples_data_path):
                samples_pa_matrix = process_samples_data(pa_matrix=pa_matrix, samples_to_include=intersection_species)
                samples_pa_matrix.to_csv(samples_data_path, header=False)

        orig_dir = os.getcwd()
        os.chdir(output_dir)
        gemma_cmd = f"gemma -g {samples_data_path} -p {samples_trait_path} -k {kinship_matrix_path} -lmm -o {test_results_suffix} "
        res = os.system(gemma_cmd)
        os.chdir(orig_dir)
        if res != 0:
            error_msg = (
                f"test association failed due to error and so output was not created in {results_path}. exist "
                f"code = {res} "
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    results = pd.read_csv(results_path, sep="\t")
    results.sort_values("p_wald", inplace=True)
    _, correct_p_wald, _, _ = multipletests(
        pvals=results.p_wald, alpha=0.05, method=multiple_test_correction_method, is_sorted=True,
    )  # add correction for multiple testing
    results["corrected_p_wald"] = correct_p_wald
    results.to_csv(results_path)
    return results


def infer_novel_seeds(species: t.List[str], sequence_alignment_path: str, novel_seeds_dir: str, workdir: str):
    """
    :param species:
    :param sequence_alignment_path:
    :param novel_seeds_dir:
    :param workdir:
    :return:
    """
    os.makedirs(novel_seeds_dir, exist_ok=True)
    parent_path = f"'/groups/itay_mayrose/halabikereb'"
    # for each species, infer structure-guided alignments of suspected structural regions across their genomic alignment
    alignments_paths = [
        f"{sequence_alignment_path}{re.sub('[^A-Za-z0-9]+', '_', sp)}_aligned.fasta"
        for sp in species
        if os.path.exists(f"{sequence_alignment_path}{re.sub('[^A-Za-z0-9]+', '_', sp)}_aligned.fasta")
    ]
    logger.info(f"out of {len(species)} species of interest, {len(alignments_paths)} have available genomic alignments")
    jobs_dir = f"{workdir}jobs/"
    os.makedirs(jobs_dir, exist_ok=True)
    output_dir = f"{workdir}infer_structural_regions/"
    os.makedirs(output_dir, exist_ok=True)
    output_to_wait_for, jobs_paths = [], []
    for alignment_path in alignments_paths:
        species_filename = os.path.basename(alignment_path).replace("_aligned.fasta", "")
        species_jobdir = f"{jobs_dir}/{species_filename}/"
        os.makedirs(species_jobdir, exist_ok=True)
        species_outdir = f"{output_dir}/{species_filename}/"
        os.makedirs(species_outdir, exist_ok=True)
        cmd_alignment_path = f"'{alignment_path}'"
        cmd_workdir = f"'{species_outdir}'"
        log_init = "logging.basicConfig(level=logging.INFO,format='%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s', handlers=[logging.StreamHandler(sys.stdout),],force=True)"
        cmd = f'python -c "import logging, sys;{log_init};sys.path.append({parent_path});from utils.rna_struct_utils import RNAStructUtils;RNAStructUtils.infer_structural_regions(alignment_path={cmd_alignment_path}, workdir={cmd_workdir})"'
        job_name = species_filename
        job_path = f"{jobs_dir}/{job_name}.sh"
        if not os.path.exists(job_path):
            PBSService.create_job_file(
                job_path=job_path,
                job_name=job_name,
                job_output_dir=species_jobdir,
                commands=[cmd],
                cpus_num=2,
                ram_gb_size=8,
            )
        if len(os.listdir(species_outdir)) == 0:
            output_to_wait_for.append(species_jobdir)
            jobs_paths.append(job_path)
    logger.info(f"# jobs to submit = {len(jobs_paths)}")
    for job_path in jobs_paths:
        res = os.system(f"qsub {job_path}")
    complete = np.all([len(os.listdir(outdir)) > 0 for outdir in output_to_wait_for])
    while not complete:
        sleep(2 * 60)
        complete = np.all([len(os.listdir(outdir)) > 0 for outdir in output_to_wait_for])

    # create covariance models for each available alignment of suspected structural region
    cov_models_inference_jobs_dir = f"{workdir}infer_cov_models/"
    os.makedirs(cov_models_inference_jobs_dir, exist_ok=True)
    cov_models_paths = []
    jobs_paths = []
    for sp in species:
        species_filename = re.sub("[^A-Za-z0-9]+", "_", sp)
        species_cov_models_inference_jobs_dir = f"{cov_models_inference_jobs_dir}/{species_filename}/"
        os.makedirs(species_cov_models_inference_jobs_dir, exist_ok=True)
        refined_alignments_dir = f"{output_dir}/{species_filename}/rnaz_candidates_mlocarna_aligned/"
        clustal_refined_alignments_paths = [
            f"{refined_alignments_dir}{path}" for path in os.listdir(refined_alignments_dir) if path.endswith("clustal")
        ]
        fasta_refined_alignments_paths = []
        # convert formats from clustal to fasta
        for clustal_path in clustal_refined_alignments_paths:
            fasta_path = clustal_path.replace(".clustal", ".fasta")
            records = list(SeqIO.parse(clustal_path, format="clustal"))
            SeqIO.write(records, fasta_path, format="fasta")
            os.remove(clustal_path)
            fasta_refined_alignments_paths.append(fasta_path)
        # infer covariance models
        species_cov_models_dir = f"{novel_seeds_dir}/{species_filename}"
        os.makedirs(species_cov_models_dir)
        for alignment_path in fasta_refined_alignments_paths:
            job_output_path = (
                f"{species_cov_models_inference_jobs_dir}/{os.path.basename(alignment_path).replace('.fasta', '')}"
            )
            cov_model_path = f"{species_cov_models_dir}/{os.path.basename(alignment_path).replace('.fasta', '.cm')}"
            cmds = [
                "source /groups/itay_mayrose/halabikeren/.bashrc",
                f"cmbuild --noss {cov_model_path} {alignment_path}",
            ]
            job_name = f"{species_filename}_{os.path.basename(alignment_path).replace('.fasta', '')}"
            job_path = f"{species_cov_models_inference_jobs_dir}/{job_name}.sh"
            if not os.path.exists(job_path):
                PBSService.create_job_file(
                    job_path=job_path,
                    job_name=job_name,
                    job_output_dir=job_output_path,
                    commands=cmds,
                    cpus_num=1,
                    ram_gb_size=8,
                )
            if not os.path.exists(cov_model_path):
                cov_models_paths.append(cov_model_path)
                jobs_paths.append(job_path)

    logger.info(f"{len(jobs_paths)} jobs to submit for inferring covariance models from candidate structural regions")
    job_index = 0
    while PBSService.compute_curr_jobs_num() < 1500:
        res = os.system(f"qsub {jobs_paths[job_index]}")
        job_index += 1

    complete = np.all([os.path.exists(path) and os.path.getsize(path) > 0 for path in cov_models_paths])
    while not complete:
        sleep(2 * 60)
        complete = np.all([os.path.exists(path) for path in cov_models_paths])

    # calibrate covariance models
    cov_models_inference_jobs_dir = f"{workdir}calibrate_cov_models/"
    # for cov_model_path in cov_models_paths:


@click.command()
@click.option(
    "--rfam_data_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path to dataframe with rfam ids corresponding to viral secondary structures and the viral species name they are associated with",
    required=False,
    default=None,
)
@click.option(
    "--sequence_alignments_dir",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="directory that holds genomic msa files for the species of interest, on which inference of rna-structure guided alignment will be applied for the purpose of creating additional seeds to the existing rfam ids",
    required=False,
    default=None,
)
@click.option(
    "--associations_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to virus-host associations data, based on which the 'phenotype' of each category (columns) will be determined",
    required=True,
)
@click.option(
    "--sequence_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to viral sequence dataframe",
    required=True,
)
@click.option(
    "--tree_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to a tree in newick format that will be used to compute a kinship matrix across the tested viral species",
    required=True,
)
@click.option(
    "--virus_taxonomic_filter_column_name",
    type=click.Choice(["genus", "family", "class"]),
    help="taxonomic rank to select viral data for analysis based on",
    required=False,
    default="family",
)
@click.option(
    "--virus_taxonomic_filter_column_value",
    type=str,
    help="taxonomic rank to select viral data for analysis based on",
    required=False,
    default="flaviviridae",
)
@click.option(
    "--host_taxonomic_group",
    type=click.Choice(["species", "genus", "family", "class"]),
    help="path to virus-host associations data, based on which the 'phenotype' of each category (columns) will be determined",
    required=False,
    default="class",
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory to hold the pipeline files in",
    required=False,
    default=None,
)
@click.option(
    "--rfam_cm_models_wget_path",
    type=str,
    help="url to the rfam fasta ftp services",
    required=False,
    default="https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.tar.gz",
)
@click.option(
    "--log_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the logging of the script",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output of the association tests",
)
@click.option(
    "--multiple_test_correction_method",
    type=click.Choice(["fdr_bh", "bonferroni"]),
    help="method for correction for multiple testing across categories (rows)",
    required=False,
    default="fdr_bh",
)
def test_structs_host_associations(
    rfam_data_path: str,
    associations_data_path: str,
    sequence_data_path: str,
    tree_path: str,
    sequence_alignments_dir: str,
    virus_taxonomic_filter_column_name: t.Optional[str],
    virus_taxonomic_filter_column_value: str,
    host_taxonomic_group: str,
    workdir: str,
    rfam_cm_models_wget_path: str,
    log_path: str,
    output_dir: str,
    multiple_test_correction_method: str,
):
    # initialize the logger
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(log_path)),],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    # process input data
    associations_df = pd.read_csv(associations_data_path)
    relevant_associations_df = associations_df
    if virus_taxonomic_filter_column_name is not None:
        relevant_associations_df = associations_df.loc[
            associations_df[f"virus_{virus_taxonomic_filter_column_name}_name"] == virus_taxonomic_filter_column_value
        ]
    viral_species_names = list(relevant_associations_df.virus_species_name.unique())
    logger.info(
        f"processed {relevant_associations_df.shape[0]} associations across {len(viral_species_names)} viral species for analysis"
    )

    # collect rfam data, if not already available
    if rfam_data_path is None or not os.path.exists(rfam_data_path):
        logger.info(f"extracting viral RFAM data ('genotype') data")
        if rfam_data_path is None:
            rfam_data_path = f"{output_dir}/rfam_data.csv"
        rfam_data = get_viral_rfam_data(output_path=rfam_data_path)
    else:
        rfam_data = pd.read_csv(rfam_data_path)
    logger.info(f"collected data of {len(list(rfam_data.rfam_acc.unique()))} rfam records")
    rfam_core_ids = list(rfam_data.rfam_acc.unique())

    # create additional rfam-like seeds based on the cm models
    # corresponding to structure-guided alignments of suspected structural regions within genomic alignments

    # create a sequence "database" in the form a a single giant fasta file (for downstream cmsearch executions)
    seq_db_path = f"{workdir}/sequence_database.fasta"
    sequence_data = pd.read_csv(sequence_data_path, usecols=["accession", "species_name", "sequence"])
    relevant_sequence_data = sequence_data.loc[sequence_data.species_name.isin(viral_species_names)]
    accession_to_species_map = relevant_sequence_data.drop_duplicates("accession").set_index("accession")[
        "species_name"
    ]
    write_sequence_db(sequence_data=relevant_sequence_data, seq_db_path=seq_db_path)
    logger.info(f"wrote sequence database to {seq_db_path}")

    # for each unique rfam id, get the alignment that conferred it from the rfam ftp service: http://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/
    rfam_workdir = f"{workdir}/rfam/"
    rfam_cm_models_dir = f"{rfam_workdir}cm_models/"
    get_rfam_cm_models(
        required_rfam_ids=rfam_core_ids, output_dir=rfam_cm_models_dir, wget_path=rfam_cm_models_wget_path,
    )
    logger.info(f"downloaded {len(os.listdir(rfam_cm_models_dir))} rfam cm models to {rfam_cm_models_dir}")

    # apply rfam based search on each alignment
    infernal_workdir = f"{rfam_workdir}search_jobs/"
    infernal_results_dir = f"{rfam_workdir}/cmsearch_results/"
    apply_infernal_search(
        cm_models_dir=rfam_cm_models_dir,
        workdir=infernal_workdir,
        output_dir=infernal_results_dir,
        db_path=seq_db_path,
    )
    logger.info(
        f"submitted infernal pipeline jobs for the {len(os.listdir(rfam_cm_models_dir))} collected cm models against the sequence db at {seq_db_path}"
    )

    # parse the species mapped to each relevant rfam id
    rfam_pa_matrix_path = f"{output_dir}/rfam_pa_matrix.csv"
    rfam_pa_matrix = get_rfam_pa_matrix(
        viral_species=viral_species_names,
        rfam_ids=rfam_core_ids,
        infernal_results_dir=infernal_results_dir,
        accession_to_species_map=accession_to_species_map,
        output_path=rfam_pa_matrix_path,
    )

    # perform gemma association test
    association_test_output_dir = f"{output_dir}gemma/"
    apply_lmm_association_test(
        pa_matrix=rfam_pa_matrix,
        samples_trait_data=relevant_associations_df,
        sample_id_name="virus_species_name",
        trait_name=f"host_{host_taxonomic_group}_name",
        tree_path=tree_path,
        output_dir=association_test_output_dir,
        multiple_test_correction_method=multiple_test_correction_method,
    )


if __name__ == "__main__":
    test_structs_host_associations()
