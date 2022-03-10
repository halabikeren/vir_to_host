import os
import re
import shutil
import sys
import typing as t
from Bio import SeqIO

sys.path.append("..")
from serivces.pbs_service import PBSService
from utils.data_collecting.rfam_collecting_utils import RfamCollectingUtils
from utils.data_generation.rna_struct_utils import RNAStructPredictionUtils
from utils.programs import *

import click
import logging

import pandas as pd

logger = logging.getLogger(__name__)

from settings import get_settings


def generate_pa_matrix(
    seed_to_species: t.Dict[str, t.List[str]], species: t.List[str], output_path: str
) -> pd.DataFrame:
    species_to_seed_ids = {
        species: {
            seed: 1 if seed in seed_to_species and species in seed_to_species[seed] else 0 for seed in seed_to_species
        }
        for species in species
    }
    degenerate_rfam_pa_matrix = pd.DataFrame(species_to_seed_ids).transpose()
    nondegenerate_rfam_pa_matrix = degenerate_rfam_pa_matrix.loc[:, (degenerate_rfam_pa_matrix != 0).any()]
    nondegenerate_rfam_pa_matrix.to_csv(output_path, index_label="species_name")
    return nondegenerate_rfam_pa_matrix


def get_rfam_pa_matrix(
    viral_species: t.List[str],
    ids: t.List[str],
    infernal_results_dir: str,
    accession_to_species_map: t.Dict[str, str],
    output_path: str,
):
    """
    :param viral_species: list of viral species names (corresponds to rows)
    :param ids: list of ids of interest (corresponds to columns)
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

    rfam_id_to_species = Infernal.get_hits(
        ids=ids, search_results_dir=infernal_results_dir, hit_id_to_required_id_map=accession_to_species_map,
    )
    nondegenerate_rfam_pa_matrix = generate_pa_matrix(
        seed_to_species=rfam_id_to_species, species=viral_species, output_path=output_path
    )
    return nondegenerate_rfam_pa_matrix


def get_novel_seeds_pa_matrix(
    viral_species: t.List[str], infernal_results_dir: str, accession_to_species_map: t.Dict[str, str], output_path: str,
):
    if os.path.exists(output_path):
        pa_matrix = pd.read_csv(output_path).rename(columns={"Unnamed: 0": "species_name"})
        if "species_name" in pa_matrix.columns:
            pa_matrix.set_index("species_name", inplace=True)
        return pa_matrix

    novel_seed_to_species = dict()
    for species in viral_species:
        species_filename = re.sub("[^A-Za-z0-9]+", "_", species)
        species_novel_seeds_dir = f"{infernal_results_dir}{species_filename}/"
        if os.path.exists(species_novel_seeds_dir):
            local_seed_ids = os.listdir(species_novel_seeds_dir)
            local_seed_id_to_species = Infernal.get_hits(
                ids=local_seed_ids,
                search_results_dir=species_novel_seeds_dir,
                hit_id_to_required_id_map=accession_to_species_map,
            )
            novel_seed_to_species.update(
                {
                    f"{species_filename}_{local_id}": local_seed_id_to_species[local_id]
                    for local_id in local_seed_id_to_species
                }
            )
    nondegenerate_rfam_pa_matrix = generate_pa_matrix(
        seed_to_species=novel_seed_to_species, species=viral_species, output_path=output_path
    )
    return nondegenerate_rfam_pa_matrix


def apply_infernal_pipeline(alignment_path: str, workdir: str, output_dir: str, db_path: str):
    infernal_executor = Infernal(sequence_db_path=db_path)
    infernal_executor.apply_struct_search_pipeline(
        alignment_path=alignment_path, work_dir=workdir, output_dir=output_dir
    )


def infer_novel_seeds_from_genomic_alignment(alignment_path: str, work_dir: str, output_dir: str, db_path: str):
    # infer from genomic alignment structure-guided alignments of suspected structural regions
    structural_regions_work_dir = f"{work_dir}/structural_regions_alignments/"
    structural_alignments_dir = RNAStructPredictionUtils.infer_structural_regions(
        alignment_path=alignment_path, workdir=structural_regions_work_dir
    )

    # apply search struct pipeline on each alignment in parallel using a job array
    log_init = "logging.basicConfig(level=logging.INFO,format='%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s', handlers=[logging.StreamHandler(sys.stdout),],force=True)"
    cmd = (
        f'python -c "import logging, sys;{log_init};sys.path.append({get_settings().PYTHON_BASH_CODE_DIR});from .test_structs_host_association import apply_infernal_pipeline;'
        + 'apply_infernal_pipeline(alignment_path={alignment_path}, workdir={workdir}, output_dir={output_dir}, db_path={db_path})"'
    )
    infernal_work_dir = f"{work_dir}/infernal/"
    infernal_output_dir = f"{output_dir}/infernal/"
    cmd_argname_to_varname = {
        "alignment_path": "input_path",
        "workdir": infernal_work_dir + "/{os.path.basename(input_path).split('.')[0]}/",
        "output_dir": infernal_output_dir + "/{os.path.basename(input_path).split('.')[0]}/",
        "db_path": db_path,
    }
    PBSService.execute_job_array(
        input_dir=structural_alignments_dir,
        output_dir=output_dir,
        work_dir=infernal_work_dir,
        output_format="/",
        commands=[cmd],
        cmd_argname_to_varname=cmd_argname_to_varname,
    )


# previous function, working "bfs" - finish each step of the pipeline per dataset, and then continue to the last step
def infer_novel_seeds_from_genomic_alignments(
    species: t.List[str], alignments_dir: str, novel_seeds_dir: str, workdir: str
):
    """
    :param species:
    :param alignments_dir:
    :param novel_seeds_dir:
    :param workdir:
    :return:
    """
    genomic_alignments_dir = f"{workdir}/genomic_alignments/"
    os.makedirs(genomic_alignments_dir, exist_ok=True)
    for sp in species:
        source_path = f"{alignments_dir}{re.sub('[^A-Za-z0-9]+', '_', sp)}_aligned.fasta"
        dest_path = f"{genomic_alignments_dir}{re.sub('[^A-Za-z0-9]+', '_', sp)}_aligned.fasta"
        if os.path.exists(source_path):
            shutil.copyfile(source_path, dest_path)
    logger.info(
        f"out of {len(species)} species of interest, {len(os.listdir(genomic_alignments_dir))} have available genomic alignments"
    )

    # infer structure-guided alignments for each genomic alignment
    log_init = "logging.basicConfig(level=logging.INFO,format='%(asctime)s module: %(module)s function: %(funcName)s line %(lineno)d: %(message)s', handlers=[logging.StreamHandler(sys.stdout),],force=True)"
    cmd = (
        f'python -c "import logging, sys;{log_init};sys.path.append({get_settings().PYTHON_BASH_CODE_DIR});from utils.rna_struct_utils import RNAStructUtils;'
        + 'RNAStructUtils.infer_structural_regions(alignment_path={input_path}, workdir={output_ath})"'
    )
    refined_structs_inference_dir = f"{novel_seeds_dir}/inferred_structural_regions/"
    if not os.path.exists(refined_structs_inference_dir):
        PBSService.execute_job_array(
            input_dir=genomic_alignments_dir,
            output_dir=refined_structs_inference_dir,
            work_dir=f"{workdir}/infer_structural_regions/",
            output_format="/",
            commands=[cmd],
        )

        # convert all clustal alignments to fasta ones
        refined_structs_alignments_dir = f"{novel_seeds_dir}/inferred_structural_alignments/"
        os.makedirs(refined_structs_alignments_dir, exist_ok=True)
        for sp in species:
            species_filename = re.sub("[^A-Za-z0-9]+", "_", sp)
            refined_alignments_dir = (
                f"{refined_structs_inference_dir}/{species_filename}/rnaz_candidates_mlocarna_aligned/"
            )
            species_alignments_dir = f"{refined_structs_alignments_dir}/{species_filename}"
            for path in refined_alignments_dir:
                fasta_path = f"{species_alignments_dir}/{path.replace('.clustal', '.fasta')}"
                records = list(SeqIO.parse(f"{refined_alignments_dir}/{path}", format="clustal"))
                SeqIO.write(records, fasta_path, format="fasta")
        shutil.rmtree(refined_structs_inference_dir, ignore_errors=True)

    # create covariance models for each available alignment of suspected structural region
    cov_models_dir = f"{novel_seeds_dir}/cov_models/"
    if not os.path.exists(cov_models_dir):
        Infernal.infer_covariance_models(
            alignments_dir=refined_structs_alignments_dir,
            covariance_models_dir=cov_models_dir,
            workdir=f"{workdir}/infer_cov_models/",
        )

        # calibrate covariance models
        Infernal.calibrate_covariance_models(
            covariance_models_dir=cov_models_dir, workdir=f"{workdir}/calibrate_cov_models/",
        )

    # search hits
    cov_model_hits_dir = f"{novel_seeds_dir}/cov_models_hits/"
    if not os.path.exists(cov_model_hits_dir):
        Infernal.apply_search(
            cm_models_dir=cov_models_dir,
            workdir=f"{workdir}/search_cov_models_against_db/",
            output_dir=cov_model_hits_dir,
        )

    return cov_model_hits_dir


@click.command()
@click.option(
    "--rfam_data_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path to dataframe with rfam ids corresponding to viral secondary structures and the viral species name they "
    "are associated with",
    required=False,
    default=None,
)
@click.option(
    "--sequence_alignments_dir",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="directory that holds genomic msa files for the species of interest, on which inference of rna-structure "
    "guided alignment will be applied for the purpose of creating additional seeds to the existing rfam ids",
    required=False,
    default=None,
)
@click.option(
    "--associations_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to virus-host associations data, based on which the 'phenotype' of each category (columns) will be "
    "determined",
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
    help="path to virus-host associations data, based on which the 'phenotype' of each category (columns) will be "
    "determined",
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
    default="bonferroni",
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
    sequence_data = pd.read_csv(sequence_data_path, usecols=["species_name", "family_name", "accession"])
    relevant_associations_df = associations_df
    relevant_sequence_data = sequence_data
    if virus_taxonomic_filter_column_name is not None:
        relevant_associations_df = associations_df.loc[
            associations_df[f"virus_{virus_taxonomic_filter_column_name}_name"] == virus_taxonomic_filter_column_value
        ]
        relevant_sequence_data = relevant_sequence_data.loc[
            relevant_sequence_data[f"{virus_taxonomic_filter_column_name}_name"] == virus_taxonomic_filter_column_value
        ]
    viral_species_names = list(relevant_sequence_data.species_name.unique())
    logger.info(
        f"processed {relevant_associations_df.shape[0]} associations across {len(viral_species_names)} viral species for analysis"
    )

    # collect rfam data, if not already available
    if rfam_data_path is None or not os.path.exists(rfam_data_path):
        logger.info(f"extracting viral RFAM data ('genotype') data")
        if rfam_data_path is None:
            rfam_data_path = f"{output_dir}/rfam_data.csv"
        rfam_collector = RfamCollectingUtils()
        rfam_data = rfam_collector.get_viral_rfam_data(output_path=rfam_data_path)
        del rfam_collector
    else:
        rfam_data = pd.read_csv(rfam_data_path)
    logger.info(f"collected data of {len(list(rfam_data.rfam_acc.unique()))} rfam records")
    rfam_core_ids = list(rfam_data.rfam_acc.unique())

    # for each unique rfam id, get the alignment that conferred it from the rfam ftp service: http://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/
    rfam_workdir = f"{workdir}/rfam/"
    rfam_cm_models_dir = f"{rfam_workdir}cm_models/"
    RfamCollectingUtils.get_cm_models(
        required_rfam_ids=rfam_core_ids, output_dir=rfam_cm_models_dir, wget_path=rfam_cm_models_wget_path,
    )
    logger.info(f"downloaded {len(os.listdir(rfam_cm_models_dir))} rfam cm models to {rfam_cm_models_dir}")

    # apply rfam based search on each alignment
    seq_db_path = f"{workdir}/sequence_database.fasta"
    infernal_executor = Infernal(sequence_db_path=seq_db_path)

    infernal_workdir = f"{rfam_workdir}search_jobs/"
    infernal_results_dir = f"{rfam_workdir}/cmsearch_results/"
    infernal_executor.apply_search(
        cm_models_dir=rfam_cm_models_dir, workdir=infernal_workdir, output_dir=infernal_results_dir,
    )
    logger.info(
        f"submitted infernal pipeline jobs for the {len(os.listdir(rfam_cm_models_dir))} collected cm models against the sequence db at {seq_db_path}"
    )

    # parse the species mapped to each relevant rfam id
    accession_to_species_map = (
        sequence_data.drop_duplicates("accession").set_index("accession")["species_name"].to_dict()
    )
    rfam_pa_matrix_path = f"{output_dir}/rfam_pa_matrix.csv"
    rfam_pa_matrix = get_rfam_pa_matrix(
        viral_species=viral_species_names,
        ids=rfam_core_ids,
        infernal_results_dir=infernal_results_dir,
        accession_to_species_map=accession_to_species_map,
        output_path=rfam_pa_matrix_path,
    )

    # create additional rfam-like seeds based on the cm models
    # corresponding to structure-guided alignments of suspected structural regions within genomic alignments
    novel_seeds_dir = f"{output_dir}/novel_seeds/"
    novel_seeds_hits_dir = infer_novel_seeds_from_genomic_alignments(
        species=viral_species_names,
        alignments_dir=sequence_alignments_dir,
        novel_seeds_dir=novel_seeds_dir,
        workdir=novel_seeds_dir,
    )

    # create a sequence "database" in the form a a single giant fasta file (for downstream cmsearch executions)
    # relevant sequence data will consist of genomic segments for which a structure was predicted, namely, ones extracted from novel_seeds_dir
    infernal_executor.write_sequence_db(sequence_data_dir=f"{novel_seeds_dir}/inferred_structural_regions/")
    logger.info(f"wrote sequence database to {seq_db_path}")

    novel_seeds_pa_matrix_path = f"{output_dir}/novel_seeds_pa_matrix.csv"
    novel_seeds_pa_matrix = get_novel_seeds_pa_matrix(
        viral_species=viral_species_names,
        infernal_results_dir=novel_seeds_hits_dir,
        accession_to_species_map=accession_to_species_map,
        output_path=novel_seeds_pa_matrix_path,
    )

    # join the two matrices
    joint_pa_matrix_path = f"{output_dir}/joint_pa_matrix.csv"
    joint_pa_matrix = pd.concat([rfam_pa_matrix, novel_seeds_pa_matrix], axis=1)
    joint_pa_matrix.to_csv(joint_pa_matrix_path)

    # perform gemma association test
    association_test_output_dir = f"{output_dir}gemma/"
    Gemma.apply_lmm_association_test(
        pa_matrix=joint_pa_matrix,
        samples_trait_data=relevant_associations_df,
        sample_id_name="virus_species_name",
        trait_name=f"host_{host_taxonomic_group}_name",
        tree_path=tree_path,
        output_dir=association_test_output_dir,
        multiple_test_correction_method=multiple_test_correction_method,
    )


if __name__ == "__main__":
    test_structs_host_associations()
