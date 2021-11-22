import logging

import numpy as np
from Bio import SeqIO

logger = logging.getLogger(__name__)

import os
import re
import sys
import click
import pandas as pd
import typing as t

sys.path.append("..")
from utils.rna_pred_utils import RNAPredUtils

df = pd.DataFrame({"id": [1, 2, 3, 4]})


def get_secondary_struct(
    sequence_data_path: str, workdir: str, significance_score_cutoff: float = 0.9
) -> t.Tuple[
    t.List[str],
    t.List[str],
    t.List[float],
    t.List[bool],
    t.List[float],
    t.List[float],
    t.List[float],
]:
    """
    this pipeline follows the one of RNASIV, which can be found in: https://www.mdpi.com/1999-4915/11/5/401/htm#B30-viruses-11-00401
    :param sequence_data_path: alignment data path to provide as input to the rna secondary structures prediction
    :param workdir: directory to write the pipeline output files in
    :param significance_score_cutoff: threshold between 0 and 1 determining the cutoff of secondary structure RNAz
    probability based on which the structure will be determined as significant or not
    :return: a dataframe corresponding to the secondary structures inferred for the respective cluster id
    """

    (
        struct_representation,
        struct_sequence,
        struct_prob,
        struct_significance,
        struct_mfe,
        struct_entropy,
        struct_conservation_index,
    ) = ([], [], [], [], [], [], [])

    if not os.path.exists(sequence_data_path):
        logger.error(
            f"no MSA is available at {sequence_data_path} and thus no secondary structures will be computed"
        )
        return (
            struct_representation,
            struct_sequence,
            struct_prob,
            struct_significance,
            struct_mfe,
            struct_entropy,
            struct_conservation_index,
        )

    num_sequences = len(list(SeqIO.parse(sequence_data_path, format="fasta")))
    secondary_structures = []
    if num_sequences > 1:
        logger.info(f"computing rnaz reliable windows for prediction")
        rnaz_window_output_path = f"{workdir}/rnaz_window.out"
        RNAPredUtils.exec_rnaz_window(input_path=sequence_data_path, output_path=rnaz_window_output_path)
        logger.info(f"executing RNAz predictor on refined windows")
        rnaz_output_path = f"{workdir}/rnaz_initial.out"
        RNAPredUtils.exec_rnaz(input_path=rnaz_window_output_path, output_path=rnaz_output_path)
        logger.info(f"clustering RNAz hits of overlapping windows")
        rnaz_cluster_output_path = f"{workdir}/rnaz_cluster.dat"
        RNAPredUtils.exec_rnaz_cluster(input_path=rnaz_output_path, output_path=rnaz_cluster_output_path)
        logger.info(f"extracting sequence data per selected window for mlocarna refinement")
        rnaz_candidates_output_dir = f"{workdir}/rnaz_candidates_sequence_data/"
        RNAPredUtils.parse_candidates(candidates_info_path=rnaz_cluster_output_path, sequence_data_path=rnaz_window_output_path, output_dir=rnaz_candidates_output_dir)
        logger.info(f"creating refined alignments of candidates with mlocarna")
        mlocarna_output_dir = f"{workdir}/rnaz_candidates_mlocarna_aligned/"
        for path in os.listdir(rnaz_candidates_output_dir):
            input_path = f"{rnaz_candidates_output_dir}{path}"
            output_path = f"{mlocarna_output_dir}{path.replace('.fasta', '.clustal')}"
            RNAPredUtils.exec_mlocarna(input_path=input_path, output_path=output_path)
        logger.info(f"executing prediction on aligned windows with rnaz to be able to classify the selected structures")
        rnaz_refined_output_dir = f"{workdir}/rnaz_final_output/"
        os.makedirs(rnaz_refined_output_dir, exist_ok=True)
        for path in os.listdir(mlocarna_output_dir):
            input_path=f"{mlocarna_output_dir}{path}"
            output_path = f"{rnaz_refined_output_dir}{path.replace('.clustal', '_rnaz.out')}"
            RNAPredUtils.exec_rnaz(input_path=input_path, output_path=output_path)
        logger.info(f"parsing the obtained rna structures")
        for path in os.listdir(rnaz_refined_output_dir):
            struct = RNAPredUtils.parse_rnaz_output(rnaz_output_path=f"{rnaz_refined_output_dir}{path}", significance_score_cutoff=significance_score_cutoff)
            secondary_structures.append(struct)
    else:
        logger.info(f"executing RNALfold on the single sequence obtained for the species")
        rnalfold_output_path = f"{workdir}/rnalfold.out"
        RNAPredUtils.exec_rnalfold(input_path=sequence_data_path, output_path=rnalfold_output_path)
        secondary_structures = RNAPredUtils.parse_rnalfold_result(rnalfold_path=rnalfold_output_path)

    functional_structures = [struct for struct in secondary_structures if bool(struct.is_significant) and bool(struct.is_functional_structure)]
    logger.info(f"out of {len(secondary_structures)}, {len(functional_structures)} are significant and functional")
    logger.info(f"the mean z-score for the predicted structures is {np.mean([struct.mean_zscore for struct in functional_structures])} and standard deviation of {np.std([struct.mean_zscore for struct in functional_structures])}")
    for struct in secondary_structures: # here, I will save all the structures and filter out weight them by svm_rna_probability (= prb > 0.5 means it is a functional RNA, prob larger than 0.9 is more stringent and what was used in RNASIV)
        struct_representation.append(struct.consensus_representation)
        struct_sequence.append(struct.consensus_sequence)
        struct_prob.append(struct.svm_rna_probability)
        struct_mfe.append(struct.consensus_mfe)
        struct_entropy.append(struct.shannon_entropy)
        struct_conservation_index.append(struct.structure_conservation_index)

    return (
        struct_representation,
        struct_sequence,
        struct_prob,
        struct_significance,
        struct_mfe,
        struct_entropy,
        struct_conservation_index,
    )


def compute_rna_secondary_structures(
    input_df: pd.DataFrame,
    sequence_data_dir: str,
    workdir: str,
    output_path: str,
    significance_score_cutoff: float = 0.9,
):
    """
    :param input_df: dataframe with viral species of interest
    :param sequence_data_dir: directory holding sequence data of the viral species of interest
    :param workdir: directory to
    :param output_path: path of output dataframe
    :param significance_score_cutoff: significance_score_cutoff: threshold between 0 and 1 determining the cutoff of secondary structure RNAz
    probability based on which the structure will be determined as significant or not
    :return:
    """
    secondary_structures_df = pd.DataFrame(
        {"virus_species_name": input_df["virus_species_name"].unique()}
    )
    secondary_struct_fields = [
        "struct_representation",
        "struct_sequence",
        "struct_prob",
        "struct_significance",
        "struct_mfe",
        "struct_entropy",
        "struct_conservation_index",
    ]
    secondary_structures_df[secondary_struct_fields] = secondary_structures_df.groupby(
        "virus_species_name", group_keys=False
    ).apply(
        lambda group: get_secondary_struct(
            sequence_data_path=f"{sequence_data_dir}{re.sub('[^0-9a-zA-Z]+', '_', group.iloc[0]['virus_species_name'])}_aligned.fasta",
            workdir=f"{workdir}/{re.sub('[^0-9a-zA-Z]+', '_', group.iloc[0]['virus_species_name'])}/",
            significance_score_cutoff=significance_score_cutoff,
        )
    )
    secondary_structures_df = secondary_structures_df.explode(secondary_struct_fields)
    secondary_structures_df.to_csv(output_path, index=False)


@click.command()
@click.option(
    "--associations_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="input path of associations grouped viral species and host species",
)
@click.option(
    "--sequence_data_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory holding sequence data files per species with their collected sequences",
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
@click.option(
    "--significance_score_cutoff",
    type=click.FloatRange(min=0, max=1),
    help="significance_score_cutoff: threshold between 0 and 1 determining the cutoff of secondary structure RNAz probability based on which the structure will be determined as significant or not",
    required=False,
    default = 0.9,
)
def predict_secondary_structures(
    associations_data_path: click.Path,
    sequence_data_dir: click.Path,
    workdir: t.Optional[click.Path],
    log_path: click.Path,
    df_output_path: click.Path,
    significance_score_cutoff: float,
):
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

    if not workdir:
        workdir = f"{os.path.dirname(str(associations_data_path))}/rna_pred_aux/"
        logger.info(f"creating working directory {workdir}")
        os.makedirs(workdir, exist_ok=True)

    compute_rna_secondary_structures(
        input_df=pd.read_csv(associations_data_path),
        sequence_data_dir=str(sequence_data_dir),
        workdir=str(workdir),
        output_path=str(df_output_path),
        significance_score_cutoff=significance_score_cutoff
    )

if __name__ == '__main__':
    predict_secondary_structures()