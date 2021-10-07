import logging

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

    os.makedirs(workdir, exist_ok=True)
    rnalalifold_output_dir = f"{workdir}/RNALalifold/"
    os.makedirs(rnalalifold_output_dir, exist_ok=True)
    res = RNAPredUtils.exec_rnalalifold(
        input_path=sequence_data_path, output_dir=rnalalifold_output_dir
    )
    mlocarna_input_dir = f"{workdir}/mLocaRNA/input/"
    os.makedirs(mlocarna_input_dir, exist_ok=True)
    RNAPredUtils.parse_rnaalifold_output(
        rnaalifold_output_dir=rnalalifold_output_dir,
        mlocarna_input_dir=mlocarna_input_dir,
    )
    mlocarna_output_dir = f"{workdir}/mLocaRNA/output/"
    os.makedirs(mlocarna_output_dir, exist_ok=True)
    os.makedirs(mlocarna_output_dir, exist_ok=True)
    for path in os.listdir(mlocarna_input_dir):
        res = RNAPredUtils.exec_mlocarna(
            input_path=f"{mlocarna_input_dir}{path}",
            output_path=f"{mlocarna_output_dir}{path.replace('.fasta', '_mlocrna.out')}",
        )

    rnaz_input_dir = f"{workdir}/RNAz/input/"
    os.makedirs(rnaz_input_dir, exist_ok=True)
    for path in os.listdir(mlocarna_output_dir):
        RNAPredUtils.parse_mlocarna_output(
            mlocarna_output_path=f"{mlocarna_output_dir}{path}",
            rnaz_input_path=f"{rnaz_input_dir}{path.replace('_mlocrna.out', '_rnaz.in')}",
        )

    rnaz_output_dir = f"{workdir}/RNAz/output/"
    os.makedirs(rnaz_output_dir, exist_ok=True)
    for path in os.listdir(rnaz_input_dir):
        res = RNAPredUtils.exec_rnaz(
            input_path=f"{rnaz_input_dir}{path}",
            output_path=f"{rnaz_output_dir}{path.replace('_rnaz.in', 'rnaz.out')}",
        )

    secondary_structures = []
    for path in os.listdir(rnaz_output_dir):
        secondary_structures.append(
            RNAPredUtils.parse_rnaz_output(
                rnaz_output_path=f"{rnaz_output_dir}{path}",
                significance_score_cutoff=significance_score_cutoff,
            )
        )

    for struct in secondary_structures:
        struct_representation.append(struct.consensus_representation)
        struct_sequence.append(struct.consensus_sequence)
        struct_prob.append(struct.svm_rna_probability)
        struct_significance.append(struct.is_significant)
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
    clusters_df: pd.DataFrame,
    cluster_field_name: str,
    sequence_data_dir: str,
    workdir: str,
    output_path: str,
    significance_score_cutoff: float = 0.9,
):
    """
    :param clusters_df:
    :param cluster_field_name:
    :param sequence_data_dir:
    :param workdir:
    :param output_path:
    :param significance_score_cutoff: significance_score_cutoff: threshold between 0 and 1 determining the cutoff of secondary structure RNAz
    probability based on which the structure will be determined as significant or not
    :return:
    """
    secondary_structures_df = pd.DataFrame(
        {cluster_field_name: clusters_df[cluster_field_name].unique()}
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
        cluster_field_name, group_keys=False
    ).apply(
        lambda group: get_secondary_struct(
            sequence_data_path=f"{sequence_data_dir}{re.sub('[^0-9a-zA-Z]+', '_', group.irow(0)[cluster_field_name])}_aligned.fasta",
            workdir=f"{workdir}/{re.sub('[^0-9a-zA-Z]+', '_', group.irow(0)[cluster_field_name])}/",
            significance_score_cutoff=significance_score_cutoff,
        )
    )
    secondary_structures_df = secondary_structures_df.explode(secondary_struct_fields)
    secondary_structures_df.to_csv(output_path, index=False)


@click.command()
@click.option(
    "--associations_clusters_data_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="input path, holding associations grouped by some rational",
)
@click.option(
    "--cluster_field_name",
    type=str,
    help="name of id of clusters, corresponding to unique cluster values",
    required=False,
    default="virus_species_name",
)
@click.option(
    "--sequence_data_dir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory holding sequence data files per species with their collected sequences",
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory to hold the RNA prediction pipeline fils in",
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
)
def compute_seq_similarities(
    associations_clusters_data_path: click.Path,
    cluster_field_name: str,
    sequence_data_dir: click.Path,
    workdir: click.Path,
    log_path: click.Path,
    df_output_path: click.Path,
):
    # initialize the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )

    compute_rna_secondary_structures(
        clusters_df=pd.read_csv(associations_clusters_data_path),
        cluster_field_name=cluster_field_name,
        sequence_data_dir=str(sequence_data_dir),
        workdir=str(workdir),
        output_path=str(df_output_path),
    )
