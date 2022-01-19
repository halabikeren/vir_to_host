from pandarallel import pandarallel

pandarallel.initialize()

import click
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import sys

sys.path.append("..")
from utils.rna_struct_utils import RNAStructUtils

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--secondary_structures_df_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to dataframe holding the names of species for which data should be collected. this dataframe is  expected to hold both the virus_species_name and the grouping_field in it",
)
@click.option(
    "--grouping_field", type=str, help="column name to group data by", required=False, default="virus_family_name",
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
    "--partition_by",
    type=click.Choice(["range", "annotation"]),
    help="indicator weather tp partition the data by constant ranges across the genome or by annotations (UTRs / CDS / ect.)",
    required=False,
    default="annotation",
)
@click.option(
    "--partition_size",
    type=click.IntRange(100, float("inf")),
    help="value to use as partition size, in case that partition_by=range",
    required=False,
    default=120,
)
@click.option(
    "--vadr_annotation_path",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="path to annotations inspired by vadr, already processed in the form of a csv file"
    "file suffix should be .ftr according to https://github.com/ncbi/vadr/blob/master/documentation/formats.md#annotate",
    required=False,
    default=None,
)
@click.option(
    "--workdir",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="directory to write family sequence data and align it in",
    required=True,
)
@click.option(
    "--log_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the logging of the script",
    required=True,
)
@click.option(
    "--df_output_path",
    type=click.Path(exists=False, file_okay=True, readable=True),
    help="path holding the output dataframe to write",
    required=True,
)
def partition_secondary_structures(
    secondary_structures_df_path: click.Path,
    grouping_field: str,
    sequence_data_dir: str,
    species_wise_msa_dir: str,
    partition_by: str,
    partition_size: int,
    vadr_annotation_path: click.Path,
    workdir: str,
    log_path: click.Path,
    df_output_path: click.Path,
):

    # initialize the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(str(log_path)),],
        force=True,  # run over root logger settings to enable simultaneous writing to both stdout and file handler
    )

    # create a family alignment by selecting genome representative per species within the family and aligning the representatives
    secondary_structures_df = pd.read_csv(secondary_structures_df_path)
    logger.info(f"applying partitioning pipeline for {secondary_structures_df.shape[0]} structures")
    if grouping_field not in secondary_structures_df.columns:
        error = (
            f"grouping field {grouping_field} is absent from the dataframe columns at {secondary_structures_df_path}"
        )
        logger.error(error)
        raise ValueError(error)
    if "virus_species_name" not in secondary_structures_df.columns:
        error_msg = f"virus_species_name is absent from the dataframe columns at {secondary_structures_df_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    secondary_structures_groups = secondary_structures_df.groupby(grouping_field)
    logger.info(f"dataframe consists of {len(secondary_structures_groups.groups.keys())} groups of {grouping_field}")
    output_dfs = []
    for g in secondary_structures_groups.groups.keys():
        if g == "flaviviridae":  # remove after analysis to generalize to all viral families

            logger.info(f"handling group {g}")
            df = secondary_structures_groups.get_group(g)

            # map the start position of each secondary structure from its original value (determined by the species-wise alignment) to its new value (determined by the family-wise alignment) in the output df
            logger.info(f"mapping species-wise structures positions to group-wise positions")
            df = RNAStructUtils.map_species_wise_pos_to_group_wise_pos(
                df=df,
                seq_data_dir=sequence_data_dir,
                species_wise_msa_dir=species_wise_msa_dir,
                workdir=f"{workdir}/{g}/",
            )

            # obtain the annotations of each species in the family, and the range of each annotation within the representative genome (throw away annotations that do not appear in all the species within the family)
            logger.info(f"assigning partitions to structures based on {partition_by}")
            if partition_by == "range":
                df = RNAStructUtils.assign_partition_by_size(df=df, partition_size=partition_size)
            elif partition_by == "annotation":
                df = RNAStructUtils.assign_partition_by_annotation(df=df, vadr_annotation_path=vadr_annotation_path)
            output_dfs.append(df)

    output_df = pd.concat(output_dfs)
    output_df.to_csv(df_output_path, index=False)


if __name__ == "__main__":
    partition_secondary_structures()
