import sys
import os

import click
import typing as t
import json


import numpy as np
import pandas as pd
import swifter

import logging

from utils.data_collecting_utils import DataCollectingUtils, RefSource


def parse_association_data(
    input_path: str,
    columns_translator: t.Dict[str, t.Dict[str, str]],
    logger: logging.log,
) -> pd.DataFrame:
    """
    :param input_path: path to input file
    :param columns_translator: map of unique original filed names to required field names
    :param logger: logging instance
    :return: dataframe of the parsed associations based on the n
    """

    processed_data_path = f"{os.path.dirname(input_path)}/{os.path.splitext(os.path.basename(input_path))[0]}_processed.csv"
    if os.path.exists(processed_data_path):
        d = pd.read_csv(processed_data_path)
        d.drop(
            labels=[col for col in d.columns if "Unnamed" in col], axis=1, inplace=True
        )
        return d

    data_columns_translator = columns_translator[
        os.path.splitext(os.path.basename(input_path))[0]
    ]
    df = pd.read_csv(input_path, sep="," if ".csv" in input_path else "\t", header=0)
    df.drop(
        labels=[col for col in df.columns if "Unnamed" in col], axis=1, inplace=True
    )

    if "wardeh_et_al_2020" in input_path:
        df = df.loc[df["pc1"] == "virus"]
    if "albery_et_al_2020" in input_path:
        df = df.loc[df["Cargo classification"] == "Virus"]

    # collect DOIs from references
    references_field = None
    source_type = None
    if "virushostdb" in input_path:
        references_field = "pmid"
        source_type = RefSource.PUBMED_ID
    elif "albery_et_al_2020" in input_path:
        references_field = "Publications"
        source_type = RefSource.GENE_ID
    elif "pandit_et_al_2018" in input_path:
        df["association_references"] = (
            df["Title/accession number"]
            + " "
            + df["Authors"]
            + " "
            + df["Year"]
            + " "
            + df["Journal"]
        )
        references_field = "association_references"
        source_type = RefSource.PAPER_DETAILS
    elif "wardeh_et_al_2021_2" in input_path:
        references_field = "reference"
        source_type = RefSource.SEQ_ID

    if references_field:
        DataCollectingUtils.collect_dois(
            df=df,
            output_field_name="association_references_doi",
            references_field=references_field,
            source_type=source_type,
            logger=logger,
        )

    cols_to_include = list(data_columns_translator.keys())
    if references_field:
        cols_to_include.append("association_references_doi")
    df = df[cols_to_include]
    df.rename(
        columns={"association_references_doi": "association_references"}, inplace=True
    )
    df.rename(columns=data_columns_translator, inplace=True)

    if "viprdb" in input_path:
        df["virus_taxon_name"] = (
            df["virus_species_name"] + " " + df["virus_strain_name"]
        )

    df.to_csv(processed_data_path, index=False)

    return df


def unite_references(association_record: pd.Series) -> str:
    """
    :param association_record: pandas series corresponding to all columns holding reference data in the form of DOIs
    :return: string representing all the unique DOIs
    """
    references = set()
    for value in association_record.unique():
        if type(value) is list:
            references.add(value.split(","))
        elif type(value) is str:
            references.add(value)
    return ",".join(list(references))


def get_data_from_prev_studies(
    input_dir: click.Path, output_path: str, logger: logging.log
) -> pd.DataFrame:
    """
    :param input_dir: directory of data from previous studies
    :param output_path: path to write to the united collected data
    :param logger: logging instance
    :return: dataframe with the united data
    """

    if os.path.exists(output_path):
        logger.info(
            f"united data from previous studies is already available at {output_path}"
        )
        d = pd.read_csv(output_path)
        d.drop(
            labels=[col for col in d.columns if "Unnamed" in col], axis=1, inplace=True
        )
        return d

    # collect data into dataframes
    logger.info(
        f"Processing data from previous studies and writing it to {output_path}"
    )

    columns_translator = dict()
    paths = []
    for path in os.listdir(str(input_dir)):
        if ".json" in path:
            with open(f"{input_dir}/{path}", "r") as j:
                columns_translator = json.load(j)
        elif os.path.isfile(f"{input_dir}/{path}") and not "processed" in path:
            paths.append(f"{input_dir}/{path}")
    dfs = [
        parse_association_data(
            input_path=path, columns_translator=columns_translator, logger=logger
        )
        for path in paths
    ]

    # merge and process united data
    intersection_cols = ["virus_taxon_name", "host_taxon_name"]
    udf = dfs[0].merge(dfs[1], on=intersection_cols, how="outer")
    for df in dfs[2:]:
        udf = udf.merge(df, on=intersection_cols, how="outer")

    # deal with duplicated columns caused by inequality of Nan values
    DataCollectingUtils.handle_duplicated_columns(
        colname="virus_taxon_id", df=udf, logger=logger
    )
    # udf.virus_taxon_id_x.fillna(udf.virus_taxon_id_y, inplace=True)
    # udf.rename(columns={"virus_taxon_id_x": "virus_taxon_id"}, inplace=True)
    # udf.drop("virus_taxon_id_y", axis="columns", inplace=True)
    DataCollectingUtils.handle_duplicated_columns(
        colname="host_taxon_id", df=udf, logger=logger
    )
    # udf.host_taxon_id_x.fillna(udf.host_taxon_id_y, inplace=True)
    # udf.rename(columns={"host_taxon_id_x": "host_taxon_id"}, inplace=True)
    # udf.drop("host_taxon_id_y", axis="columns", inplace=True)

    # translate references to dois and unite them
    udf["references"] = udf[
        [col for col in udf.columns if "association_references" in col]
    ].swifter.apply(lambda x: unite_references(x), axis=1)
    udf.drop(
        [col for col in udf.columns if "association_references" in col],
        axis="columns",
        inplace=True,
    )
    udf["references_num"] = udf["references"].apply(
        lambda x: x.count(",") + 1 if type(x) is str else 0,
    )

    # drop duplicates caused by contradiction in insignificant fields
    udf.drop_duplicates(subset=intersection_cols, keep="first", inplace=True)

    # save intermediate output
    udf.to_csv(output_path, index=False)
    logger.info(
        f"data from previous studies collected and saved successfully to {output_path}"
    )

    return udf


def get_data_from_databases(
    input_dir: click.Path, output_path: str, logger: logging.log
) -> pd.DataFrame:
    if os.path.exists(output_path):
        logger.info(f"united data from databases is already available at {output_path}")
        d = pd.read_csv(output_path, dtype=object,)
        d.drop(
            labels=[col for col in d.columns if "Unnamed" in col], axis=1, inplace=True
        )
        return d

    # collect data into dataframes
    logger.info(f"Processing data from databases and writing it to {output_path}")

    columns_translator = dict()
    paths = []
    for path in os.listdir(str(input_dir)):
        if ".json" in path:
            with open(f"{input_dir}/{path}", "r") as j:
                columns_translator = json.load(j)
        elif os.path.isfile(f"{input_dir}/{path}") and not "processed" in path:
            paths.append(f"{input_dir}/{path}")
    dfs = [
        parse_association_data(
            input_path=path, columns_translator=columns_translator, logger=logger
        )
        for path in paths
    ]

    # merge and process united data
    intersection_cols = ["virus_taxon_name", "host_taxon_name"]
    udf = dfs[0].merge(dfs[1], on=intersection_cols, how="outer")
    for df in dfs[2:]:
        udf = udf.merge(df, on=intersection_cols, how="outer")

    # deal with duplicated columns caused by inequality of Nan values
    DataCollectingUtils.handle_duplicated_columns(
        colname="virus_genbank_accession", df=udf, logger=logger
    )
    # udf.virus_genbank_accession_x.fillna(udf.virus_genbank_accession_y, inplace=True)
    # udf.rename(
    #     columns={"virus_genbank_accession_x": "virus_genbank_accession"}, inplace=True
    # )
    # udf.drop("virus_genbank_accession_y", axis="columns", inplace=True)

    # translate references to dois and unite them
    udf["references"] = udf[
        [col for col in udf.columns if "association_references" in col]
    ].swifter.apply(lambda x: unite_references(x), axis=1)
    udf.drop(
        [col for col in udf.columns if "association_references" in col],
        axis="columns",
        inplace=True,
    )
    udf["references_num"] = udf["references"].apply(
        lambda x: x.count(",") + 1 if type(x) is str else 0,
    )

    # drop duplicates caused by contradiction in insignificant fields
    udf.drop_duplicates(subset=intersection_cols, keep="first", inplace=True)
    # save intermediate output
    udf.to_csv(output_path, index=False)
    logger.info(
        f"data from previous studies collected and saved successfully to {output_path}"
    )

    return udf


@click.command()
@click.option(
    "--previous_studies_dir",
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="directory that holds the data collected from previous studies",
    default="./data/previous_studies/associations/",
)
@click.option(
    "--database_sources_dir",
    type=click.Path(exists=True, dir_okay=True),
    help="directory holding associations data from databases",
    default="./data/databases/associations/",
)
@click.option(
    "--filter_to_flaviviridae",
    type=bool,
    help="indicator weather data should be filtered out to contain only associations with viruses from the "
    "Flaviviridae family",
    default=False,
)
@click.option(
    "--logger_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to logging file",
    default="./data/collect_associations_data.log",
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    help="boolean indicating weather script should be executed in debug mode",
    default=False,
)
@click.option(
    "--output_path",
    type=click.Path(exists=False, file_okay=True),
    help="path to output file consisting of the united associations data",
    default="./data/associations_united.csv",
)
def collect_virus_host_associations(
    previous_studies_dir: click.Path,
    database_sources_dir: click.Path,
    filter_to_flaviviridae: bool,
    logger_path: click.Path,
    debug_mode: bool,
    output_path: click.Path,
):
    # initialize the logger
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format="%(asctime)s module: %(module)s function: %(funcName)s line: %(lineno)d %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logger_path),],
    )
    logger = logging.getLogger(__name__)

    # process data from previous studies
    prev_studies_df = get_data_from_prev_studies(
        input_dir=previous_studies_dir,
        output_path=f"{os.path.dirname(str(output_path))}/united_previous_studies_associations{'_test' if debug_mode else ''}.csv",
        logger=logger,
    )

    # process data from databases
    databases_df = get_data_from_databases(
        input_dir=database_sources_dir,
        output_path=f"{os.path.dirname(str(output_path))}/united_databases_associations{'_test' if debug_mode else ''}.csv",
        logger=logger,
    )

    # merge dataframes
    final_df = prev_studies_df.merge(
        databases_df, on=["virus_taxon_name", "host_taxon_name"], how="outer"
    )

    # unite duplicated columns
    DataCollectingUtils.handle_duplicated_columns(
        colname="virus_genbank_accession", df=final_df, logger=logger
    )
    DataCollectingUtils.handle_duplicated_columns(
        colname="virus_taxon_id", df=final_df, logger=logger
    )
    DataCollectingUtils.handle_duplicated_columns(
        colname="host_taxon_id", df=final_df, logger=logger
    )
    DataCollectingUtils.handle_duplicated_columns(
        colname="virus_species_name", df=final_df, logger=logger
    )
    DataCollectingUtils.handle_duplicated_columns(
        colname="virus_genus_name", df=final_df, logger=logger
    )

    # unite references
    reference_columns = [col for col in final_df.columns if "references_" in col]
    final_df["references"] = final_df[reference_columns].swifter.apply(
        lambda x: unite_references(x), axis=1
    )
    for col in reference_columns:
        final_df.drop(col, axis="columns", inplace=True)
    references_num_columns = [
        col for col in final_df.columns if "references_num_" in col
    ]
    final_df["references_num"] = final_df["references"].apply(
        lambda x: x.count(",") + 1 if len(x) > 1 else 0
    )
    for col in references_num_columns:
        final_df.drop(col, axis="columns", inplace=True)

    # collect taxonomy data
    final_df = DataCollectingUtils.collect_taxonomy_data(
        df=final_df, taxonomy_data_prefix="virus", logger=logger
    )
    final_df = DataCollectingUtils.collect_taxonomy_data(
        df=final_df, taxonomy_data_prefix="host", logger=logger
    )

    # filter data if needed
    if filter_to_flaviviridae:
        final_df = final_df.loc[final_df["virus_taxon_family"] == "Flaviviridae"]

    # write to output
    final_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    collect_virus_host_associations()
