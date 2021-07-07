import sys
import os

import click
import typing as t
from enum import Enum
import json

import numpy as np
import pandas as pd
from habanero import Crossref
import logging
from Bio import Entrez


class RefSource(Enum):
    PAPER_DETAILS = 1
    SEQ_ID = 2
    GENE_ID = 3
    PUBMED_ID = 3
    OTHER = 4


def collect_dois(df: pd.DataFrame, output_field_name: str, references_field: str, source_type: RefSource,
                 logger: logging.log):
    """
    :param df: dataframe to add a references by DOIs columns to
    :param output_field_name: the name of the added field
    :param references_field: a list of fields by which be extracted
    :param source_type: the type of query to conduct to ge the doi
    :param logger instance
    :return: none
    """
    Entrez.email = "halabikeren@gmail.com"

    df[references_field] = df[references_field].apply(lambda x: str(x).split(",|;"))
    for chunk in np.array_split(df, (len(df.index) + 2) / 100):
        references = set([y for x in chunk[references_field] for y in x])
        ref_to_doi = {ref: [] for ref in references}
        refs_query = ",".join(ref_to_doi.keys())
        if source_type in [RefSource.SEQ_ID, RefSource.GENE_ID, RefSource.PUBMED_ID]:
            try:
                matches = [record for record in Entrez.parse(
                    Entrez.efetch(db="pubmed" if source_type == RefSource.PUBMED_ID else "nucleotide", id=refs_query,
                                  retmode="xml", retmax=len(ref_to_doi.keys())))]
                for match in matches:
                    doi = []
                    if source_type == RefSource.PUBMED_ID and "DOI" in match:
                        doi = [match["DOI"]]
                    elif source_type != RefSource.PUBMED_ID:
                        for ref in match['GBSeq_references']:
                            if 'GBReference_xref' in ref and 'GBXref_dbname' in ref['GBReference_xref'][0] and \
                                    ref['GBReference_xref'][0]['GBXref_dbname'] == 'doi':
                                doi.append(ref['GBReference_xref'][0]['GBXref_id'])
                    key = match["Id"] if source_type == RefSource.PUBMED_ID else (
                        match['GBSeq_accession-version'] if source_type == RefSource.SEQ_ID else
                        [s.split("|")[-1] for s in match['GBSeq_other-seqids'] if "gi|" in s][0])
                    for item in doi:
                        ref_to_doi[key].append(item)
            except Exception as e:
                logger.error(
                    f"failed to extract doi from references {refs_query} by {source_type.name} due to error {e}")
        elif source_type == RefSource.PAPER_DETAILS:
            cr = Crossref()
            for ref in ref_to_doi.keys():
                res = cr.works(query_bibliographic=ref, limit=1)  # couldn't find a batch option
                try:
                    ref_to_doi[ref].append(res['message']['items'][0]['DOI'])
                except Exception as e:
                    logger.error(f"failed to extract DOI for ref {ref} based on {source_type.name} due to error {e}")
        else:
            logger.error(f"No mechanism is available for extraction of DOI from source type {source_type.name}")
        df.loc[df.index.isin(chunk.index), output_field_name] = df.apply(
            lambda x: ",".join(",".join(y) for y in [ref_to_doi[y] for x in chunk[references_field] for y in x]),
            axis=1)


def parse_association_data(input_path: str, columns_translator: t.Dict[str, t.Dict[str, str]],
                           logger: logging.log) -> pd.DataFrame:
    """
    :param input_path: path to input file
    :param columns_translator: map of unique original filed names to required field names
    :param logger: logging instance
    :return: dataframe of the parsed associations based on the n
    """

    processed_data_path = f"{os.path.dirname(input_path)}/{os.path.splitext(os.path.basename(input_path))[0]}_processed.csv"
    if os.path.exists(processed_data_path):
        return pd.read_csv(processed_data_path)

    data_columns_translator = columns_translator[os.path.splitext(os.path.basename(input_path))[0]]
    df = pd.read_csv(input_path, sep="," if ".csv" in input_path else "\t", header=0)

    if "wardeh_et_al_2020" in input_path:
        df = df.loc[df["pc1"] == "virus"]
    if "albery_et_al_2020" in input_path:
        df = df.loc[df["Cargo classification"] == "Virus"]

    # collect DOIs from references
    references_field = np.nan
    source_type = None
    if "virushostdb" in input_path:
        references_field = "pmid"
        source_type = RefSource.PUBMED_ID
    elif "albery_et_al_2020" in input_path:
        references_field = "Publications"
        source_type = RefSource.GENE_ID
    elif "pandit_et_al_2018" in input_path:
        df["association_references"] = df["Title/accession number"] + " " + df["Authors"] + " " + df["Year"] + " " + df[
            "Journal"]
        references_field = "association_references"
        source_type = RefSource.PAPER_DETAILS
    elif "wardeh_et_al_2021_2" in input_path:
        references_field = "reference"
        source_type = RefSource.SEQ_ID
    collect_dois(df=df, output_field_name="association_references", references_field=references_field,
                 source_type=source_type, logger=logger)

    df = df[list(data_columns_translator.keys())]
    df.rename(columns=data_columns_translator, inplace=True)

    if "viprdb" in input_path:
        df["virus_taxon_name"] = df["virus_species_name"] + " " + df["virus_strain_name"]

    df.to_csv(processed_data_path)

    return df


def unite_references(association_record: pd.Series) -> str:
    """
    :param association_record: pandas series corresponding to all columns holding reference data in the form of DOIs
    :return: string representing all the unique DOIs
    """
    references = set()
    for value in association_record.values:
        references.add(value.split(","))
    return ",".join(list(references))


def get_data_from_prev_studies(input_dir: click.Path, output_path: str, logger: logging.log) -> pd.DataFrame:
    """
    :param input_dir: directory of data from previous studies
    :param output_path: path to write to the united collected data
    :param logger: logging instance
    :return: dataframe with the united data
    """

    if os.path.exists(output_path):
        logger.info(f"united data from previous studies is already available at {output_path}")
        return pd.read_csv(output_path)

    # collect data into dataframes
    logger.info(f"Processing data from previous studies and writing it to {output_path}")

    columns_translator = dict()
    paths = []
    for path in os.listdir(str(input_dir)):
        if ".json" in path:
            with open(f"{input_dir}/{path}", "r") as j:
                columns_translator = json.load(j)
        elif os.path.isfile(f"{input_dir}/{path}") and not "processed" in path:
            paths.append(f"{input_dir}/{path}")
    dfs = [parse_association_data(input_path=path, columns_translator=columns_translator, logger=logger) for path in
           paths]

    # merge and process united data
    intersection_cols = ["virus_taxon_name", "host_taxon_name"]
    udf = dfs[0].merge(dfs[1], on=intersection_cols, how="outer")
    for df in dfs[2:]:
        udf = udf.merge(df, on=intersection_cols, how="outer")
    udf.virus_taxon_id_x.fillna(udf.virus_taxon_id_y, inplace=True)

    # deal with duplicated columns caused by inequality of Nan values
    udf.rename(columns={"virus_taxon_id_x": "virus_taxon_id"}, inplace=True)
    udf.drop("virus_taxon_id_y", axis="columns", inplace=True)
    udf.host_taxon_id_x.fillna(udf.host_taxon_id_y, inplace=True)
    udf.rename(columns={"host_taxon_id_x": "host_taxon_id"}, inplace=True)
    udf.drop("host_taxon_id_y", axis="columns", inplace=True)

    # translate references to dois and unite them
    udf["association_references"] = udf[[col for col in udf.columns if "association_references" in col]].apply(
        lambda x: unite_references(x))
    udf["association_references_num"] = udf["association_references"].apply(lambda x: x.count(",") + 1, axis=1)

    # drop duplicates caused by contradiction in insignificant fields
    udf.drop_duplicates(subset=intersection_cols, keep='first',
                        inplace=True)

    # save intermediate output
    udf.to_csv(output_path)
    logger.info(f"data from previous studies collected and saved successfully to {output_path}")

    return udf


def get_data_from_databases(input_dir: click.Path, output_path: str, logger: logging.log) -> pd.DataFrame:
    if os.path.exists(output_path):
        logger.info(f"united data from databases is already available at {output_path}")
        return pd.read_csv(output_path)

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
    dfs = [parse_association_data(input_path=path, columns_translator=columns_translator, logger=logger) for path in
           paths]

    # merge and process united data
    intersection_cols = ["virus_taxon_name", "host_taxon_name"]
    udf = dfs[0].merge(dfs[1], on=intersection_cols, how="outer")
    for df in dfs[2:]:
        udf = udf.merge(df, on=intersection_cols, how="outer")
    udf.virus_taxon_id_x.fillna(udf.virus_taxon_id_y, inplace=True)

    # deal with duplicated columns caused by inequality of Nan values
    udf.virus_genbank_accession_x.fillna(udf.virus_genbank_accession_y, inplace=True)
    udf.rename(columns={"virus_genbank_accession_x": "virus_genbank_accession"}, inplace=True)
    udf.drop("virus_genbank_accession_y", axis="columns", inplace=True)

    # translate references to dois and unite them
    udf["association_references"] = udf[[col for col in udf.columns if "association_references" in col]].apply(
        lambda x: unite_references(x))
    udf["association_references_num"] = udf["association_references"].apply(lambda x: x.count(",") + 1, axis=1)

    # drop duplicates caused by contradiction in insignificant fields
    udf.drop_duplicates(subset=intersection_cols, keep='first',
                        inplace=True)
    # save intermediate output
    udf.to_csv(output_path)
    logger.info(f"data from previous studies collected and saved successfully to {output_path}")

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
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logger_path),
        ],
    )
    logger = logging.getLogger(__name__)

    # process data from previous studies
    prev_studies_df = get_data_from_prev_studies(input_dir=previous_studies_dir,
                                                 output_path=f"{os.path.dirname(str(output_path))}/united_previous_studies_associations.csv",
                                                 logger=logger)

    # process data from databases
    databases_df = get_data_from_databases(input_dir=database_sources_dir,
                                           output_path=f"{os.path.dirname(str(output_path))}/united_databases_associations.csv",
                                           logger=logger)

    # merge dataframes
    final_df = prev_studies_df.merge(databases_df, on=["virus_taxon_name", "host_taxon_name"], how="outer")

    # add missing taxonomy data

    # filter data if needed
    if filter_to_flaviviridae:
        final_df = final_df.loc[final_df["virus_taxon_family"] == "Flaviviridae"]

    # write to output
    final_df.to_csv(output_path)


if __name__ == "__main__":
    collect_virus_host_associations()
