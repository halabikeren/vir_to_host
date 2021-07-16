import logging
from enum import Enum
import typing as t
from collections import defaultdict

import pandas as pd
import numpy as np
import re

from Bio import Entrez

Entrez.email = "halabikeren@mail.tau.ac.il"
from habanero import Crossref

from multiprocessing import Pool
from functools import partial
import signal


class RefSource(Enum):
    PAPER_DETAILS = 1
    SEQ_ID = 2
    GENE_ID = 3
    PUBMED_ID = 4
    OTHER = 5


def exit_handler(
    df: pd.DataFrame, output_path: str, logger: logging.log, signum, frame
):
    logger.error(
        f"closing on signal {signum} and saving temporary output to {output_path}"
    )
    df.to_csv(output_path, index=False)
    exit(0)


def parallelize(
    df: pd.DataFrame, func: partial.func, output_path: str, num_of_processes: int = 1,
):
    df_split = np.array_split(df, num_of_processes)
    pool = Pool(num_of_processes)
    df = pd.concat(pool.map(partial(func, output_path), df_split))
    df.to_csv(output_path, index=False)
    pool.close()
    pool.join()
    return df


def run_on_subset(func: partial.func, df_subset: pd.DataFrame, output_path: str):
    df = df_subset.apply(func, axis=1)
    df.to_csv(output_path, index=False)
    return df


def parallelize_on_rows(
    df: pd.DataFrame, func: partial.func, output_path: str, num_of_processes: int = 1,
):
    return parallelize(
        df, partial(run_on_subset, func, output_path), output_path, num_of_processes,
    )


def handle_duplicated_columns(colname: str, df: pd.DataFrame, logger: logging.log):
    """
    :param colname: name of the column that was duplicated as a result of the merge
    :param df: dataframe to remove duplicated columns from
    :param logger: logger instance
    :return:
    """
    duplicated_columns = [col for col in df.columns if colname in col]
    main_colname = duplicated_columns[0]
    for col in duplicated_columns[1:]:
        # check that there are no contradictions
        contradictions = df.loc[
            (
                (not isinstance(df[main_colname].iloc[0], np.floating))
                & (type(df[main_colname].iloc[0]) is not float)
            )
            & (type(df[col].iloc[0]) not in [np.floating, float])
            & (df[main_colname] != df[main_colname])
        ]
        if contradictions.shape[0] > 0:
            logger.error(
                f"Contradictions found between column values in {main_colname} and {col}"
            )
            raise ValueError(
                f"Contradictions found between column values in {main_colname} and {col}"
            )
        df[main_colname].fillna(df[col], inplace=True)
    df.rename(columns={main_colname: colname}, inplace=True)
    for col in duplicated_columns[1:]:
        df.drop(col, axis="columns", inplace=True)


def get_references(
    record: pd.Series, references_field: str, ref_to_doi: t.Dict[str, list]
) -> t.Optional[str]:
    """
    :param record: data record in the form od a pandas series
    :param references_field: name of the column which holds references data
    :param ref_to_doi: dictionary mapping items in the references field to dois
    :return:
    """
    references = record[references_field]
    if type(references) is float and np.isnan(references):
        return references
    if type(references) is str:
        references = [references]
    dois_united = []
    for reference in references:
        if reference in ref_to_doi:
            dois_united += ref_to_doi[reference]
    return ",".join(dois_united)


def collect_dois(
    df: pd.DataFrame,
    output_field_name: str,
    references_field: str,
    source_type: RefSource,
    logger: logging.log,
    output_path: str,
    ncpus: int = 1,
):
    """
    :param df: dataframe to add a references by DOIs columns to
    :param output_field_name: the name of the added field
    :param references_field: a list of fields by which be extracted
    :param source_type: the type of query to conduct to ge the doi
    :param logger instance
    :param output_path: path to write temporary output to
    :param ncpus: number of cpus available for run, in case of parallelization
    :return: none
    """

    # set signal handling
    signal.signal(
        signal.SIGINT, partial(exit_handler, df, output_path),
    )
    signal.signal(
        signal.SIGTERM, partial(exit_handler, df, output_path),
    )

    if source_type != RefSource.PAPER_DETAILS:
        df.loc[df[references_field].notnull(), references_field] = df.loc[
            df[references_field].notnull(), references_field
        ].apply(lambda x: re.split(",|;", str(x)))
    num_records = 0
    for chunk in np.array_split(df, (len(df.index) + 2) / 42):
        num_records += chunk.shape[0]
        if source_type != RefSource.PAPER_DETAILS:
            references = set([y for x in chunk[references_field].dropna() for y in x])
        else:
            references = set([x for x in chunk[references_field].dropna()])
        if len(references) == 0:
            continue
        ref_to_doi = defaultdict(list)
        refs_query = ",".join(references)
        if source_type in [
            RefSource.SEQ_ID,
            RefSource.GENE_ID,
            RefSource.PUBMED_ID,
        ]:
            try:
                db = "pubmed" if source_type == RefSource.PUBMED_ID else "nucleotide"
                getter = (
                    Entrez.esummary
                    if source_type == RefSource.PUBMED_ID
                    else Entrez.efetch
                )
                matches = [
                    record
                    for record in Entrez.read(
                        getter(
                            db=db, id=refs_query, retmode="xml", retmax=len(references),
                        )
                    )
                ]
                for match in matches:
                    doi = []
                    if source_type == RefSource.PUBMED_ID and "DOI" in match:
                        doi = [match["DOI"]]
                    elif source_type != RefSource.PUBMED_ID:
                        for ref in match["GBSeq_references"]:
                            if (
                                "GBReference_xref" in ref
                                and "GBXref_dbname" in ref["GBReference_xref"][0]
                                and ref["GBReference_xref"][0]["GBXref_dbname"] == "doi"
                            ):
                                doi.append(ref["GBReference_xref"][0]["GBXref_id"])
                    key = (
                        match["Id"]
                        if source_type == RefSource.PUBMED_ID
                        else (
                            match["GBSeq_accession-version"]
                            if source_type == RefSource.SEQ_ID
                            else [
                                s.split("|")[-1]
                                for s in match["GBSeq_other-seqids"]
                                if "gi|" in s
                            ][0]
                        )
                    )
                    for item in doi:
                        ref_to_doi[key].append(item)
            except Exception as e:
                logger.error(
                    f"failed to extract doi from references {refs_query} by {source_type.name} due to error {e}"
                )
        elif source_type == RefSource.PAPER_DETAILS:
            cr = Crossref()
            for ref in references:
                try:
                    res = cr.works(
                        query_bibliographic=ref, limit=1
                    )  # couldn't find a batch option
                    ref_to_doi[ref].append(res["message"]["items"][0]["DOI"])
                except Exception as e:
                    logger.error(
                        f"failed to extract DOI for ref {ref} based on {source_type.name} due to error {e}"
                    )
        else:
            logger.error(
                f"No mechanism is available for extraction of DOI from source type {source_type.name}"
            )

        df.loc[
            (df.index.isin(chunk.index)) & (df[references_field].notnull()),
            output_field_name,
        ] = df.loc[
            (df.index.isin(chunk.index)) & (df[references_field].notnull())
        ].apply(
            func=lambda x: get_references(x, references_field, ref_to_doi),
        )
        df.to_csv(output_path, ignore_index=True)
        logger.info(f"Processed DOI data for {num_records} records")


def get_taxon_id(taxon_name: str, logger: logging.log) -> t.Optional[np.float64]:
    """
    :param taxon_name: name of taxon
    :param logger: logging instance
    :return: taxon id
    """
    try:
        search = Entrez.esearch(
            term=taxon_name.replace(" ", "+"), db="taxonomy", retmode="xml", retmax=1,
        )
        data = Entrez.read(search)
        if len(data["IdList"]) == 0:
            logger.error(f"did not find a taxon id matching name {taxon_name}")
        return np.float64(data["IdList"][0])
    except Exception as e:
        logger.error(
            f"failed to retrieve id for taxon name {taxon_name} due to error {e}"
        )
        return np.nan


def collect_taxonomy_data(
    df: pd.DataFrame,
    taxonomy_data_prefix: str,
    output_dir: str,
    logger: logging.log,
    ncpus: int = 1,
) -> pd.DataFrame:
    """
    :param df: dataframe holding taxon names (and possibly ids) by which taxon data should be extracted
    :param taxonomy_data_prefix: prefix of taxon_name and taxon_id to search by and add new columns by (either "virus_" or "host_" in this context)
    :param logger: logging instance
    :param output_dir: directory to write temporary output to in case of signals
    :param ncpus: number of required cpus
    :return: the updated dataframe
    """
    taxon_name_field = f"{taxonomy_data_prefix}_taxon_name"
    taxon_id_field = f"{taxonomy_data_prefix}_taxon_id"
    df.set_index(taxon_id_field)
    i = 0
    num_records = 0
    batches_num = int((len(df.index) + 2) / 5)
    if batches_num == 0:
        batches_num += 1
    for chunk in np.array_split(df, batches_num):

        # define output path for signal handling
        num_records += chunk.shape[0]
        i += 1
        output_path = f"{output_dir}/collect_taxonomy_data_chunk_{i}.csv"
        signal.signal(
            signal.SIGINT, partial(exit_handler, chunk, output_path),
        )
        signal.signal(
            signal.SIGTERM, partial(exit_handler, chunk, output_path),
        )

        try:

            # fill in missing taxon ids
            taxa_name_to_id = pd.DataFrame(
                {
                    taxon_name_field: chunk.loc[
                        pd.isnull(chunk[taxon_id_field]), taxon_name_field
                    ].unique()
                }
            )

            taxa_name_to_id[taxon_id_field] = taxa_name_to_id.apply(
                func=lambda x: get_taxon_id(
                    taxon_name=x[taxon_name_field], logger=logger,
                ),
                axis=1,
            )

            logger.info(
                f"Collected taxonomy ids for {num_records} records successfully on field {taxon_id_field}"
            )

            chunk.set_index(taxon_name_field, inplace=True)
            taxa_name_to_id.set_index(taxon_name_field, inplace=True)
            chunk.fillna(taxa_name_to_id, inplace=True)

            # extract taxa data in batch
            taxa_ids = [
                str(int(item)) for item in chunk[taxon_id_field].dropna().unique()
            ]
            if len(taxa_ids) == 0:
                continue

            query = ",".join(taxa_ids)
            try:
                taxa_data = list(
                    Entrez.read(
                        Entrez.efetch(
                            id=query,
                            db="taxonomy",
                            retmode="xml",
                            retmax=len(taxa_ids),
                        )
                    )
                )
                for taxon_data in taxa_data:
                    try:
                        taxon_id = np.float64(taxon_data["TaxId"])
                        chunk.loc[
                            chunk[taxon_id_field] == taxon_id,
                            f"{taxonomy_data_prefix}_taxon_rank",
                        ] = taxon_data["Rank"]
                        for item in taxon_data["LineageEx"]:
                            chunk.loc[
                                chunk[taxon_id_field] == taxon_id,
                                f"{taxonomy_data_prefix}_{item['Rank']}_id",
                            ] = np.int64(item["TaxId"])
                            chunk.loc[
                                chunk[taxon_id_field] == taxon_id,
                                f"{taxonomy_data_prefix}_{item['Rank']}_name",
                            ] = item["ScientificName"]
                    except Exception as e:
                        logger.error(
                            f"Parsing of taxon data {taxon_data} failed due to error {e}"
                        )
                logger.info(
                    f"Collected taxonomy data for {num_records} records successfully on ids of type {taxon_id_field}"
                )
            except Exception as e:
                logger.error(
                    f"batch query to taxonomy browser on {query} failed due to error {e}"
                )
        finally:
            new_cols = [col for col in chunk.columns if col not in df.columns]
            for col in new_cols:
                df[col] = np.nan
            df.set_index(taxon_name_field, inplace=True)
            df.fillna(chunk, inplace=True)
            df.reset_index(inplace=True)

    return df
