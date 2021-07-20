import logging
import os
import time

logger = logging.getLogger(__name__)
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


def exit_handler(df: pd.DataFrame, output_path: str, signum, frame):
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


# this is less efficient than merge + drop duplicated columns
def my_fillna(
    df_to_fill: pd.DataFrame,
    df_by_fill: pd.DataFrame,
    by_column: str,
    to_fill_columns: t.List[str],
) -> pd.DataFrame:
    """
    :param df_to_fill: dataframe with missing values in the by columns
    :param df_by_fill: dataframe with possibly some values to fill in the by columns
    :param by_column: name of column that can be used as index. must be shared between df_to_fill and df_by_fill
    :param to_fill_columns: columns with missing data
    :return: dataframe where missing data that was available in df_by_fill is now available
    """
    for column in to_fill_columns:
        df_to_fill.loc[df_to_fill[column].isnull(), column] = df_to_fill.loc[
            df_to_fill[column].isnull(), by_column
        ].apply(
            lambda x: df_by_fill.loc[df_by_fill[by_column] == x, column].values[0]
            if df_by_fill.loc[df_by_fill[by_column] == x, column].shape[0] > 0
            else np.nan
        )
    return df_to_fill


def handle_duplicated_columns(colname: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    :param colname: name of the column that was duplicated as a result of the merge
    :param df: dataframe to remove duplicated columns from
    :return:
    """
    duplicated_columns = [
        col for col in df.columns if colname in col and col != colname
    ]
    if len(duplicated_columns) == 0:
        return df
    main_colname = duplicated_columns[0]
    for col in duplicated_columns[1:]:
        # check that there are no contradictions
        contradictions = df.dropna(subset=[main_colname, col], how="any", inplace=False)
        contradictions = contradictions.loc[
            contradictions[main_colname] != contradictions[col]
        ]
        if contradictions.shape[0] > 0:
            logger.error(
                f"{contradictions.shape[0]} contradictions found between column values in {main_colname} and {col}. original column values will be overridden"
            )
            df.loc[df.index.isin(contradictions.index), main_colname] = df.loc[
                df.index.isin(contradictions.index), main_colname
            ].apply(
                lambda x: contradictions.loc[
                    contradictions[main_colname] == x, col
                ].values[0]
            )
        df[main_colname] = df[main_colname].fillna(df[col])
    df = df.rename(columns={main_colname: colname})
    for col in duplicated_columns[1:]:
        df = df.drop(col, axis="columns")
    return df


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
    output_path: str,
):
    """
    :param df: dataframe to add a references by DOIs columns to
    :param output_field_name: the name of the added field
    :param references_field: a list of fields by which be extracted
    :param source_type: the type of query to conduct to ge the doi
    :param output_path: path to write temporary output to
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


def collect_taxonomy_data(df: pd.DataFrame, taxonomy_data_dir: str,) -> pd.DataFrame:
    """
    :param df: dataframe holding taxon names (and possibly ids) by which taxon data should be extracted
    :param taxonomy_data_dir: directory holding dump files of the NCBI taxonomy FTP services https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/
    :return: the processed dataframe
    """

    output_path = f"{os.getcwd()}/collect_taxonomy_data.csv"

    # set signal handling
    signal.signal(
        signal.SIGINT, partial(exit_handler, df, output_path),
    )
    signal.signal(
        signal.SIGTERM, partial(exit_handler, df, output_path),
    )

    df = df.applymap(
        lambda s: s.lower() if isinstance(s, str) else s
    )  # make all strings lowercase to account for inconsistency between databases

    logger.info("complementing missing virus and host taxon ids from names.dmp")
    taxonomy_names_df = pd.read_csv(
        f"{taxonomy_data_dir}/names.dmp",
        sep="|",
        header=None,
        index_col=False,
        names=["tax_id", "name_txt", "unique_name", "class_name"],
    )
    taxonomy_names_df.replace(to_replace="\t", value="", regex=True, inplace=True)
    taxonomy_names_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
    taxonomy_names_df = taxonomy_names_df.applymap(
        lambda s: s.lower() if isinstance(s, str) else s
    )
    logger.info(
        f"#missing virus taxon ids before addition= {df.loc[df.virus_taxon_id.isna()].shape[0]}"
    )
    virus_taxon_names_df = taxonomy_names_df.loc[
        taxonomy_names_df.name_txt.isin(df.virus_taxon_name.unique())
    ][["tax_id", "name_txt"]]
    df.set_index(["virus_taxon_name"], inplace=True)
    df["virus_taxon_id"].fillna(
        value=virus_taxon_names_df.set_index("name_txt")["tax_id"].to_dict(),
        inplace=True,
    )
    df.reset_index(inplace=True)
    logger.info(
        (
            f"#missing virus taxon ids after addition = {df.loc[df.virus_taxon_id.isna()].shape[0]}"
        )
    )

    logger.info(
        (
            f"#missing host taxon ids before addition = {df.loc[df.host_taxon_id.isna()].shape[0]}"
        )
    )
    host_taxon_names_df = taxonomy_names_df.loc[
        taxonomy_names_df.name_txt.isin(df.host_taxon_name.unique())
    ][["tax_id", "name_txt"]]
    df.set_index(["host_taxon_name"], inplace=True)
    df["host_taxon_id"].fillna(
        value=host_taxon_names_df.set_index("name_txt")["tax_id"].to_dict(),
        inplace=True,
    )
    df.reset_index(inplace=True)
    logger.info(
        (
            f"#missing host taxon ids after addition = {df.loc[df.host_taxon_id.isna()].shape[0]}"
        )
    )
    df.reset_index(inplace=True)

    # fill in virus and host lineage info
    logger.info(
        "complementing missing virus and host taxon lineage info from rankedlineage.dmp"
    )
    logger.info(f"# missing cells before = {df.isnull().sum().sum()}")
    taxonomy_lineage_df = pd.read_csv(
        f"{taxonomy_data_dir}/rankedlineage.dmp",
        sep="|",
        header=None,
        index_col=False,
        names=[
            "tax_id",
            "tax_name",
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
            "superkingdom",
        ],
        dtype={"tax_id": np.float64},
    )
    taxonomy_lineage_df.replace(to_replace="\t", value="", regex=True, inplace=True)
    taxonomy_lineage_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
    taxonomy_lineage_df = taxonomy_lineage_df.applymap(
        lambda s: s.lower() if isinstance(s, str) else s
    )

    virus_taxonomy_lineage_df = taxonomy_lineage_df.loc[
        taxonomy_lineage_df.tax_name.isin(df.virus_taxon_name.unique())
    ].rename(
        columns={
            col: f"virus_{col.replace('tax_id', 'taxon').replace('tax_name', 'taxon')}_{'id' if 'id' in col else 'name'}"
            for col in taxonomy_lineage_df.columns
        },
    )
    df.set_index(["virus_taxon_name"], inplace=True)
    virus_taxonomy_lineage_df.set_index(["virus_taxon_name"], inplace=True)
    for col in virus_taxonomy_lineage_df.columns:
        if col in df.columns:
            values = virus_taxonomy_lineage_df[col].to_dict()
            df[col].fillna(value=values, inplace=True)
    df.reset_index(inplace=True)
    df.set_index(["virus_species_name"], inplace=True)
    for col in virus_taxonomy_lineage_df.columns:
        if col in df.columns:
            values = virus_taxonomy_lineage_df[col].to_dict()
            df[col].fillna(value=values, inplace=True)
    df.reset_index(inplace=True)

    host_taxonomy_lineage_df = taxonomy_lineage_df.loc[
        taxonomy_lineage_df.tax_name.isin(df.host_taxon_name.unique())
    ].rename(
        columns={
            col: f"host_{col.replace('tax_id', 'taxon').replace('tax_name', 'taxon')}_{'id' if 'id' in col else 'name'}"
            for col in taxonomy_lineage_df.columns
        },
    )
    host_taxonomy_lineage_df["host_is_mammalian"] = host_taxonomy_lineage_df[
        "host_class_name"
    ].apply(lambda x: 1 if x == "mammalia" else 0)
    df.set_index(["host_taxon_name"], inplace=True)
    host_taxonomy_lineage_df.set_index(["host_taxon_name"], inplace=True)
    for col in host_taxonomy_lineage_df.columns:
        if col in df.columns:
            values = host_taxonomy_lineage_df[col].to_dict()
            df[col].fillna(value=values, inplace=True)
    df.reset_index(inplace=True)

    logger.info(f"# missing cells after = {df.isnull().sum().sum()}")

    # fill rank of virus and host taxa
    logger.info("extracting rank of virus and host taxa")
    logger.info(f"# missing cells before = {df.isnull().sum().sum()}")
    taxonomy_ranks_df = pd.read_csv(
        f"{taxonomy_data_dir}/nodes.dmp",
        sep="|",
        header=None,
        index_col=False,
        names=[
            "tax_id",
            "parent_tax_id",
            "rank",
            "embl_code",
            "division_id",
            "inherited_div_flag",
            "genetic_code_id",
            "inherited_GC_flag",
            "mitochondrial_genetic_code_id",
            "inherited_MGC_flag",
            "GenBank_hidden_flag",
            "hidden_subtree_root_flag",
            "comments",
            "plastid_genetic_code_id",
            "inherited_PGC_flag",
            "specified_species",
            "hydrogenosome_genetic_code_id",
            "inherited_HGC_flag",
        ],
    )
    taxonomy_ranks_df = taxonomy_ranks_df.applymap(
        lambda s: s.lower() if isinstance(s, str) else s
    )
    taxonomy_ranks_df.replace(to_replace="\t", value="", regex=True, inplace=True)
    taxonomy_ranks_df.replace(to_replace="", value=np.nan, regex=True, inplace=True)
    virus_rank_df = taxonomy_ranks_df.loc[
        taxonomy_ranks_df.tax_id.isin(df.virus_taxon_id.unique())
    ]
    df["virus_taxon_rank"] = np.nan
    df.set_index(["virus_taxon_id"], inplace=True)
    values = virus_rank_df.set_index("tax_id")["rank"].to_dict()
    df["virus_taxon_rank"].fillna(value=values, inplace=True)
    df.reset_index(inplace=True)

    host_rank_df = taxonomy_ranks_df.loc[
        taxonomy_ranks_df.tax_id.isin(df.host_taxon_id.unique())
    ]
    df["host_taxon_rank"] = np.nan
    df.set_index(["host_taxon_id"], inplace=True)
    values = host_rank_df.set_index("tax_id")["rank"].to_dict()
    df["host_taxon_rank"].fillna(value=values, inplace=True)
    logger.info(f"# missing cells after = {df.isnull().sum().sum()}")
    df.reset_index(inplace=True)

    df.loc[df.virus_strain_name.notnull(), "virus_is_species"] = 0
    df.loc[df.virus_strain_name.isnull(), "virus_is_species"] = 1
    df.loc[
        (df.virus_species_name.isnull()) & (df.virus_is_species == 1),
        "virus_species_name",
    ] = df.loc[
        (df.virus_species_name.isnull()) & (df.virus_is_species == 1),
        "virus_taxon_name",
    ]
    df.loc[df.virus_strain_name.notnull(), "virus_taxon_rank"] = "strain"
    df = df[[col for col in df if "_id" not in col and col != "index"]]

    return df
