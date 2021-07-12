from pydantic import BaseModel
import logging
from enum import Enum
import typing as t
from collections import defaultdict

import pandas as pd
import numpy as np
import re
import math

from Bio import Entrez
from habanero import Crossref


class RefSource(Enum):
    PAPER_DETAILS = 1
    SEQ_ID = 2
    GENE_ID = 3
    PUBMED_ID = 4
    OTHER = 5


class DataCollectingUtils(BaseModel):
    Entrez.email = "halabikeren@mail.tau.ac.il"

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def collect_dois(
        df: pd.DataFrame,
        output_field_name: str,
        references_field: str,
        source_type: RefSource,
        logger: logging.log,
    ):
        """
        :param df: dataframe to add a references by DOIs columns to
        :param output_field_name: the name of the added field
        :param references_field: a list of fields by which be extracted
        :param source_type: the type of query to conduct to ge the doi
        :param logger instance
        :return: none
        """

        if source_type != RefSource.PAPER_DETAILS:
            df.loc[df[references_field].notnull(), references_field] = df.loc[
                df[references_field].notnull(), references_field
            ].apply(lambda x: re.split(",|;", str(x)))
        for chunk in np.array_split(df, (len(df.index) + 2) / 42):
            if source_type != RefSource.PAPER_DETAILS:
                references = set(
                    [y for x in chunk[references_field].dropna() for y in x]
                )
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
                    db = (
                        "pubmed" if source_type == RefSource.PUBMED_ID else "nucleotide"
                    )
                    getter = (
                        Entrez.esummary
                        if source_type == RefSource.PUBMED_ID
                        else Entrez.efetch
                    )
                    matches = [
                        record
                        for record in Entrez.read(
                            getter(
                                db=db,
                                id=refs_query,
                                retmode="xml",
                                retmax=len(references),
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
                                    and ref["GBReference_xref"][0]["GBXref_dbname"]
                                    == "doi"
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
            ] = df.apply(
                lambda x: DataCollectingUtils.get_references(
                    x, references_field, ref_to_doi
                ),
                axis=1,
            )

    @staticmethod
    def get_taxon_id(taxon_name: str, logger: logging.log) -> t.Optional[np.float64]:
        """
        :param taxon_name: name of taxon
        :param logger: logging instance
        :return: taxon id
        """
        try:
            search = Entrez.esearch(
                term=taxon_name.replace(" ", "+"),
                db="taxonomy",
                retmode="xml",
                retmax=1,
            )
            data = Entrez.read(search)
            return np.float64(data["IdList"][0])
        except Exception as e:
            logger.error(
                f"failed to retrieve id for taxon name {taxon_name} due to error {e}"
            )
            return np.nan

    @staticmethod
    def collect_taxonomy_data(
        df: pd.DataFrame, taxonomy_data_prefix: str, logger: logging.log,
    ) -> pd.DataFrame:
        """
        :param df: dataframe holding taxon names (and possibly ids) by which taxon data should be extracted
        :param taxonomy_data_prefix: prefix of taxon_name and taxon_id to search by and add new columns by (either "virus_" or "host_" in this context)
        :param logger: logging instance
        :return: the updated dataframe
        """
        taxon_name_field = f"{taxonomy_data_prefix}_taxon_name"
        taxon_id_field = f"{taxonomy_data_prefix}_taxon_id"
        for chunk in np.array_split(df, (len(df.index) + 2) / 5):

            # fill in missing taxon ids
            taxa_name_to_id = pd.DataFrame(
                {
                    taxon_name_field: chunk.loc[
                        pd.isnull(chunk[taxon_id_field]), taxon_name_field
                    ].unique()
                }
            )

            taxa_name_to_id[taxon_id_field] = taxa_name_to_id[taxon_name_field].apply(
                lambda x: DataCollectingUtils.get_taxon_id(taxon_name=x, logger=logger),
            )
            df = df.merge(taxa_name_to_id, on=[taxon_name_field], how="left")
            DataCollectingUtils.handle_duplicated_columns(
                colname=taxon_id_field, df=df, logger=logger
            )

            # extract taxa data in batch
            taxa_ids = [
                str(item)
                for item in df.loc[
                    df[taxon_name_field].isin(chunk[taxon_name_field]), taxon_id_field
                ].unique()
            ]
            if "nan" in taxa_ids:
                taxa_ids.remove("nan")
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
                        df.loc[
                            df[taxon_id_field] == taxon_id,
                            f"{taxonomy_data_prefix}_taxon_rank",
                        ] = taxon_data["Rank"]
                        for item in taxon_data["LineageEx"]:
                            df.loc[
                                df[taxon_id_field] == taxon_id,
                                f"{taxonomy_data_prefix}_{item['Rank']}_id",
                            ] = item["TaxId"]
                            df.loc[
                                df[taxon_id_field] == taxon_id,
                                f"{taxonomy_data_prefix}_{item['Rank']}_name",
                            ] = item["ScientificName"]
                    except Exception as e:
                        logger.error(
                            f"Parsing of taxon data {taxon_data} failed due to error {e}"
                        )
            except Exception as e:
                logger.error(
                    f"batch query to taxonomy browser on {query} failed due to error {e}"
                )
        return df
