import re
import signal
import typing as t

import logging
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
from Bio import Entrez
from habanero import Crossref

logger = logging.getLogger(__name__)
from enum import Enum

from .signal_handling_service import SignalHandlingService


class RefSource(Enum):
    PAPER_DETAILS = 1
    SEQ_ID = 2
    GENE_ID = 3
    PUBMED_ID = 4
    OTHER = 5


class ReferenceCollectingUtils:
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
            signal.SIGINT, partial(SignalHandlingService.exit_handler, df, output_path),
        )
        signal.signal(
            signal.SIGTERM,
            partial(SignalHandlingService.exit_handler, df, output_path),
        )

        if source_type != RefSource.PAPER_DETAILS:
            df.loc[df[references_field].notnull(), references_field] = df.loc[
                df[references_field].notnull(), references_field
            ].apply(lambda x: re.split(",|;", str(x)))
        num_records = 0
        for chunk in np.array_split(df, (len(df.index) + 2) / 42):
            num_records += chunk.shape[0]
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
            ] = df.loc[
                (df.index.isin(chunk.index)) & (df[references_field].notnull())
                ].apply(
                func=lambda x: ReferenceCollectingUtils.get_references(
                    x, references_field, ref_to_doi
                ),
            )
            df.to_csv(output_path, ignore_index=True)
            logger.info(f"Processed DOI data for {num_records} records")

    @staticmethod
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
